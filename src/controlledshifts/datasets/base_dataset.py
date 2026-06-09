"""Base dataset loader class for the Trajectory Forecasting Datasets."""

import json
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from functools import cache
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from characterization.features.safeshift_features import SafeShiftFeatures
from characterization.schemas import (
    AgentData,
    Scenario,
    ScenarioMetadata,
    ScenarioScores,
    StaticMapData,
    TracksToPredict,
)
from characterization.scorer.safeshift_scorer import SafeShiftScorer
from characterization.utils.common import AgentTrajectoryMasker, AgentType
from characterization.utils.geometric_utils import find_closest_lanes, find_conflict_points
from omegaconf import DictConfig
from torch.utils.data import Dataset

from controlledshifts.utils import data_utils, pylogger
from controlledshifts.utils.constants import LARGE_FLOAT, DataSplits, SampleSelection


_LOGGER = pylogger.get_pylogger(__name__)


class BaseDataset(Dataset, ABC):
    """Base dataset loader class for trajectory datasets.

    This class handles loading and processing of scenario data, including caching, multiprocessing, and transformation
    into an agent-centric format. Subclasses should implement the `load_as_open_scenario` method to load scenarios in
    their specific format.
    """

    def __init__(self, config: DictConfig) -> None:
        """Dataset constructor.

        Args:
            config (DictConfig): Configuration object containing dataset parameters.
        """
        super().__init__()

        self.split = config.split
        self.config = config
        self.num_data_to_consider = config.num_data_to_consider
        self.past_len = config.past_len
        self.future_len = config.future_len
        self.current_time_idx = self.past_len - 1
        self.total_steps = self.past_len + self.future_len
        self.subset_data_tag = None

        # Scenario characterization
        self.autolabel_agents = self.config.get("autolabel_agents", False)
        self.conflict_points_config = self.config.get("conflict_points", None)
        self.closest_lanes_config = self.config.get("closest_lanes", None)

        self.scenario_features_processor, self.scenario_scores_processor = None, None
        scenario_characterization_config = self.config.get("scenario_characterization", None)
        if self.autolabel_agents:
            if self.conflict_points_config is None or self.closest_lanes_config is None:
                error_message = "autolabel_agents is True but conflict_points or closest_lanes config is missing."
                raise ValueError(error_message)

            if scenario_characterization_config is None:
                error_message = "autolabel_agents is True but scenario_characterization config is missing."
                raise ValueError(error_message)

            self.scenario_features_processor = SafeShiftFeatures(scenario_characterization_config)
            self.scenario_scores_processor = SafeShiftScorer(scenario_characterization_config)

        if self.config.load_data:
            match self.split:
                case DataSplits.TRAINING:
                    self.data_path = config.train_data_path
                    self.data_tag = config.train_data_tag
                case DataSplits.VALIDATION:
                    self.data_path = config.val_data_path
                    self.data_tag = config.val_data_tag
                case DataSplits.TESTING:
                    self.data_path = config.test_data_path
                    self.data_tag = config.test_data_tag
                case _:
                    error_message = f"Unsupported split value: {self.split}"
                    raise ValueError(error_message)

            if len(self.data_path) != len(self.data_tag):
                error_message = f"There should be tag {self.data_tag} for each dataset path {self.data_path}."
                raise ValueError(error_message)

            self.data_loaded_in_memory = []
            self.file_cache = {}
            self.file_list = config.get("filelist_cache", "file_list.pkl")

            self.blacklist = []
            sample_selection_strategy = SampleSelection(self.config.sample_selection_strategy)
            if sample_selection_strategy != SampleSelection.ALL:
                sample_selection_filepath = Path(self.config.sample_selection_filepath)
                with sample_selection_filepath.open("r") as f:
                    selected_samples = json.load(f)
                    self.blacklist = selected_samples["drop"]
                    _LOGGER.info("Removing %s  samples using %s", len(self.blacklist), sample_selection_strategy)
            self.load_data()

    @abstractmethod
    def load_as_open_scenario(self, path: Path) -> Scenario:
        """Load the given path and return it as a Scenario object."""

    def _get_dataset_summary(self, data_path: str) -> dict[str, str]:
        """Get a summary of filenames and paths for the pickle dataset.

        Args:
            data_path (str): Path to the directory containing the pickle dataset.

        Returns:
            dict[str, str]: A dictionary mapping filenames to their full paths.
        """
        mapping = {filepath.name: str(filepath) for filepath in Path(data_path).rglob("*.pkl")}
        _LOGGER.info("Got %s scenarios.", len(mapping))
        return mapping

    def load_data(self) -> None:
        """Loads and processes scenario data into N chunks."""
        _LOGGER.info("Loading %s data...", self.split)
        self.data_loaded = {}
        for data_path, data_tag in zip(self.data_path, self.data_tag, strict=False):
            # NOTE: self.subset_data_tag will be used in repack-scenario to add a subset tag to the scenario.
            self.subset_data_tag = data_tag
            self.cache_path = Path(self.config.cache_path, self.subset_data_tag)
            _LOGGER.info("Cache path %s", self.cache_path)

            if self.config.use_cache or data_utils.is_ddp():
                file_list = self.get_data_list()
            elif Path(self.cache_path).exists() and self.config.get("overwrite_cache", False) is False:
                _LOGGER.warning("Cache path %s already exists, skip ", self.cache_path)
                file_list = self.get_data_list()
            else:
                _LOGGER.info("Creating cache...")
                mapping = self._get_dataset_summary(data_path)
                if self.cache_path.exists():
                    shutil.rmtree(self.cache_path)
                self.cache_path.mkdir(parents=True, exist_ok=True)

                cpu_count = os.cpu_count()
                process_num = (
                    cpu_count // 2 if (cpu_count is not None and cpu_count > 1 and len(mapping) > cpu_count) else 1
                )
                _LOGGER.info("Using %s processes to load data...", process_num)

                summary_splits = np.array_split(list(mapping.keys()), process_num)
                data_splits = [
                    (data_path, mapping, filenames_split, self.subset_data_tag) for filenames_split in summary_splits
                ]

                # Save data_splits in a temporary directory.
                Path(self.config.temp_path).mkdir(parents=True, exist_ok=True)
                for i, data_split in enumerate(data_splits):
                    with Path(self.config.temp_path, f"{i}.pkl").open("wb") as f:
                        pickle.dump(data_split, f)

                with Pool(processes=process_num) as pool:
                    results = pool.map(self.process_data_chunk, list(range(process_num)))
                # Concatenate the results.
                file_list = {}
                for result in results:
                    file_list.update(result)

                filelist_cache = Path(self.cache_path, self.file_list)
                with filelist_cache.open("wb") as f:
                    pickle.dump(file_list, f)

                data_list = list(file_list.items())
                np.random.shuffle(data_list)  # noqa: NPY002
                if self.split == DataSplits.TRAINING:
                    # Randomly sample the configured number of scenarios.
                    file_list = dict(data_list[: self.num_data_to_consider])

            num_scenarios = len(file_list)
            _LOGGER.info("Loaded %s samples from %s", num_scenarios, data_path)

            # Remove scenarios in specified blacklist. Needed for sample-selection experiments and to avoid creating a
            # different cache for each type of selection experiment.
            file_list_post_blacklist = {}
            for file_id, file_info in file_list.items():
                if file_info["scenario_id"][0] in self.blacklist:
                    continue
                file_list_post_blacklist[file_id] = file_info

            num_scenarios_post_blacklist = len(file_list_post_blacklist)
            num_removed_scenarios = num_scenarios - num_scenarios_post_blacklist
            _LOGGER.info("Total instances: %s (removed: %s)", num_scenarios_post_blacklist, num_removed_scenarios)
            if not num_scenarios_post_blacklist:
                err_msg = "No scenarios left after applying blacklist"
                raise RuntimeError(err_msg)

            self.data_loaded.update(file_list_post_blacklist)

            if self.config.store_data_in_memory:
                _LOGGER.info("Loading data into memory...")
                for data_filepath in file_list_post_blacklist:
                    with Path(data_filepath).open("rb") as f:
                        data = pickle.load(f)
                    self.data_loaded_memory.append(data)
                _LOGGER.info("Loaded %s data into memory", len(self.data_loaded_memory))

        self.data_loaded_keys = list(self.data_loaded.keys())
        _LOGGER.info("Data loaded")

    def process_data_chunk(self, worker_index: int) -> dict[str, dict[str, Any]]:
        """Processes scenario data for a given chunk index.

        Args:
            worker_index (int): The index of the worker process for which to process the data chunk.

        Returns:
            dict[str, dict[str, Any]]: A dictionary containing the processed scenario data.
        """
        with Path(self.config.temp_path, f"{worker_index}.pkl").open("rb") as f:
            data_path, mapping, data_list, dataset_name = pickle.load(f)

        file_list = {}
        hdf5_path = Path(self.cache_path, f"{worker_index}.h5")

        with h5py.File(hdf5_path, "w") as f:
            for cnt, filename in enumerate(data_list):
                if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                    _LOGGER.info("%s/%s data processed", cnt, len(data_list))

                output = self.load_and_process_scenario(Path(mapping[filename]))
                if output is None:
                    continue

                for i, record in enumerate(output):
                    grp_name = dataset_name + "-" + str(worker_index) + "-" + str(cnt) + "-" + str(i)
                    grp = f.create_group(grp_name)
                    for key, value in record.items():
                        if value is None:
                            continue
                        if isinstance(value, str):
                            value = np.bytes_(value)  # noqa: PLW2901
                        grp.create_dataset(key, data=value)
                    file_info = {}
                    kalman_difficulty = np.stack([x["kalman_difficulty"] for x in output])
                    file_info["scenario_id"] = [x["scenario_id"].item() for x in output]
                    file_info["kalman_difficulty"] = kalman_difficulty
                    file_info["h5_path"] = hdf5_path
                    file_list[grp_name] = file_info
                del output
        return file_list

    def compute_scenario_map_metadata(self, scenario: Scenario) -> Scenario:
        """Computes map-related metadata for the scenario, such as conflict points and closest lanes.

        Args:
            scenario (Scenario): The input scenario for which to compute map metadata.

        Returns:
            Scenario: The scenario with updated map metadata.
        """
        # Compute conflict point information from static map data
        conflict_points_info = find_conflict_points(
            scenario,
            resample_factor=self.conflict_points_config.get("resample_factor", 1),
            intersection_threshold=self.conflict_points_config.get("intersection_threshold", 0.5),
            return_static_conflict_points=self.conflict_points_config.get("return_static_conflict_points", False),
            return_lane_conflict_points=self.conflict_points_config.get("return_lane_conflict_points", False),
            return_dynamic_conflict_points=self.conflict_points_config.get("return_dynamic_conflict_points", False),
        )
        agent_distances_to_conflict_points, conflict_points = None, None
        if conflict_points_info is not None:
            agent_distances_to_conflict_points = (
                None
                if conflict_points_info["agent_distances_to_conflict_points"] is None
                else conflict_points_info["agent_distances_to_conflict_points"][:, : self.total_steps, :]
            )
            conflict_points = (
                None
                if conflict_points_info["all_conflict_points"] is None
                else conflict_points_info["all_conflict_points"]
            )
            scenario.static_map_data.map_conflict_points = conflict_points
            scenario.static_map_data.agent_distances_to_conflict_points = agent_distances_to_conflict_points

        # Compute closest lane information
        closest_lanes_info = find_closest_lanes(
            scenario,
            k_closest=self.closest_lanes_config.get("num_lanes", 16),
            threshold_distance=self.closest_lanes_config.get("threshold_distance", 10.0),
            subsample_factor=self.closest_lanes_config.get("subsample_factor", 2),
        )
        if closest_lanes_info is not None:
            agent_closest_lanes = closest_lanes_info["agent_closest_lanes"][:, : self.total_steps, :, :]
            scenario.static_map_data.agent_closest_lanes = agent_closest_lanes

        return scenario

    def load_and_process_scenario(self, path: Path) -> list[dict[str, Any]] | None:
        """Process a scenario into an agent-centric format.

        Args:
            path (Path): The file path to the scenario data to be loaded and processed.

        Returns:
            list[dict[str, Any]] | None: A list of dictionaries containing the processed scenario data in agent-centric
                format, or None if processing fails.
        """
        scenario = self.load_as_open_scenario(path)
        # TODO: Resolve bare except from UniTraj.
        try:
            scenario_scores = None
            if self.autolabel_agents:
                scenario = self.compute_scenario_map_metadata(scenario)
                scenario_features = self.scenario_features_processor.compute(scenario)
                scenario_scores = self.scenario_scores_processor.compute(scenario, scenario_features)

            # Process intermediate format into final format.
            # NOTE: Currently, this is UniTraj's scenario representation. It returns a dictionary, not a schema.
            ac_scenario = self.process_agent_centric_scenario(scenario, scenario_scores=scenario_scores)
            if ac_scenario is not None:
                # Compute UniTraj's characterizations: Kalman Difficulty and Trajectory Type features.
                # NOTE: Currently, these are derived from the agent-centric scenario representation. Later, they will
                # be moved to ScenarioCharacterization.
                ac_scenario = self.characterize_scenario(ac_scenario)

        except Exception:
            _LOGGER.exception("error processing scenario: %s", path)
            ac_scenario = None
        return ac_scenario

    def get_data_list(self) -> dict[str, Any]:
        """Get the data list if it has already been preprocessed into a file_list cache."""
        filelist_cache = Path(self.cache_path, self.file_list)
        if filelist_cache.exists():
            data_loaded = pickle.load(filelist_cache.open("rb"))
        else:
            error_message = f"Error: {filelist_cache} not found."
            raise ValueError(error_message)

        data_list = list(data_loaded.items())
        np.random.shuffle(data_list)  # noqa: NPY002
        return dict(data_list[: self.num_data_to_consider])

    def process_agent_centric_scenario(
        self, scenario: Scenario, scenario_scores: ScenarioScores | None = None
    ) -> list[dict[str, Any]] | None:
        """Processes a scenario from an internal format into an agent-centric format.

        Args:
            scenario (Scenario): The input scenario in internal format to be transformed into agent-centric format.
            scenario_scores (ScenarioScores | None): optional scenario scores to be added to the agent-centric format.

        Returns:
            list[dict[str, Any]] | None: A list of dictionaries containing the processed scenario data in agent-centric
                format, or None if processing fails.
        """
        agent_data = scenario.agent_data
        tracks_to_predict = scenario.tracks_to_predict
        metadata = scenario.metadata

        # Process the trajectory information
        center_objects, track_index_to_predict = self.get_agents_of_interest_center_points(
            agent_data=agent_data, tracks_to_predict=tracks_to_predict, metadata=metadata
        )
        if center_objects is None:
            return None

        # Create return dict with agent data
        ret_dict = self.get_centered_agent_data(
            agent_data=agent_data,
            center_objects=center_objects,
            track_index_to_predict=track_index_to_predict,
            metadata=metadata,
            scenario_scores=scenario_scores,
        )

        # Add center objects and scenario information
        scenario_id = metadata.scenario_id
        ret_dict["scenario_id"] = np.array([scenario_id] * len(track_index_to_predict))
        ret_dict["center_objects_world"] = center_objects
        ret_dict["center_objects_id"] = np.array(agent_data.agent_ids)[track_index_to_predict]
        agent_types_int = [agent_type.value for agent_type in agent_data.agent_types]
        ret_dict["center_objects_type"] = np.array(agent_types_int)[track_index_to_predict]
        ret_dict["center_gt_trajs_src"] = agent_data.agent_trajectories[track_index_to_predict]

        # Process the map information
        if self.config.get("manually_split_lane", False):
            map_dict = self.get_manually_split_centered_map_data(
                map_data=scenario.static_map_data, center_objects=center_objects, metadata=metadata
            )
        else:
            map_dict = self.get_centered_map_data(
                map_data=scenario.static_map_data, center_objects=center_objects, metadata=metadata
            )
        ret_dict.update(map_dict)

        # Mask out unused attributes.
        BaseDataset._mask_out_attributes(ret_dict, self.config.masked_attributes)
        # Cast NumPy arrays.
        BaseDataset._cast_dictionary(ret_dict)

        # Propagate the dataset name to each of the centered scenarios
        sample_num = center_objects.shape[0]
        ret_dict["dataset_name"] = [scenario.metadata.dataset] * sample_num

        # Batch
        scenario_list = []
        for i in range(sample_num):
            ret_dict_i = {}
            for k, v in ret_dict.items():
                # values such as individual_agent_scores can be None rather than an array
                ret_dict_i[k] = None if v is None else v[i]
            scenario_list.append(ret_dict_i)
        return scenario_list

    @staticmethod
    def _mask_out_attributes(scenario_dict: dict, attributes_to_mask: list[str]) -> None:
        """Masks out specified attributes in the scenario dictionary by setting them to zero.

        Args:
            scenario_dict (dict): The dictionary containing scenario data, which will be modified in-place.
            attributes_to_mask (list[str]): A list of attribute names to be masked out in the scenario dictionary.
        """
        if "z_axis" in attributes_to_mask:
            scenario_dict["obj_trajs"][..., 2] = 0
            scenario_dict["map_polylines"][..., 2] = 0
        if "size" in attributes_to_mask:
            scenario_dict["obj_trajs"][..., 3:6] = 0
        if "velocity" in attributes_to_mask:
            scenario_dict["obj_trajs"][..., 25:27] = 0
        if "acceleration" in attributes_to_mask:
            scenario_dict["obj_trajs"][..., 27:29] = 0
        if "heading" in attributes_to_mask:
            scenario_dict["obj_trajs"][..., 23:25] = 0

    @staticmethod
    def _cast_dictionary(
        scenario_dict: dict, from_dtype: np.dtype = np.float64, to_dtype: np.dtype = np.float32
    ) -> None:
        """Casts all NumPy arrays in the scenario dictionary from one data type to another in-place.

        Args:
            scenario_dict (dict): The dictionary containing scenario data, which will be modified in-place.
            from_dtype (np.dtype): The original data type of the arrays.
            to_dtype (np.dtype): The target data type of the arrays.
        """
        for k, v in scenario_dict.items():
            if isinstance(v, np.ndarray) and v.dtype == from_dtype:
                scenario_dict[k] = v.astype(to_dtype)

    def get_agents_of_interest_center_points(
        self, agent_data: AgentData, tracks_to_predict: TracksToPredict | None, metadata: ScenarioMetadata
    ) -> tuple[np.ndarray | None, np.ndarray | list]:
        """Get center points for agents of interest in the scenario.

        Notation:
            N: number of agents
            D: agent attributes

        Args:
            agent_data (AgentData): Object containing agent trajectory data obtained in
                `self.preprocess_scenario()`.
            tracks_to_predict (TracksToPredict | None): Object containing tracks in `agent_data` to predict.
            metadata (ScenarioMetadata): Object containing scenario metadata.

        Returns:
            agent_centerpoints (np.ndarray[N, D]): NumPy array with center points for agents of interest.
            agent_idxs (np.ndarray[N]): NumPy array with indices of agents of interest.
        """
        if not tracks_to_predict:
            return None, []
        agent_centerpoints_list = []
        agents_of_interest_idx_list = []
        selected_type = [AgentType[x] for x in self.config.object_type]

        scenario_id = metadata.scenario_id
        agents_trajectories = agent_data.agent_trajectories
        masker = AgentTrajectoryMasker(agents_trajectories)
        agent_valid = masker.agent_valid
        agents_types = agent_data.agent_types
        agents_to_predict_idxs = tracks_to_predict.track_index

        # Get the agents of interest (agents to predict) center points
        for agent_idx in agents_to_predict_idxs:
            # Check if the agent of interest is valid at the last observation index.
            if not agent_valid[agent_idx, self.current_time_idx]:
                print(f"Warning: agent={agent_idx} of scene={scenario_id} is not valid at time {self.current_time_idx}")
                continue
            # Check if the agent type is in the expected training types.
            if agents_types[agent_idx] not in selected_type:
                continue

            agent_centerpoints_list.append(agents_trajectories[agent_idx, self.current_time_idx])
            agents_of_interest_idx_list.append(agent_idx)

        if len(agent_centerpoints_list) == 0:
            print(f"Warning: no center objects at time step {self.current_time_idx}, scene_id={scenario_id}")
            return None, []

        return np.stack(agent_centerpoints_list, axis=0), np.array(agents_of_interest_idx_list)

    def get_centered_agent_data(  # noqa: PLR0915
        self,
        agent_data: AgentData,
        center_objects: np.ndarray,
        track_index_to_predict: np.ndarray,
        metadata: ScenarioMetadata,
        scenario_scores: ScenarioScores | None = None,
    ) -> dict[str, Any]:
        """Computes the agent-centric data.

        Notation:
            C: number of center objects (agents of interest)
            N: number of agents
            T: number of timesteps
            Dpost: number of attributes in the centered agent trajectories after processing

        Args:
            agent_data (AgentData): Object containing agent trajectory data obtained in `self.preprocess_scenario()`.
            center_objects (np.ndarray): NumPy array containing the center points of the agents of interest.
            track_index_to_predict (np.ndarray): NumPy array containing the indices of the agents of interest.
            metadata (ScenarioMetadata): Object containing scenario metadata.
            scenario_scores (ScenarioScores | None): optional scenario scores to be added to the agent-centric format.

        Returns:
            dict[str, Any]: A dictionary containing the processed agent-centric data for the scenario.
        """
        # Transform the agent trajectories
        center_points = AgentTrajectoryMasker(center_objects)

        # Transform the histories
        agent_trajectories = agent_data.agent_trajectories
        agent_histories_pre = AgentTrajectoryMasker(agent_trajectories[:, : self.current_time_idx + 1])
        centered_histories = BaseDataset.transform_trajectories_wrt_center_points(agent_histories_pre, center_points)

        num_center_points, num_agents, num_timesteps, num_dims = centered_histories.agent_trajectories.shape
        # Tile the agent ids to be of shape (C, N)
        agent_ids = np.array(agent_data.agent_ids)
        agent_ids = np.tile(agent_ids[None, :], (num_center_points, 1))

        # Create agent type mask (C, N, T, 5)
        agent_types = np.array(agent_data.agent_types)
        agents_onehot_type_mask = np.zeros((num_center_points, num_agents, num_timesteps, 5))
        agents_onehot_type_mask[:, agent_types == AgentType.TYPE_VEHICLE, :, 0] = 1
        agents_onehot_type_mask[:, agent_types == AgentType.TYPE_PEDESTRIAN, :, 1] = 1
        agents_onehot_type_mask[:, agent_types == AgentType.TYPE_CYCLIST, :, 2] = 1
        agents_onehot_type_mask[np.arange(num_center_points), track_index_to_predict, :, 3] = 1
        agents_onehot_type_mask[:, metadata.ego_vehicle_index, :, 4] = 1

        # Create temporal embedding (C, N, Th, Th+1)
        history_timestamps = np.array(metadata.timestamps_seconds[: self.current_time_idx + 1], dtype=np.float32)
        agents_time_embeddings = np.zeros((num_center_points, num_agents, num_timesteps, num_timesteps + 1))
        for i in range(num_timesteps):
            agents_time_embeddings[:, :, i, i] = 1
        agents_time_embeddings[:, :, :, -1] = history_timestamps

        # Create heading embedding (C, N, Th, 2)
        centered_headings = centered_histories.agent_headings.squeeze(-1)
        agent_heading_embedding = np.zeros((num_center_points, num_agents, num_timesteps, 2))
        agent_heading_embedding[:, :, :, 0] = np.sin(centered_headings)
        agent_heading_embedding[:, :, :, 1] = np.cos(centered_headings)

        # Calculate accelerations (C, N, Th, 2)
        centered_velocities = centered_histories.agent_xy_vel
        centered_velocities_pre = np.roll(centered_velocities, shift=1, axis=2)
        dt = np.pad(history_timestamps[1:] - history_timestamps[:-1], (0, 1), "mean")
        acceleration = (centered_velocities - centered_velocities_pre) / dt.reshape(1, 1, -1, 1)
        acceleration[:, :, 0, :] = acceleration[:, :, 1, :]

        # Concatenate all history features (C, N, Th, Dpost=P+D+O+Te+He+V+A)
        agent_histories = np.concatenate(
            [
                centered_histories.agent_xyz_pos,  # P=(x, y, z)
                centered_histories.agent_dims,  # D=(length, width, height)
                agents_onehot_type_mask,  # O=one-hot vector of dim=5
                agents_time_embeddings,  # Te=history embedding of dim hist-timesteps+1
                agent_heading_embedding,  # He=heading embedding of dim=2
                centered_histories.agent_xy_vel,  # V=(vx, vy)
                acceleration,  # A=(ax, ay)
            ],
            axis=-1,
        )
        # Agent history mask (C, N, Th)
        agent_histories_mask = centered_histories.agent_valid.squeeze(-1)
        agent_histories[agent_histories_mask == 0] = 0
        assert agent_trajectories.__len__() == agent_histories.shape[1]

        # Transform the futures
        agent_futures = AgentTrajectoryMasker(agent_trajectories[:, self.current_time_idx + 1 :])
        centered_futures = BaseDataset.transform_trajectories_wrt_center_points(agent_futures, center_points)

        # Agent futures (C, N, Tf, S)
        agent_futures = centered_futures.agent_state  # S=(x, y, vx, vy)
        agent_futures_mask = centered_futures.agent_valid.squeeze(-1)
        agent_futures[agent_futures_mask == 0] = 0

        # Only get the GT trajectories of the agents to predict (C, F, S)
        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = agent_futures[center_obj_idxs, track_index_to_predict]
        center_gt_trajs_mask = agent_futures_mask[center_obj_idxs, track_index_to_predict]
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        # Get mask of valid agents shape: (N)
        valid_past_mask = np.logical_not(agent_histories_pre.agent_valid.squeeze(-1).sum(axis=-1) == 0)
        # agent histories shape (C, M, Th, Dpost) # M = N - invalid
        agent_histories_mask = agent_histories_mask[:, valid_past_mask]
        agent_histories = agent_histories[:, valid_past_mask]
        # agent futures shape (C, M, Tf, 4)
        agent_futures = agent_futures[:, valid_past_mask]
        agent_futures_mask = agent_futures_mask[:, valid_past_mask]
        # agent ids shape (C, M)
        agent_ids = agent_ids[:, valid_past_mask]

        # Get the history's last valid position
        agent_histories_pos = agent_histories[:, :, :, 0:3]
        num_center_objects, num_agents, num_timestamps, _ = agent_histories_pos.shape
        agent_histories_last_pos = np.zeros((num_center_objects, num_agents, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = agent_histories_mask[:, :, k] > 0
            agent_histories_last_pos[cur_valid_mask] = agent_histories_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0
            center_gt_final_valid_idx[cur_valid_mask] = k

        # Get the context agents. Here, context agents are the agents closest to the ego-vehicle at the last observed
        # timestep
        max_num_agents = self.config.max_num_agents
        # shape: (C, M)
        agent_dists_to_center_points = np.linalg.norm(agent_histories[..., -1, 0:2], axis=-1)
        agent_dists_to_center_points[agent_histories_mask[..., -1] == 0] = LARGE_FLOAT
        # shape: (C, max_num_agents, 1, 1)
        topk_idxs = np.argsort(agent_dists_to_center_points, axis=-1)[:, :max_num_agents, None, None]

        # Get the information from the topk_idxs
        agent_ids = np.take_along_axis(agent_ids[..., None, None], topk_idxs, axis=1)
        agent_histories = np.take_along_axis(agent_histories, topk_idxs, axis=1)
        agent_histories_mask = np.take_along_axis(agent_histories_mask, topk_idxs[..., 0], axis=1)
        agent_histories_pos = np.take_along_axis(agent_histories_pos, topk_idxs, axis=1)
        agent_histories_last_pos = np.take_along_axis(agent_histories_last_pos, topk_idxs[..., 0], axis=1)
        agent_futures = np.take_along_axis(agent_futures, topk_idxs, axis=1)
        agent_futures_mask = np.take_along_axis(agent_futures_mask, topk_idxs[..., 0], axis=1)
        track_index_to_predict_new = np.zeros(len(track_index_to_predict), dtype=np.int64)

        # Pad information if the scene has fewer agents than the maximum.
        size_to_pad = max_num_agents - agent_histories_pos.shape[1]
        agent_ids = np.pad(agent_ids, ((0, 0), (0, size_to_pad), (0, 0), (0, 0)), constant_values=-1)
        agent_histories = np.pad(agent_histories, ((0, 0), (0, size_to_pad), (0, 0), (0, 0)))
        agent_histories_mask = np.pad(agent_histories_mask, ((0, 0), (0, size_to_pad), (0, 0)))
        agent_histories_pos = np.pad(agent_histories_pos, ((0, 0), (0, size_to_pad), (0, 0), (0, 0)))
        agent_histories_last_pos = np.pad(agent_histories_last_pos, ((0, 0), (0, size_to_pad), (0, 0)))
        agent_futures = np.pad(agent_futures, ((0, 0), (0, size_to_pad), (0, 0), (0, 0)))
        agent_futures_mask = np.pad(agent_futures_mask, ((0, 0), (0, size_to_pad), (0, 0)))

        # Extract score information if available
        individual_agent_scores, individual_scene_scores = None, None
        interaction_agent_scores, interaction_scene_scores = None, None
        individual_agent_scores_mask, interaction_agent_scores_mask = None, None
        if scenario_scores is not None:
            # Get valid agent scores
            individual_agent_scores = scenario_scores.individual_scores.agent_scores[valid_past_mask].clip(min=1, max=4)  # pyright: ignore[reportOptionalSubscript]
            individual_agent_scores_valid = scenario_scores.individual_scores.agent_scores_valid[valid_past_mask]  # pyright: ignore[reportOptionalSubscript]
            individual_agent_scores[~individual_agent_scores_valid] = 0
            individual_agent_scores = np.tile(individual_agent_scores[None, :], (num_center_points, 1))

            interaction_agent_scores = scenario_scores.interaction_scores.agent_scores[valid_past_mask].clip(
                min=1, max=4
            )  # pyright: ignore[reportOptionalSubscript]
            interaction_agent_scores_valid = scenario_scores.interaction_scores.agent_scores_valid[valid_past_mask]  # pyright: ignore[reportOptionalSubscript]
            interaction_agent_scores[~interaction_agent_scores_valid] = 0
            interaction_agent_scores = np.tile(interaction_agent_scores[None, :], (num_center_points, 1))

            # Build validity masks: True where the scorer produced a valid score, False for invalid or padded agents.
            individual_agent_scores_mask = np.tile(individual_agent_scores_valid[None, :], (num_center_points, 1))
            interaction_agent_scores_mask = np.tile(interaction_agent_scores_valid[None, :], (num_center_points, 1))

            # Take only the top-k agents
            individual_agent_scores = np.take_along_axis(individual_agent_scores[..., None, None], topk_idxs, axis=1)
            interaction_agent_scores = np.take_along_axis(interaction_agent_scores[..., None, None], topk_idxs, axis=1)
            individual_agent_scores_mask = np.take_along_axis(
                individual_agent_scores_mask[..., None, None], topk_idxs, axis=1
            )
            interaction_agent_scores_mask = np.take_along_axis(
                interaction_agent_scores_mask[..., None, None], topk_idxs, axis=1
            )

            # Pad scores and masks; padded entries are marked invalid (mask = False / 0).
            individual_agent_scores = np.pad(individual_agent_scores, ((0, 0), (0, size_to_pad), (0, 0), (0, 0)))
            interaction_agent_scores = np.pad(interaction_agent_scores, ((0, 0), (0, size_to_pad), (0, 0), (0, 0)))
            individual_agent_scores_mask = np.pad(
                individual_agent_scores_mask, ((0, 0), (0, size_to_pad), (0, 0), (0, 0))
            )
            interaction_agent_scores_mask = np.pad(
                interaction_agent_scores_mask, ((0, 0), (0, size_to_pad), (0, 0), (0, 0))
            )

            # Extract scene scores
            individual_scene_scores = [scenario_scores.individual_scores.scene_score] * num_center_objects
            interaction_scene_scores = [scenario_scores.interaction_scores.scene_score] * num_center_objects

        return {
            "pad": size_to_pad * np.ones(shape=(num_center_objects)),
            "obj_ids": agent_ids,
            "obj_trajs": agent_histories,
            "obj_trajs_mask": agent_histories_mask.astype(bool),
            "obj_trajs_pos": agent_histories_pos,
            "obj_trajs_last_pos": agent_histories_last_pos,
            "obj_trajs_future_state": agent_futures,
            "obj_trajs_future_mask": agent_futures_mask,
            "individual_agent_scores": individual_agent_scores,
            "individual_agent_scores_mask": individual_agent_scores_mask,
            "individual_scene_scores": individual_scene_scores,
            "interaction_agent_scores": interaction_agent_scores,
            "interaction_agent_scores_mask": interaction_agent_scores_mask,
            "interaction_scene_scores": interaction_scene_scores,
            "center_gt_trajs": center_gt_trajs,
            "center_gt_trajs_mask": center_gt_trajs_mask,
            "center_gt_final_valid_idx": center_gt_final_valid_idx,
            "track_index_to_predict": track_index_to_predict_new,
        }

    @staticmethod
    def transform_trajectories_wrt_center_points(
        agent_tracks: AgentTrajectoryMasker, center_points: AgentTrajectoryMasker
    ) -> AgentTrajectoryMasker:
        """Transform trajectories with respect to center points.

        Notation:
            C: number of center points
            N: number of agents
            T: number of timesteps
            D: number of features

        Args:
            agent_tracks (AgentTrajectoryMasker): Object containing agent track information.
            center_points (AgentTrajectoryMasker): Object containing center-point track information.

        Returns:
            AgentTrajectoryMasker: Object containing transformed track information.
        """
        trajectories = agent_tracks.agent_trajectories
        num_objects, num_timestamps, _ = trajectories.shape

        center_position = center_points.agent_xyz_pos
        center_heading = center_points.agent_headings
        num_center_objects = center_position.shape[0]
        assert center_position.shape[0] == center_heading.shape[0]

        # TODO: refactor this method.
        # Tile the agent trajectories to be of shape (C, N, T, D)
        trajectories = np.tile(trajectories[None, :, :, :], (num_center_objects, 1, 1, 1))
        masker = AgentTrajectoryMasker(trajectories)

        # Shift the points to the center
        trajectories[..., masker.xyz_pos_mask] -= center_position[:, None, None, :]  # (1, 1, 1, D=XYZ)

        # Rotate the positions
        trajectories[:, :, :, masker.xy_pos_mask] = data_utils.rotate_points_along_z(
            points=masker.agent_xy_pos.reshape(num_center_objects, -1, 2),  # (C, N, T, D=XY) -> (C, N*T, D=XY)
            angle=-center_heading,
        ).reshape(num_center_objects, num_objects, num_timestamps, 2)  # (C, N * T, D=XY) -> (C, N, T, D=XY)
        trajectories[:, :, :, masker.heading_mask] -= center_heading[:, None, None]

        # Rotate the velocities
        trajectories[:, :, :, masker.xy_vel_mask] = data_utils.rotate_points_along_z(
            points=masker.agent_xy_vel.reshape(num_center_objects, -1, 2),
            angle=-center_heading,
        ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        return AgentTrajectoryMasker(trajectories)

    @staticmethod
    def transform_polylines_wrt_center_points(
        polylines: np.ndarray, center_points: AgentTrajectoryMasker
    ) -> np.ndarray:
        """Transform map polylines with respect to center points.

        Notation:
            C: number of center points
            P: number of polylines
            M: max number of segments per polyline
            D: number of features

        Args:
            polylines (np.ndarray): NumPy array containing map polyline information of shape (C, P, M, D)
            center_points (AgentTrajectoryMasker): Object containing center-point track information.

        Returns:
            np.ndarray: Transformed map polyline information of shape (C, P, M, D).
        """
        center_position = center_points.agent_xyz_pos
        center_heading = center_points.agent_headings

        polylines[..., 0:3] -= center_position[:, None, 0:3]
        polylines[..., 0:2] = data_utils.rotate_points_along_z(points=polylines[..., 0:2], angle=-center_heading)
        polylines[:, :, 3:5] = data_utils.rotate_points_along_z(points=polylines[:, :, 3:5], angle=-center_heading)
        return polylines

    def get_centered_map_data(
        self, map_data: StaticMapData, center_objects: np.ndarray, metadata: ScenarioMetadata
    ) -> dict[str, np.ndarray]:
        """Get map information centered with respect to center agents.

        Notation:
            C: number of center agents
            P: number of polylines
            M: max number of segments
            N: max number of points per segment
            Dt: agent tracks dimension
            Dp: polyline input feature dimension
            Dm: map output dimension

        Args:
            map_data (StaticMapData): Object containing all map information.
            center_objects (np.ndarray(C, Dt)): NumPy array containing center points.
            metadata (ScenarioMetadata): Object containing scenario metadata.

        Returns:
            map_data_dict (dict): Dictionary containing centered map information:
                map_polylines (np.ndarray(C, M, N, Dm)): Centered map information.
                map_polylines_mask (np.ndarray(C, M, N)): Centered map mask information.
                map_polylines_center (np.ndarray(C, M, 3)): Map center XYZ positions.
        """
        if len(map_data.map_polylines) == 0:
            print(f"Warning: empty HDMap {metadata.scenario_id}")
            map_data.map_polylines = np.zeros((2, 7), dtype=np.float32)

        num_center_agents = center_objects.shape[0]
        # Expand the polylines to the number of center objects polylines: (C, M, D=7)
        polylines = np.expand_dims(map_data.map_polylines.copy(), axis=0).repeat(num_center_agents, axis=0)

        # Transform the polylines w.r.t the center objects
        center_points = AgentTrajectoryMasker(center_objects)
        centered_polylines = BaseDataset.transform_polylines_wrt_center_points(polylines, center_points)

        map_data_dict = map_data.model_dump()
        max_points_per_lane = self.config.max_points_per_lane
        center_offset = self.config.center_offset_of_map

        polyline_list = []
        polyline_mask_list = []
        for polyline_type in self.config.line_type:
            key = f"{polyline_type}_polyline_idxs"

            # Check if polylines of the desired type exist in the map dictionary
            # Polyline idxs shape: (P, 2)
            polyline_idxs = map_data_dict.get(key, None)
            if polyline_idxs is None or not len(polyline_idxs):
                continue

            # Process each polyline segment
            for start, end in zip(polyline_idxs[:, 0], polyline_idxs[:, 1], strict=False):
                # Get valid segments within the [start, end]
                # polyline segment: (1, num_polylines, 7)
                # segment_list: (1, max_segments, max_points_per_segment, 7)
                segments, segments_mask = self.get_valid_segments(centered_polylines[:, start:end])
                polyline_list.append(segments)
                polyline_mask_list.append(segments_mask)

        if len(polyline_list) == 0:
            return (
                np.zeros((num_center_agents, 0, max_points_per_lane, centered_polylines.shape[-1])),
                np.zeros((num_center_agents, 0, max_points_per_lane)),
            )

        # Polylines shape: (C, N, M, 7)
        # Polylines mask shape: (C, N, M)
        polylines = np.concatenate(polyline_list, axis=1)
        polylines_mask = np.concatenate(polyline_mask_list, axis=1)

        # Get the distance of each road/polyline to the center offset shape: (C, N)
        polyline_centered = polylines[..., 0:2] - np.reshape(center_offset, (1, 1, 1, 2))
        mask_sum = polylines_mask.sum(axis=-1)
        num_valid_points = np.clip(mask_sum.astype(float), a_min=1.0, a_max=None)
        polyline_centered_dist = np.linalg.norm(polyline_centered, axis=-1).sum(-1) / num_valid_points
        polyline_centered_dist[mask_sum == 0] = LARGE_FLOAT

        # Get max_num_roads that are closest to the ego
        # topk_idxs shape: (C, N, 1, 1)
        max_num_roads = self.config.max_num_roads
        topk_idxs = np.argsort(polyline_centered_dist, axis=-1)[:, :max_num_roads, None, None]
        map_polylines = np.take_along_axis(polylines, topk_idxs, axis=1)
        map_polylines_mask = np.take_along_axis(polylines_mask, topk_idxs[..., 0], axis=1)

        # Pad map_polylines and map_polylines_mask
        size_to_pad = max_num_roads - map_polylines.shape[1]
        map_polylines = np.pad(map_polylines, ((0, 0), (0, size_to_pad), (0, 0), (0, 0)))
        map_polylines_mask = np.pad(map_polylines_mask, ((0, 0), (0, size_to_pad), (0, 0)))

        # Get the polylines center (C, N, 3)
        temp_sum = (map_polylines[..., 0:3] * map_polylines_mask[..., None].astype(float)).sum(axis=-2)
        denom = np.clip(map_polylines_mask.sum(axis=-1).astype(float)[:, :, None], a_min=1.0, a_max=None)
        map_polylines_center = temp_sum / denom

        # Get final map information
        xy_pos_pre = map_polylines[:, :, :, 0:3]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]

        # Get one-hot encoding for map types.
        map_types = map_polylines[:, :, :, -1]
        map_types = np.eye(self.config.total_map_types)[map_types.astype(int)]

        map_polylines = map_polylines[:, :, :, :-1]

        # map_polylines shape: (C, N, M, Dm)
        map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)
        map_polylines[map_polylines_mask == 0] = 0
        return {
            "map_polylines": map_polylines,
            "map_polylines_mask": map_polylines_mask.astype(bool),
            "map_polylines_center": map_polylines_center,
        }

    def get_manually_split_centered_map_data(
        self, map_data: StaticMapData, center_objects: np.ndarray, metadata: ScenarioMetadata
    ) -> dict[str, np.ndarray]:
        """Get map information using geometric polyline splitting, centered w.r.t. center agents.

        Splits the flat polyline array by detecting large inter-point distance breaks rather than using
        per-feature-type index ranges. Used by MTR, which requires this segmentation style. Gated by config key
        ``manually_split_lane: True``.

        Notation:
            C: number of center agents
            K: number of geometrically-split polyline segments (before top-k selection)
            N: num_points_each_polyline (fixed points per segment after chunking)
            Dp: raw polyline feature dimension (7)
            Dm: map output dimension (6 + 3 + total_map_types)

        Args:
            map_data (StaticMapData): Object containing all map information.
            center_objects (np.ndarray(C, Dt)): NumPy array containing center agent state vectors.
            metadata (ScenarioMetadata): Object containing scenario metadata.

        Returns:
            map_data_dict (dict): Dictionary containing centered map information:
                map_polylines (np.ndarray(C, max_num_roads, N, Dm)): Centered map information.
                map_polylines_mask (np.ndarray(C, max_num_roads, N)): Centered map mask.
                map_polylines_center (np.ndarray(C, max_num_roads, 3)): Map center XYZ positions.
        """
        required_keys = ["point_sampled_interval", "vector_break_dist_thresh", "num_points_each_polyline"]
        missing = [k for k in required_keys if self.config.get(k) is None]
        if missing:
            error_message = f"manually_split_lane is True but the following config keys are missing: {missing}"
            raise ValueError(error_message)

        if len(map_data.map_polylines) == 0:
            print(f"Warning: empty HDMap {metadata.scenario_id}")
            map_data.map_polylines = np.zeros((2, 7), dtype=np.float32)

        num_center_objects = center_objects.shape[0]
        point_sampled_interval = self.config.point_sampled_interval
        vector_break_dist_thresh = self.config.vector_break_dist_thresh
        num_points_each_polyline = self.config.num_points_each_polyline
        num_of_src_polylines = self.config.max_num_roads

        # Subsample the flat polyline array and detect geometric break points.
        polylines = map_data.map_polylines.copy()
        point_dim = polylines.shape[-1]
        sampled = polylines[::point_sampled_interval]  # (P', Dp)
        sampled_shift = np.roll(sampled, shift=1, axis=0)
        buffer = np.concatenate((sampled[:, 0:2], sampled_shift[:, 0:2]), axis=-1)  # (P', 4)
        buffer[0, 2:4] = buffer[0, 0:2]  # no break at the very first point
        break_idxs = (np.linalg.norm(buffer[:, 0:2] - buffer[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]

        # Chunk each segment into fixed-size windows of num_points_each_polyline.
        polyline_list = np.array_split(sampled, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []
        for segment in polyline_list:
            if len(segment) == 0:
                continue
            for idx in range(0, len(segment), num_points_each_polyline):
                chunk = segment[idx : idx + num_points_each_polyline]
                cur = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
                cur_mask = np.zeros((num_points_each_polyline,), dtype=np.int32)
                cur[: len(chunk)] = chunk
                cur_mask[: len(chunk)] = 1
                ret_polylines.append(cur)
                ret_polylines_mask.append(cur_mask)

        if ret_polylines:
            batch_polylines = np.stack(ret_polylines, axis=0)  # (K, N, Dp)
            batch_polylines_mask = np.stack(ret_polylines_mask, axis=0)  # (K, N)
        else:
            batch_polylines = np.zeros((0, num_points_each_polyline, point_dim), dtype=np.float32)
            batch_polylines_mask = np.zeros((0, num_points_each_polyline), dtype=np.int32)

        # Select the top-k closest polylines per center agent using world-frame distances.
        center_offset = np.array(self.config.center_offset_of_map, dtype=np.float32)  # (2,)
        center_heading = center_objects[:, 6]  # (C,)

        if len(batch_polylines) > num_of_src_polylines:
            # Compute each polyline's centroid in world coordinates: (K, 2)
            polyline_center = np.sum(batch_polylines[:, :, 0:2], axis=1) / np.clip(
                np.sum(batch_polylines_mask, axis=1)[:, None].astype(float), a_min=1.0, a_max=None
            )
            # Rotate center_offset into world frame for each center agent: (C, 1, 2) -> (C, 2)
            center_offset_rot = data_utils.rotate_points_along_z(
                points=np.tile(center_offset[None, None, :], (num_center_objects, 1, 1)),
                angle=center_heading,
            )[:, 0, :]  # (C, 2)
            pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot  # (C, 2)
            dist = np.linalg.norm(pos_of_map_centers[:, None, :] - polyline_center[None, :, :], axis=-1)  # (C, K)
            topk_idxs = np.argsort(dist, axis=1)[:, :num_of_src_polylines]  # (C, R)
            map_polylines = batch_polylines[topk_idxs]  # (C, R, N, Dp)
            map_polylines_mask = batch_polylines_mask[topk_idxs]  # (C, R, N)
        else:
            map_polylines = batch_polylines[None].repeat(num_center_objects, 0)  # (C, K, N, Dp)
            map_polylines_mask = batch_polylines_mask[None].repeat(num_center_objects, 0)  # (C, K, N)
            size_to_pad = num_of_src_polylines - map_polylines.shape[1]
            map_polylines = np.pad(map_polylines, ((0, 0), (0, size_to_pad), (0, 0), (0, 0)))
            map_polylines_mask = np.pad(map_polylines_mask, ((0, 0), (0, size_to_pad), (0, 0)))

        # Transform polylines to agent-centric coordinates (translate + rotate).
        num_center_objects, num_roads, num_pts, _ = map_polylines.shape
        map_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
        map_polylines[:, :, :, 0:2] = data_utils.rotate_points_along_z(
            points=map_polylines[:, :, :, 0:2].reshape(num_center_objects, num_roads * num_pts, 2),
            angle=-center_heading,
        ).reshape(num_center_objects, num_roads, num_pts, 2)
        map_polylines[:, :, :, 3:5] = data_utils.rotate_points_along_z(
            points=map_polylines[:, :, :, 3:5].reshape(num_center_objects, num_roads * num_pts, 2),
            angle=-center_heading,
        ).reshape(num_center_objects, num_roads, num_pts, 2)

        # Append previous-point XYZ features and zero out masked entries.
        xy_pos_roll = np.roll(map_polylines[:, :, :, 0:3], shift=1, axis=-2)
        xy_pos_roll[:, :, 0, :] = xy_pos_roll[:, :, 1, :]
        map_polylines = np.concatenate((map_polylines, xy_pos_roll), axis=-1)  # (C, R, N, Dp+3)
        map_polylines[map_polylines_mask == 0] = 0

        # Compute polyline centers in agent-centric coordinates.
        temp_sum = (map_polylines[..., 0:3] * map_polylines_mask[..., None].astype(float)).sum(axis=-2)
        denom = np.clip(map_polylines_mask.sum(axis=-1).astype(float)[:, :, None], a_min=1.0, a_max=None)
        map_polylines_center = temp_sum / denom  # (C, R, 3)

        # One-hot encode map types and assemble final feature vector.
        map_types = map_polylines[:, :, :, 6]
        xy_pos_pre = map_polylines[:, :, :, 7:]  # (C, R, N, 3)
        map_polylines = map_polylines[:, :, :, :6]  # (C, R, N, 6)
        map_types = np.eye(self.config.total_map_types)[map_types.astype(int)]  # (C, R, N, total_map_types)
        map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)
        map_polylines[map_polylines_mask == 0] = 0

        return {
            "map_polylines": map_polylines,
            "map_polylines_mask": map_polylines_mask.astype(bool),
            "map_polylines_center": map_polylines_center,
        }

    def get_valid_segments(self, polyline_segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gets the valid segments within a polyline segment.

        Notation:
            C: number of center agents
            P: number of polylines
            D: dimension of each polyline.
            M: max number of segments
            N: max number of points per segment

        Args:
            polyline_segment (np.ndarray(C, P, D)): a numpy array containing all polylines that make a segment.

        Returns:
            segments (np.ndarray(C, M, N, D)): a numpy array containing the valid segments within the input array.
            segments_mask (np.ndarray(C, M, N)): a numpy array containing the segment mask.
        """
        max_points = self.config.max_points_per_lane
        map_range = self.config.map_range
        center_offset = self.config.center_offset_of_map

        num_center_agents, _, num_polyline_dims = polyline_segment.shape

        polyline_segment_x = polyline_segment[:, :, 0] - center_offset[0]
        polyline_segment_y = polyline_segment[:, :, 1] - center_offset[1]
        # Check if the polyline segment is within a desired range
        in_range_mask = (abs(polyline_segment_x) < map_range) * (abs(polyline_segment_y) < map_range)

        # For each of the center agents, extract segments of continuous 'True' values in the mask.
        segment_index_list = [data_utils.find_true_segments(in_range_mask[i]) for i in range(num_center_agents)]
        max_segments = max([len(x) for x in segment_index_list])

        segments = np.zeros([num_center_agents, max_segments, max_points, num_polyline_dims], dtype=np.float32)
        segments_mask = np.zeros([num_center_agents, max_segments, max_points], dtype=np.int32)

        # For each of the center agents, get the valid segment and corresponding mask
        for i in range(num_center_agents):
            if in_range_mask[i].sum() == 0:
                continue
            segment_i = polyline_segment[i]
            for num, seg_index in enumerate(segment_index_list[i]):
                segment = segment_i[seg_index]
                segment_size = segment.shape[0]
                # If there are more points than the maximum allowed, select indices using linspace.
                if segment_size > max_points:
                    segments[i, num] = segment[np.linspace(0, segment_size - 1, max_points, dtype=int)]
                    segments_mask[i, num] = 1
                # Otherwise, fill the segment and mask up to segment_size
                else:
                    segments[i, num, :segment_size] = segment
                    segments_mask[i, num, :segment_size] = 1
        return segments, segments_mask

    def characterize_scenario(self, output: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add scenario characterization information to the output dictionary.

        Args:
            output (list[dict[str, Any]]): list of dictionaries containing scenario information for each scenario in
                the batch.

        Returns:
            output (list[dict[str, Any]]): list of dictionaries containing scenario information for each scenario in
                the batch, with added scenario characterization information.
        """
        # TODO: Add SafeShift features here.
        # Add the trajectory difficulty
        data_utils.get_kalman_difficulty(output)
        # Add the trajectory type (stationary, straight, right turn...)
        data_utils.get_trajectory_type(output)
        # Add causal label information
        for out in output:
            scenario_id = out["scenario_id"]
            agent_ids = out["obj_ids"].squeeze(-1).squeeze(-1)
            causal_idxs = np.zeros_like(agent_ids)
            # Validity mask: True where a proper ground-truth causal label is available.
            causal_mask = np.ones_like(agent_ids, dtype=bool)
            causal_ids = data_utils.load_causal_agent_ids(self.config.causal_labels_path, scenario_id)
            if causal_ids is not None:
                # If there are no causal IDs in the scene, let's assume for now that all agents are causal
                if causal_ids.shape[0] != 0:
                    causal_idxs = np.isin(agent_ids, causal_ids)
                    causal_idxs[out["track_index_to_predict"]] = True
                # out['causal_ids_votes'] = np.array(causal_labels['labeler_votes'], dtype=int)
            else:
                # No ground-truth labels available: mark all agents as invalid for training.
                causal_mask[:] = False

            # Mask out padded agents and/or agents with invalid histories (i.e., full mask is False).
            invalid_hists = out["obj_trajs_mask"].sum(axis=1) == 0
            causal_idxs[invalid_hists] = False
            causal_mask[invalid_hists] = False
            out["causal_idxs"] = causal_idxs.astype(np.float32)
            out["causal_mask"] = causal_mask.astype(np.float32)
        return output

    def collate_fn(self, data_list: list) -> dict:
        """Collate a list of scenario samples into a batch dictionary.

        Args:
            data_list (list): List of per-sample dictionaries.

        Returns:
            batch (dict): dictionary containing batch information as follows:
                'batch_size' (int): size (B) of the input batch.
                'input_dict' (dict): dictionary containing the following scenario data:
                    TODO: annotate
                    'center_gt_final_valid_idx' (torch.tensor([B]):
                    'center_gt_trajs' (torch.tensor([B, F, 4])): agent-centric gt future trajs.
                    'center_gt_trajs_mask' (torch.tensor([B, F])): agent-centric future masks.
                    'center_gt_trajs_src' (torch.tensor([B, T, 10])): agent-centric full trajectories following:
                        idx 0 to 2: the agent's (x, y, z) center coordinates.
                        idx 3 to 5: the agent's length, width and height in meters.
                        idx 6: the agent's yaw angle (heading) of the forward direction in radians
                        idx 7 to 8: the agent's (x, y) velocity in meters/second
                        idx 9: a flag indicating if the information is valid
                    'center_objects_id' (torch.tensor([B])):
                    'center_objects_type' (torch.tensor([B])):
                    'center_objects_world' (torch.tensor([B, 10])):
                    'dataset_name' (list[str, size(b)]): list of strings containing the dataset name.
                    'kalman_difficulty' (torch.tensor(B, 3)): tensor with Kalman difficulty values at 2, 4, 6 seconds.
                    'map_center' (torch.tensor(B, 3)): tensor containing each map's center XYZ coordinate.
                    'map_polylines' (torch.tensor(B, P, M, D)): tensor containing polyline (P) information.
                        TODO: figure out M, D
                    'map_polylines_center' (torch.tensor(B, P, 3)): tensor of polyline (P) center XYZ coordinates.
                    'map_polylines_mask' (torch.tensor(B, P, M)): tensor of polyline (P) masks.
                    'obj_trajs' (torch.tensor(B, N, H, 29)): tensor containing agent historical information.
                        TODO: figure out what the 29 size is.
                    'obj_trajs_future_mask' (torch.tensor(B, N, F)): tensor containing agents (N) future (F) masks.
                    'obj_trajs_future_state' (torch.tensor(B, N, F, 4)): tensor containing agents (N) future (F) states
                        as XYZ coordinates + heading.
                    'obj_trajs_last_pos' (torch.tensor(B, N, 3)): tensor containing each agent's last XYZ position.
                    'obj_trajs_mask' (torch.tensor(B, N, H)): tensor containing each agents mask.
                    'agent_histories_pos' (torch.tensor(B, N, H, 3)): tensor containing agents (N) historical (H)
                        positions as XYZ (3) coordinates.
                    'scenario_id' (list[str, size(B)]): list of strings containing scenario IDs.
                    'track_index_to_predict' (torch.tensor(B)): index of the track predict.
                    'trajectory_type' (torch.tensor(B)): type of the ego-agent's trajectory by TrajectoryType
        """
        batch_list = data_list.copy()

        batch_size = len(batch_list)
        key_to_list = {}
        for key in batch_list[0]:
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():
            # TODO: Handle bare exception.
            try:
                input_dict[key] = torch.from_numpy(np.stack(val_list, axis=0))
            except:  # noqa: E722
                input_dict[key] = val_list

        input_dict["center_objects_type"] = input_dict["center_objects_type"].numpy()
        return {"batch_size": batch_size, "input_dict": input_dict, "batch_sample_count": batch_size}

    def __len__(self) -> int:
        """Returns the number of samples in the dataset, which is equal to the number of loaded data keys."""
        return len(self.data_loaded_keys)

    @cache  # noqa: B019
    def _get_file(self, file_path: str) -> h5py.File:
        """Gets the file object for a given file path, using caching to avoid reopening files.

        Args:
            file_path (h5py.File): path to the file to open.
        """
        return h5py.File(file_path, "r")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Gets the data for a given index.

        Args:
            idx (int): index of the data to retrieve.

        Returns:
            dict[str, Any]: dictionary containing the data for the given index.
        """
        file_key = self.data_loaded_keys[idx]
        file_info = self.data_loaded[file_key]
        file_path = file_info["h5_path"]

        if file_path not in self.file_cache:
            self.file_cache[file_path] = self._get_file(file_path)

        group = self.file_cache[file_path][file_key]
        return {k: group[k][()].decode("utf-8") if group[k].dtype.type == np.bytes_ else group[k][()] for k in group}

    def close_files(self) -> None:
        """Closes all open files in the file cache."""
        for f in self.file_cache.values():
            f.close()
        self.file_cache.clear()
