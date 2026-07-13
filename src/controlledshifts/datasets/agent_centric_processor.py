"""Generic agent-centric scenario processing (Stage B of the data pipeline).

Transforms an open ``Scenario`` (produced by a dataset-specific preprocessor such as ``waymo.preprocessor``) into the
agent-centric record format consumed by the models. This module is dataset-agnostic: it has no dependency on the torch
``Dataset``/loader (``base_dataset``) or on any raw dataset format. It also defines the *processing profile* -- the hash
of tensor-affecting config keys that keys the agent-centric cache (``ac_cache/<variant>/<profile_hash>/``).
"""

import hashlib
import json
from typing import Any

import numpy as np
from characterization.schemas import (
    AgentData,
    Scenario,
    ScenarioMetadata,
    ScenarioScores,
    StaticMapData,
    TracksToPredict,
)
from characterization.utils.common import AgentTrajectoryMasker, AgentType
from omegaconf import DictConfig, OmegaConf

from controlledshifts.datasets.scenario_scorer import ScenarioScorer, add_scenario_map_metadata
from controlledshifts.utils import data_utils, pylogger
from controlledshifts.utils.constants import LARGE_FLOAT


_LOGGER = pylogger.get_pylogger(__name__)

# Config keys that change the agent-centric tensors. The cache directory is keyed by a hash of these (see
# ``processing_profile``); any key that alters ``process_agent_centric_scenario`` / ``get_centered_*`` /
# ``characterize_scenario`` / the profile-shaping in ``shape_scenario`` MUST appear here, or training could silently
# read a mismatched cache (a unit test guards this list). Keys that only affect selection (num_data_to_consider) or
# labelling (the per-source tag) are deliberately excluded so they do not fragment the cache.
PROFILE_KEYS: tuple[str, ...] = (
    "past_len",
    "future_len",
    "max_num_agents",
    "max_num_roads",
    "max_points_per_lane",
    "map_range",
    "center_offset_of_map",
    "manually_split_lane",
    "point_sampled_interval",
    "num_points_each_polyline",
    "vector_break_dist_thresh",
    "total_map_types",
    "object_type",
    "line_type",
    "only_train_on_ego",
    "trajectory_sample_interval",
    "masked_attributes",
    "autolabel_agents",
    "causal_labels_path",
)


def processing_profile(config: DictConfig) -> dict[str, Any]:
    """Returns the resolved config values that determine the cached agent-centric tensors.

    Only the keys in :data:`PROFILE_KEYS` (plus the SafeShift ``scenario_characterization`` block when
    ``autolabel_agents`` is enabled) are included, so selection/labelling options do not fragment the cache.
    """
    profile: dict[str, Any] = {}
    for key in PROFILE_KEYS:
        value = config.get(key, None)
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        profile[key] = value
    if config.get("autolabel_agents", False):
        characterization = config.get("scenario_characterization", None)
        profile["scenario_characterization"] = (
            OmegaConf.to_container(characterization, resolve=True) if characterization is not None else None
        )
    return profile


def processing_profile_hash(config: DictConfig) -> str:
    """Returns a short, stable hash of :func:`processing_profile`, used as the agent-centric cache directory name."""
    payload = json.dumps(processing_profile(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class AgentCentricProcessor:
    """Transforms open ``Scenario`` objects into agent-centric records (Stage B).

    Holds the processing config and (optionally) the SafeShift autolabel processors. The profile-dependent shaping
    (``trajectory_sample_interval`` frequency mask, ``past_len``/``future_len`` slicing, ``only_train_on_ego`` track
    selection) is applied here -- NOT baked into the stored ``Scenario`` -- so a single canonical Scenario store serves
    every processing profile.
    """

    def __init__(self, config: DictConfig) -> None:
        """Builds the processor from a dataset config (the processing-profile keys plus optional SafeShift config)."""
        self.config = config
        self.past_len = config.past_len
        self.future_len = config.future_len
        self.current_time_idx = self.past_len - 1
        self.total_steps = self.past_len + self.future_len
        self.trajectory_sample_interval = config.get("trajectory_sample_interval", 1)
        self.only_train_on_ego = config.get("only_train_on_ego", True)

        self.autolabel_agents = config.get("autolabel_agents", False)
        self.conflict_points_config = config.get("conflict_points", None)
        self.closest_lanes_config = config.get("closest_lanes", None)
        self.scorer: ScenarioScorer | None = None
        if self.autolabel_agents:
            # ScenarioScorer validates the presence of the scoring config blocks.
            self.scorer = ScenarioScorer(config)

    def processing_profile(self) -> dict[str, Any]:
        """Returns this processor's resolved processing profile (see module-level :func:`processing_profile`)."""
        return processing_profile(self.config)

    def processing_profile_hash(self) -> str:
        """Returns this processor's processing-profile hash (the agent-centric cache directory name)."""
        return processing_profile_hash(self.config)

    def shape_scenario(self, scenario: Scenario) -> Scenario:
        """Applies profile-dependent shaping to a freshly loaded ``Scenario`` (mutated in place).

        This is the shaping that must vary per processing profile and therefore is NOT baked into the stored Scenario:
        the ``trajectory_sample_interval`` frequency mask on trajectory validity, slicing to ``total_steps``, and the
        ``only_train_on_ego`` track-to-predict selection.
        """
        agent_data = scenario.agent_data
        trajectories = np.asarray(agent_data.agent_trajectories)[:, : self.total_steps].copy()
        frequency_mask = data_utils.generate_mask(
            self.current_time_idx, self.total_steps, self.trajectory_sample_interval
        )
        trajectories[..., -1] *= frequency_mask[np.newaxis]
        agent_data.agent_trajectories = trajectories

        metadata = scenario.metadata
        metadata.timestamps_seconds = list(metadata.timestamps_seconds)[: self.total_steps]

        if self.only_train_on_ego:
            scenario.tracks_to_predict = TracksToPredict(
                track_index=[metadata.ego_vehicle_index], difficulty=[0], object_type=[AgentType.TYPE_VEHICLE]
            )
        return scenario

    def process(self, scenario: Scenario, scenario_scores: ScenarioScores | None = None) -> list[dict[str, Any]] | None:
        """Shapes a ``Scenario`` and transforms it into agent-centric records (or ``None`` if it cannot be processed).

        Args:
            scenario: The canonical open scenario to process.
            scenario_scores: Precomputed scores (e.g. from the visualizer); when ``None`` and
                ``autolabel_agents`` is enabled, scores are computed here.

        Returns:
            One record per center agent, or ``None`` on failure / no valid center agents.
        """
        # TODO: Resolve bare except inherited from UniTraj.
        try:
            scenario = self.shape_scenario(scenario)
            scores = scenario_scores
            if self.autolabel_agents:
                scores = self.compute_scores(scenario)

            records = self.process_agent_centric_scenario(scenario, scenario_scores=scores)
            if records is not None:
                # UniTraj characterizations (Kalman difficulty, trajectory type) + causal labels.
                records = self.characterize_scenario(records)
        except Exception:
            scenario_id = getattr(getattr(scenario, "metadata", None), "scenario_id", "<unknown>")
            _LOGGER.exception("error processing scenario: %s", scenario_id)
            records = None
        return records

    def compute_scenario_map_metadata(self, scenario: Scenario) -> Scenario:
        """Computes map metadata (conflict points, closest lanes), truncated to this processor's ``total_steps``."""
        return add_scenario_map_metadata(
            scenario, self.conflict_points_config, self.closest_lanes_config, total_steps=self.total_steps
        )

    def compute_scores(self, scenario: Scenario) -> ScenarioScores:
        """Computes SafeShift scores for a (shaped) scenario; requires an autolabel-enabled processor.

        Map metadata is computed truncated to ``total_steps`` to match the processor's shaped trajectories, then the
        SafeShift features and scores are computed.
        """
        if self.scorer is None:
            error_message = (
                "Scoring requires an autolabel-enabled processor. Re-run with the override "
                "`dataset.config.autolabel_agents=true`."
            )
            raise ValueError(error_message)
        scenario = self.compute_scenario_map_metadata(scenario)
        return self.scorer.score_features(scenario)

    def process_agent_centric_scenario(
        self, scenario: Scenario, scenario_scores: ScenarioScores | None = None
    ) -> list[dict[str, Any]] | None:
        """Processes a scenario from an internal format into an agent-centric format.

        Args:
            scenario: The input scenario in internal format to be transformed into agent-centric format.
            scenario_scores: optional scenario scores to be added to the agent-centric format.

        Returns:
            A list of dictionaries containing the processed scenario data in agent-centric format, or None if
            processing fails.
        """
        agent_data = scenario.agent_data
        tracks_to_predict = scenario.tracks_to_predict
        metadata = scenario.metadata

        center_objects, track_index_to_predict = self.get_agents_of_interest_center_points(
            agent_data=agent_data, tracks_to_predict=tracks_to_predict, metadata=metadata
        )
        if center_objects is None:
            return None

        ret_dict = self.get_centered_agent_data(
            agent_data=agent_data,
            center_objects=center_objects,
            track_index_to_predict=track_index_to_predict,
            metadata=metadata,
            scenario_scores=scenario_scores,
        )

        scenario_id = metadata.scenario_id
        ret_dict["scenario_id"] = np.array([scenario_id] * len(track_index_to_predict))
        ret_dict["center_objects_world"] = center_objects
        ret_dict["center_objects_id"] = np.array(agent_data.agent_ids)[track_index_to_predict]
        agent_types_int = [agent_type.value for agent_type in agent_data.agent_types]
        ret_dict["center_objects_type"] = np.array(agent_types_int)[track_index_to_predict]
        ret_dict["center_gt_trajs_src"] = agent_data.agent_trajectories[track_index_to_predict]

        if self.config.get("manually_split_lane", False):
            map_dict = self.get_manually_split_centered_map_data(
                map_data=scenario.static_map_data, center_objects=center_objects, metadata=metadata
            )
        else:
            map_dict = self.get_centered_map_data(
                map_data=scenario.static_map_data, center_objects=center_objects, metadata=metadata
            )
        ret_dict.update(map_dict)

        AgentCentricProcessor._mask_out_attributes(ret_dict, self.config.masked_attributes)
        AgentCentricProcessor._cast_dictionary(ret_dict)

        sample_num = center_objects.shape[0]
        ret_dict["dataset_name"] = [scenario.metadata.dataset] * sample_num

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
            scenario_dict: The dictionary containing scenario data, which will be modified in-place.
            attributes_to_mask: A list of attribute names to be masked out in the scenario dictionary.
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
            scenario_dict: The dictionary containing scenario data, which will be modified in-place.
            from_dtype: The original data type of the arrays.
            to_dtype: The target data type of the arrays.
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
            agent_data: Object containing agent trajectory data obtained in `self.preprocess_scenario()`.
            tracks_to_predict: Object containing tracks in `agent_data` to predict.
            metadata: Object containing scenario metadata.

        Returns:
            agent_centerpoints: center points for agents of interest, shape (N, D).
            agent_idxs: indices of agents of interest, shape (N,).
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

        for agent_idx in agents_to_predict_idxs:
            if not agent_valid[agent_idx, self.current_time_idx]:
                print(f"Warning: agent={agent_idx} of scene={scenario_id} is not valid at time {self.current_time_idx}")
                continue
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
            agent_data: Object containing agent trajectory data obtained in `self.preprocess_scenario()`.
            center_objects: center points of the agents of interest.
            track_index_to_predict: indices of the agents of interest.
            metadata: Object containing scenario metadata.
            scenario_scores: optional scenario scores to be added to the agent-centric format.

        Returns:
            A dictionary containing the processed agent-centric data for the scenario.
        """
        center_points = AgentTrajectoryMasker(center_objects)

        agent_trajectories = agent_data.agent_trajectories
        agent_histories_pre = AgentTrajectoryMasker(agent_trajectories[:, : self.current_time_idx + 1])
        centered_histories = AgentCentricProcessor.transform_trajectories_wrt_center_points(
            agent_histories_pre, center_points
        )

        num_center_points, num_agents, num_timesteps, num_dims = centered_histories.agent_trajectories.shape
        # agent ids (C, N)
        agent_ids = np.array(agent_data.agent_ids)
        agent_ids = np.tile(agent_ids[None, :], (num_center_points, 1))

        # agent type mask (C, N, T, 5)
        agent_types = np.array(agent_data.agent_types)
        agents_onehot_type_mask = np.zeros((num_center_points, num_agents, num_timesteps, 5))
        agents_onehot_type_mask[:, agent_types == AgentType.TYPE_VEHICLE, :, 0] = 1
        agents_onehot_type_mask[:, agent_types == AgentType.TYPE_PEDESTRIAN, :, 1] = 1
        agents_onehot_type_mask[:, agent_types == AgentType.TYPE_CYCLIST, :, 2] = 1
        agents_onehot_type_mask[np.arange(num_center_points), track_index_to_predict, :, 3] = 1
        agents_onehot_type_mask[:, metadata.ego_vehicle_index, :, 4] = 1

        # temporal embedding (C, N, Th, Th+1)
        history_timestamps = np.array(metadata.timestamps_seconds[: self.current_time_idx + 1], dtype=np.float32)
        agents_time_embeddings = np.zeros((num_center_points, num_agents, num_timesteps, num_timesteps + 1))
        for i in range(num_timesteps):
            agents_time_embeddings[:, :, i, i] = 1
        agents_time_embeddings[:, :, :, -1] = history_timestamps

        # heading embedding (C, N, Th, 2)
        centered_headings = centered_histories.agent_headings.squeeze(-1)
        agent_heading_embedding = np.zeros((num_center_points, num_agents, num_timesteps, 2))
        agent_heading_embedding[:, :, :, 0] = np.sin(centered_headings)
        agent_heading_embedding[:, :, :, 1] = np.cos(centered_headings)

        # accelerations (C, N, Th, 2)
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
        # agent history mask (C, N, Th)
        agent_histories_mask = centered_histories.agent_valid.squeeze(-1)
        agent_histories[agent_histories_mask == 0] = 0
        assert agent_trajectories.__len__() == agent_histories.shape[1]

        agent_futures = AgentTrajectoryMasker(agent_trajectories[:, self.current_time_idx + 1 :])
        centered_futures = AgentCentricProcessor.transform_trajectories_wrt_center_points(agent_futures, center_points)

        # agent futures (C, N, Tf, S)
        agent_futures = centered_futures.agent_state  # S=(x, y, vx, vy)
        agent_futures_mask = centered_futures.agent_valid.squeeze(-1)
        agent_futures[agent_futures_mask == 0] = 0

        # GT trajectories of the agents to predict (C, F, S)
        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = agent_futures[center_obj_idxs, track_index_to_predict]
        center_gt_trajs_mask = agent_futures_mask[center_obj_idxs, track_index_to_predict]
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        # valid-agent mask (N,)
        valid_past_mask = np.logical_not(agent_histories_pre.agent_valid.squeeze(-1).sum(axis=-1) == 0)
        # agent histories (C, M, Th, Dpost), M = N - invalid
        agent_histories_mask = agent_histories_mask[:, valid_past_mask]
        agent_histories = agent_histories[:, valid_past_mask]
        # agent futures (C, M, Tf, 4)
        agent_futures = agent_futures[:, valid_past_mask]
        agent_futures_mask = agent_futures_mask[:, valid_past_mask]
        # agent ids (C, M)
        agent_ids = agent_ids[:, valid_past_mask]

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
        # (C, M)
        agent_dists_to_center_points = np.linalg.norm(agent_histories[..., -1, 0:2], axis=-1)
        agent_dists_to_center_points[agent_histories_mask[..., -1] == 0] = LARGE_FLOAT
        # (C, max_num_agents, 1, 1)
        topk_idxs = np.argsort(agent_dists_to_center_points, axis=-1)[:, :max_num_agents, None, None]

        agent_ids = np.take_along_axis(agent_ids[..., None, None], topk_idxs, axis=1)
        agent_histories = np.take_along_axis(agent_histories, topk_idxs, axis=1)
        agent_histories_mask = np.take_along_axis(agent_histories_mask, topk_idxs[..., 0], axis=1)
        agent_histories_pos = np.take_along_axis(agent_histories_pos, topk_idxs, axis=1)
        agent_histories_last_pos = np.take_along_axis(agent_histories_last_pos, topk_idxs[..., 0], axis=1)
        agent_futures = np.take_along_axis(agent_futures, topk_idxs, axis=1)
        agent_futures_mask = np.take_along_axis(agent_futures_mask, topk_idxs[..., 0], axis=1)
        track_index_to_predict_new = np.zeros(len(track_index_to_predict), dtype=np.int64)

        # Pad up to max_num_agents when the scene has fewer agents.
        size_to_pad = max_num_agents - agent_histories_pos.shape[1]
        agent_ids = np.pad(agent_ids, ((0, 0), (0, size_to_pad), (0, 0), (0, 0)), constant_values=-1)
        agent_histories = np.pad(agent_histories, ((0, 0), (0, size_to_pad), (0, 0), (0, 0)))
        agent_histories_mask = np.pad(agent_histories_mask, ((0, 0), (0, size_to_pad), (0, 0)))
        agent_histories_pos = np.pad(agent_histories_pos, ((0, 0), (0, size_to_pad), (0, 0), (0, 0)))
        agent_histories_last_pos = np.pad(agent_histories_last_pos, ((0, 0), (0, size_to_pad), (0, 0)))
        agent_futures = np.pad(agent_futures, ((0, 0), (0, size_to_pad), (0, 0), (0, 0)))
        agent_futures_mask = np.pad(agent_futures_mask, ((0, 0), (0, size_to_pad), (0, 0)))

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
            agent_tracks: Object containing agent track information.
            center_points: Object containing center-point track information.

        Returns:
            Object containing transformed track information.
        """
        trajectories = agent_tracks.agent_trajectories
        num_objects, num_timestamps, _ = trajectories.shape

        center_position = center_points.agent_xyz_pos
        center_heading = center_points.agent_headings
        num_center_objects = center_position.shape[0]
        assert center_position.shape[0] == center_heading.shape[0]

        # TODO: refactor this method.
        # (C, N, T, D)
        trajectories = np.tile(trajectories[None, :, :, :], (num_center_objects, 1, 1, 1))
        masker = AgentTrajectoryMasker(trajectories)

        trajectories[..., masker.xyz_pos_mask] -= center_position[:, None, None, :]  # (1, 1, 1, D=XYZ)

        trajectories[:, :, :, masker.xy_pos_mask] = data_utils.rotate_points_along_z(
            points=masker.agent_xy_pos.reshape(num_center_objects, -1, 2),  # (C, N, T, D=XY) -> (C, N*T, D=XY)
            angle=-center_heading,
        ).reshape(num_center_objects, num_objects, num_timestamps, 2)  # (C, N * T, D=XY) -> (C, N, T, D=XY)
        trajectories[:, :, :, masker.heading_mask] -= center_heading[:, None, None]

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
            polylines: map polyline information, shape (C, P, M, D).
            center_points: Object containing center-point track information.

        Returns:
            Transformed map polyline information, shape (C, P, M, D).
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
            map_data: Object containing all map information.
            center_objects: center points, shape (C, Dt).
            metadata: Object containing scenario metadata.

        Returns:
            Dictionary containing centered map information:
                map_polylines: Centered map information, shape (C, M, N, Dm).
                map_polylines_mask: Centered map mask information, shape (C, M, N).
                map_polylines_center: Map center XYZ positions, shape (C, M, 3).
        """
        if len(map_data.map_polylines) == 0:
            print(f"Warning: empty HDMap {metadata.scenario_id}")
            map_data.map_polylines = np.zeros((2, 7), dtype=np.float32)

        num_center_agents = center_objects.shape[0]
        # (C, M, D=7)
        polylines = np.expand_dims(map_data.map_polylines.copy(), axis=0).repeat(num_center_agents, axis=0)

        center_points = AgentTrajectoryMasker(center_objects)
        centered_polylines = AgentCentricProcessor.transform_polylines_wrt_center_points(polylines, center_points)

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

            for start, end in zip(polyline_idxs[:, 0], polyline_idxs[:, 1], strict=False):
                # input (1, num_polylines, 7) -> segments (1, max_segments, max_points_per_segment, 7)
                segments, segments_mask = self.get_valid_segments(centered_polylines[:, start:end])
                polyline_list.append(segments)
                polyline_mask_list.append(segments_mask)

        if len(polyline_list) == 0:
            # No polylines of any configured type were found in range; emit a single zero polyline so the normal
            # top-k/padding path below produces correctly-shaped, all-zero map tensors (and a dict, not a tuple).
            polyline_list.append(
                np.zeros((num_center_agents, 1, max_points_per_lane, centered_polylines.shape[-1]), dtype=np.float32)
            )
            polyline_mask_list.append(np.zeros((num_center_agents, 1, max_points_per_lane), dtype=np.int32))

        # polylines (C, N, M, 7), mask (C, N, M)
        polylines = np.concatenate(polyline_list, axis=1)
        polylines_mask = np.concatenate(polyline_mask_list, axis=1)

        # distance of each polyline to the center offset (C, N)
        polyline_centered = polylines[..., 0:2] - np.reshape(center_offset, (1, 1, 1, 2))
        mask_sum = polylines_mask.sum(axis=-1)
        num_valid_points = np.clip(mask_sum.astype(float), a_min=1.0, a_max=None)
        polyline_centered_dist = np.linalg.norm(polyline_centered, axis=-1).sum(-1) / num_valid_points
        polyline_centered_dist[mask_sum == 0] = LARGE_FLOAT

        # topk_idxs (C, N, 1, 1): the max_num_roads polylines closest to the ego
        max_num_roads = self.config.max_num_roads
        topk_idxs = np.argsort(polyline_centered_dist, axis=-1)[:, :max_num_roads, None, None]
        map_polylines = np.take_along_axis(polylines, topk_idxs, axis=1)
        map_polylines_mask = np.take_along_axis(polylines_mask, topk_idxs[..., 0], axis=1)

        size_to_pad = max_num_roads - map_polylines.shape[1]
        map_polylines = np.pad(map_polylines, ((0, 0), (0, size_to_pad), (0, 0), (0, 0)))
        map_polylines_mask = np.pad(map_polylines_mask, ((0, 0), (0, size_to_pad), (0, 0)))

        # polylines center (C, N, 3)
        temp_sum = (map_polylines[..., 0:3] * map_polylines_mask[..., None].astype(float)).sum(axis=-2)
        denom = np.clip(map_polylines_mask.sum(axis=-1).astype(float)[:, :, None], a_min=1.0, a_max=None)
        map_polylines_center = temp_sum / denom

        xy_pos_pre = map_polylines[:, :, :, 0:3]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]

        map_types = map_polylines[:, :, :, -1]
        map_types = np.eye(self.config.total_map_types)[map_types.astype(int)]

        map_polylines = map_polylines[:, :, :, :-1]

        # map_polylines (C, N, M, Dm)
        map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)
        map_polylines[map_polylines_mask == 0] = 0
        return {
            "map_polylines": map_polylines,
            "map_polylines_mask": map_polylines_mask.astype(bool),
            "map_polylines_center": map_polylines_center,
        }

    def get_manually_split_centered_map_data(  # noqa: PLR0915
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
            map_data: Object containing all map information.
            center_objects: center agent state vectors, shape (C, Dt).
            metadata: Object containing scenario metadata.

        Returns:
            Dictionary containing centered map information:
                map_polylines: Centered map information, shape (C, max_num_roads, N, Dm).
                map_polylines_mask: Centered map mask, shape (C, max_num_roads, N).
                map_polylines_center: Map center XYZ positions, shape (C, max_num_roads, 3).
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
            polyline_segment: all polylines that make a segment, shape (C, P, D).

        Returns:
            segments: the valid segments within the input array, shape (C, M, N, D).
            segments_mask: the segment mask, shape (C, M, N).
        """
        max_points = self.config.max_points_per_lane
        map_range = self.config.map_range
        center_offset = self.config.center_offset_of_map

        num_center_agents, _, num_polyline_dims = polyline_segment.shape

        polyline_segment_x = polyline_segment[:, :, 0] - center_offset[0]
        polyline_segment_y = polyline_segment[:, :, 1] - center_offset[1]
        in_range_mask = (abs(polyline_segment_x) < map_range) * (abs(polyline_segment_y) < map_range)

        segment_index_list = [data_utils.find_true_segments(in_range_mask[i]) for i in range(num_center_agents)]
        max_segments = max([len(x) for x in segment_index_list])

        segments = np.zeros([num_center_agents, max_segments, max_points, num_polyline_dims], dtype=np.float32)
        segments_mask = np.zeros([num_center_agents, max_segments, max_points], dtype=np.int32)

        for i in range(num_center_agents):
            if in_range_mask[i].sum() == 0:
                continue
            segment_i = polyline_segment[i]
            for num, seg_index in enumerate(segment_index_list[i]):
                segment = segment_i[seg_index]
                segment_size = segment.shape[0]
                if segment_size > max_points:
                    # Downsample to max_points evenly-spaced indices
                    segments[i, num] = segment[np.linspace(0, segment_size - 1, max_points, dtype=int)]
                    segments_mask[i, num] = 1
                else:
                    segments[i, num, :segment_size] = segment
                    segments_mask[i, num, :segment_size] = 1
        return segments, segments_mask

    def characterize_scenario(self, output: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add scenario characterization information to the output dictionary.

        Args:
            output: list of dictionaries containing scenario information for each scenario in the batch.

        Returns:
            The same list with added scenario characterization information.
        """
        # TODO: Add SafeShift features here.
        data_utils.get_kalman_difficulty(output)
        data_utils.get_trajectory_type(output)
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
