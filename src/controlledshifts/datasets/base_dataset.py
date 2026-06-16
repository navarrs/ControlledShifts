"""Agent-centric dataset loader (Stage D of the data pipeline).

A dataset-agnostic ``torch`` ``Dataset`` that assembles training/eval samples by reading per-scenario agent-centric
records from the canonical cache (``ac_cache/<variant>/<profile_hash>/``, built offline by
``controlledshifts.build_ac_cache``). It does no scenario processing itself: each split source declares
``{variant, split_json, split, tag}``; the split JSON selects scenario IDs, the variant's cache supplies the records,
and the tag labels the source (injected as ``dataset_name`` at read time). The processing-profile hash that locates the
cache is defined in :mod:`controlledshifts.datasets.agent_centric_processor`.
"""

import json
import pickle  # nosec B403
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from controlledshifts.datasets.agent_centric_processor import processing_profile, processing_profile_hash
from controlledshifts.utils import pylogger
from controlledshifts.utils.constants import DataSplits, SampleSelection


_LOGGER = pylogger.get_pylogger(__name__)


class BaseDataset(Dataset):
    """Loads agent-centric samples for a split from the canonical per-variant caches.

    This class is concrete and dataset-agnostic: the dataset-specific repack (raw -> ``Scenario``) and the agent-centric
    transform (``Scenario`` -> records) live in the preprocessing/builder modules, not here.
    """

    def __init__(self, config: DictConfig) -> None:
        """Dataset constructor.

        Args:
            config: Configuration object containing dataset parameters.
        """
        super().__init__()

        self.split = config.split
        self.config = config
        self.num_data_to_consider = config.num_data_to_consider

        if self.config.load_data:
            match self.split:
                case DataSplits.TRAINING:
                    self.sources = config.train_sources
                case DataSplits.VALIDATION:
                    self.sources = config.val_sources
                case DataSplits.TESTING:
                    self.sources = config.test_sources
                case _:
                    error_message = f"Unsupported split value: {self.split}"
                    raise ValueError(error_message)

            self.blacklist = []
            sample_selection_strategy = SampleSelection(self.config.sample_selection_strategy)
            if sample_selection_strategy != SampleSelection.ALL:
                sample_selection_filepath = Path(self.config.sample_selection_filepath)
                with sample_selection_filepath.open("r") as f:
                    selected_samples = json.load(f)
                    self.blacklist = selected_samples["drop"]
                    _LOGGER.info("Removing %s  samples using %s", len(self.blacklist), sample_selection_strategy)
            self.load_data()

    def _cache_dir(self, variant: str) -> Path:
        """Returns the agent-centric cache directory for a variant under the current processing profile."""
        return Path(self.config.ac_cache_root, variant, processing_profile_hash(self.config))

    def _load_cache_index(self, cache_dir: Path, variant: str) -> dict[str, Any]:
        """Loads a variant's ``_index.pkl`` and verifies ``_profile.json`` matches the current processing profile."""
        index_path = cache_dir / "_index.pkl"
        profile_path = cache_dir / "_profile.json"
        if not index_path.exists() or not profile_path.exists():
            error_message = (
                f"Agent-centric cache for variant '{variant}' not found at {cache_dir}. "
                f"Build it first, e.g.: uv run -m controlledshifts.build_ac_cache variant={variant}"
            )
            raise FileNotFoundError(error_message)

        with profile_path.open("r") as f:
            cached_profile = json.load(f)
        current_profile = json.loads(json.dumps(processing_profile(self.config), sort_keys=True, default=str))
        if cached_profile != current_profile:
            error_message = (
                f"Processing-profile mismatch for variant '{variant}' at {cache_dir}: the cache was built with a "
                f"different config. Rebuild it (build_ac_cache variant={variant}) or align the dataset/model config."
            )
            raise ValueError(error_message)

        with index_path.open("rb") as f:
            return pickle.load(f)  # nosec B301

    def _load_split_ids(self, split_json: str, split: str) -> list[str]:
        """Returns the scenario IDs for ``split`` (training/validation/testing) from ``splits_root/<split_json>.json``."""
        split_path = Path(self.config.splits_root, f"{split_json}.json")
        with split_path.open("r") as f:
            payload = json.load(f)
        return payload[split]

    def load_data(self) -> None:
        """Builds the flat sample index for the current split from the canonical per-variant agent-centric caches.

        Each source declares ``{variant, split_json, split, tag}``: the split JSON selects scenario IDs, the variant's
        cache supplies the per-scenario agent-centric records, and the tag labels the source (injected as
        ``dataset_name`` at read time). No scenario is reprocessed or copied here.
        """
        _LOGGER.info("Loading %s data...", self.split)
        self.data_loaded: dict[str, dict[str, Any]] = {}

        for source in self.sources:
            variant, split_json, split, tag = (
                source["variant"],
                source["split_json"],
                source["split"],
                source["tag"],
            )
            cache_dir = self._cache_dir(variant)
            index = self._load_cache_index(cache_dir, variant)
            scenarios = index["scenarios"]

            split_ids = self._load_split_ids(split_json, split)
            num_missing, num_empty = 0, 0
            for scenario_id in split_ids:
                info = scenarios.get(scenario_id)
                if info is None:
                    num_missing += 1
                    continue
                if info["num_records"] == 0:
                    num_empty += 1
                    continue
                file_path = cache_dir / info["rel_path"]
                for record_idx in range(info["num_records"]):
                    key = f"{tag}::{variant}::{scenario_id}::{record_idx}"
                    self.data_loaded[key] = {
                        "file_path": file_path,
                        "record_idx": record_idx,
                        "scenario_id": scenario_id,
                        "kalman_difficulty": info["kalman_difficulty"][record_idx],
                        "tag": tag,
                    }
            _LOGGER.info(
                "Source %s (variant=%s, %s): %d ids (%d missing from cache, %d empty)",
                tag,
                variant,
                split,
                len(split_ids),
                num_missing,
                num_empty,
            )

        # Blacklisting lets a single cache serve every sample-selection variant without rebuilding.
        if self.blacklist:
            blacklist = set(self.blacklist)
            self.data_loaded = {k: v for k, v in self.data_loaded.items() if v["scenario_id"] not in blacklist}

        if not self.data_loaded:
            err_msg = f"No samples loaded for split {self.split} (after blacklist)."
            raise RuntimeError(err_msg)

        # Shuffle deterministically (seed-driven) so all DDP ranks agree on ordering and the train-only subsample.
        keys = list(self.data_loaded.keys())
        rng = np.random.default_rng(self.config.get("seed", 0))
        rng.shuffle(keys)
        if self.split == DataSplits.TRAINING and self.num_data_to_consider is not None:
            keys = keys[: self.num_data_to_consider]
        self.data_loaded = {key: self.data_loaded[key] for key in keys}
        self.data_loaded_keys = list(self.data_loaded.keys())

        if self.config.store_data_in_memory:
            _LOGGER.info("Loading data into memory...")
            for entry in self.data_loaded.values():
                records = self._load_records(str(entry["file_path"]))
                entry["record"] = records[entry["record_idx"]]
            _LOGGER.info("Loaded %s samples into memory", len(self.data_loaded))

        _LOGGER.info("Loaded %s samples", len(self.data_loaded_keys))

    def collate_fn(self, data_list: list) -> dict:
        """Collate a list of scenario samples into a batch dictionary.

        Args:
            data_list: List of per-sample dictionaries.

        Returns:
            Batch dictionary containing:
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
        """Returns the number of samples in the dataset."""
        return len(self.data_loaded_keys)

    @lru_cache(maxsize=128)  # noqa: B019
    def _load_records(self, file_path: str) -> list[dict[str, Any]]:
        """Loads the list of agent-centric records cached for one scenario.

        A small LRU cache bounds memory while avoiding repeated unpickling when consecutive samples come from the same
        scenario file (e.g. multi-record scenarios when ``only_train_on_ego`` is disabled).

        Args:
            file_path: path to the per-scenario agent-centric ``.pkl`` file.

        Returns:
            The cached records (one per center agent).
        """
        with Path(file_path).open("rb") as f:
            return pickle.load(f)  # nosec B301

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Gets the data for a given index.

        Reads the record from the canonical agent-centric cache and injects the per-source ``dataset_name`` tag (used
        downstream by ``BaseModel.split_by_dataset_name`` to report metrics per evaluation source).

        Args:
            idx: index of the data to retrieve.

        Returns:
            Dictionary containing the data for the given index.
        """
        entry = self.data_loaded[self.data_loaded_keys[idx]]
        if "record" in entry:
            record = entry["record"]
        else:
            record = self._load_records(str(entry["file_path"]))[entry["record_idx"]]
        record = dict(record)
        record["dataset_name"] = f"waymo-{entry['tag']}"
        return record
