"""Tests for the agent-centric processing/builder/loader pipeline (Stages B, C, D)."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from controlledshifts.datasets import agent_centric_cacher, agent_centric_processor
from controlledshifts.datasets.agent_centric_processor import (
    AgentCentricProcessor,
    processing_profile,
    processing_profile_hash,
)
from controlledshifts.datasets.base_dataset import BaseDataset
from controlledshifts.datasets.waymo.repacker import repack_scenario
from controlledshifts.utils.constants import DataSplits


TOTAL_STEPS = 91  # past_len(11) + future_len(80), matching the base config below.


def _base_config(tmp_path: Path, **overrides) -> OmegaConf:
    """Creates a minimal dataset config covering the keys read by the loader and the processing profile."""
    config = {
        "load_data": False,
        "split": None,
        "num_data_to_consider": None,
        "seed": 0,
        "past_len": 11,
        "future_len": 80,
        "ac_cache_root": str(tmp_path / "ac_cache"),
        "splits_root": str(tmp_path / "splits"),
        "profile_alias": "test",
        "train_sources": None,
        "val_sources": None,
        "test_sources": None,
        "sample_selection_strategy": "all",
        "sample_selection_filepath": None,
        "store_data_in_memory": False,
        # Tensor-affecting (profile) keys.
        "max_num_agents": 32,
        "max_num_roads": 384,
        "max_points_per_lane": 30,
        "map_range": 100,
        "center_offset_of_map": [30.0, 0.0],
        "manually_split_lane": False,
        "point_sampled_interval": 1,
        "num_points_each_polyline": 20,
        "vector_break_dist_thresh": 1.0,
        "total_map_types": 20,
        "object_type": ["TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"],
        "line_type": ["lane", "stop_sign", "road_edge", "road_line", "crosswalk", "speed_bump"],
        "only_train_on_ego": True,
        "trajectory_sample_interval": 1,
        "masked_attributes": ["z_axis", "size"],
        "autolabel_agents": False,
        "causal_labels_path": str(tmp_path / "labels"),
    }
    config.update(overrides)
    cfg = OmegaConf.create(config)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _raw_scenario(scenario_id: str, *, ego_valid: bool) -> dict:
    """Builds a minimal decoded Waymo raw scenario dict (one ego vehicle) suitable for repack + processing."""
    trajs = np.zeros((1, TOTAL_STEPS, 10), dtype=np.float32)
    if ego_valid:
        trajs[0, :, -1] = 1.0  # mark all timesteps valid
    return {
        "scenario_id": scenario_id,
        "timestamps_seconds": [i * 0.1 for i in range(TOTAL_STEPS)],
        "current_time_index": 10,
        "sdc_track_index": 0,
        "objects_of_interest": [],
        "tracks_to_predict": {"track_index": [0], "difficulty": [0], "object_type": ["TYPE_VEHICLE"]},
        "track_infos": {"object_id": [123], "object_type": ["TYPE_VEHICLE"], "trajs": trajs},
        # Mirror decode_map_features_from_proto, which always emits these keys (empty lists when absent).
        "map_infos": {
            "all_polylines": np.zeros((2, 7), dtype=np.float32),
            "lane": [],
            "road_line": [],
            "road_edge": [],
            "stop_sign": [],
            "crosswalk": [],
            "speed_bump": [],
        },
        "dynamic_map_infos": {"stop_point": [], "lane_id": [], "state": []},
    }


def _record(scenario_id: str, kalman: tuple[float, float, float]) -> dict:
    """Builds a minimal agent-centric record with the fields the loader/index rely on."""
    return {
        "scenario_id": np.array([scenario_id]),
        "kalman_difficulty": np.array(kalman, dtype=np.float32),
        "obj_trajs": np.zeros((2, 3), dtype=np.float32),
    }


def _write_fake_cache(tmp_path: Path, cfg, variant: str, scenarios: dict[str, list | None]) -> Path:
    """Writes a fake agent-centric cache (records + _index.pkl + _profile.json) for the loader tests."""
    cache_dir = Path(cfg.ac_cache_root) / variant / processing_profile_hash(cfg)
    (cache_dir / "scenarios").mkdir(parents=True, exist_ok=True)
    index_scenarios, empty = {}, []
    for scenario_id, records in scenarios.items():
        rel = f"scenarios/{scenario_id}.pkl"
        if records:
            with (cache_dir / rel).open("wb") as f:
                pickle.dump(records, f)
            kd = np.stack([r["kalman_difficulty"] for r in records])
            index_scenarios[scenario_id] = {"num_records": len(records), "kalman_difficulty": kd, "rel_path": rel}
        else:
            index_scenarios[scenario_id] = {
                "num_records": 0,
                "kalman_difficulty": np.zeros((0, 3), dtype=np.float32),
                "rel_path": rel,
            }
            empty.append(scenario_id)
    with (cache_dir / "_index.pkl").open("wb") as f:
        pickle.dump({"variant": variant, "scenarios": index_scenarios, "empty_scenarios": empty}, f)
    with (cache_dir / "_profile.json").open("w") as f:
        json.dump(processing_profile(cfg), f, sort_keys=True, default=str)
    return cache_dir


def _write_split(tmp_path: Path, name: str, **splits) -> None:
    """Writes a benchmark split JSON under splits_root."""
    splits_root = tmp_path / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)
    payload = {"training": [], "validation": [], "testing": [], "invalid": [], "benchmark_name": name}
    payload.update(splits)
    with (splits_root / f"{name}.json").open("w") as f:
        json.dump(payload, f)


# --------------------------------------------------------------------------------------------------------------------
# Processing profile / cache key
# --------------------------------------------------------------------------------------------------------------------


def test_profile_hash_ignores_selection_keys(tmp_path):
    """Selection-only keys (num_data_to_consider, sample_selection, seed) must not change the cache identity."""
    base = processing_profile_hash(_base_config(tmp_path))
    other = processing_profile_hash(
        _base_config(tmp_path, num_data_to_consider=10, sample_selection_strategy="random_drop", seed=123)
    )
    assert base == other


def test_profile_hash_tracks_tensor_keys(tmp_path):
    """Tensor-affecting keys (e.g. max_num_agents, masked_attributes) must change the cache identity."""
    base = processing_profile_hash(_base_config(tmp_path))
    assert base != processing_profile_hash(_base_config(tmp_path, max_num_agents=64))
    assert base != processing_profile_hash(_base_config(tmp_path, masked_attributes=["z_axis"]))


def test_profile_keys_are_actually_read_by_the_transform():
    """Guard: every PROFILE_KEYS entry should be a config key the agent-centric module actually reads."""
    source = Path(agent_centric_processor.__file__).read_text()
    for key in agent_centric_processor.PROFILE_KEYS:
        assert f"config.{key}" in source or f'get("{key}"' in source, f"PROFILE_KEYS entry not read: {key}"


# --------------------------------------------------------------------------------------------------------------------
# Repack (raw dict -> Scenario) and profile shaping
# --------------------------------------------------------------------------------------------------------------------


def test_repack_scenario_builds_open_scenario():
    """repack_scenario converts a decoded raw dict into a Scenario with full (unshaped) trajectories."""
    scenario = repack_scenario(_raw_scenario("s0", ego_valid=True))
    assert scenario.metadata.scenario_id == "s0"
    assert scenario.metadata.ego_vehicle_index == 0
    assert scenario.agent_data.agent_trajectories.shape == (1, TOTAL_STEPS, 10)
    # Thin repack keeps the full track_to_predict and does not apply the only_train_on_ego shaping.
    assert scenario.tracks_to_predict.track_index == [0]


def test_shape_scenario_applies_profile_shaping(tmp_path):
    """shape_scenario applies the trajectory_sample_interval mask and the only_train_on_ego track selection."""
    processor = AgentCentricProcessor(_base_config(tmp_path, trajectory_sample_interval=2))
    scenario = processor.shape_scenario(repack_scenario(_raw_scenario("s0", ego_valid=True)))
    # only_train_on_ego -> a single ego track.
    assert scenario.tracks_to_predict.track_index == [scenario.metadata.ego_vehicle_index]
    # trajectory_sample_interval=2 zeroes validity at the non-sampled history steps (mask is not all-ones).
    validity = scenario.agent_data.agent_trajectories[0, :, -1]
    assert validity.min() == 0.0


# --------------------------------------------------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------------------------------------------------


def test_build_variant_cache_writes_index_and_profile(tmp_path):
    """The builder repacks + processes each raw scenario, writing an index and a matching profile.

    Uses an ego-invalid scenario so the transform yields no records, which deterministically exercises the index /
    empty-scenario / profile plumbing without depending on the full agent-centric transform succeeding.
    """
    cfg = _base_config(tmp_path)
    variant_dir = tmp_path / "variants" / "base"
    variant_dir.mkdir(parents=True, exist_ok=True)
    with (variant_dir / "s0.pkl").open("wb") as f:
        pickle.dump(_raw_scenario("s0", ego_valid=False), f)

    processor = AgentCentricProcessor(cfg)
    cache_dir = Path(cfg.ac_cache_root) / "base" / processor.processing_profile_hash()
    index = agent_centric_cacher.build_variant_cache(
        processor, "base", variant_dir, cache_dir, num_workers=1, overwrite=False
    )

    assert index["scenarios"]["s0"]["num_records"] == 0
    assert index["empty_scenarios"] == ["s0"]
    assert (cache_dir / "_index.pkl").exists()

    with (cache_dir / "_profile.json").open("r") as f:
        cached_profile = json.load(f)
    expected = json.loads(json.dumps(processing_profile(cfg), sort_keys=True, default=str))
    assert cached_profile == expected


# --------------------------------------------------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------------------------------------------------


def test_loader_assembles_samples_across_composite_sources(tmp_path):
    """Composite test sources combine variants, skip missing/empty ids, and tag each sample by source."""
    cfg = _base_config(tmp_path, load_data=True)
    _write_fake_cache(
        tmp_path, cfg, "base", {"s0": [_record("s0", (1, 2, 3))], "s1": [_record("s1", (1, 2, 3))], "s2": None}
    )
    _write_fake_cache(tmp_path, cfg, "pert", {"s0": [_record("s0", (1, 2, 3))], "s1": [_record("s1", (1, 2, 3))]})
    # 'sX' is in the split but absent from both caches; 's2' is present but empty in base.
    _write_split(tmp_path, "uniform", testing=["s0", "s1", "s2", "sX"])

    cfg.test_sources = [
        {"variant": "base", "split_json": "uniform", "split": "testing", "tag": "base-testing"},
        {"variant": "pert", "split_json": "uniform", "split": "testing", "tag": "pert-testing"},
    ]
    cfg.split = DataSplits.TESTING
    dataset = BaseDataset(cfg)

    # base contributes s0,s1 (s2 empty, sX missing) = 2; pert contributes s0,s1 = 2.
    assert len(dataset) == 4
    tags = {dataset[i]["dataset_name"] for i in range(len(dataset))}
    assert tags == {"waymo-base-testing", "waymo-pert-testing"}
    assert "kalman_difficulty" in dataset[0]


def test_loader_blacklist_and_train_subsample(tmp_path):
    """Blacklisted scenario ids are dropped and the train-only subsample caps the sample count."""
    cfg = _base_config(tmp_path, load_data=True, num_data_to_consider=2)
    ids = [f"s{i}" for i in range(6)]
    _write_fake_cache(tmp_path, cfg, "base", {sid: [_record(sid, (1, 2, 3))] for sid in ids})
    _write_split(tmp_path, "uniform", training=ids)

    blacklist_path = tmp_path / "blacklist.json"
    with blacklist_path.open("w") as f:
        json.dump({"drop": ["s0", "s1"]}, f)

    cfg.train_sources = [{"variant": "base", "split_json": "uniform", "split": "training", "tag": "base-training"}]
    cfg.sample_selection_strategy = "random_drop"
    cfg.sample_selection_filepath = str(blacklist_path)
    cfg.split = DataSplits.TRAINING
    dataset = BaseDataset(cfg)

    assert len(dataset) == 2  # 6 - 2 blacklisted = 4 eligible, capped to num_data_to_consider=2 (train-only).
    loaded_ids = {dataset.data_loaded[key]["scenario_id"] for key in dataset.data_loaded_keys}
    assert "s0" not in loaded_ids and "s1" not in loaded_ids


def test_loader_missing_cache_raises(tmp_path):
    """Loading a variant with no built cache fails fast with a build hint."""
    cfg = _base_config(tmp_path, load_data=True)
    _write_split(tmp_path, "uniform", validation=["s0"])
    cfg.val_sources = [{"variant": "base", "split_json": "uniform", "split": "validation", "tag": "base-val"}]
    cfg.split = DataSplits.VALIDATION
    with pytest.raises(FileNotFoundError, match="build_ac_cache"):
        BaseDataset(cfg)


def test_loader_profile_mismatch_raises(tmp_path):
    """A cache built under a different processing profile is rejected rather than silently used."""
    cfg = _base_config(tmp_path, load_data=True)
    cache_dir = _write_fake_cache(tmp_path, cfg, "base", {"s0": [_record("s0", (1, 2, 3))]})
    with (cache_dir / "_profile.json").open("r") as f:
        profile = json.load(f)
    profile["max_num_agents"] = 999
    with (cache_dir / "_profile.json").open("w") as f:
        json.dump(profile, f)

    _write_split(tmp_path, "uniform", validation=["s0"])
    cfg.val_sources = [{"variant": "base", "split_json": "uniform", "split": "validation", "tag": "base-val"}]
    cfg.split = DataSplits.VALIDATION
    with pytest.raises(ValueError, match="[Pp]rofile"):
        BaseDataset(cfg)
