"""Tests for the per-scenario model-output cache (`save_cache` / `load_batches`)."""

import pickle
from pathlib import Path

import pytest
import torch

from controlledshifts.schemas.output_schemas import ModelOutput, ScenarioEmbedding
from controlledshifts.utils.data_utils import load_batches, load_batches_for_ids, load_batches_per_model, save_cache
from controlledshifts.utils.model_runs import ModelCacheSpec


# `_build_batch` writes under the default `waymo` source; the scenario ids the round-trip tests may request.
_ALL_IDS = [f"scen{suffix}" for suffix in ("A", "B", "C", "D")] + [f"scen{i}" for i in range(20)]


def _specs(**caches: Path) -> list[ModelCacheSpec]:
    """One flat (unlabelled row) spec per named cache directory."""
    return [ModelCacheSpec(name=name, cache_path=path, source="waymo") for name, path in caches.items()]


def _build_batch(scenario_ids: list[str], dataset_names: list[str] | None = None) -> ModelOutput:
    """Builds a minimal batched `ModelOutput` with a leading batch dim equal to len(scenario_ids)."""
    n = len(scenario_ids)
    return ModelOutput(
        scenario_embedding=ScenarioEmbedding(scenario_enc=torch.rand(n, 8), scenario_dec=torch.rand(n, 8)),
        history_ground_truth=torch.rand(n, 11, 2),
        future_ground_truth=torch.rand(n, 5, 2),
        dataset_name=["waymo"] * n if dataset_names is None else list(dataset_names),
        scenario_id=list(scenario_ids),
        agent_ids=torch.zeros(n, 4, dtype=torch.int32),
    )


def test_save_cache_writes_one_file_per_scenario_namespaced_by_source(tmp_path):
    save_cache(_build_batch(["scenA", "scenB"]), tmp_path, "val")

    # Outputs are namespaced by source (dataset_name) so variants of the same scene cannot overwrite each other.
    source_dir = tmp_path / "val" / "waymo"
    assert sorted(p.name for p in source_dir.glob("*.pkl")) == ["scenA.pkl", "scenB.pkl"]


def test_save_cache_strips_batch_dim_and_moves_to_cpu(tmp_path):
    batch = _build_batch(["scenA", "scenB"])
    save_cache(batch, tmp_path, "val")

    with (tmp_path / "val" / "waymo" / "scenA.pkl").open("rb") as f:
        loaded = pickle.load(f)  # nosec B301

    assert loaded.scenario_id == ["scenA"]
    # resplit_batch indexes [n], so the per-scenario output drops the leading batch dimension.
    assert loaded.history_ground_truth.value.shape == batch.history_ground_truth.value.shape[1:]
    assert loaded.history_ground_truth.value.device.type == "cpu"


def test_load_batches_round_trip(tmp_path):
    batch = _build_batch(["scenA", "scenB"])
    save_cache(batch, tmp_path, "val")

    loaded = load_batches(tmp_path, num_batches=None, num_scenarios=None, seed=0, tag="val")

    assert set(loaded) == {"scenA", "scenB"}
    for scenario_id, scenario_output in loaded.items():
        assert scenario_output.scenario_id[0] == scenario_id
        # The loader must not resplit again: tensors keep their per-scenario rank (history stays 2D, not 1D).
        assert scenario_output.history_ground_truth.value.shape == batch.history_ground_truth.value.shape[1:]


def test_duplicate_scenario_id_within_one_source_overwrites(tmp_path):
    save_cache(_build_batch(["dup", "dup"]), tmp_path, "val")

    assert [p.name for p in (tmp_path / "val" / "waymo").glob("*.pkl")] == ["dup.pkl"]


def test_same_scenario_id_from_two_sources_both_survive(tmp_path):
    # causal-agents evaluates the base and remove_noncausal variants of the SAME scene ids in its test split. Keying
    # cache files on the scenario id alone silently kept only whichever variant was written last.
    save_cache(_build_batch(["scenA"], ["base"]), tmp_path, "test")
    save_cache(_build_batch(["scenA"], ["remove_noncausal"]), tmp_path, "test")

    assert (tmp_path / "test" / "base" / "scenA.pkl").is_file()
    assert (tmp_path / "test" / "remove_noncausal" / "scenA.pkl").is_file()


def test_load_batches_source_selects_one_variant(tmp_path):
    save_cache(_build_batch(["scenA", "scenB"], ["base", "base"]), tmp_path, "test")
    save_cache(_build_batch(["scenA"], ["remove_noncausal"]), tmp_path, "test")

    base = load_batches(tmp_path, num_batches=None, num_scenarios=None, seed=0, tag="test", source="base")
    perturbed = load_batches(
        tmp_path, num_batches=None, num_scenarios=None, seed=0, tag="test", source="remove_noncausal"
    )

    assert set(base) == {"scenA", "scenB"}
    assert set(perturbed) == {"scenA"}
    assert base["scenA"].dataset_name == ["base"]
    assert perturbed["scenA"].dataset_name == ["remove_noncausal"]


def test_load_batches_without_source_spans_every_source(tmp_path):
    save_cache(_build_batch(["scenA"], ["base"]), tmp_path, "test")
    save_cache(_build_batch(["scenB"], ["remove_noncausal"]), tmp_path, "test")

    loaded = load_batches(tmp_path, num_batches=None, num_scenarios=None, seed=0, tag="test")

    # Non-overlapping ids across sources: every scenario is reachable without naming a source.
    assert set(loaded) == {"scenA", "scenB"}


def test_load_batches_for_ids_loads_only_the_requested_scenarios(tmp_path):
    save_cache(_build_batch(["scenA", "scenB", "scenC"]), tmp_path, "val")

    loaded = load_batches_for_ids(tmp_path, ["scenA", "scenC", "absent"], tag="val", source="waymo")

    # Absent ids are skipped rather than raising: a pane simply has nothing cached for them.
    assert set(loaded) == {"scenA", "scenC"}


def test_load_batches_per_model_intersects_and_aligns(tmp_path):
    dir_a, dir_b = tmp_path / "modelA", tmp_path / "modelB"
    save_cache(_build_batch(["scenA", "scenB", "scenC"]), dir_a, "val")
    save_cache(_build_batch(["scenB", "scenC", "scenD"]), dir_b, "val")

    result = load_batches_per_model(_specs(A=dir_a, B=dir_b), _ALL_IDS, num_scenarios=None, seed=0, tag="val")

    # Only the scenarios cached by every pane are returned, each carrying an output from every pane.
    assert set(result) == {"scenB", "scenC"}
    for rows in result.values():
        assert set(rows[""]) == {"A", "B"}


def test_load_batches_per_model_samples_within_the_requested_ids(tmp_path):
    """Sampling must happen inside the whitelist, not across the whole cache and filtered afterwards."""
    dir_a = tmp_path / "modelA"
    save_cache(_build_batch([f"scen{i}" for i in range(20)]), dir_a, "val")

    whitelist = ["scen3", "scen7"]
    for seed in range(8):
        result = load_batches_per_model(_specs(A=dir_a), whitelist, num_scenarios=1, seed=seed, tag="val")
        assert set(result) <= set(whitelist)


def test_load_batches_per_model_sampling_is_deterministic(tmp_path):
    dir_a, dir_b = tmp_path / "modelA", tmp_path / "modelB"
    ids = [f"scen{i}" for i in range(10)]
    save_cache(_build_batch(ids), dir_a, "val")
    save_cache(_build_batch(ids), dir_b, "val")

    specs = _specs(A=dir_a, B=dir_b)
    first = load_batches_per_model(specs, ids, num_scenarios=3, seed=0, tag="val")
    second = load_batches_per_model(specs, ids, num_scenarios=3, seed=0, tag="val")

    assert len(first) == 3
    assert first.keys() == second.keys()


def test_load_batches_per_model_groups_panes_into_rows_and_columns(tmp_path):
    dir_a, dir_b = tmp_path / "uniformWayformer", tmp_path / "envWayformer"
    save_cache(_build_batch(["scenA"]), dir_a, "val")
    save_cache(_build_batch(["scenA"]), dir_b, "val")

    specs = [
        ModelCacheSpec(name="wayformer", cache_path=dir_a, source="waymo", group="Uniform"),
        ModelCacheSpec(name="wayformer", cache_path=dir_b, source="waymo", group="Environments"),
    ]
    result = load_batches_per_model(specs, ["scenA"], num_scenarios=None, seed=0, tag="val")

    # One row per benchmark, one column per model within it.
    assert list(result["scenA"]) == ["Uniform", "Environments"]
    assert list(result["scenA"]["Uniform"]) == ["wayformer"]


def test_load_batches_per_model_raises_without_overlap(tmp_path):
    dir_a, dir_b = tmp_path / "modelA", tmp_path / "modelB"
    save_cache(_build_batch(["scenA"]), dir_a, "val")
    save_cache(_build_batch(["scenB"]), dir_b, "val")

    with pytest.raises(ValueError, match="cached by every pane"):
        load_batches_per_model(_specs(A=dir_a, B=dir_b), _ALL_IDS, num_scenarios=None, seed=0, tag="val")
