"""Tests for the Ego-SafeShift benchmark split creation."""

import pandas as pd
import pytest
from omegaconf import OmegaConf

from controlledshifts.benchmarks import ego_safeshift as ego_safeshift_module
from controlledshifts.benchmarks import resolve_split_name
from controlledshifts.benchmarks.common import Benchmark
from controlledshifts.benchmarks.ego_safeshift import (
    SCORE_COLUMNS,
    create_ego_safeshift_benchmark,
    ego_safeshift_split_name,
)


SCORE_TYPE = "gt_critical_continuous_safeshift"


def _write_scenarios(input_dir, ids) -> None:
    """Creates empty .pkl scenario files (stems only) under input_dir."""
    input_dir.mkdir(parents=True, exist_ok=True)
    for scenario_id in ids:
        (input_dir / f"{scenario_id}.pkl").write_bytes(b"")


def _write_scores_csv(csv_path, ids_with_ext) -> None:
    """Writes a scenario-to-score mapping CSV whose scenario_ids carry the .pkl extension."""
    frame = pd.DataFrame({"scenario_ids": ids_with_ext, SCORE_TYPE: list(range(len(ids_with_ext)))})
    frame.to_csv(csv_path, index=False)


def test_pkl_suffixed_ids_are_placed_not_invalid(tmp_path):
    """CSV scenario_ids carrying the .pkl extension should still match the extensionless input stems."""
    stems = [f"scenario_{i:02d}" for i in range(10)]
    input_dir = tmp_path / "input"
    _write_scenarios(input_dir, stems)

    csv_path = tmp_path / "scenario_to_scores_mapping.csv"
    _write_scores_csv(csv_path, [f"{stem}.pkl" for stem in stems])

    config = OmegaConf.create(
        {
            "input_data_path": str(input_dir),
            "scenario_score_mapping_filepath": str(csv_path),
            "score_type": SCORE_TYPE,
            "split_ratios": [0.70, 0.15, 0.15],
            "seed": 42,
        }
    )

    split = create_ego_safeshift_benchmark(config)

    placed = set(split.training) | set(split.validation) | set(split.testing)
    assert placed == set(stems)  # every scenario landed in a split
    assert split.invalid == []  # the .pkl suffix no longer causes a mismatch
    # the hardest (highest-scored) scenario is the last stem and must be in the test set
    assert "scenario_09" in split.testing


def _fallback_config(tmp_path, input_dir, *, score_csv=None, score_type=SCORE_TYPE):
    """A config that takes the score-computation fallback (no usable scenario_score_mapping_filepath)."""
    return OmegaConf.create(
        {
            "input_data_path": str(input_dir),
            "scenario_score_mapping_filepath": score_csv,
            "score_type": score_type,
            "split_ratios": [0.70, 0.15, 0.15],
            "seed": 42,
            "splits_path": str(tmp_path / "splits"),
            "num_workers": 2,
            "overwrite": True,
            "scoring": {"scenario_characterization": {"feature_type": "categorical"}},
        }
    )


def _fake_scores_frame(stems):
    """A computed-scores frame with the three score columns; the safeshift score increases with the stem index."""
    return pd.DataFrame(
        {
            "scenario_ids": list(stems),
            SCORE_COLUMNS["safeshift"]: list(range(len(stems))),
            SCORE_COLUMNS["individual"]: list(range(len(stems))),
            SCORE_COLUMNS["interaction"]: list(range(len(stems))),
        }
    )


def test_missing_csv_branches_to_fallback(tmp_path, monkeypatch):
    """A null score CSV path computes scores via the fallback, writes the hashed CSV, and produces the split."""
    stems = [f"scenario_{i:02d}" for i in range(10)]
    input_dir = tmp_path / "input"
    _write_scenarios(input_dir, stems)
    config = _fallback_config(tmp_path, input_dir, score_csv=None)

    monkeypatch.setattr(ego_safeshift_module, "compute_scores_dataframe", lambda *a, **k: _fake_scores_frame(stems))

    split = create_ego_safeshift_benchmark(config)

    placed = set(split.training) | set(split.validation) | set(split.testing)
    assert placed == set(stems)
    assert split.invalid == []
    assert "scenario_09" in split.testing  # highest computed score -> test set
    assert ego_safeshift_module._resolve_scores_csv_path(config).exists()  # scores cached to disk


def test_nonexistent_csv_triggers_fallback(tmp_path, monkeypatch):
    """A score CSV path that does not exist on disk also takes the fallback."""
    stems = [f"scenario_{i:02d}" for i in range(10)]
    input_dir = tmp_path / "input"
    _write_scenarios(input_dir, stems)
    config = _fallback_config(tmp_path, input_dir, score_csv=str(tmp_path / "does_not_exist.csv"))

    monkeypatch.setattr(ego_safeshift_module, "compute_scores_dataframe", lambda *a, **k: _fake_scores_frame(stems))

    split = create_ego_safeshift_benchmark(config)
    assert set(split.training) | set(split.validation) | set(split.testing) == set(stems)


def test_invalid_score_type_raises(tmp_path, monkeypatch):
    """A score_type absent from the computed columns raises a clear error."""
    stems = [f"scenario_{i:02d}" for i in range(10)]
    input_dir = tmp_path / "input"
    _write_scenarios(input_dir, stems)
    config = _fallback_config(tmp_path, input_dir, score_csv=None, score_type="gt_critical_categorical_safeshift")

    monkeypatch.setattr(ego_safeshift_module, "compute_scores_dataframe", lambda *a, **k: _fake_scores_frame(stems))

    with pytest.raises(ValueError, match="not among the computed score columns"):
        create_ego_safeshift_benchmark(config)


def test_failed_scenarios_recorded_invalid(tmp_path, monkeypatch):
    """On-disk scenarios that failed to score (absent from the computed frame) are recorded as invalid."""
    stems = [f"scenario_{i:02d}" for i in range(10)]
    input_dir = tmp_path / "input"
    _write_scenarios(input_dir, stems)
    config = _fallback_config(tmp_path, input_dir, score_csv=None)

    scored = stems[:-1]  # scenario_09 fails to score
    monkeypatch.setattr(ego_safeshift_module, "compute_scores_dataframe", lambda *a, **k: _fake_scores_frame(scored))

    split = create_ego_safeshift_benchmark(config)
    placed = set(split.training) | set(split.validation) | set(split.testing)
    assert "scenario_09" not in placed
    assert "scenario_09" in split.invalid


def test_split_name_explicit_overrides(tmp_path):
    """An explicit split_name is used verbatim as the split filename stem."""
    config = _fallback_config(tmp_path, tmp_path / "input", score_csv=None)
    config.split_name = "ego_safeshift_scores8"
    assert ego_safeshift_split_name(config) == "ego_safeshift_scores8"
    assert resolve_split_name(Benchmark.EGO_SAFESHIFT, config) == "ego_safeshift_scores8"


def test_split_name_auto_derived_varies_by_scoring(tmp_path):
    """Without split_name, the auto-derived name is stable per scoring config and differs across scoring configs."""
    config = _fallback_config(tmp_path, tmp_path / "input", score_csv=None)
    name = ego_safeshift_split_name(config)
    assert name.startswith("ego_safeshift_")
    assert ego_safeshift_split_name(config) == name  # stable

    other = _fallback_config(tmp_path, tmp_path / "input", score_csv=None)
    other.scoring.scenario_characterization.feature_type = "continuous"
    assert ego_safeshift_split_name(other) != name  # different scoring config -> different split file


def test_split_name_varies_by_csv(tmp_path):
    """Two different provided score CSVs yield different auto-derived split names."""
    csv_a, csv_b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_scores_csv(csv_a, [f"scenario_{i:02d}.pkl" for i in range(10)])
    pd.DataFrame({"scenario_ids": ["scenario_00.pkl"], SCORE_TYPE: [1]}).to_csv(csv_b, index=False)

    config_a = _fallback_config(tmp_path, tmp_path / "input", score_csv=str(csv_a))
    config_b = _fallback_config(tmp_path, tmp_path / "input", score_csv=str(csv_b))
    assert ego_safeshift_split_name(config_a) != ego_safeshift_split_name(config_b)


def test_resolve_split_name_non_ego_uses_benchmark_name(tmp_path):
    """Non-Ego-SafeShift benchmarks keep using benchmark_name as the split filename stem."""
    config = OmegaConf.create({"benchmark_name": "uniform"})
    assert resolve_split_name(Benchmark.UNIFORM, config) == "uniform"
