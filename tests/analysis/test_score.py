"""Tests for distribution-shift scaled-error (robustness) scoring."""

import logging

import matplotlib as mpl
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


mpl.use("Agg")

from controlledshifts.utils.analysis.robustness_scores import (
    COMBINED_COLUMN,
    COMBINED_TERM,
    ID_TERM,
    OOD_TERM,
    _geometric_mean,
    _geometric_mean_combined,
    _metric_label,
    _score,
    compute_robustness_scores,
    run_robustness_scores_analysis,
)


def _metrics_df(records) -> pd.DataFrame:
    """Build a combined-results frame from per-run dicts; missing columns become NaN."""
    return pd.DataFrame(records)


def _frame(values: dict[str, list[float]]) -> pd.DataFrame:
    """Build a per-model frame indexed by ``Model`` (rows A, B, ...) for the standardization tests."""
    index = pd.Index([chr(ord("A") + i) for i in range(len(next(iter(values.values()))))], name="Model")
    return pd.DataFrame(values, index=index)


def test_score_value_and_guards():
    # Lower model error than the reference -> > 1; reference vs itself -> 1.0; worse -> < 1.
    assert np.isclose(_score(5.0, 10.0), 2.0)  # better than the reference
    assert np.isclose(_score(10.0, 10.0), 1.0)  # equal / reference vs itself
    assert np.isclose(_score(20.0, 10.0), 0.5)  # worse than the reference
    # NaN on missing or non-positive arguments.
    assert np.isnan(_score(np.nan, 10.0))
    assert np.isnan(_score(5.0, 0.0))
    assert np.isnan(_score(-1.0, 10.0))
    assert np.isnan(_score(5.0, -1.0))


def test_naive_relative_id_and_ood_values():
    # Single benchmark; reference is Naive within the benchmark.
    records = [
        {"Name": "bench_naive", "s_b/m1": 10.0, "u_b/m1": 12.0},
        {"Name": "bench_autobot", "s_b/m1": 5.0, "u_b/m1": 8.0},
    ]
    benchmarks = [("bench", "Bench", "s_b", "u_b")]
    scores = compute_robustness_scores(
        _metrics_df(records),
        benchmarks,
        ["m1"],
        ["naive", "autobot"],
        reference_mode="naive_relative",
    )

    # Naive vs itself: both axes are exactly 1.0.
    for term in (ID_TERM, OOD_TERM):
        assert np.isclose(scores[term].at["Naive", "m1"], 1.0)

    assert np.isclose(scores[ID_TERM].at["AutoBot", "m1"], 10.0 / 5.0)
    assert np.isclose(scores[OOD_TERM].at["AutoBot", "m1"], 12.0 / 8.0)


def test_uniform_relative_references_same_model_and_excludes_uniform():
    records = [
        {"Name": "uniform_naive", "s_uni/m1": 11.0, "u_uni/m1": 11.0},
        {"Name": "uniform_autobot", "s_uni/m1": 4.0, "u_uni/m1": 5.0},
        {"Name": "bench_naive", "s_b/m1": 10.0, "u_b/m1": 12.0},
        {"Name": "bench_autobot", "s_b/m1": 5.0, "u_b/m1": 8.0},
    ]
    benchmarks = [
        ("uniform", "Uniform", "s_uni", "u_uni"),
        ("bench", "Bench", "s_b", "u_b"),
    ]
    scores = compute_robustness_scores(
        _metrics_df(records),
        benchmarks,
        ["m1"],
        ["naive", "autobot"],
        reference_mode="uniform_relative",
    )

    # AutoBot scaled by AutoBot-in-Uniform (seen=4, unseen=5), not by Naive; Uniform benchmark excluded.
    assert np.isclose(scores[ID_TERM].at["AutoBot", "m1"], 4.0 / 5.0)
    assert np.isclose(scores[OOD_TERM].at["AutoBot", "m1"], 5.0 / 8.0)
    # Differs from the naive_relative ID term, confirming a different reference is used.
    assert not np.isclose(scores[ID_TERM].at["AutoBot", "m1"], 10.0 / 5.0)


def test_nan_propagation_skips_missing_benchmark_cells():
    # bench_autobot has no unseen value -> OOD NaN there and skipped; only the valid benchmark counts.
    records = [
        {"Name": "good_naive", "s_g/m1": 10.0, "u_g/m1": 11.0},
        {"Name": "good_autobot", "s_g/m1": 5.0, "u_g/m1": 6.0},
        {"Name": "bad_naive", "s_x/m1": 10.0, "u_x/m1": 11.0},
        {"Name": "bad_autobot", "s_x/m1": 5.0, "u_x/m1": np.nan},
    ]
    benchmarks = [
        ("good", "Good", "s_g", "u_g"),
        ("bad", "Bad", "s_x", "u_x"),
    ]
    scores = compute_robustness_scores(
        _metrics_df(records),
        benchmarks,
        ["m1"],
        ["naive", "autobot"],
        reference_mode="naive_relative",
    )

    # AutoBot's "good" benchmark cell is valid; the "bad" cell is NaN and dropped from the mean.
    assert np.isclose(scores[OOD_TERM].at["AutoBot", "m1"], 11.0 / 6.0)


def test_geometric_mean_value_and_guards():
    left = pd.Series([4.0, 1.0, np.nan, 2.0, -1.0])
    right = pd.Series([9.0, 0.0, 3.0, np.nan, 5.0])
    result = _geometric_mean(left, right)
    assert np.isclose(result.iloc[0], 6.0)  # sqrt(4 * 9)
    assert np.isnan(result.iloc[1])  # non-positive factor -> NaN
    assert np.isnan(result.iloc[2])  # missing left -> NaN
    assert np.isnan(result.iloc[3])  # missing right -> NaN
    assert np.isnan(result.iloc[4])  # negative factor -> NaN


def test_geometric_mean_combined_is_per_metric_gm():
    # combined[metric] = sqrt(id * ood); Combined = mean across metrics.
    id_df = _frame({"m1": [4.0], "m2": [1.0]})
    ood_df = _frame({"m1": [9.0], "m2": [9.0]})
    combined = _geometric_mean_combined(id_df, ood_df, ["m1", "m2"])
    assert np.isclose(combined.at["A", "m1"], 6.0)  # sqrt(4*9)
    assert np.isclose(combined.at["A", "m2"], 3.0)  # sqrt(1*9)
    assert np.isclose(combined.at["A", COMBINED_COLUMN], 4.5)  # mean(6, 3)


def test_geometric_mean_combined_drops_metric_with_missing_axis():
    # m1 has a missing ID cell -> its GM is undefined -> that metric drops from the Combined mean.
    id_df = _frame({"m1": [np.nan], "m2": [4.0]})
    ood_df = _frame({"m1": [5.0], "m2": [9.0]})
    combined = _geometric_mean_combined(id_df, ood_df, ["m1", "m2"])
    assert np.isnan(combined.at["A", "m1"])
    assert np.isclose(combined.at["A", "m2"], 6.0)  # sqrt(4*9)
    assert np.isclose(combined.at["A", COMBINED_COLUMN], 6.0)  # only m2 contributes


def test_combined_demotes_naive_and_keeps_worse_model_below():
    # Naive-relative combined: Naive (its own reference) is pinned at 1.0; a strong model's GM beats it, and a model
    # genuinely worse than Naive stays below 1.0.
    records = [
        {"Name": "bench_naive", "s_b/m1": 10.0, "u_b/m1": 11.0},
        {"Name": "bench_good", "s_b/m1": 1.0, "u_b/m1": 2.0},
        {"Name": "bench_weak", "s_b/m1": 12.0, "u_b/m1": 13.0},
    ]
    benchmarks = [("bench", "Bench", "s_b", "u_b")]
    scores = compute_robustness_scores(
        _metrics_df(records),
        benchmarks,
        ["m1"],
        ["naive", "good", "weak"],
        reference_mode="naive_relative",
    )
    combined = scores[COMBINED_TERM][COMBINED_COLUMN]
    assert np.isclose(combined.at["Naive"], 1.0)  # reference sits exactly on 1.0
    assert np.isclose(combined.at["good"], np.sqrt((10.0 / 1.0) * (11.0 / 2.0)))  # sqrt(id * ood)
    assert combined.at["good"] > combined.at["Naive"]  # strong model beats the baseline
    assert combined.at["weak"] < combined.at["Naive"]  # genuinely-worse model stays below it


def test_metric_label_uses_clean_name():
    assert _metric_label("brierFDE") == "BrierFDE"
    assert _metric_label("minADE6") == "MinADE"
    assert "↓" not in _metric_label("brierFDE")
    assert _metric_label("unmapped_metric") == "unmapped_metric"  # falls back to raw name


def test_run_score_analysis_writes_artifacts(tmp_path):
    records = [
        {"Name": "uniform_naive", "test/s_uni/m1": 11.0, "test/u_uni/m1": 11.0},
        {"Name": "uniform_autobot", "test/s_uni/m1": 4.0, "test/u_uni/m1": 5.0},
        {"Name": "bench_naive", "test/s_b/m1": 10.0, "test/u_b/m1": 12.0},
        {"Name": "bench_autobot", "test/s_b/m1": 5.0, "test/u_b/m1": 8.0},
    ]
    csv_path = tmp_path / "results.csv"
    _metrics_df(records).to_csv(csv_path, index=False)

    config = OmegaConf.create(
        {
            "benchmarks_filepath": str(csv_path),
            "trajectory_forecasting_metrics": ["m1"],
            "models_to_compare": ["naive", "autobot"],
            "score_colormap": "colorblind",
            "benchmarks": [
                {"uniform": {"name": "Uniform", "seen": "test/s_uni", "unseen": "test/u_uni"}},
                {"bench": {"name": "Bench", "seen": "test/s_b", "unseen": "test/u_b"}},
            ],
            "score": {
                "reference_modes": ["naive_relative", "uniform_relative"],
                "uniform_key": "uniform",
                "aggregate": "mean",
            },
        }
    )

    output_path = tmp_path / "out"
    run_robustness_scores_analysis(config, logging.getLogger("test"), output_path)

    for folder in ("quality_naive", "stability_uniform"):
        for stem in ("seen_score", "unseen_score"):
            assert (output_path / folder / f"{stem}_radar.png").exists()
            assert (output_path / folder / f"{stem}_scores.csv").exists()
            assert (output_path / folder / f"{stem}_scores.tex").exists()
        assert (output_path / folder / "score_decomposition.png").exists()
        assert (output_path / folder / "combined_robustness_ranking.png").exists()
        assert (output_path / folder / "combined_robustness_scores.csv").exists()
        assert (output_path / folder / "combined_robustness_scores.tex").exists()

    # Single combined Quality-vs-Stability summary at the top level (no per-mode summary files).
    assert (output_path / "robustness_summary.png").exists()
    for folder in ("quality_naive", "stability_uniform"):
        assert not (output_path / folder / "robustness_summary.png").exists()
