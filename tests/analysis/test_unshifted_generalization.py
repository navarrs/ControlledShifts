"""Tests for the unshifted-generalization analysis (single reference vs many benchmarks)."""

import logging

import pandas as pd
from omegaconf import OmegaConf

from controlledshifts.utils.analysis.distribution_shift import build_benchmark_df
from controlledshifts.utils.analysis.unshifted_generalization import (
    Block,
    _build_block_rows,
    run_unshifted_generalization_analysis,
)


REF_SPLIT = "val/waymo-ref"
BENCH_SPLIT = "test/waymo-bench"
METRICS = ["minADE6"]


def _write_results_csv(csv_path) -> None:
    """Two models; the benchmark doubles model A's metric (gap +100%) and halves model B's (gap -50%)."""
    frame = pd.DataFrame(
        {
            "Name": ["mini_naive", "mini_mtr"],
            f"{REF_SPLIT}/minADE6": [1.0, 2.0],
            f"{BENCH_SPLIT}/minADE6": [2.0, 1.0],
        }
    )
    frame.to_csv(csv_path, index=False)


def _make_config(csv_path, output_path):
    return OmegaConf.create(
        {
            "analysis_name": "unshifted_generalization",
            "benchmarks_filepath": str(csv_path),
            "output_path": str(output_path),
            "benchmark_colormap": "colorblind",
            "show_run_id": False,
            "models_to_compare": ["naive", "mtr"],
            "trajectory_forecasting_metrics": METRICS,
            "reference": {"name": "Ref", "split": REF_SPLIT},
            "benchmarks": [{"bench": {"name": "Bench", "split": BENCH_SPLIT}}],
        }
    )


def test_full_run_writes_table_and_plots(tmp_path):
    """A full run writes results.tex and both figures, and the table carries the expected gap annotations."""
    csv_path = tmp_path / "mini.csv"
    _write_results_csv(csv_path)
    output_path = tmp_path / "out"

    run_unshifted_generalization_analysis(_make_config(csv_path, output_path), logging.getLogger(__name__), output_path)

    assert (output_path / "results.tex").exists()
    assert (output_path / "benchmark_values.png").exists()
    assert (output_path / "gap_heatmap.png").exists()

    table = (output_path / "results.tex").read_text()
    assert "+100.00\\%" in table  # model A degrades by 100% on the benchmark
    assert "-50.00\\%" in table  # model B improves by 50%
    assert "ForestGreen" in table and "OrangeRed" in table


def test_block_rows_gap_and_bolding():
    """The benchmark block annotates gaps relative to the reference and bolds the best (minimum) value."""
    metrics_df = pd.DataFrame(
        {
            "Name": ["mini_naive", "mini_mtr"],
            f"{REF_SPLIT}/minADE6": [1.0, 2.0],
            f"{BENCH_SPLIT}/minADE6": [2.0, 1.0],
        }
    )
    df = build_benchmark_df(metrics_df, (REF_SPLIT, BENCH_SPLIT), METRICS, ["naive", "mtr"], show_run_id=False)

    ref_rows = _build_block_rows(df, Block("Ref", REF_SPLIT), REF_SPLIT, METRICS, is_reference=True)
    assert all("\\%" not in row for row in ref_rows[:-1])  # reference block carries no gap annotations
    assert "\\textbf{1.000}" in ref_rows[0]  # best reference value is bolded

    bench_rows = _build_block_rows(df, Block("Bench", BENCH_SPLIT), REF_SPLIT, METRICS, is_reference=False)
    assert "+100.00\\%" in bench_rows[0]
    assert "-50.00\\%" in bench_rows[1]
    assert "\\textbf{1.000}" in bench_rows[1]  # best benchmark value (model B) is bolded
