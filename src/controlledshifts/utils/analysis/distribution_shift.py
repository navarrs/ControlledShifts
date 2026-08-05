"""Distribution-shift / benchmark analysis utilities (ID vs OOD comparisons, LaTeX tables).

See `docs/ANALYSIS.md` for usage details.
"""

import math
from itertools import product
from logging import Logger
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import NDArray
from omegaconf import DictConfig

from controlledshifts.utils.analysis.common import (
    COMPACT_ANNOT_FONTSIZE,
    COMPACT_LABEL_FONTSIZE,
    COMPACT_LEGEND_FONTSIZE,
    COMPACT_SUPTITLE_FONTSIZE,
    COMPACT_TICK_FONTSIZE,
    COMPACT_TITLE_FONTSIZE,
    QUALITY_COLUMN,
    STABILITY_COLUMN,
    build_benchmark_df,
    iter_benchmarks,
    load_results_csv,
    relative_gap_pct,
    save_figure,
    set_yaxis_limits,
)
from controlledshifts.utils.analysis.latex import format_gap, format_value
from controlledshifts.utils.analysis.robustness_scores import compute_benchmark_robustness
from controlledshifts.utils.plotting import set_analysis_theme


# Macros the paper's preamble defines per model; the LaTeX table prints these instead of the display name. Benchmarks
# carry their own macro in the config (`latex`), because a benchmark's macro does not always follow from its name.
MODEL_MACRO_MAP = {
    "Naive": "\\naive",
    "AutoBot": "\\autobot",
    "SceneTransformer": "\\scenetransformer",
    "Wayformer": "\\wayformer",
    "MTR": "\\mtr",
}

# Column headers for the LaTeX table. Kept separate from METRIC_NAME_MAP, which abbreviates MinFDE6/MinADE6 for plot
# axes; the table spells the number out.
METRIC_HEADER_MAP = {
    "brierFDE": "BrierFDE",
    "minFDE6": "MinFDE6",
    "minADE6": "MinADE6",
    "missRate": "MissRate",
    "collisionRate0.25": "CollisionRate",
}

# Comment rule delimiting each benchmark block, so blocks stay findable when the table is edited by hand.
BLOCK_BANNER_RULE = "% " + "-" * 56

# Trailing robustness columns, in display order. The table's benchmark frames carry them as columns of the same name;
# unlike the error metrics these are scores, so higher is better.
ROBUSTNESS_COLUMNS = (QUALITY_COLUMN, STABILITY_COLUMN)


def _plot_distribution_shift_comparison(
    summary_df: pd.DataFrame, output_path: Path, colormap: str, id_metric: str, ood_metric: str
) -> None:
    """Plots a comparison of In-Distribution (ID) vs Out-of-Distribution (OOD) performance for different models,
    highlighting the performance gaps.

    Args:
        summary_df: Model metrics, one row per model.
        output_path: Directory to save the generated plot.
        colormap: Matplotlib colormap name.
        id_metric: ID metric column.
        ood_metric: OOD metric column.
    """
    assert id_metric in summary_df.columns, f"ID metric '{id_metric}' not found in summary_df columns"
    assert ood_metric in summary_df.columns, f"OOD metric '{ood_metric}' not found in summary_df columns"

    palette = sns.color_palette(colormap, len(summary_df))
    models = summary_df["Model"].to_numpy()

    def _plot_bars(ax: Axes, metric: str, title: str) -> None:
        values = summary_df[metric].to_numpy()
        bars = ax.bar(models, values, color=palette, alpha=0.8, edgecolor="black", linewidth=1.5)

        ax.set_ylabel(metric, fontsize=COMPACT_LABEL_FONTSIZE, fontweight="bold")
        ax.set_title(title, fontsize=COMPACT_TITLE_FONTSIZE, fontweight="bold")
        ax.tick_params(axis="x", labelsize=COMPACT_TICK_FONTSIZE, rotation=30)

        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                x = bar.get_x() + bar.get_width() / 2.0
                ax.text(x, height, f"{height:.3f}", ha="center", va="bottom", fontsize=COMPACT_ANNOT_FONTSIZE)
        ax.yaxis.grid(visible=True, alpha=0.3)

        set_yaxis_limits(ax, list(values), padding_factor=0.15, lower_factor=0.4, min_padding=0.1)

    n_models = models.shape[0]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.5 * n_models * 3, 6))
    fig.suptitle("Distribution Shift Analysis", fontsize=COMPACT_SUPTITLE_FONTSIZE, fontweight="bold")

    _plot_bars(ax1, id_metric, "In-Distribution (ID) Performance")
    _plot_bars(ax2, ood_metric, "Out-of-Distribution (OOD) Performance")

    # Performance Gap (OOD - ID, relative %)
    id_values = summary_df[id_metric].to_numpy()
    ood_values = summary_df[ood_metric].to_numpy()
    gap_values: NDArray = relative_gap_pct(ood_values, id_values)  # pyright: ignore[reportArgumentType, reportAssignmentType]

    gap_colors = ["#f07569" if gap > 0 else "#7cbf7c" for gap in gap_values]
    bars = ax3.bar(models, gap_values, color=gap_colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
    ax3.set_ylabel("Performance Gap (OOD - ID)", fontsize=COMPACT_LABEL_FONTSIZE, fontweight="bold")
    ax3.set_title("Generalization Gap", fontsize=COMPACT_TITLE_FONTSIZE, fontweight="bold")
    ax3.tick_params(axis="x", labelsize=COMPACT_TICK_FONTSIZE, rotation=30)

    for bar, gap in zip(bars, gap_values, strict=False):
        height = bar.get_height()
        if not np.isnan(height):
            va = "bottom" if height > 0 else "top"
            x = bar.get_x() + bar.get_width() / 2.0
            ax3.text(x, height, f"{gap:.3f}", ha="center", va=va, fontsize=COMPACT_ANNOT_FONTSIZE, fontweight="bold")
    ax3.yaxis.grid(visible=True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path / "distribution_shift_comparison.png")


def _plot_benchmark_comparison(
    summary_df: pd.DataFrame, metrics: dict[str, str], output_path: Path, colormap: str
) -> None:
    """Plots a benchmark comparison across different models for specified metrics.

    Args:
        summary_df: Model metrics, one row per model.
        metrics: Metric column names mapped to display names.
        output_path: Directory to save the generated plot.
        colormap: Matplotlib colormap name.
    """
    num_metrics = len(metrics)
    n_models = summary_df["Model"].shape[0]
    n_cols = min(2, num_metrics)
    n_rows = math.ceil(num_metrics / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_models * n_cols, 4.0 * n_rows), constrained_layout=True)
    fig.suptitle("Model Performance Comparison", fontsize=COMPACT_SUPTITLE_FONTSIZE, fontweight="bold")

    axes = np.atleast_1d(axes).flatten()

    palette = sns.color_palette(colormap, len(summary_df))
    model_order = summary_df["Model"].to_numpy()

    for idx, metric_name in enumerate(metrics.values()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        values = summary_df[metric_name].to_numpy()
        bars = ax.bar(model_order, values, color=palette, edgecolor="black", linewidth=1.0, alpha=0.8)

        ax.set_title(metric_name, pad=12)
        ax.set_ylabel("Metric Value", fontsize=COMPACT_LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=COMPACT_TICK_FONTSIZE)
        ax.set_axisbelow(True)

        set_yaxis_limits(ax, list(values), padding_factor=0.15, lower_factor=0.4, min_padding=0.1)

        # Value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=COMPACT_ANNOT_FONTSIZE,
                    fontweight="medium",
                )

        # Highlight best model
        best_idx = np.nanargmin(values) if "↓" in metric_name else np.nanargmax(values)
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(4)

    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    # tight=False preserves this figure's current cropping; it is one of the few that omits it, likely an oversight.
    save_figure(fig, output_path / "benchmark_comparison.png", tight=False)


def _plot_performance_gaps(
    summary_df: pd.DataFrame, output_path: Path, metric_pairs: list[tuple[str, str, str]]
) -> None:
    """Plots comprehensive performance gaps (absolute and percentage) between OOD and ID metrics for multiple metrics.

    Args:
        summary_df: Model metrics, one row per model.
        output_path: Directory to save the generated plot.
        metric_pairs: ``(id_metric_col, ood_metric_col, display_name)`` tuples.
    """
    gap_data = {}
    for id_col, ood_col, metric_name in metric_pairs:
        if id_col in summary_df.columns and ood_col in summary_df.columns:
            id_vals = summary_df[id_col].to_numpy()
            ood_vals = summary_df[ood_col].to_numpy()
            ood_id_diff = ood_vals - id_vals
            gap_data[metric_name] = {
                "absolute": ood_id_diff,
                "percent": relative_gap_pct(ood_vals, id_vals),
            }

    if gap_data:
        num_metrics = len(gap_data)
        num_models = summary_df["Model"].shape[0]
        horizontal_size = num_models * num_metrics * 1.5
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(horizontal_size, 6))
        fig.suptitle("Performance Gaps (OOD - ID)", fontsize=COMPACT_SUPTITLE_FONTSIZE, fontweight="bold")
        x = np.arange(len(summary_df))
        width = 0.25
        for i, (metric_name, gaps) in enumerate(gap_data.items()):
            offset = width * i
            ax1.bar(x + offset, gaps["absolute"], width, label=metric_name, alpha=0.8, edgecolor="black", linewidth=1)
            ax2.bar(x + offset, gaps["percent"], width, label=metric_name, alpha=0.8, edgecolor="black", linewidth=1)

        ax1.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
        ax1.set_xlabel("Model", fontsize=COMPACT_LABEL_FONTSIZE, fontweight="bold")
        ax1.set_ylabel("Absolute Gap (OOD - ID)", fontsize=COMPACT_LABEL_FONTSIZE, fontweight="bold")
        ax1.set_title(
            "Absolute Performance Gaps\n(Positive = OOD performs worse)",
            fontsize=COMPACT_TITLE_FONTSIZE,
            fontweight="bold",
        )
        ax1.set_xticks(x + (num_metrics - 1) * width)
        ax1.set_xticklabels(summary_df["Model"].values, ha="right", fontsize=COMPACT_TICK_FONTSIZE)
        ax1.legend(fontsize=COMPACT_LEGEND_FONTSIZE)
        ax1.yaxis.grid(visible=True, alpha=0.3)
        ax1.set_axisbelow(True)

        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
        ax2.set_xlabel("Model", fontsize=COMPACT_LABEL_FONTSIZE, fontweight="bold")
        ax2.set_ylabel("Percentage Gap (%)", fontsize=COMPACT_LABEL_FONTSIZE, fontweight="bold")
        ax2.set_title(
            "Percentage Performance Gaps\n(Positive = OOD worse, % relative to ID)",
            fontsize=COMPACT_TITLE_FONTSIZE,
            fontweight="bold",
        )
        ax2.set_xticks(x + (num_metrics - 1) * width)
        ax2.set_xticklabels(summary_df["Model"].values, ha="right", fontsize=COMPACT_TICK_FONTSIZE)
        ax2.legend(fontsize=COMPACT_LEGEND_FONTSIZE)
        ax2.yaxis.grid(visible=True, alpha=0.3)
        ax2.set_axisbelow(True)

        plt.tight_layout()
        save_figure(fig, output_path / "performance_gaps.png")

        # Print gap statistics
        print("\n" + "=" * 80)
        print("Performance Gap Analysis (OOD - ID):")
        print("=" * 80)
        for metric_name, gaps in gap_data.items():
            abs_gaps = gaps["absolute"]
            pct_gaps = gaps["percent"]
            print(f"\n{metric_name}:")
            for i, model in enumerate(summary_df["Model"].values):
                print(f"  {model:30s}: {abs_gaps[i]:+.4f} (Absolute) | {pct_gaps[i]:+.2f}% (Relative)")
            print(f"  Average Gap: {np.mean(abs_gaps):+.4f} | {np.mean(pct_gaps):+.2f}%")
            print(f"  Max Gap:     {np.max(abs_gaps):+.4f} | {np.max(pct_gaps):+.2f}%")


def _plot_grouped_bar_chart(
    summary_df: pd.DataFrame, metrics: dict[str, str], output_path: Path, key_metrics_display: list[str]
) -> None:
    """Plots a grouped bar chart comparing multiple key metrics across different models.

    Args:
        summary_df: Model metrics, one row per model.
        metrics: Metric column names mapped to display names.
        output_path: Directory to save the generated plot.
        key_metrics_display: Key metric column names to include in the chart.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    available_metrics = [m for m in key_metrics_display if m in summary_df.columns]

    if available_metrics:
        x = np.arange(len(summary_df))
        width = 0.2
        all_values: list[float] = []
        for i, metric in enumerate(available_metrics):
            values = summary_df[metric].to_numpy()
            all_values.extend(values)
            ax.bar(x + width * i, values, width, label=metric, alpha=0.8, edgecolor="black", linewidth=1)

        set_yaxis_limits(ax, all_values, padding_factor=0.15, lower_factor=0.4, min_padding=0.1)

        ax.set_xlabel("Model", fontsize=COMPACT_LABEL_FONTSIZE, fontweight="bold")
        ax.set_ylabel("Metric Value", fontsize=COMPACT_LABEL_FONTSIZE, fontweight="bold")
        ax.set_title("Multi-Metric Comparison", fontsize=COMPACT_TITLE_FONTSIZE, fontweight="bold")
        ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
        ax.set_xticklabels(summary_df["Model"].values, rotation=35, ha="right")
        ax.legend(loc="upper left", fontsize=COMPACT_LEGEND_FONTSIZE)
        ax.yaxis.grid(visible=True, alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        save_figure(fig, output_path / "grouped_comparison.png")

    # Print best performing model for each metric
    print("\n" + "=" * 80)
    print("Best Performing Models (Lower is Better):")
    print("=" * 80)
    for metric_name in metrics.values():
        if metric_name in summary_df.columns:
            best_idx = summary_df[metric_name].idxmin()
            if pd.notna(best_idx):
                best_model = summary_df.loc[best_idx, "Model"]
                best_value = summary_df.loc[best_idx, metric_name]
                print(f"{metric_name:30s}: {best_model:30s} ({best_value:.4f})")


def _plot_benchmark(
    benchmark_df: pd.DataFrame, splits: tuple[str, str], metrics: list[str], colormap: str, output_path: Path
) -> None:
    """Generates the per-benchmark ID-vs-OOD comparison plots for a single benchmark.

    Args:
        benchmark_df: Per-model frame from :func:`build_benchmark_df`.
        splits: ``(seen_split, unseen_split)`` column prefixes for the ID/OOD splits.
        metrics: Metric names to plot.
        colormap: Seaborn/matplotlib palette name.
        output_path: Directory to save the generated plots.
    """
    seen_split, unseen_split = splits
    id_split_name = seen_split.split("/")[-1]
    ood_split_name = unseen_split.split("/")[-1]
    metrics_map = {
        f"{split}/{metric}": f"{metric} ({split.split('/')[-1]},↓)" for metric, split in product(metrics, splits)
    }

    summary_data = []
    for _, row in benchmark_df.iterrows():
        model_metrics: dict[str, float | str] = {"Model": row["Model"]}
        for metric_col, metric_name in metrics_map.items():
            if metric_col in benchmark_df.columns:
                model_metrics[metric_name] = row[metric_col]
        summary_data.append(model_metrics)
    summary_df = pd.DataFrame(summary_data)
    print("Metrics Summary:")
    print(summary_df.to_string(index=False, float_format="{:.3f}".format))

    _plot_benchmark_comparison(summary_df, metrics_map, output_path, colormap)
    _plot_distribution_shift_comparison(
        summary_df,
        output_path,
        colormap,
        id_metric=f"{metrics[0]} ({id_split_name},↓)",
        ood_metric=f"{metrics[0]} ({ood_split_name},↓)",
    )
    metric_pairs = [(f"{metric} ({id_split_name},↓)", f"{metric} ({ood_split_name},↓)", metric) for metric in metrics]
    _plot_performance_gaps(summary_df, output_path, metric_pairs)
    key_metrics_display = [
        f"{metrics[0]} ({id_split_name},↓)",
        f"{metrics[0]} ({ood_split_name},↓)",
    ]
    _plot_grouped_bar_chart(summary_df, metrics_map, output_path, key_metrics_display=key_metrics_display)


class _MetricMeans(NamedTuple):
    """Column means for a set of rows: seen/unseen values and OOD gap (percent) per metric, plus the robustness
    scores keyed by :data:`ROBUSTNESS_COLUMNS`.
    """

    seen: dict[str, float]
    unseen: dict[str, float]
    gap: dict[str, float]
    robustness: dict[str, float]


def _benchmark_metric_means(
    benchmark_df: pd.DataFrame, id_split: str, ood_split: str, metrics: list[str]
) -> _MetricMeans:
    """Compute per-metric mean seen/unseen values and mean per-model OOD gap for one benchmark block.

    The mean gap is the mean of the per-model relative ID->OOD gaps (NaNs skipped) -- i.e. the average degradation
    across the models in the benchmark, not the gap between the averaged values.

    Args:
        benchmark_df: Per-model frame from :func:`build_benchmark_df`, carrying the robustness columns.
        id_split: Column prefix of the ID split.
        ood_split: Column prefix of the OOD split.
        metrics: Metric names to summarize.

    Returns:
        Per-metric mean seen value, mean unseen value and mean OOD gap, plus the mean robustness scores.
    """
    mean_seen: dict[str, float] = {}
    mean_unseen: dict[str, float] = {}
    mean_gap: dict[str, float] = {}
    for metric in metrics:
        id_vals = benchmark_df[f"{id_split}/{metric}"]
        ood_vals = benchmark_df[f"{ood_split}/{metric}"]
        gaps: pd.Series = relative_gap_pct(ood_vals, id_vals)  # pyright: ignore[reportAssignmentType, reportArgumentType]
        mean_seen[metric] = float(id_vals.mean())
        mean_unseen[metric] = float(ood_vals.mean())
        mean_gap[metric] = float(gaps.mean())
    mean_robustness = {column: float(benchmark_df[column].mean()) for column in ROBUSTNESS_COLUMNS}
    return _MetricMeans(mean_seen, mean_unseen, mean_gap, mean_robustness)


def _build_mean_row(
    benchmark_label: str, model_label: str, means: _MetricMeans, metrics: list[str], gray_level: float
) -> str:
    r"""Render a gray-shaded summary row of per-metric mean seen/unseen values and mean OOD gaps (plain text).

    The row matches the model rows' column layout (benchmark + model + seen metrics + two spacers + unseen metrics +
    two spacers + robustness scores) and is shaded via ``\rowcolor`` so it reads as an aggregate. Unlike the model
    rows, the gap is plain text (no color/bold).

    Args:
        benchmark_label: Leading benchmark cell (empty inside a block; a label such as ``Overall`` otherwise).
        model_label: Text for the model cell (e.g. ``Mean``).
        means: Per-metric mean seen/unseen values and mean OOD gap.
        metrics: Metric names, in column order.
        gray_level: ``\rowcolor[gray]`` level (0=black, 1=white); lower is darker.

    Returns:
        The LaTeX row string, prefixed with the row-color directive.
    """
    row_parts = [benchmark_label, model_label]

    id_values = [f"{means.seen[metric]:.3f}" if pd.notna(means.seen[metric]) else "---" for metric in metrics]
    id_values.append("")  # spacer column

    ood_values = [""]  # spacer column
    for metric in metrics:
        if pd.notna(means.unseen[metric]) and pd.notna(means.gap[metric]):
            ood_values.append(f"{means.unseen[metric]:.3f} ({means.gap[metric]:+.2f}\\%)")
        else:
            ood_values.append("---")

    row_parts.extend(id_values)
    row_parts.extend(ood_values)

    row_parts.extend(["", ""])  # spacer columns before the robustness group
    for column in ROBUSTNESS_COLUMNS:
        score = means.robustness[column]
        row_parts.append(f"{score:.3f}" if pd.notna(score) else "---")

    return f"\\rowcolor[gray]{{{gray_level}}}\n" + " & ".join(row_parts) + " \\\\"


def _best_score(values: pd.Series) -> float:
    """Highest score in a robustness column, or NaN when every model ties.

    A fully tied column carries no ranking -- the Uniform block's stability is 1.000 for every model, since each is
    referenced against its own row -- and bolding all of it would suggest one.
    """
    low, high = values.min(), values.max()
    return float("nan") if pd.isna(high) or np.isclose(low, high) else float(high)


def _build_benchmark_rows(
    benchmark_df: pd.DataFrame,
    benchmark_latex: str,
    id_split: str,
    ood_split: str,
    metrics: list[str],
) -> list[str]:
    """Builds the LaTeX ``tabular`` rows for a single benchmark block (with gap annotations/coloring).

    Best ID/OOD values are bolded and OOD gap severity is colored relative to this benchmark's own min/max gap, so each
    block is self-scaled. The trailing robustness scores are bolded on their highest value instead (they are scores,
    not errors).

    Args:
        benchmark_df: Per-model frame from :func:`build_benchmark_df`, carrying the robustness columns.
        benchmark_latex: Benchmark macro for the leading multirow cell.
        id_split: Column prefix of the ID split.
        ood_split: Column prefix of the OOD split.
        metrics: Metric names to include, in column order.

    Returns:
        One LaTeX row string per model in the benchmark.
    """
    # Precompute best ID/OOD and gap severity per metric
    best_id, best_ood, gap_stats = {}, {}, {}
    for metric in metrics:
        id_vals = benchmark_df[f"{id_split}/{metric}"]
        best_id[metric] = id_vals.min()

        ood_vals = benchmark_df[f"{ood_split}/{metric}"]
        best_ood[metric] = ood_vals.min()

        gaps: pd.Series = relative_gap_pct(ood_vals, id_vals)  # pyright: ignore[reportAssignmentType, reportArgumentType]
        gap_stats[metric] = (gaps.min(), gaps.max())  # best, worst

    best_score = {column: _best_score(benchmark_df[column]) for column in ROBUSTNESS_COLUMNS}

    table_rows = []
    first_row = True
    for _, row in benchmark_df.iterrows():
        row_parts = []
        if first_row:
            row_parts.append(f"\\multirow{{{len(benchmark_df)}}}{{*}}{{{benchmark_latex}}}")
            first_row = False
        else:
            row_parts.append("")
        model = str(row["Model"])
        row_parts.append(MODEL_MACRO_MAP.get(model, model))

        id_values, ood_values = [], []
        for metric in metrics:
            id_val = row[f"{id_split}/{metric}"]
            ood_val = row[f"{ood_split}/{metric}"]

            id_values.append(format_value(id_val, best_id[metric]))

            # The OOD cell needs both values: without the ID value there is no gap to annotate it with.
            if pd.notna(id_val) and pd.notna(ood_val):
                gap = relative_gap_pct(ood_val, id_val)
                gap_str = format_gap(gap, *gap_stats[metric])
                ood_str = f"{format_value(ood_val, best_ood[metric])} ({gap_str})"
            else:
                ood_str = "---"

            ood_values.append(ood_str)

        id_values.append("")  # spacer column
        row_parts.extend(id_values)
        ood_values = ["", *ood_values]  # spacer column
        row_parts.extend(ood_values)

        row_parts.extend(["", ""])  # spacer columns before the robustness group
        row_parts.extend(format_value(row[column], best_score[column]) for column in ROBUSTNESS_COLUMNS)
        table_rows.append(" & ".join(row_parts) + " \\\\")

    # Light-gray per-benchmark mean row; benchmark cell left empty so the multirow label stays over the model rows.
    means = _benchmark_metric_means(benchmark_df, id_split, ood_split, metrics)
    table_rows.append(_build_mean_row("", "Mean", means, metrics, gray_level=0.95))

    return table_rows


def _write_combined_tex_table(
    blocks: list[tuple[str, str, str, str, pd.DataFrame]],
    metrics: list[str],
    output_path: Path,
    table_config: DictConfig,
) -> str:
    r"""Writes a single LaTeX table spanning all benchmarks, each as a multirow block separated by ``\midrule``.

    Args:
        blocks: One ``(benchmark_name, benchmark_latex, id_split, ood_split, benchmark_df)`` tuple per benchmark, in
            display order; each frame carries the robustness columns.
        metrics: Metric names to include, in column order.
        output_path: Directory to save the generated ``results.tex`` file.
        table_config: The config's ``table`` node: ``caption`` and the ``seen_label``/``unseen_label``/
            ``robustness_label`` column-group headers, copied verbatim (they may use macros the consuming document
            defines), plus ``include_overall_mean`` -- when false the closing overall-mean row is written as
            commented-out LaTeX, so it can be re-enabled by hand without rerunning the analysis.

    Returns:
        The rendered LaTeX table string.
    """
    body_rows: list[str] = []
    for benchmark_name, benchmark_latex, id_split, ood_split, benchmark_df in blocks:
        banner = [BLOCK_BANNER_RULE, f"% {benchmark_name.upper()} BENCHMARK HERE:", BLOCK_BANNER_RULE, "\\midrule"]
        body_rows.extend(banner)
        body_rows.extend(_build_benchmark_rows(benchmark_df, benchmark_latex, id_split, ood_split, metrics))

    # Gray overall mean row across the entire sweep (grand mean over all benchmark x model entries). Always computed:
    # the commented-out form still carries the real numbers.
    acc_id: dict[str, list[pd.Series]] = {metric: [] for metric in metrics}
    acc_ood: dict[str, list[pd.Series]] = {metric: [] for metric in metrics}
    acc_gap: dict[str, list[pd.Series]] = {metric: [] for metric in metrics}
    acc_score: dict[str, list[pd.Series]] = {column: [] for column in ROBUSTNESS_COLUMNS}
    for _, _, id_split, ood_split, benchmark_df in blocks:
        for metric in metrics:
            id_vals = benchmark_df[f"{id_split}/{metric}"]
            ood_vals = benchmark_df[f"{ood_split}/{metric}"]
            acc_id[metric].append(id_vals)
            acc_ood[metric].append(ood_vals)
            acc_gap[metric].append(pd.Series(relative_gap_pct(ood_vals, id_vals)))
        for column in ROBUSTNESS_COLUMNS:
            acc_score[column].append(benchmark_df[column])
    overall_means = _MetricMeans(
        seen={metric: float(pd.concat(acc_id[metric], ignore_index=True).mean()) for metric in metrics},
        unseen={metric: float(pd.concat(acc_ood[metric], ignore_index=True).mean()) for metric in metrics},
        gap={metric: float(pd.concat(acc_gap[metric], ignore_index=True).mean()) for metric in metrics},
        robustness={
            column: float(pd.concat(acc_score[column], ignore_index=True).mean()) for column in ROBUSTNESS_COLUMNS
        },
    )
    # The mean row is two lines (\rowcolor + the row itself), so each line is commented individually.
    overall_row = _build_mean_row("\\textsc{Overall}", "Mean", overall_means, metrics, gray_level=0.89)
    prefix = "" if table_config.include_overall_mean else "% "
    body_rows.extend(f"{prefix}{line}" for line in ["\\midrule", *overall_row.split("\n")])

    n_metrics = len(metrics)
    n_scores = len(ROBUSTNESS_COLUMNS)
    # Two spacer columns precede the unseen and the robustness group; each group's header spans its own pair.
    col_spec = "l l " + "c" * (2 * n_metrics + n_scores + 4)
    headers = [METRIC_HEADER_MAP.get(metric, metric) for metric in metrics]

    latex_lines: list[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{table_config.caption}}}",
        "\\label{tab:distribution_shift_results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        (
            f"\\multirow{{2}}{{*}}{{\\textbf{{Benchmark}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & "
            f"\\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{{table_config.seen_label}}}}} & "
            f"\\multicolumn{{{n_metrics + 2}}}{{c}}{{\\textbf{{{table_config.unseen_label}}}}} & "
            f"\\multicolumn{{{n_scores + 2}}}{{c}}{{\\textbf{{{table_config.robustness_label}}}}} \\\\"
        ),
        "& & " + " & ".join([*headers, "", "", *headers, "", "", *ROBUSTNESS_COLUMNS]) + " \\\\",
        *body_rows,
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\vspace{-0.3cm}",
        "\\end{table*}",
    ]
    latex_table_str = "\n".join(latex_lines)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "results.tex"
    output_file.write_text(latex_table_str + "\n")
    print(f"\n✓ Combined LaTeX table saved as '{output_file}'")

    return latex_table_str


def run_distribution_shift_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Runs ID-vs-OOD distribution-shift analysis for every benchmark using a single combined results file.

    For each benchmark listed in ``config.benchmarks`` the seen/unseen split columns are looked up in the shared
    results file, per-benchmark comparison plots are written under ``output_path/<benchmark>/``, and a single combined
    LaTeX table spanning all benchmarks is written to ``output_path/results.tex``. The table's trailing robustness
    columns are scored per benchmark over the same set of benchmarks (see :func:`compute_benchmark_robustness`).

    Args:
        config: Analysis configuration (``benchmarks_filepath``, ``benchmarks``, ``models_to_compare``,
            ``trajectory_forecasting_metrics``, ``benchmark_colormap``, ``show_run_id``, ``table``).
        log: Logger.
        output_path: Directory to save the generated plots and table.
    """
    set_analysis_theme(log=log)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_df = load_results_csv(Path(config.benchmarks_filepath), log)
    if metrics_df is None:
        return

    metrics = list(config.trajectory_forecasting_metrics)
    models_to_compare = list(config.models_to_compare)
    colormap = config.benchmark_colormap

    benchmarks = [(key, spec.name, spec.seen, spec.unseen) for key, spec in iter_benchmarks(config)]
    robustness = compute_benchmark_robustness(
        metrics_df, benchmarks, metrics, models_to_compare, uniform_key=str(config.table.uniform_key)
    )

    blocks: list[tuple[str, str, str, str, pd.DataFrame]] = []
    for key, spec in iter_benchmarks(config):
        log.info("Analyzing benchmark '%s' (%s): seen=%s, unseen=%s", key, spec.name, spec.seen, spec.unseen)

        splits = (spec.seen, spec.unseen)
        benchmark_df = build_benchmark_df(
            metrics_df, splits, metrics, models_to_compare, show_run_id=config.show_run_id
        )
        if benchmark_df.empty:
            log.warning("No matching models found for benchmark '%s'; skipping.", key)
            continue

        print(f"\n=== Benchmark: {spec.name} ({len(benchmark_df)} models) ===")
        benchmark_output = output_path / key
        benchmark_output.mkdir(parents=True, exist_ok=True)
        _plot_benchmark(benchmark_df, splits, metrics, colormap, benchmark_output)

        # Table-only columns, attached after plotting so the figures keep seeing the raw metric frame.
        block_scores = robustness.get(key)
        model_names = benchmark_df["Model"].str.split("[").str[0]  # scores are keyed without the [run-id] suffix
        for column in ROBUSTNESS_COLUMNS:
            benchmark_df[column] = model_names.map(block_scores[column]) if block_scores is not None else float("nan")

        blocks.append((spec.name, spec.latex, spec.seen, spec.unseen, benchmark_df))

    if blocks:
        _write_combined_tex_table(blocks, metrics, output_path, config.table)

    print("\n✓ Analysis complete!")
    log.info("Distribution shift analysis complete!")
