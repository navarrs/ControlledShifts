"""Distribution-shift / benchmark analysis utilities (ID vs OOD comparisons, LaTeX tables).

See `docs/ANALYSIS.md` for usage details.
"""

import math
from itertools import product
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import NDArray
from omegaconf import DictConfig

from controlledshifts.utils.analysis.common import (
    MODEL_NAME_MAP,
    MODEL_SIZE_MAP,
    relative_gap_pct,
    set_yaxis_limits,
)
from controlledshifts.utils.constants import EPSILON


# Minimum gap color intensity percentage (0-100); higher makes small OOD gaps more vibrant in the LaTeX table.
GAP_MIN_COLOR_VALUE = 20.0


def _plot_distribution_shift_comparison(
    summary_df: pd.DataFrame, output_path: Path, colormap: str, id_metric: str, ood_metric: str
) -> None:
    """Plots a comparison of In-Distribution (ID) vs Out-of-Distribution (OOD) performance for different models,
    highlighting the performance gaps.

    Args:
        summary_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        output_path (Path): Directory to save the generated plot.
        colormap (str): Name of the matplotlib colormap to use for consistent coloring.
        id_metric (str): Name of the ID metric.
        ood_metric (str): Name of the OOD metric.
    """
    assert id_metric in summary_df.columns, f"ID metric '{id_metric}' not found in summary_df columns"
    assert ood_metric in summary_df.columns, f"OOD metric '{ood_metric}' not found in summary_df columns"

    palette = sns.color_palette(colormap, len(summary_df))
    models = summary_df["Model"].to_numpy()

    def _plot_bars(ax: Axes, metric: str, title: str) -> None:
        values = summary_df[metric].to_numpy()
        bars = ax.bar(models, values, color=palette, alpha=0.8, edgecolor="black", linewidth=1.5)

        ax.set_ylabel(metric, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9, rotation=30)

        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                x = bar.get_x() + bar.get_width() / 2.0
                ax.text(x, height, f"{height:.3f}", ha="center", va="bottom", fontsize=10)
        ax.yaxis.grid(visible=True, alpha=0.3)

        set_yaxis_limits(ax, list(values), padding_factor=0.15, lower_factor=0.4, min_padding=0.1)

    n_models = models.shape[0]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.5 * n_models * 3, 6))
    fig.suptitle("Distribution Shift Analysis", fontsize=14, fontweight="bold")

    _plot_bars(ax1, id_metric, "In-Distribution (ID) Performance")
    _plot_bars(ax2, ood_metric, "Out-of-Distribution (OOD) Performance")

    # Performance Gap (OOD - ID, relative %)
    id_values = summary_df[id_metric].to_numpy()
    ood_values = summary_df[ood_metric].to_numpy()
    gap_values: NDArray = relative_gap_pct(ood_values, id_values)  # pyright: ignore[reportArgumentType, reportAssignmentType]

    gap_colors = ["#f07569" if gap > 0 else "#7cbf7c" for gap in gap_values]
    bars = ax3.bar(models, gap_values, color=gap_colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
    ax3.set_ylabel("Performance Gap (OOD - ID)", fontsize=11, fontweight="bold")
    ax3.set_title("Generalization Gap", fontsize=12, fontweight="bold")
    ax3.tick_params(axis="x", labelsize=12, rotation=30)

    for bar, gap in zip(bars, gap_values, strict=False):
        height = bar.get_height()
        if not np.isnan(height):
            va = "bottom" if height > 0 else "top"
            x = bar.get_x() + bar.get_width() / 2.0
            ax3.text(x, height, f"{gap:.3f}", ha="center", va=va, fontsize=8, fontweight="bold")
    ax3.yaxis.grid(visible=True, alpha=0.3)

    plt.tight_layout()
    output_file = output_path / "distribution_shift_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Plot saved as '{output_file}'")


def _plot_benchmark_comparison(
    summary_df: pd.DataFrame, metrics: dict[str, str], output_path: Path, colormap: str
) -> None:
    """Plots a benchmark comparison across different models for specified metrics.

    Args:
        summary_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        metrics (dict[str, str]): Dictionary mapping metric column names to display names.
        output_path (Path): Directory to save the generated plot.
        colormap (str): Name of the matplotlib colormap to use for consistent coloring.
    """
    num_metrics = len(metrics)
    n_models = summary_df["Model"].shape[0]
    n_cols = min(2, num_metrics)
    n_rows = math.ceil(num_metrics / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_models * n_cols, 4.0 * n_rows), constrained_layout=True)
    fig.suptitle("Model Performance Comparison", fontsize=20, fontweight="bold")

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
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.tick_params(axis="x", labelsize=10)
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
                    fontsize=10,
                    fontweight="medium",
                )

        # Highlight best model
        best_idx = np.nanargmin(values) if "↓" in metric_name else np.nanargmax(values)
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(4)

    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    output_file = output_path / "benchmark_comparison.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"\n✓ Plot saved as '{output_file}'")


def _plot_performance_gaps(
    summary_df: pd.DataFrame, output_path: Path, metric_pairs: list[tuple[str, str, str]]
) -> None:
    """Plots comprehensive performance gaps (absolute and percentage) between OOD and ID metrics for multiple metrics.

    Args:
        summary_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        output_path (Path): Directory to save the generated plot.
        metric_pairs (list[tuple[str, str, str]]): List of tuples containing (ID metric column name, OOD metric column
            name, metric display name).
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
        fig.suptitle("Performance Gaps (OOD - ID)", fontsize=14, fontweight="bold")
        x = np.arange(len(summary_df))
        width = 0.25
        for i, (metric_name, gaps) in enumerate(gap_data.items()):
            offset = width * i
            ax1.bar(x + offset, gaps["absolute"], width, label=metric_name, alpha=0.8, edgecolor="black", linewidth=1)
            ax2.bar(x + offset, gaps["percent"], width, label=metric_name, alpha=0.8, edgecolor="black", linewidth=1)

        ax1.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
        ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Absolute Gap (OOD - ID)", fontsize=12, fontweight="bold")
        ax1.set_title("Absolute Performance Gaps\n(Positive = OOD performs worse)", fontsize=12, fontweight="bold")
        ax1.set_xticks(x + (num_metrics - 1) * width)
        ax1.set_xticklabels(summary_df["Model"].values, ha="right", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.yaxis.grid(visible=True, alpha=0.3)
        ax1.set_axisbelow(True)

        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
        ax2.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Percentage Gap (%)", fontsize=12, fontweight="bold")
        ax2.set_title(
            "Percentage Performance Gaps\n(Positive = OOD worse, % relative to ID)", fontsize=12, fontweight="bold"
        )
        ax2.set_xticks(x + (num_metrics - 1) * width)
        ax2.set_xticklabels(summary_df["Model"].values, ha="right", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.yaxis.grid(visible=True, alpha=0.3)
        ax2.set_axisbelow(True)

        plt.tight_layout()
        output_file = output_path / "performance_gaps.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved as '{output_file}'")

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
        summary_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        metrics (dict[str, str]): Dictionary mapping metric column names to display names.
        output_path (Path): Directory to save the generated plot.
        key_metrics_display (list[str]): List of key metric column names to include in the grouped bar chart.
    """
    _fig, ax = plt.subplots(figsize=(14, 7))

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

        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Metric Value", fontsize=12, fontweight="bold")
        ax.set_title("Multi-Metric Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
        ax.set_xticklabels(summary_df["Model"].values, rotation=35, ha="right")
        ax.legend(loc="upper left", fontsize=10)
        ax.yaxis.grid(visible=True, alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        output_file = output_path / "grouped_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved as '{output_file}'")

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


def build_benchmark_df(
    metrics_df: pd.DataFrame,
    splits: tuple[str, str],
    metrics: list[str],
    models_to_compare: list[str],
    *,
    show_run_id: bool,
) -> pd.DataFrame:
    """Reconstruct a per-model benchmark DataFrame from the combined results file.

    For each model in ``models_to_compare``, the seen (ID) and unseen (OOD) metric values are looked up independently
    and joined by model name. A benchmark's seen and unseen splits may therefore come from different training runs (or
    even different datasets); each value is taken from the first row that populates the corresponding column.

    Args:
        metrics_df (pd.DataFrame): The combined results file with a ``Name`` (``<dataset>_<model>``) column and
            ``<split>/<metric>`` value columns.
        splits (tuple[str, str]): ``(seen_split, unseen_split)`` column prefixes, e.g.
            ``("test/waymo-uniform-validation", "test/waymo-uniform-testing")``.
        metrics (list[str]): Metric names to extract for each split.
        models_to_compare (list[str]): Raw model identifiers (as they appear after ``<dataset>_``) to include.
        show_run_id (bool): Whether to append the source run ID to the displayed model name.

    Returns:
        pd.DataFrame: One row per model with a ``Model`` column and ``<split>/<metric>`` value columns (NaN when a
            metric is absent), shape-compatible with the plotting and LaTeX-table helpers.
    """
    metrics_df = metrics_df.copy()
    metrics_df["model_name"] = metrics_df["Name"].str.rsplit("_", n=1).str[-1]
    if "ID" not in metrics_df.columns:
        metrics_df["ID"] = np.arange(len(metrics_df))

    rows: list[dict[str, float | str]] = []
    for model in models_to_compare:
        model_rows = metrics_df[metrics_df["model_name"] == model]
        if model_rows.empty:
            continue
        display_name = MODEL_NAME_MAP.get(model, model)
        record: dict[str, float | str] = {}
        run_id: str | None = None
        for split in splits:
            for metric in metrics:
                col = f"{split}/{metric}"
                value = float("nan")
                if col in model_rows.columns:
                    non_null = model_rows[col].dropna()
                    if not non_null.empty:
                        value = float(non_null.iloc[0])
                        if run_id is None:
                            run_id = str(model_rows.loc[non_null.index[0], "ID"])
                record[col] = value
        record["Model"] = f"{display_name}[{run_id}]" if show_run_id and run_id is not None else display_name
        rows.append(record)

    return pd.DataFrame(rows)


def _plot_benchmark(
    benchmark_df: pd.DataFrame, splits: tuple[str, str], metrics: list[str], colormap: str, output_path: Path
) -> None:
    """Generates the per-benchmark ID-vs-OOD comparison plots for a single benchmark.

    Args:
        benchmark_df (pd.DataFrame): Per-model frame from :func:`build_benchmark_df`.
        splits (tuple[str, str]): ``(seen_split, unseen_split)`` column prefixes for the ID/OOD splits.
        metrics (list[str]): Metric names to plot.
        colormap (str): Name of the seaborn/matplotlib palette to use.
        output_path (Path): Directory to save the generated plots.
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


def _build_benchmark_rows(  # noqa: PLR0912, PLR0915
    benchmark_df: pd.DataFrame,
    benchmark_name: str,
    id_split: str,
    ood_split: str,
    metrics: list[str],
) -> list[str]:
    """Builds the LaTeX ``tabular`` rows for a single benchmark block (with gap annotations/coloring).

    Best ID/OOD values are bolded and OOD gap severity is colored relative to this benchmark's own min/max gap, so each
    block is self-scaled.

    Args:
        benchmark_df (pd.DataFrame): Per-model frame from :func:`build_benchmark_df`.
        benchmark_name (str): Display name of the benchmark for the leading multirow cell.
        id_split (str): Column prefix of the In-Distribution split.
        ood_split (str): Column prefix of the Out-of-Distribution split.
        metrics (list[str]): Metric names to include, in column order.

    Returns:
        list[str]: One LaTeX row string per model in the benchmark.
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

    table_rows = []
    first_row = True
    for _, row in benchmark_df.iterrows():
        row_parts = []
        if first_row:
            row_parts.append(f"\\multirow{{{len(benchmark_df)}}}{{*}}{{\\texttt{{{benchmark_name}}}}}")
            first_row = False
        else:
            row_parts.append("")
        row_parts.append(str(row["Model"]))

        # Model size
        if "model/params/total" in row and pd.notna(row["model/params/total"]):
            size_val = row["model/params/total"]
            size_str = f"{size_val:.2e}" if isinstance(size_val, (int, float)) else str(size_val)
        else:
            size_str = MODEL_SIZE_MAP.get(row["Model"], "---")
        row_parts.append(size_str)

        id_values, ood_values = [], []
        for metric in metrics:
            id_val = row[f"{id_split}/{metric}"]
            ood_val = row[f"{ood_split}/{metric}"]

            # In-distribution value
            if pd.notna(id_val):
                id_str = f"{id_val:.3f}"
                if np.isclose(id_val, best_id[metric]):
                    id_str = f"\\textbf{{{id_str}}}"
            else:
                id_str = "---"
            id_values.append(id_str)

            # Out-of-distribution value with gap annotation and coloring
            if pd.notna(id_val) and pd.notna(ood_val):
                gap = relative_gap_pct(ood_val, id_val)

                best_gap, worst_gap = gap_stats[metric]
                denom = max(abs(worst_gap - best_gap), EPSILON)
                severity = np.clip(abs(gap - best_gap) / denom, 0, 1)
                intensity = int(GAP_MIN_COLOR_VALUE + severity * (100 - GAP_MIN_COLOR_VALUE))

                color = "OrangeRed" if gap > 0 else "ForestGreen"
                gap_str = f"\\textcolor{{{color}!{intensity}}}{{{gap:+.2f}\\%}}"

                ood_str = f"{ood_val:.3f}"
                if np.isclose(ood_val, best_ood[metric]):
                    ood_str = f"\\textbf{{{ood_str}}}"
                ood_str = f"{ood_str} ({gap_str})"
            else:
                ood_str = "---"

            ood_values.append(ood_str)

        id_values.append("")  # spacer column
        row_parts.extend(id_values)
        ood_values = ["", *ood_values]  # spacer column
        row_parts.extend(ood_values)
        table_rows.append(" & ".join(row_parts) + " \\\\")

    return table_rows


def _write_combined_tex_table(
    blocks: list[tuple[str, str, str, pd.DataFrame]], metrics: list[str], output_path: Path
) -> str:
    r"""Writes a single LaTeX table spanning all benchmarks, each as a multirow block separated by ``\midrule``.

    Args:
        blocks (list[tuple[str, str, str, pd.DataFrame]]): One ``(benchmark_name, id_split, ood_split, benchmark_df)``
            tuple per benchmark, in display order.
        metrics (list[str]): Metric names to include, in column order.
        output_path (Path): Directory to save the generated ``results.tex`` file.

    Returns:
        str: The rendered LaTeX table string.
    """
    body_rows: list[str] = []
    for i, (benchmark_name, id_split, ood_split, benchmark_df) in enumerate(blocks):
        if i > 0:
            body_rows.append("\\midrule")
        body_rows.extend(_build_benchmark_rows(benchmark_df, benchmark_name, id_split, ood_split, metrics))

    n_metrics = len(metrics)
    col_spec = "l l c " + "c" * (2 * n_metrics) + "cc"

    latex_lines: list[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Distribution Shift Results}",
        "\\label{tab:distribution_shift_results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        (
            f"\\multirow{{2}}{{*}}{{\\textbf{{Benchmark}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & "
            f"\\multirow{{2}}{{*}}{{\\textbf{{Model Size}}}} & "
            f"\\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{Seem}}}} & "
            f"\\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{Unseen}}}} \\\\"
        ),
        " & & & " + " & ".join([*metrics, "", "", *metrics]) + " \\\\",
        "\\midrule",
        *body_rows,
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ]
    latex_table_str = "\n".join(latex_lines)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "results.tex"
    output_file.write_text(latex_table_str)
    print(f"\n✓ Combined LaTeX table saved as '{output_file}'")

    return latex_table_str


def run_distribution_shift_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Runs ID-vs-OOD distribution-shift analysis for every benchmark using a single combined results file.

    For each benchmark listed in ``config.benchmarks`` the seen/unseen split columns are looked up in the shared
    results file, per-benchmark comparison plots are written under ``output_path/<benchmark>/``, and a single combined
    LaTeX table spanning all benchmarks is written to ``output_path/results.tex``.

    Args:
        config (DictConfig): Analysis configuration (``benchmarks_filepath``, ``benchmarks``, ``models_to_compare``,
            ``trajectory_forecasting_metrics``, ``benchmark_colormap``, ``show_run_id``).
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots and table.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        },
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_filepath = Path(config.benchmarks_filepath)
    if not metrics_filepath.exists():
        log.error("Results file not found at %s", metrics_filepath)
        return
    metrics_df = pd.read_csv(metrics_filepath)
    if "Name" not in metrics_df.columns:
        log.error("CSV must contain a 'Name' column")
        return

    metrics = list(config.trajectory_forecasting_metrics)
    models_to_compare = list(config.models_to_compare)
    colormap = config.benchmark_colormap

    blocks: list[tuple[str, str, str, pd.DataFrame]] = []
    for benchmark_entry in config.benchmarks:
        key, spec = next(iter(benchmark_entry.items()))
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
        blocks.append((spec.name, spec.seen, spec.unseen, benchmark_df))

    if blocks:
        _write_combined_tex_table(blocks, metrics, output_path)

    print("\n✓ Analysis complete!")
    log.info("Distribution shift analysis complete!")
