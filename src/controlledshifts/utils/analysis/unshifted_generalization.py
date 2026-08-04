"""Unshifted-generalization analysis: one in-distribution reference vs many evaluation benchmarks.

NOTE: Unlike the distribution-shift study (see ``distribution_shift.py``), the training and validation data here is not
subjected to any controlled shift, so there is a single shared in-distribution reference split (e.g. mini-validation)
and every benchmark's performance gap is measured relative to it. See ``docs/ANALYSIS.md`` for usage details.
"""

import math
from logging import Logger
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from omegaconf import DictConfig

from controlledshifts.utils.analysis.common import model_colors, relative_gap_pct, save_figure, set_yaxis_limits
from controlledshifts.utils.analysis.distribution_shift import GAP_MIN_COLOR_VALUE, build_benchmark_df
from controlledshifts.utils.constants import EPSILON
from controlledshifts.utils.plotting import set_analysis_theme


class Block(NamedTuple):
    """One table/plot block: a display ``name`` and the ``split`` column prefix it reads from."""

    name: str
    split: str


def _parse_blocks(config: DictConfig) -> tuple[Block, list[Block]]:
    """Return the reference block and the list of benchmark blocks from the config."""
    reference = Block(config.reference.name, config.reference.split)
    benchmarks = [Block(spec.name, spec.split) for entry in config.benchmarks for spec in entry.values()]
    return reference, benchmarks


def _format_value(value: float, best_value: float) -> str:
    """Render a metric value, bolding it when it ties the block's best (minimum, lower-is-better)."""
    if pd.isna(value):
        return "---"
    value_str = f"{value:.3f}"
    if pd.notna(best_value) and np.isclose(value, best_value):
        value_str = f"\\textbf{{{value_str}}}"
    return value_str


def _format_gap(gap: float, best_gap: float, worst_gap: float) -> str:
    """Render a colored gap annotation; severity is scaled to the block's own ``[best, worst]`` gap range."""
    denom = max(abs(worst_gap - best_gap), EPSILON)
    severity = float(np.clip(abs(gap - best_gap) / denom, 0, 1))
    intensity = int(GAP_MIN_COLOR_VALUE + severity * (100 - GAP_MIN_COLOR_VALUE))
    color = "OrangeRed" if gap > 0 else "ForestGreen"
    return f"\\textcolor{{{color}!{intensity}}}{{{gap:+.2f}\\%}}"


def _build_mean_row(
    df: pd.DataFrame, eval_split: str, ref_split: str, metrics: list[str], *, is_reference: bool
) -> str:
    """Build a light-gray block mean row: per-metric mean value and (benchmark blocks) the mean per-model gap."""
    parts = ["", "Mean"]
    for metric in metrics:
        values = df[f"{eval_split}/{metric}"]
        mean_value = values.mean()
        if pd.isna(mean_value):
            parts.append("---")
            continue
        cell = f"{mean_value:.3f}"
        if not is_reference:
            gaps: pd.Series = relative_gap_pct(values, df[f"{ref_split}/{metric}"])  # pyright: ignore[reportAssignmentType, reportArgumentType]
            mean_gap = gaps.mean()
            if pd.notna(mean_gap):
                cell = f"{cell} ({mean_gap:+.2f}\\%)"
        parts.append(cell)
    return "\\rowcolor[gray]{0.9}\n" + " & ".join(parts) + " \\\\"


def _build_block_rows(
    df: pd.DataFrame, block: Block, ref_split: str, metrics: list[str], *, is_reference: bool
) -> list[str]:
    """Build the LaTeX rows for one block (reference or benchmark): one model per row plus a gray mean row.

    Reference blocks show plain absolute values; benchmark blocks annotate each value with its colored gap relative to
    the model's reference value. Best (minimum) values are bolded and gap severity is self-scaled within the block.
    """
    eval_split = block.split
    best_value = {metric: df[f"{eval_split}/{metric}"].min() for metric in metrics}
    gap_stats: dict[str, tuple[float, float]] = {}
    if not is_reference:
        for metric in metrics:
            gaps: pd.Series = relative_gap_pct(df[f"{eval_split}/{metric}"], df[f"{ref_split}/{metric}"])  # pyright: ignore[reportAssignmentType, reportArgumentType]
            gap_stats[metric] = (gaps.min(), gaps.max())

    rows: list[str] = []
    first_row = True
    for _, row in df.iterrows():
        parts = [f"\\multirow{{{len(df)}}}{{*}}{{\\texttt{{{block.name}}}}}" if first_row else ""]
        first_row = False
        parts.append(str(row["Model"]))

        for metric in metrics:
            value = row[f"{eval_split}/{metric}"]
            cell = _format_value(value, best_value[metric])
            if not is_reference and pd.notna(value):
                ref_value = row[f"{ref_split}/{metric}"]
                if pd.notna(ref_value):
                    gap = relative_gap_pct(value, ref_value)
                    cell = f"{cell} ({_format_gap(gap, *gap_stats[metric])})"
            parts.append(cell)
        rows.append(" & ".join(parts) + " \\\\")

    rows.append(_build_mean_row(df, eval_split, ref_split, metrics, is_reference=is_reference))
    return rows


def _build_overall_mean_row(df: pd.DataFrame, ref_split: str, benchmarks: list[Block], metrics: list[str]) -> str:
    """Build the closing gray row: grand mean value and gap over every benchmark x model entry (reference excluded)."""
    parts = ["\\texttt{Overall}", "Mean"]
    for metric in metrics:
        value_chunks, gap_chunks = [], []
        for block in benchmarks:
            values = df[f"{block.split}/{metric}"]
            value_chunks.append(values)
            gap_chunks.append(pd.Series(relative_gap_pct(values, df[f"{ref_split}/{metric}"])))
        mean_value = pd.concat(value_chunks, ignore_index=True).mean()
        mean_gap = pd.concat(gap_chunks, ignore_index=True).mean()
        parts.append(f"{mean_value:.3f} ({mean_gap:+.2f}\\%)" if pd.notna(mean_value) else "---")
    return "\\rowcolor[gray]{0.8}\n" + " & ".join(parts) + " \\\\"


def _write_tex_table(df: pd.DataFrame, blocks: list[Block], metrics: list[str], output_path: Path) -> str:
    """Write the vertical per-benchmark LaTeX table (reference block, one block per benchmark, overall mean)."""
    reference, benchmarks = blocks[0], blocks[1:]
    body: list[str] = list(_build_block_rows(df, reference, reference.split, metrics, is_reference=True))
    for block in benchmarks:
        body.append("\\midrule")
        body.extend(_build_block_rows(df, block, reference.split, metrics, is_reference=False))
    body.append("\\midrule")
    body.append(_build_overall_mean_row(df, reference.split, benchmarks, metrics))

    header = " & ".join(f"\\textbf{{{metric}}}" for metric in metrics)
    latex_lines = [
        "% Requires \\usepackage[table]{xcolor} (\\rowcolor) and \\usepackage{multirow}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{Unshifted-Generalization Results (gaps relative to \\texttt{{{reference.name}}})}}",
        "\\label{tab:unshifted_generalization_results}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l l " + "c" * len(metrics) + "}",
        "\\toprule",
        f"\\textbf{{Benchmark}} & \\textbf{{Model}} & {header} \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ]
    latex_table_str = "\n".join(latex_lines)

    output_file = output_path / "results.tex"
    output_file.write_text(latex_table_str)
    print(f"\n✓ LaTeX table saved as '{output_file}'")
    return latex_table_str


def _plot_benchmark_values(
    df: pd.DataFrame, blocks: list[Block], metrics: list[str], colormap: str, output_path: Path
) -> None:
    """Grouped bars of each metric per benchmark across models; the reference block (first) is hatched."""
    block_names = [block.name for block in blocks]
    block_splits = [block.split for block in blocks]
    models = df["Model"].to_numpy()
    n_models = len(models)
    palette = model_colors(models, colormap)

    n_cols = min(2, len(metrics))
    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(8.0, 2.2 * len(block_names)) * n_cols, 4.5 * n_rows),
        constrained_layout=True,
    )
    fig.suptitle(f"Per-Benchmark Performance (reference: {blocks[0].name})", fontsize=18, fontweight="bold")
    axes = np.atleast_1d(axes).flatten()

    x = np.arange(len(block_names))
    width = 0.8 / n_models
    for idx, metric in enumerate(metrics):
        ax: Axes = axes[idx]
        all_values: list[float] = []
        for model_idx, model in enumerate(models):
            values = [float(df.loc[df["Model"] == model, f"{split}/{metric}"].iloc[0]) for split in block_splits]
            all_values.extend(value for value in values if not np.isnan(value))
            offset = (model_idx - (n_models - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                color=palette[model_idx],
                edgecolor="black",
                linewidth=0.8,
                label=str(model),
                alpha=0.85,
            )
            bars[0].set_hatch("//")  # reference block
        ax.set_title(metric, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(block_names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Value (↓)", fontweight="bold")
        set_yaxis_limits(ax, all_values, padding_factor=0.15, lower_factor=0.4, min_padding=0.1)
        ax.yaxis.grid(visible=True, alpha=0.3)
        ax.set_axisbelow(True)
        if idx == 0:
            ax.legend(fontsize=8, ncol=2, title="Model")
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    # tight=False preserves this figure's current cropping; it is one of the few that omits it, likely an oversight.
    save_figure(fig, output_path / "benchmark_values.png", tight=False)


def _plot_gap_heatmaps(
    df: pd.DataFrame, ref_split: str, benchmarks: list[Block], metrics: list[str], output_path: Path
) -> None:
    """Benchmark x model heatmap of the percentage gap vs the reference, one panel per metric (red = worse)."""
    models = df["Model"].to_numpy()
    bench_names = [block.name for block in benchmarks]

    n_cols = min(2, len(metrics))
    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(7.0, 1.4 * len(models)) * n_cols, 1.0 * len(bench_names) * n_rows + 2.0),
        constrained_layout=True,
    )
    fig.suptitle("Performance Gap vs Reference (%)", fontsize=18, fontweight="bold")
    axes = np.atleast_1d(axes).flatten()

    for idx, metric in enumerate(metrics):
        ax: Axes = axes[idx]
        gap_matrix = np.full((len(benchmarks), len(models)), np.nan)
        for bench_idx, block in enumerate(benchmarks):
            for model_idx, model in enumerate(models):
                model_row = df[df["Model"] == model]
                value = float(model_row[f"{block.split}/{metric}"].iloc[0])
                reference = float(model_row[f"{ref_split}/{metric}"].iloc[0])
                gap_matrix[bench_idx, model_idx] = relative_gap_pct(value, reference)
        vmax = float(np.nanmax(np.abs(gap_matrix))) if np.isfinite(gap_matrix).any() else 1.0
        sns.heatmap(
            gap_matrix,
            ax=ax,
            cmap="RdYlGn_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt="+.1f",
            annot_kws={"fontsize": 8},
            xticklabels=models,
            yticklabels=bench_names,
            cbar_kws={"label": "Gap %"},
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(metric, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    # tight=False preserves this figure's current cropping; it is one of the few that omits it, likely an oversight.
    save_figure(fig, output_path / "gap_heatmap.png", tight=False)


def _print_summary(df: pd.DataFrame, ref_split: str, benchmarks: list[Block], metrics: list[str]) -> None:
    """Print the per-benchmark mean gap (over models) for the primary metric and flag the worst benchmark."""
    primary = metrics[0]
    print("\n" + "=" * 80)
    print(f"Mean gap vs reference (over models, metric={primary}):")
    print("=" * 80)
    bench_mean_gaps: dict[str, float] = {}
    for block in benchmarks:
        gaps: pd.Series = relative_gap_pct(df[f"{block.split}/{primary}"], df[f"{ref_split}/{primary}"])  # pyright: ignore[reportAssignmentType, reportArgumentType]
        bench_mean_gaps[block.name] = float(gaps.mean())
        print(f"  {block.name:25s}: {bench_mean_gaps[block.name]:+.2f}%")
    worst = max(bench_mean_gaps, key=lambda name: bench_mean_gaps[name])
    print(f"\nWorst benchmark (largest mean gap): {worst} ({bench_mean_gaps[worst]:+.2f}%)")


def run_unshifted_generalization_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Compare many evaluation benchmarks against a single in-distribution reference from a combined results file.

    For each model in ``config.models_to_compare`` the reference and per-benchmark metric values are looked up and
    joined by model name. Writes a vertical per-benchmark LaTeX table (``results.tex``) with gaps relative to the
    reference, grouped value bars (``benchmark_values.png``), and a benchmark x model gap heatmap (``gap_heatmap.png``).

    Args:
        config: Analysis configuration (``benchmarks_filepath``, ``reference``, ``benchmarks``, ``models_to_compare``,
            ``trajectory_forecasting_metrics``, ``benchmark_colormap``, ``show_run_id``).
        log: Logger.
        output_path: Directory to save the generated table and plots.
    """
    set_analysis_theme(log=log)

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
    reference, benchmarks = _parse_blocks(config)
    blocks = [reference, *benchmarks]

    splits = tuple(block.split for block in blocks)
    df = build_benchmark_df(metrics_df, splits, metrics, models_to_compare, show_run_id=config.show_run_id)
    if df.empty:
        log.warning("No matching models found in '%s'; nothing to analyze.", metrics_filepath)
        return

    log.info("Reference '%s' vs %d benchmarks over %d models", reference.name, len(benchmarks), len(df))
    _write_tex_table(df, blocks, metrics, output_path)
    _plot_benchmark_values(df, blocks, metrics, config.benchmark_colormap, output_path)
    _plot_gap_heatmaps(df, reference.split, benchmarks, metrics, output_path)
    _print_summary(df, reference.split, benchmarks, metrics)

    print("\n✓ Analysis complete!")
    log.info("Unshifted-generalization analysis complete!")
