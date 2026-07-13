"""Shared, general-purpose utilities for model analysis.

See `docs/ANALYSIS.md` for usage details.
"""

from dataclasses import dataclass
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.patches import Patch
from matplotlib.text import Text
from numpy.typing import NDArray

from controlledshifts.utils.constants import EPSILON


MODEL_NAME_MAP = {
    "autobot": "AutoBot",
    "scenetransformer": "SceneTransformer",
    "wayformer": "Wayformer",
    "mtr": "MTR",
    "safe-wayformer": "Safe-Wayformer",
    "naive": "Naive",
    "cvm": "CVM",
}

MODEL_COLOR_MAP = {
    "Naive": "#000000",
    "AutoBot": "#FFA358",
    "SceneTransformer": "#FF705D",
    "Wayformer": "#5EAEFE",
    "MTR": "#B774DE",
    "CVM": "#5AD495",
}

SPLIT_COLOR_MAP = {
    "training": "#355070",
    "validation": "#8f769f",
    "testing": "#b56576",
}

METRIC_NAME_MAP = {
    "brierFDE": "BrierFDE",
    "minFDE6": "MinFDE",
    "minADE6": "MinADE",
    "missRate": "MissRate",
    "collisionRate0.25": "CollisionRate",
}

MODEL_SIZE_MAP = {
    "Naive": "624k",
    "AutoBot": "1.5M",
    "SceneTransformer": "7.6M",
    "Wayformer": "15.1M",
    "Safe-Wayformer": "15.2M",
    "MTR": "27.2M",  # This is the size with d_model=256. The original MTR with d_model=512 has 65M parameters.
}


def relative_gap_pct(value: float | NDArray, reference: float | NDArray) -> float | NDArray:
    """Compute ``(value - reference) / |reference| * 100``. Works for scalars and numpy arrays."""
    return ((value - reference) / (np.abs(reference) + EPSILON)) * 100


def model_colors(models: pd.Index | NDArray, colormap: str) -> list[str | tuple[float, float, float]]:
    """Per-model colors aligned to ``models``, using the fixed :data:`MODEL_COLOR_MAP` where available.

    Models without a fixed color fall back to the configured ``colormap`` palette (assigned in order among the
    unmapped models), so the Naive baseline stays black and every model keeps the same color across plots.
    """
    fallback = iter(sns.color_palette(colormap, sum(model not in MODEL_COLOR_MAP for model in models)))
    return [MODEL_COLOR_MAP[model] if model in MODEL_COLOR_MAP else next(fallback) for model in models]


def set_yaxis_limits(
    ax: Axes,
    values: list[float],
    *,
    padding_factor: float = 0.5,
    lower_factor: float = 1.0,
    min_padding: float = 0.05,
) -> None:
    """Set y-axis limits with padding around the data range.

    Args:
        ax: Matplotlib axis to modify.
        values: Data values used to compute the range.
        padding_factor: Fraction of the data range to use as padding.
        lower_factor: Multiplier applied to padding on the lower end (useful for bar charts).
        min_padding: Minimum padding when the data range is zero.
    """
    if not values:
        return
    ymin, ymax = np.nanmin(values), np.nanmax(values)
    padding = padding_factor * (ymax - ymin) if ymax > ymin else min_padding
    ax.set_ylim(ymin - padding * lower_factor, ymax + padding)


def flatten_metrics(data: dict, prefix: str = "") -> dict[str, float | int | str | bool | None]:
    """Recursively flatten a nested dict, joining keys with dots."""
    flat: dict[str, float | int | str | bool | None] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name))
        else:
            flat[name] = value
    return flat


def plot_heatmap(  # noqa: PLR0913
    heatmap: npt.NDArray[np.float64],
    title: str,
    x_label: str,
    y_label: str,
    cbar_label: str,
    output_filepath: Path,
    colormap: str = "viridis",
) -> None:
    """Visualizes a heatmap matrix.

    Args:
        heatmap: the matrix to plot.
        title: the title of the heatmap.
        x_label: the label of the x-axis.
        y_label: the label of the y-axis.
        cbar_label: the label of the colorbar.
        colormap: the colormap to use.
        output_filepath: filepath to save the visualization.
    """
    plt.figure(figsize=(35, 30))

    plt.imshow(heatmap, cmap=colormap, aspect="auto")
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=40)
    cbar.set_label(cbar_label, size=40)

    plt.title(title, fontsize=50)
    plt.xlabel(x_label, fontsize=40)
    plt.ylabel(y_label, fontsize=40)
    plt.xticks(range(heatmap.shape[0]))
    plt.yticks(range(heatmap.shape[1]))
    plt.grid(visible=False)

    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()
    print(f"Heatmap saved to {output_filepath}")


# --- Per-scenario distribution plots across train/val/test splits ----------------------------------------------------
# Shared by the causal-agents and ego-safeshift score distribution analyses: each renders one quantity per figure, with
# the three splits compared side by side (violin/density) or stacked (ridgeline), one panel per benchmark. The
# quantity-to-label mapping is passed in by the caller.

SPLIT_ORDER: tuple[str, ...] = ("training", "validation", "testing")
SPLIT_LABELS: dict[str, str] = {"training": "Train", "validation": "Val", "testing": "Test"}

_TITLE_FONTSIZE = 16
_LABEL_FONTSIZE = 15
_TICK_FONTSIZE = 13
_SUPTITLE_FONTSIZE = 17
_LEGEND_FONTSIZE = 14
TEXT_COLOR = "#808080"


@dataclass(frozen=True)
class SplitDistributionPlotConfig:
    """Shared rendering context for the split-distribution plots.

    Attributes:
        benchmark_names: Benchmark display names, in panel/column order.
        palette: One color per split, aligned with ``SPLIT_ORDER``.
        quantity_labels: Maps each quantity column to its axis display label.
        output_path: Directory the figures and summary are written to.
        title_suffix: Appended to the quantity label to form the figure title (axis labels omit it).
    """

    benchmark_names: list[str]
    palette: list[str]
    quantity_labels: dict[str, str]
    output_path: Path
    title_suffix: str = ""


def _add_split_legend(fig: Figure, palette: list[str]) -> Legend:
    """Attaches a single split legend to ``fig``, outside the panels on the right.

    Args:
        fig: Figure to attach the legend to.
        palette: One color per split (aligned with ``SPLIT_ORDER``).

    Returns:
        The attached legend.
    """
    handles = [
        Patch(facecolor=color, edgecolor="none", label=SPLIT_LABELS[split_key])
        for split_key, color in zip(SPLIT_ORDER, palette, strict=False)
    ]
    legend = fig.legend(
        handles=handles,
        title="Split",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=_LEGEND_FONTSIZE,
        title_fontsize=_LEGEND_FONTSIZE,
        labelcolor=TEXT_COLOR,
        frameon=False,
    )
    legend.get_title().set_color(TEXT_COLOR)
    return legend


def _center_suptitle_over_content(fig: Figure, suptitle: Text, legend: Legend) -> None:
    """Recenters ``suptitle`` over the union of the panels and the outside legend.

    The figure legend sits to the right of the panels, and ``bbox_inches="tight"`` expands the saved image to include
    it, so a title left at the default figure-center (x=0.5) lands left of the visible center. This moves the title to
    the horizontal midpoint of the panels-plus-legend extent so it reads as centered in the saved figure.

    Args:
        fig: The figure, already laid out (after ``tight_layout``).
        suptitle: The suptitle text to reposition.
        legend: The outside legend that widens the saved figure.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_width = fig.bbox.width
    # Tight axes bboxes include the y-axis label and tick numbers on the left; the legend widens the right. Centering
    # over their union (the same extent bbox_inches="tight" crops to) makes the title read as centered in the save.
    bboxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
    bboxes.append(legend.get_window_extent(renderer))
    left = min(bbox.x0 for bbox in bboxes) / fig_width
    right = max(bbox.x1 for bbox in bboxes) / fig_width
    suptitle.set_x((left + right) / 2)


def plot_violin(long_df: pd.DataFrame, quantity: str, plot_config: SplitDistributionPlotConfig) -> None:
    """Saves a side-by-side violin plot of ``quantity`` per split, one panel per benchmark.

    Args:
        long_df: Long-form frame with ``benchmark``, ``split`` and the quantity columns.
        quantity: Column to plot on the y-axis.
        plot_config: Shared rendering context (benchmark names, palette, quantity labels, output path).
    """
    benchmark_names = plot_config.benchmark_names
    order = list(SPLIT_ORDER)
    fig, axes = plt.subplots(1, len(benchmark_names), figsize=(6 * len(benchmark_names), 6), sharey=True, squeeze=False)
    for ax, benchmark in zip(axes[0], benchmark_names, strict=False):
        data = long_df[long_df["benchmark"] == benchmark]
        sns.violinplot(
            data=data,
            x="split",
            y=quantity,
            order=order,
            hue="split",
            hue_order=order,
            palette=plot_config.palette,
            legend=False,
            cut=0,
            density_norm="width",
            alpha=0.6,
            ax=ax,
        )
        ax.set_title(benchmark, fontsize=_TITLE_FONTSIZE, color=TEXT_COLOR)
        ax.set_xlabel("")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([SPLIT_LABELS[split_key] for split_key in order])
        label = plot_config.quantity_labels[quantity] if ax is axes[0, 0] else ""
        ax.set_ylabel(label, fontsize=_LABEL_FONTSIZE, color=TEXT_COLOR)
        ax.tick_params(labelsize=_TICK_FONTSIZE, colors=TEXT_COLOR)

    title = f"{plot_config.quantity_labels[quantity]}{plot_config.title_suffix}"
    suptitle = fig.suptitle(title, fontsize=_SUPTITLE_FONTSIZE, color=TEXT_COLOR)
    legend = _add_split_legend(fig, plot_config.palette)
    fig.tight_layout()
    _center_suptitle_over_content(fig, suptitle, legend)
    output_file = plot_config.output_path / f"{quantity}_violin.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def plot_ridge(long_df: pd.DataFrame, quantity: str, plot_config: SplitDistributionPlotConfig) -> None:
    """Saves a side-by-side ridgeline plot of ``quantity``: one overlapping density row per split, per benchmark.

    Each split's distribution gets its own row (stacked with a slight vertical overlap) so the three can be read
    separately, while sharing the x-axis within a benchmark column. Densities are normalized per split.

    Args:
        long_df: Long-form frame with ``benchmark``, ``split`` and the quantity columns.
        quantity: Column to plot on the x-axis.
        plot_config: Shared rendering context (benchmark names, palette, quantity labels, output path).
    """
    benchmark_names = plot_config.benchmark_names
    order = list(SPLIT_ORDER)
    color_map = dict(zip(order, plot_config.palette, strict=False))
    n_rows, n_cols = len(order), len(benchmark_names)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 1.5 * n_rows + 1), sharex=True, sharey=True, squeeze=False
    )
    for col, benchmark in enumerate(benchmark_names):
        data = long_df[long_df["benchmark"] == benchmark]
        for row, split_key in enumerate(order):
            ax = axes[row, col]
            sns.kdeplot(
                x=data[data["split"] == split_key][quantity],
                fill=True,
                alpha=0.6,
                color=color_map[split_key],
                cut=0,
                clip=(0, None),
                linewidth=1.2,
                ax=ax,
            )
            ax.patch.set_alpha(0.0)  # transparent background so overlapping rows show through
            ax.set_ylabel("")
            ax.set_yticks([])
            ax.set_ylim(bottom=0)
            for spine in ("left", "right", "top"):
                ax.spines[spine].set_visible(False)

            if col == 0:
                ax.text(
                    0.0,
                    0.15,
                    SPLIT_LABELS[split_key],
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=_LABEL_FONTSIZE,
                    fontweight="bold",
                    color=color_map[split_key],
                )
            if row == 0:
                ax.set_title(benchmark, fontsize=_TITLE_FONTSIZE, color=TEXT_COLOR)
            if row == n_rows - 1:
                ax.set_xlabel(plot_config.quantity_labels[quantity], fontsize=_LABEL_FONTSIZE, color=TEXT_COLOR)
                ax.tick_params(axis="x", labelsize=_TICK_FONTSIZE, colors=TEXT_COLOR)
            else:
                ax.set_xlabel("")
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", length=0)

    title = f"{plot_config.quantity_labels[quantity]}{plot_config.title_suffix}"
    fig.suptitle(title, fontsize=_SUPTITLE_FONTSIZE, color=TEXT_COLOR)
    fig.subplots_adjust(hspace=-0.25, top=0.9)
    output_file = plot_config.output_path / f"{quantity}_ridge.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def plot_distribution_histogram(long_df: pd.DataFrame, quantity: str, plot_config: SplitDistributionPlotConfig) -> None:
    """Saves a side-by-side density plot of ``quantity`` per split, one panel per benchmark.

    Splits are overlaid as filled kernel-density curves, each normalized independently so they are comparable across
    splits of different sizes. Complements the ridgeline view, which separates the same distributions onto their own
    rows.

    Args:
        long_df: Long-form frame with ``benchmark``, ``split`` and the quantity columns.
        quantity: Column to plot on the x-axis.
        plot_config: Shared rendering context (benchmark names, palette, quantity labels, output path).
    """
    benchmark_names = plot_config.benchmark_names
    order = list(SPLIT_ORDER)
    fig, axes = plt.subplots(
        1, len(benchmark_names), figsize=(6 * len(benchmark_names), 6), sharex=True, sharey=True, squeeze=False
    )
    for ax, benchmark in zip(axes[0], benchmark_names, strict=False):
        data = long_df[long_df["benchmark"] == benchmark]
        sns.kdeplot(
            data=data,
            x=quantity,
            hue="split",
            hue_order=order,
            palette=plot_config.palette,
            fill=True,
            alpha=0.3,
            common_norm=False,
            cut=0,
            clip=(0, None),
            legend=False,
            ax=ax,
        )
        ax.set_title(benchmark, fontsize=_TITLE_FONTSIZE, color=TEXT_COLOR)
        ax.set_xlabel(plot_config.quantity_labels[quantity], fontsize=_LABEL_FONTSIZE, color=TEXT_COLOR)
        ax.set_ylabel("Density" if ax is axes[0, 0] else "", fontsize=_LABEL_FONTSIZE, color=TEXT_COLOR)
        ax.tick_params(labelsize=_TICK_FONTSIZE, colors=TEXT_COLOR)

    title = f"{plot_config.quantity_labels[quantity]}{plot_config.title_suffix}"
    suptitle = fig.suptitle(title, fontsize=_SUPTITLE_FONTSIZE, color=TEXT_COLOR)
    legend = _add_split_legend(fig, plot_config.palette)
    fig.tight_layout()
    _center_suptitle_over_content(fig, suptitle, legend)
    output_file = plot_config.output_path / f"{quantity}_histogram.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def write_split_distribution_summary(long_df: pd.DataFrame, quantities: list[str], output_path: Path) -> None:
    """Writes per-benchmark, per-split mean/median/std/count for every quantity to ``summary.csv``.

    Args:
        long_df: Long-form frame with ``benchmark``, ``split`` and the quantity columns.
        quantities: Quantity columns to summarize.
        output_path: Directory to save the CSV.
    """
    summary = long_df.groupby(["benchmark", "split"], observed=True)[quantities].agg(["mean", "median", "std", "count"])
    summary.columns = [f"{quantity}_{stat}" for quantity, stat in summary.columns]
    summary_file = output_path / "summary.csv"
    summary.reset_index().to_csv(summary_file, index=False)
    print(f"✓ Summary saved as '{summary_file}'")


def render_distribution_plots(
    long_df: pd.DataFrame, quantities: list[str], plot_config: SplitDistributionPlotConfig, log: Logger
) -> None:
    """Writes the per-split summary and the violin/histogram/ridge figure for every quantity.

    Args:
        long_df: Long-form frame with ``benchmark``, ``split`` and the quantity columns.
        quantities: Quantity columns to plot and summarize.
        plot_config: Shared rendering context (benchmark names, palette, quantity labels, output path).
        log: Logger for completion reporting.
    """
    write_split_distribution_summary(long_df, quantities, plot_config.output_path)
    for quantity in quantities:
        plot_violin(long_df, quantity, plot_config)
        plot_distribution_histogram(long_df, quantity, plot_config)
        plot_ridge(long_df, quantity, plot_config)
    log.info(
        "Rendered distribution plots for %d quantities across %d benchmarks",
        len(quantities),
        len(plot_config.benchmark_names),
    )
