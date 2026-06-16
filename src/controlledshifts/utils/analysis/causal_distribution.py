"""Causal vs non-causal agent distribution analysis across benchmark splits.

Compares, for each causal-agents benchmark, the train/validation/testing distributions of per-scenario agent counts
(causal, non-causal, fraction non-causal, total). The standard ``causal_agents`` benchmark reuses a random reference
split, so its distributions should match across splits; ``causal_agents_hard`` ranks scenarios by non-causal agent
count and sends the hardest scenarios to the test set, so its test distribution should be visibly shifted toward more
non-causal agents.

Counts are intrinsic to a scenario (independent of which benchmark/split it lands in), so they are computed once over
the union of scenario IDs and then bucketed per benchmark and split.

See `docs/ANALYSIS.md` for usage details.
"""

import json
import multiprocessing
import pickle  # nosec B403
from functools import partial
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from omegaconf import DictConfig
from tqdm import tqdm

from controlledshifts.benchmarks.common import get_noncausal_mask, load_benchmark_split
from controlledshifts.utils.analysis.common import SPLIT_COLOR_MAP
from controlledshifts.utils.plotting import set_analysis_theme


# Consistent ordering and display labels for the dataset splits.
_SPLIT_ORDER: tuple[str, ...] = ("training", "validation", "testing")
_SPLIT_LABELS: dict[str, str] = {"training": "Train", "validation": "Val", "testing": "Test"}

# Plot font sizes.
_TITLE_FONTSIZE = 16  # per-panel benchmark titles
_LABEL_FONTSIZE = 15  # axis labels
_TICK_FONTSIZE = 13  # tick numbers
_SUPTITLE_FONTSIZE = 17  # figure title
_LEGEND_FONTSIZE = 14

# Muted gray used for all figure text (titles, axis labels, tick numbers, legend).
_TEXT_COLOR = "#808080"

# Per-scenario quantities the analysis can plot, mapped to axis labels.
_QUANTITY_LABELS: dict[str, str] = {
    "n_causal": "Causal Agents per Scenario",
    "n_noncausal": "Non-causal agents per Scenario",
    "frac_noncausal": "Fraction Non-causal per Scenario",
    "n_total": "Total Agents per Scenario",
}


def _count_agents(input_filepath: Path, causal_labels_path: Path) -> tuple[str, int, int] | None:
    """Counts the non-causal agents and total agents in a scenario.

    Args:
        input_filepath: Path to the input scenario pkl.
        causal_labels_path: Directory with per-scenario JSON causal labels.

    Returns:
        Tuple of (scenario_id, non-causal count, total agent count), or None if the scenario or its causal labels are
        missing.
    """
    if not input_filepath.exists():
        return None

    with input_filepath.open("rb") as f:
        scenario = pickle.load(f)  # nosec B301

    scenario_id = scenario["scenario_id"]
    causal_labels_filepath = causal_labels_path / f"{scenario_id}.json"
    if not causal_labels_filepath.exists():
        return None

    with causal_labels_filepath.open("r") as f:
        causal_labels = json.load(f)

    noncausal_mask = get_noncausal_mask(scenario, causal_labels)
    return scenario_id, int(noncausal_mask.sum()), int(noncausal_mask.size)


def _collect_counts(
    scenario_ids: list[str], variants_base_path: Path, causal_labels_path: Path, num_workers: int, log: Logger
) -> pd.DataFrame:
    """Computes per-scenario causal/non-causal agent counts for a set of scenarios.

    Args:
        scenario_ids: Scenario IDs to count agents for.
        variants_base_path: Flat directory holding the unperturbed ``base`` scenario pkls.
        causal_labels_path: Directory with per-scenario JSON causal labels.
        num_workers: Number of worker processes.
        log: Logger for progress information.

    Returns:
        DataFrame indexed by scenario_id with columns ``n_noncausal``, ``n_total``, ``n_causal``, ``frac_noncausal``.
    """
    filepaths = [variants_base_path / f"{scenario_id}.pkl" for scenario_id in scenario_ids]
    chunksize = max(1, len(filepaths) // (num_workers * 8))
    with multiprocessing.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    partial(_count_agents, causal_labels_path=causal_labels_path), filepaths, chunksize=chunksize
                ),
                total=len(filepaths),
                desc="Counting agents",
            )
        )
    counts = [result for result in results if result is not None]
    missing = len(results) - len(counts)
    if missing:
        log.warning("Skipped %d scenarios missing their pkl or causal labels", missing)

    counts_df = pd.DataFrame(counts, columns=["scenario_id", "n_noncausal", "n_total"]).set_index("scenario_id")
    counts_df["n_causal"] = counts_df["n_total"] - counts_df["n_noncausal"]
    counts_df["frac_noncausal"] = counts_df["n_noncausal"] / counts_df["n_total"]
    return counts_df


def _build_long_frame(counts_df: pd.DataFrame, benchmark_name: str, split_json_path: Path) -> pd.DataFrame:
    """Joins per-scenario counts with a benchmark split into a long-form frame.

    Args:
        counts_df: Per-scenario counts indexed by scenario_id.
        benchmark_name: Display name recorded in the ``benchmark`` column.
        split_json_path: Path to the benchmark's split JSON.

    Returns:
        Long-form DataFrame with columns ``benchmark``, ``split``, ``scenario_id`` and the per-scenario count columns.
    """
    split = load_benchmark_split(split_json_path)
    frames = []
    for split_key in _SPLIT_ORDER:
        ids = [scenario_id for scenario_id in getattr(split, split_key) if scenario_id in counts_df.index]
        subset = counts_df.loc[ids].reset_index()
        subset.insert(0, "split", split_key)
        subset.insert(0, "benchmark", benchmark_name)
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def _add_split_legend(fig: Figure, palette: list[str]) -> None:
    """Attaches a single split legend to ``fig``, outside the panels on the right.

    Args:
        fig: Figure to attach the legend to.
        palette: One color per split (aligned with ``_SPLIT_ORDER``).
    """
    handles = [
        Patch(facecolor=color, edgecolor="none", label=_SPLIT_LABELS[split_key])
        for split_key, color in zip(_SPLIT_ORDER, palette, strict=False)
    ]
    legend = fig.legend(
        handles=handles,
        title="Split",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=_LEGEND_FONTSIZE,
        title_fontsize=_LEGEND_FONTSIZE,
        labelcolor=_TEXT_COLOR,
        frameon=False,
    )
    legend.get_title().set_color(_TEXT_COLOR)


def _plot_violin(
    long_df: pd.DataFrame,
    quantity: str,
    benchmark_names: list[str],
    palette: list[str],
    output_path: Path,
) -> None:
    """Saves a side-by-side violin plot of ``quantity`` per split, one panel per benchmark.

    Args:
        long_df: Long-form frame with ``benchmark``, ``split`` and the quantity columns.
        quantity: Column to plot on the y-axis.
        benchmark_names: Benchmark display names, in panel order.
        palette: One color per split (aligned with ``_SPLIT_ORDER``).
        output_path: Directory to save the figure.
    """
    order = list(_SPLIT_ORDER)
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
            palette=palette,
            legend=False,
            cut=0,
            density_norm="width",
            alpha=0.6,
            ax=ax,
        )
        ax.set_title(benchmark, fontsize=_TITLE_FONTSIZE, color=_TEXT_COLOR)
        ax.set_xlabel("")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([_SPLIT_LABELS[split_key] for split_key in order])
        ax.set_ylabel(
            _QUANTITY_LABELS[quantity] if ax is axes[0, 0] else "", fontsize=_LABEL_FONTSIZE, color=_TEXT_COLOR
        )
        ax.tick_params(labelsize=_TICK_FONTSIZE, colors=_TEXT_COLOR)

    fig.suptitle(_QUANTITY_LABELS[quantity], fontsize=_SUPTITLE_FONTSIZE, color=_TEXT_COLOR)
    _add_split_legend(fig, palette)
    fig.tight_layout()
    output_file = output_path / f"{quantity}_violin.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def _plot_ridge(
    long_df: pd.DataFrame,
    quantity: str,
    benchmark_names: list[str],
    palette: list[str],
    output_path: Path,
) -> None:
    """Saves a side-by-side ridgeline plot of ``quantity``: one overlapping density row per split, per benchmark.

    Each split's distribution gets its own row (stacked with a slight vertical overlap) so the three can be read
    separately, while sharing the x-axis within a benchmark column. Densities are normalized per split.

    Args:
        long_df: Long-form frame with ``benchmark``, ``split`` and the quantity columns.
        quantity: Column to plot on the x-axis.
        benchmark_names: Benchmark display names, in column order.
        palette: One color per split (aligned with ``_SPLIT_ORDER``).
        output_path: Directory to save the figure.
    """
    order = list(_SPLIT_ORDER)
    color_map = dict(zip(order, palette, strict=False))
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
                    _SPLIT_LABELS[split_key],
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=_LABEL_FONTSIZE,
                    fontweight="bold",
                    color=color_map[split_key],
                )
            if row == 0:
                ax.set_title(benchmark, fontsize=_TITLE_FONTSIZE, color=_TEXT_COLOR)
            if row == n_rows - 1:
                ax.set_xlabel(_QUANTITY_LABELS[quantity], fontsize=_LABEL_FONTSIZE, color=_TEXT_COLOR)
                ax.tick_params(axis="x", labelsize=_TICK_FONTSIZE, colors=_TEXT_COLOR)
            else:
                ax.set_xlabel("")
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", length=0)

    fig.suptitle(_QUANTITY_LABELS[quantity], fontsize=_SUPTITLE_FONTSIZE, color=_TEXT_COLOR)
    fig.subplots_adjust(hspace=-0.25, top=0.9)
    output_file = output_path / f"{quantity}_ridge.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def _plot_histogram(
    long_df: pd.DataFrame,
    quantity: str,
    benchmark_names: list[str],
    palette: list[str],
    output_path: Path,
) -> None:
    """Saves a side-by-side density plot of ``quantity`` per split, one panel per benchmark.

    Splits are overlaid as filled kernel-density curves, each normalized independently so they are comparable across
    splits of different sizes. Complements the ridgeline view, which separates the same distributions onto their own
    rows.

    Args:
        long_df: Long-form frame with ``benchmark``, ``split`` and the quantity columns.
        quantity: Column to plot on the x-axis.
        benchmark_names: Benchmark display names, in panel order.
        palette: One color per split (aligned with ``_SPLIT_ORDER``).
        output_path: Directory to save the figure.
    """
    order = list(_SPLIT_ORDER)
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
            palette=palette,
            fill=True,
            alpha=0.3,
            common_norm=False,
            cut=0,
            clip=(0, None),
            legend=False,
            ax=ax,
        )
        ax.set_title(benchmark, fontsize=_TITLE_FONTSIZE, color=_TEXT_COLOR)
        ax.set_xlabel(_QUANTITY_LABELS[quantity], fontsize=_LABEL_FONTSIZE, color=_TEXT_COLOR)
        ax.set_ylabel("Density" if ax is axes[0, 0] else "", fontsize=_LABEL_FONTSIZE, color=_TEXT_COLOR)
        ax.tick_params(labelsize=_TICK_FONTSIZE, colors=_TEXT_COLOR)

    fig.suptitle(_QUANTITY_LABELS[quantity], fontsize=_SUPTITLE_FONTSIZE, color=_TEXT_COLOR)
    _add_split_legend(fig, palette)
    fig.tight_layout()
    output_file = output_path / f"{quantity}_histogram.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def _write_summary(long_df: pd.DataFrame, quantities: list[str], output_path: Path) -> None:
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


def _build_distribution_frame(config: DictConfig, log: Logger, output_path: Path) -> pd.DataFrame | None:
    """Builds (and caches) the long-form distribution frame from the benchmark splits and per-scenario counts.

    Per-scenario counts are loaded from ``per_scenario_counts.csv`` when present (unless ``overwrite`` is set),
    otherwise computed from the scenario pkls and cached. The counts are then bucketed by each benchmark's split and
    written to ``causal_distribution.csv``.

    Args:
        config (DictConfig): Analysis configuration (``splits_path``, ``variants_base_path``, ``causal_labels_path``,
            ``num_workers``, ``overwrite``, ``benchmarks``).
        log (Logger): Logger for progress information.
        output_path (Path): Directory holding/receiving the cached CSVs.

    Returns:
        The long-form DataFrame, or None when no configured benchmark split could be found.
    """
    splits_path = Path(config.splits_path)
    variants_base_path = Path(config.variants_base_path)
    causal_labels_path = Path(config.causal_labels_path)

    benchmarks: list[tuple[str, Path]] = []
    all_ids: set[str] = set()
    for benchmark_entry in config.benchmarks:
        key, spec = next(iter(benchmark_entry.items()))
        split_json_path = splits_path / f"{spec.split_json}.json"
        if not split_json_path.exists():
            log.error("Split JSON not found at %s; skipping benchmark '%s'", split_json_path, key)
            continue
        split = load_benchmark_split(split_json_path)
        all_ids.update(split.training, split.validation, split.testing)
        benchmarks.append((spec.name, split_json_path))
        log.info("Loaded benchmark '%s' (%s) from %s", key, spec.name, split_json_path)

    if not benchmarks:
        log.error("No benchmark splits found under %s; nothing to analyze.", splits_path)
        return None

    counts_cache = output_path / "per_scenario_counts.csv"
    if counts_cache.exists() and not config.overwrite:
        log.info("Loading cached per-scenario counts from %s (set overwrite=true to recompute)", counts_cache)
        counts_df = pd.read_csv(counts_cache).set_index("scenario_id")
    else:
        counts_df = _collect_counts(sorted(all_ids), variants_base_path, causal_labels_path, config.num_workers, log)
        counts_df.to_csv(counts_cache)
        log.info("Saved per-scenario counts for %d scenarios to %s", len(counts_df), counts_cache)

    long_df = pd.concat(
        [_build_long_frame(counts_df, name, split_json_path) for name, split_json_path in benchmarks],
        ignore_index=True,
    )
    long_df.to_csv(output_path / "causal_distribution.csv", index=False)
    return long_df


def run_causal_distribution_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Compares causal/non-causal agent distributions across train/val/test splits for the causal-agents benchmarks.

    Renders side-by-side violin and histogram plots (one panel per benchmark) plus a per-benchmark, per-split summary
    for every configured quantity. The plots are driven entirely by the long-form ``causal_distribution.csv``: when it
    already exists (and ``overwrite`` is false) it is loaded directly, so re-rendering touches neither the scenario
    pkls nor the split JSONs. Otherwise the frame is rebuilt — reusing the cached ``per_scenario_counts.csv`` when
    present, and only falling back to loading scenarios when no cache exists or ``overwrite`` is set.

    Args:
        config (DictConfig): Analysis configuration (``splits_path``, ``variants_base_path``, ``causal_labels_path``,
            ``num_workers``, ``overwrite``, ``quantities``, ``benchmarks``).
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated counts, summary and plots.
    """
    set_analysis_theme(log=log)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    quantities = list(config.quantities)

    long_cache = output_path / "causal_distribution.csv"
    if long_cache.exists() and not config.overwrite:
        log.info("Regenerating plots from cached %s (set overwrite=true to recompute from scenarios)", long_cache)
        long_df = pd.read_csv(long_cache)
    else:
        long_df = _build_distribution_frame(config, log, output_path)
        if long_df is None:
            return

    long_df["split"] = pd.Categorical(long_df["split"], categories=list(_SPLIT_ORDER), ordered=True)
    benchmark_names = list(dict.fromkeys(long_df["benchmark"]))

    _write_summary(long_df, quantities, output_path)

    palette = [SPLIT_COLOR_MAP[split_key] for split_key in _SPLIT_ORDER]
    for quantity in quantities:
        _plot_violin(long_df, quantity, benchmark_names, palette, output_path)
        _plot_histogram(long_df, quantity, benchmark_names, palette, output_path)
        _plot_ridge(long_df, quantity, benchmark_names, palette, output_path)

    print("\n✓ Analysis complete!")
    log.info("Causal distribution analysis complete!")
