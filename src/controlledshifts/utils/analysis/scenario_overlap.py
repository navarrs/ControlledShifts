"""Scenario-overlap analysis across benchmark splits.

Quantifies how different the benchmarks are by measuring, for each split, how many scenarios their corresponding
splits share. For every configured split (training/validation/testing) it builds a symmetric benchmark x benchmark
matrix of the Jaccard index (intersection over union) of the scenario IDs and renders it as an annotated heatmap. The
raw intersection counts (and split sizes) are written alongside to a tidy CSV.

This reads only the split JSONs, so it is fast and needs no caching.

See `docs/ANALYSIS.md` for usage details.
"""

from logging import Logger
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from controlledshifts.benchmarks.common import BenchmarkSplit, load_benchmark_split
from controlledshifts.utils.plotting import set_analysis_theme


_MIN_BENCHMARKS = 2
_LABEL_FONTSIZE = 8
_ANNOT_FONTSIZE = 14


def _load_benchmarks(config: DictConfig, log: Logger) -> dict[str, BenchmarkSplit]:
    """Loads the configured benchmark splits, keyed by display name, skipping any whose JSON is missing."""
    splits_path = Path(config.splits_path)
    benchmarks: dict[str, BenchmarkSplit] = {}
    for benchmark_entry in config.benchmarks:
        _, spec = next(iter(benchmark_entry.items()))
        split_json_path = splits_path / f"{spec.split_json}.json"
        if not split_json_path.exists():
            log.error("Split JSON not found at %s; skipping benchmark '%s'", split_json_path, spec.name)
            continue
        benchmarks[spec.name] = load_benchmark_split(split_json_path)
        log.info("Loaded benchmark '%s' from %s", spec.name, split_json_path)
    return benchmarks


def _overlap_records(split: str, names: list[str], id_sets: list[set[str]]) -> list[dict[str, str | int | float]]:
    """Builds the long-form overlap records for one split over every benchmark pair (including self-pairs)."""
    records: list[dict[str, str | int | float]] = []
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            intersection = len(id_sets[i] & id_sets[j])
            union = len(id_sets[i] | id_sets[j])
            records.append(
                {
                    "split": split,
                    "benchmark_a": name_a,
                    "benchmark_b": name_b,
                    "size_a": len(id_sets[i]),
                    "size_b": len(id_sets[j]),
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union if union else 0.0,
                }
            )
    return records


def _plot_overlap_grid(
    matrices: list[npt.NDArray[np.float64]], labels: list[str], splits: list[str], output_path: Path
) -> None:
    """Saves the per-split Jaccard heatmaps side by side in one figure with a single shared colorbar."""
    panel = max(4.5, 1.3 * len(labels))
    fig, axes = plt.subplots(
        1, len(splits), figsize=(panel * len(splits), panel), squeeze=False, gridspec_kw={"wspace": 0.05}
    )
    for i, (ax, matrix, split) in enumerate(zip(axes[0], matrices, splits, strict=True)):
        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": _ANNOT_FONTSIZE},
            vmin=0.0,
            vmax=1.0,
            cmap="rocket",
            xticklabels=labels,
            yticklabels=labels if i == 0 else False,
            square=True,
            cbar=False,
        )
        ax.set_title(f"{split.capitalize()} Set", fontweight="bold")
        ax.tick_params(axis="x", rotation=0, labelsize=_LABEL_FONTSIZE)
        ax.tick_params(axis="y", rotation=0, labelsize=_LABEL_FONTSIZE)

    # Draw once so the square=True axes settle into their final boxes, then size the colorbar to match the
    # rightmost heatmap's height (rather than the taller subplot axes).
    fig.canvas.draw()
    last_pos = axes[0][-1].get_position()
    cax = fig.add_axes((last_pos.x1 + 0.01, last_pos.y0, 0.012, last_pos.height))
    mappable = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0), cmap="rocket")
    cb = fig.colorbar(mappable, cax=cax, label="Jaccard overlap")
    cb.outline.set_visible(False)

    fig.suptitle("Scenario Overlap across Benchmarks", fontweight="bold")
    output_file = output_path / "scenario_overlap.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Heatmap saved as '{output_file}'")


def run_scenario_overlap_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Measures pairwise scenario overlap between benchmarks for each split and renders per-split Jaccard heatmaps.

    For every split in ``config.splits`` a symmetric benchmark x benchmark Jaccard matrix is computed over the
    scenario IDs and saved as an annotated heatmap, with the raw intersection counts and split sizes written to
    ``scenario_overlap.csv``.

    Args:
        config: Analysis configuration (``splits_path``, ``splits``, ``benchmarks``).
        log: Logger.
        output_path: Directory to save the heatmaps and CSV.
    """
    set_analysis_theme(log=log)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    benchmarks = _load_benchmarks(config, log)
    if len(benchmarks) < _MIN_BENCHMARKS:
        log.error("Need at least two benchmarks to measure overlap; found %d.", len(benchmarks))
        return

    names = list(benchmarks)
    splits = list(config.splits)
    records: list[dict[str, str | int | float]] = []
    matrices: list[npt.NDArray[np.float64]] = []
    for split in splits:
        id_sets = [set(getattr(benchmarks[name], split)) for name in names]
        split_records = _overlap_records(split, names, id_sets)
        records.extend(split_records)

        frame = pd.DataFrame(split_records)
        matrix = frame.pivot_table(index="benchmark_a", columns="benchmark_b", values="jaccard").loc[names, names]
        matrices.append(matrix.to_numpy())

    _plot_overlap_grid(matrices, names, splits, output_path)

    csv_path = output_path / "scenario_overlap.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    log.info("Saved overlap table to %s", csv_path)

    print("\n✓ Analysis complete!")
    log.info("Scenario overlap analysis complete!")
