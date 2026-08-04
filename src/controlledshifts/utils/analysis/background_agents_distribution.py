"""Non-background vs background agent distribution analysis across benchmark splits.

Compares, for each background-agents benchmark, the train/validation/testing distributions of per-scenario agent
counts (non-background, background, fraction background, total). The standard ``background_agents`` benchmark reuses
a random reference split, so its distributions should match across splits; ``background_agents_hard`` ranks
scenarios by background agent count and sends the hardest scenarios to the test set, so its test distribution should be
visibly shifted toward more background agents.

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

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from controlledshifts.benchmarks.common import get_background_mask, load_benchmark_split
from controlledshifts.utils.analysis.common import (
    SPLIT_COLOR_MAP,
    SPLIT_ORDER,
    SplitDistributionPlotConfig,
    iter_benchmarks,
    render_distribution_plots,
)
from controlledshifts.utils.plotting import set_analysis_theme


# Keys are the CSV column names and the per-quantity plot-file stems; the label strings are display-only.
_QUANTITY_LABELS: dict[str, str] = {
    "n_non_background": "Non-Background Agents per Scenario",
    "n_background": "Background Agents per Scenario",
    "frac_background": "Fraction Background per Scenario",
    "n_total": "Total Agents per Scenario",
}


def _count_agents(input_filepath: Path, causal_labels_path: Path) -> tuple[str, int, int] | None:
    """Counts the background agents and total agents in a scenario.

    Args:
        input_filepath: Path to the input scenario pkl.
        causal_labels_path: Directory with per-scenario JSON causal labels.

    Returns:
        Tuple of (scenario_id, background count, total agent count), or None if the scenario or its causal labels are
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

    background_mask = get_background_mask(scenario, causal_labels)
    return scenario_id, int(background_mask.sum()), int(background_mask.size)


def _collect_counts(
    scenario_ids: list[str], variants_base_path: Path, causal_labels_path: Path, num_workers: int, log: Logger
) -> pd.DataFrame:
    """Computes per-scenario non-background/background agent counts for a set of scenarios.

    Args:
        scenario_ids: Scenario IDs to count agents for.
        variants_base_path: Flat directory holding the unperturbed ``base`` scenario pkls.
        causal_labels_path: Directory with per-scenario JSON causal labels.
        num_workers: Number of worker processes.
        log: Logger for progress information.

    Returns:
        DataFrame indexed by scenario_id with columns ``n_background``, ``n_total``, ``n_non_background``,
        ``frac_background``.
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

    counts_df = pd.DataFrame(counts, columns=["scenario_id", "n_background", "n_total"]).set_index("scenario_id")
    counts_df["n_non_background"] = counts_df["n_total"] - counts_df["n_background"]
    counts_df["frac_background"] = counts_df["n_background"] / counts_df["n_total"]
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
    for split_key in SPLIT_ORDER:
        ids = [scenario_id for scenario_id in getattr(split, split_key) if scenario_id in counts_df.index]
        subset = counts_df.loc[ids].reset_index()
        subset.insert(0, "split", split_key)
        subset.insert(0, "benchmark", benchmark_name)
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def _build_distribution_frame(config: DictConfig, log: Logger, output_path: Path) -> pd.DataFrame | None:
    """Builds (and caches) the long-form distribution frame from the benchmark splits and per-scenario counts.

    Per-scenario counts are loaded from ``per_scenario_counts.csv`` when present (unless ``overwrite`` is set),
    otherwise computed from the scenario pkls and cached. The counts are then bucketed by each benchmark's split and
    written to ``background_agents_distribution.csv``.

    Args:
        config: Analysis configuration (``splits_path``, ``variants_base_path``, ``causal_labels_path``,
            ``num_workers``, ``overwrite``, ``benchmarks``).
        log: Logger.
        output_path: Directory holding/receiving the cached CSVs.

    Returns:
        The long-form DataFrame, or None when no configured benchmark split could be found.
    """
    splits_path = Path(config.splits_path)
    variants_base_path = Path(config.variants_base_path)
    causal_labels_path = Path(config.causal_labels_path)

    benchmarks: list[tuple[str, Path]] = []
    all_ids: set[str] = set()
    for key, spec in iter_benchmarks(config):
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
    long_df.to_csv(output_path / "background_agents_distribution.csv", index=False)
    return long_df


def run_background_agents_distribution_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Compares non-background/background agent distributions across train/val/test splits for the background-agents
    benchmarks.

    Renders side-by-side violin and histogram plots (one panel per benchmark) plus a per-benchmark, per-split summary
    for every configured quantity. The plots are driven entirely by the long-form
    ``background_agents_distribution.csv``: when it already exists (and ``overwrite`` is false) it is loaded directly,
    so re-rendering touches neither the scenario pkls nor the split JSONs. Otherwise the frame is rebuilt — reusing the
    cached ``per_scenario_counts.csv`` when present, and only falling back to loading scenarios when no cache exists or
    ``overwrite`` is set.

    Args:
        config: Analysis configuration (``splits_path``, ``variants_base_path``, ``causal_labels_path``,
            ``num_workers``, ``overwrite``, ``quantities``, ``benchmarks``).
        log: Logger.
        output_path: Directory to save the generated counts, summary and plots.
    """
    set_analysis_theme(log=log)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    quantities = list(config.quantities)

    long_cache = output_path / "background_agents_distribution.csv"
    if long_cache.exists() and not config.overwrite:
        log.info("Regenerating plots from cached %s (set overwrite=true to recompute from scenarios)", long_cache)
        long_df = pd.read_csv(long_cache)
    else:
        long_df = _build_distribution_frame(config, log, output_path)
        if long_df is None:
            return

    long_df["split"] = pd.Categorical(long_df["split"], categories=list(SPLIT_ORDER), ordered=True)
    benchmark_names = list(dict.fromkeys(long_df["benchmark"]))

    palette = [SPLIT_COLOR_MAP[split_key] for split_key in SPLIT_ORDER]
    plot_config = SplitDistributionPlotConfig(benchmark_names, palette, _QUANTITY_LABELS, output_path)
    render_distribution_plots(long_df, quantities, plot_config, log)

    print("\n✓ Analysis complete!")
    log.info("Background distribution analysis complete!")
