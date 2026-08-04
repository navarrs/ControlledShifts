"""Ego-SafeShift score distribution analysis across benchmark splits.

Compares the per-scenario criticality-score distributions of the train/validation/testing splits. The
``ego_safeshift`` benchmark ranks scenarios by a safety score (``score_type``) and sends the hardest (highest-scoring)
scenarios to the test set, so its test distribution should be visibly shifted toward higher scores; a ``random``
baseline benchmark splits the same scenarios uniformly, so its distributions should match across splits.

The analysis is driven entirely by the scores CSV: each benchmark's split is reproduced in-script from the scenario
scores via ``split_ids_by_score`` / ``split_ids_by_ratio`` (using the configured ``score_type``, ``split_ratios`` and
``seed``), so it needs neither the scenario pkls nor a pre-existing split JSON. Unlike benchmark creation, scenarios are
not filtered by on-disk availability — every scored scenario in the CSV is included.

See `docs/ANALYSIS.md` for usage details.
"""

from logging import Logger
from pathlib import Path

import pandas as pd
from numpy.random import default_rng
from omegaconf import DictConfig

from controlledshifts.benchmarks.common import split_ids_by_ratio, split_ids_by_score, split_mapping_to_lists
from controlledshifts.utils.analysis.common import (
    SPLIT_COLOR_MAP,
    SPLIT_ORDER,
    SplitDistributionPlotConfig,
    render_distribution_plots,
)
from controlledshifts.utils.plotting import set_analysis_theme


# Axis labels for the known ego-safeshift score columns; other configured quantities fall back to a title-cased name.
# Figure titles append " Distribution" via the plot config's title_suffix.
_QUANTITY_LABELS: dict[str, str] = {
    "gt_critical_continuous_individual": "Individual Scores",
    "gt_critical_continuous_interaction": "Interaction Scores",
    "gt_critical_continuous_safeshift": "EgoSafeShift Scores",
}

# Split strategies supported per benchmark entry: score-ranked (hardest to test) or uniformly random.
_SCORE_SPLIT = "score"
_RANDOM_SPLIT = "random"


def _quantity_labels(quantities: list[str]) -> dict[str, str]:
    """Maps each configured quantity to a display label, falling back to a title-cased column name."""
    return {quantity: _QUANTITY_LABELS.get(quantity, quantity.replace("_", " ").title()) for quantity in quantities}


def _build_long_frame(
    scores_df: pd.DataFrame, benchmark_name: str, split_strategy: str, config: DictConfig
) -> pd.DataFrame:
    """Reproduces a benchmark's split from the scores and joins it with the scores into a long-form frame.

    Args:
        scores_df: Per-scenario scores indexed by ``scenario_id`` (the configured quantity columns plus ``score_type``).
        benchmark_name: Display name recorded in the ``benchmark`` column.
        split_strategy: ``"score"`` to rank by ``score_type`` (hardest to test) or ``"random"`` for a uniform split.
        config: Analysis configuration (``score_type``, ``split_ratios``, ``seed``).

    Returns:
        Long-form DataFrame with columns ``benchmark``, ``split``, ``scenario_id`` and the quantity columns.

    Raises:
        ValueError: If ``split_strategy`` is not ``"score"`` or ``"random"``.
    """
    scenario_ids = list(scores_df.index)
    split_ratios = tuple(config.split_ratios)
    random_generator = default_rng(config.seed)
    if split_strategy == _SCORE_SPLIT:
        mapping = split_ids_by_score(
            scenario_ids, scores_df[config.score_type].to_numpy(), split_ratios, random_generator, hardest_highest=True
        )
    elif split_strategy == _RANDOM_SPLIT:
        mapping = split_ids_by_ratio(scenario_ids, split_ratios, random_generator)
    else:
        error_message = f"Unknown split strategy '{split_strategy}' for benchmark '{benchmark_name}'"
        raise ValueError(error_message)

    split = split_mapping_to_lists(mapping)
    frames = []
    for split_key in SPLIT_ORDER:
        subset = scores_df.loc[getattr(split, split_key)].reset_index()
        subset.insert(0, "split", split_key)
        subset.insert(0, "benchmark", benchmark_name)
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def _build_distribution_frame(config: DictConfig, log: Logger, output_path: Path) -> pd.DataFrame:
    """Builds and caches the long-form score distribution frame from the scores CSV.

    Reads the scores CSV, strips the ``.pkl`` suffix from scenario IDs (matching benchmark creation), reproduces each
    configured benchmark's split, and writes the combined long-form frame to ``ego_safeshift_distribution.csv``.

    Args:
        config: Analysis configuration (``scores_csv_path``, ``score_type``, ``split_ratios``, ``seed``, ``quantities``,
            ``benchmarks``).
        log: Logger.
        output_path: Directory receiving the cached CSV.

    Returns:
        The long-form DataFrame.
    """
    scores_csv_path = Path(config.scores_csv_path)
    scenario_scores_df = pd.read_csv(scores_csv_path)
    scenario_scores_df["scenario_id"] = [Path(scenario_id).stem for scenario_id in scenario_scores_df["scenario_ids"]]
    columns = list(dict.fromkeys([*list(config.quantities), config.score_type]))
    scores_df = scenario_scores_df.set_index("scenario_id")[columns]
    log.info("Loaded %d scored scenarios from %s", len(scores_df), scores_csv_path)

    frames = []
    for benchmark_entry in config.benchmarks:
        key, spec = next(iter(benchmark_entry.items()))
        log.info("Building '%s' (%s) split with strategy '%s'", key, spec.name, spec.split)
        frames.append(_build_long_frame(scores_df, spec.name, spec.split, config))

    long_df = pd.concat(frames, ignore_index=True)
    long_df.to_csv(output_path / "ego_safeshift_distribution.csv", index=False)
    return long_df


def run_ego_safeshift_distribution_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Compares per-scenario criticality-score distributions across train/val/test splits for the ego-safeshift CSV.

    Renders side-by-side violin and histogram plots plus a ridgeline (one panel per benchmark) and a per-benchmark,
    per-split summary for every configured score quantity. The plots are driven entirely by the long-form
    ``ego_safeshift_distribution.csv``: when it already exists (and ``overwrite`` is false) it is loaded directly;
    otherwise the splits are reproduced from the scores CSV and the frame is rebuilt and cached.

    Args:
        config: Analysis configuration (``scores_csv_path``, ``score_type``, ``split_ratios``, ``seed``, ``overwrite``,
            ``quantities``, ``benchmarks``).
        log: Logger.
        output_path: Directory to save the generated frame, summary and plots.
    """
    set_analysis_theme(log=log)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    quantities = list(config.quantities)

    long_cache = output_path / "ego_safeshift_distribution.csv"
    if long_cache.exists() and not config.overwrite:
        log.info("Regenerating plots from cached %s (set overwrite=true to recompute from the scores CSV)", long_cache)
        long_df = pd.read_csv(long_cache)
    else:
        long_df = _build_distribution_frame(config, log, output_path)

    long_df["split"] = pd.Categorical(long_df["split"], categories=list(SPLIT_ORDER), ordered=True)
    benchmark_names = list(dict.fromkeys(long_df["benchmark"]))

    palette = [SPLIT_COLOR_MAP[split_key] for split_key in SPLIT_ORDER]
    plot_config = SplitDistributionPlotConfig(
        benchmark_names, palette, _quantity_labels(quantities), output_path, title_suffix=" Distribution"
    )
    render_distribution_plots(long_df, quantities, plot_config, log)

    print("\n✓ Analysis complete!")
    log.info("Score distribution analysis complete!")
