"""Distribution-shift robustness scoring (per-model, per-metric) and radar visualization.

Reduces raw SEEN/UNSEEN benchmark numbers into comparable *scores* per model per metric, measured against a reference:

* ``naive_relative`` -- each model vs the Naive baseline within the same benchmark.
* ``uniform_relative`` -- each model vs its own performance in the Uniform benchmark.

Following the MASE / OWA framing of the N-BEATS paper (arXiv:1905.10437), every metric is positive and lower-is-better,
and each model is characterized by two reference-relative *score* axes (the reciprocal MASE skill):

* ``id_score = ref_seen / model_seen`` -- ID score (reciprocal MASE on the seen split).
* ``ood_score = ref_unseen / model_unseen`` -- OOD score (reciprocal MASE on the unseen split).

Both are dimensionless, **higher == better**, and ``1.0`` == on par with the reference (the reference's own row is
exactly ``1.0``). Because each split is scaled by the reference *on that same split*, a good model with low absolute
OOD error stays high on ``ood_score`` regardless of its degradation *factor* -- this avoids the "robustness paradox"
where a uniformly-weak model that multiplies its error by a small factor looks more robust than a strong model. The
division is NaN-guarded (a non-finite/non-positive numerator or denominator -> ``NaN``, dropped from aggregation).

Each score is computed per ``(benchmark, model, metric)`` cell, and one of those two dimensions is then collapsed
(``aggregate_over``): collapsing ``benchmark`` leaves per-model, per-metric frames (the metrics become the score
columns / radar axes); collapsing ``metric`` leaves per-model, per-benchmark frames. Everything downstream operates on
"columns other than ``Combined``" and is agnostic to which dimension that is.

To rank models by a single value the two axes are reduced, within the same reference frame, to a ``combined`` score
that is the per-column geometric mean::

    combined_column = sqrt(id_score * ood_score)
    Combined        = mean(combined_column across columns)

and is NaN-safe (via :func:`_geometric_mean`). The geometric mean does both jobs at once: its absolute level rewards
quality, and because it punishes ID/OOD imbalance it penalizes shift degradation -- so a single frame captures both,
with no second reference. Higher is better and ``1.0`` means on par with the reference. Under ``naive_relative`` this
demotes the Naive baseline (pinned at ``1.0``, since it is its own reference) and keeps a model genuinely worse than
Naive below it. Under ``uniform_relative`` the combined is a *stability* view: because the reference is the model's own
Uniform row, the most *consistent* model (the input-agnostic Naive) ranks high there -- expected for a self-relative
stability score, not a downstream-performance ranking.

This docstring is the reference for the *methodology*; see `docs/ANALYSIS.md` for how to run the analysis and what it
writes.
"""

from collections.abc import Callable
from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig

from controlledshifts.utils.analysis.common import (
    COMBINED_COLUMN,
    MODEL_NAME_MAP,
    iter_benchmarks,
    load_results_csv,
)
from controlledshifts.utils.analysis.distribution_shift import build_benchmark_df
from controlledshifts.utils.analysis.latex import format_value
from controlledshifts.utils.analysis.robustness_plots import (
    RADAR_TERM_TITLES,
    plot_combined_ranking,
    plot_combined_summary,
    plot_robustness_decomposition,
    plot_score_radar,
    radial_ticks,
)
from controlledshifts.utils.plotting import set_analysis_theme


# Score-term keys returned by :func:`compute_robustness_scores`. The two per-metric score axes are rendered as radars;
# the combined robustness score is rendered as a sorted bar chart.
ID_TERM = "id"
OOD_TERM = "ood"
COMBINED_TERM = "combined"
# Reference-mode strings (drive the scoring/skip logic and the config ``reference_modes``); do NOT rename these.
NAIVE_RELATIVE = "naive_relative"
UNIFORM_RELATIVE = "uniform_relative"
# Aggregation axes: the dimension collapsed by ``aggregate`` (drive the config ``aggregate_over``; do NOT rename).
# The folder/label names describe what is left -- the score columns -- which is the *other* dimension.
BENCHMARK_AXIS = "benchmark"  # collapse benchmarks -> columns are metrics
METRIC_AXIS = "metric"  # collapse metrics -> columns are benchmarks
AXIS_FOLDERS = {BENCHMARK_AXIS: "per_metric", METRIC_AXIS: "per_benchmark"}
AXIS_LABELS = {BENCHMARK_AXIS: "Per-Metric", METRIC_AXIS: "Per-Benchmark"}
# Radar terms only (the two per-column score axes), mapped to their output-file stem and display title.
RADAR_TERM_STEMS = {ID_TERM: "seen_score", OOD_TERM: "unseen_score"}
COMBINED_FILE_STEM = "combined_robustness"
COMBINED_TITLE = "Combined Robustness Score"

# Output folder + semantic label per reference mode, plus the stacked-summary figure stem and its bar-panel title.
# Only OUTPUT naming lives here -- the reference-mode strings above are unchanged. Edit these to rename outputs.
SUMMARY_FOLDERS = {NAIVE_RELATIVE: "quality_naive", UNIFORM_RELATIVE: "stability_uniform"}
SUMMARY_LABELS = {NAIVE_RELATIVE: "Quality", UNIFORM_RELATIVE: "Stability"}
SUMMARY_FILE_STEM = "robustness_summary"


def _reference_label(reference_mode: str) -> str:
    """Human-readable form of a reference mode for plot subtitles, e.g. ``naive_relative`` -> ``Naive-Relative``."""
    return reference_mode.replace("_", "-").title()


def _score(model: float, ref: float) -> float:
    """MASE-style reference-relative score ``ref / model`` for positive error metrics.

    Higher is better; ``1.0`` means on par with the reference (``> 1`` beats it, ``< 1`` is worse). Returns NaN if
    either argument is missing or non-positive, so degenerate cells drop out of the NaN-safe aggregation.
    """
    if pd.isna(model) or pd.isna(ref) or model <= 0 or ref <= 0:
        return float("nan")
    return ref / model


def _select_reference(  # noqa: PLR0913
    reference_mode: str,
    *,
    frames: dict[str, pd.DataFrame],
    splits: dict[str, tuple[str, str]],
    benchmark_key: str,
    model: str,
    metric: str,
    uniform_key: str,
) -> tuple[float, float]:
    """Return ``(ref_seen, ref_unseen)`` for one ``(benchmark, model, metric)`` cell.

    ``naive_relative`` references the Naive row of the same benchmark; ``uniform_relative`` references the same model's
    row in the Uniform benchmark. Returns ``(nan, nan)`` when the reference is absent.
    """
    if reference_mode == UNIFORM_RELATIVE:
        ref_key, ref_model = uniform_key, model
    else:
        ref_key, ref_model = benchmark_key, MODEL_NAME_MAP["naive"]

    ref_frame = frames.get(ref_key)
    if ref_frame is None or ref_model not in ref_frame.index:
        return float("nan"), float("nan")

    seen_split, unseen_split = splits[ref_key]
    ref_seen = float(ref_frame.loc[ref_model, f"{seen_split}/{metric}"])
    ref_unseen = float(ref_frame.loc[ref_model, f"{unseen_split}/{metric}"])
    return ref_seen, ref_unseen


def _aggregate_term(
    accum: dict[str, dict[str, list[float]]],
    columns: list[str],
    ordered_models: list[str],
    agg_fn: Callable[[list[float]], np.floating],
) -> pd.DataFrame:
    """Aggregate the collapsed dimension's scores into a per-model, per-column frame (NaN-safe) with ``Combined``.

    Args:
        accum: ``accum[model][column]`` -> list of scores over the collapsed dimension.
        columns: Score-column names (metrics or benchmarks), in column order.
        ordered_models: Display model names, in row order.
        agg_fn: NaN-safe reducer applied across the collapsed dimension (``np.nanmean`` or ``np.nanmedian``).

    Returns:
        DataFrame indexed by ``Model`` with one column per entry of ``columns`` plus a ``Combined`` column (per-model
        mean across them).
    """
    rows: list[dict[str, float | str]] = []
    for model in ordered_models:
        if model not in accum:
            continue
        record: dict[str, float | str] = {"Model": model}
        column_scores: list[float] = []
        for column in columns:
            values = [value for value in accum[model].get(column, []) if not pd.isna(value)]
            score = float(agg_fn(values)) if values else float("nan")
            record[column] = score
            column_scores.append(score)
        valid = [value for value in column_scores if not pd.isna(value)]
        record[COMBINED_COLUMN] = float(np.mean(valid)) if valid else float("nan")
        rows.append(record)

    return pd.DataFrame(rows).set_index("Model")


def _scored_benchmarks(
    benchmarks: list[tuple[str, str, str, str]],
    frames: dict[str, pd.DataFrame],
    reference_mode: str,
    uniform_key: str,
    aggregate_over: str,
) -> list[tuple[str, str, str, str]]:
    """Benchmark tuples that actually contribute scores: those with data, minus Uniform where it does not belong.

    Uniform is dropped under ``uniform_relative`` (a model is referenced against its own Uniform row, so scoring it
    would be a degenerate uniform-vs-uniform comparison) and under :data:`METRIC_AXIS` (it is the unshifted control,
    not one of the shifts the per-benchmark view compares).
    """
    skip_uniform = reference_mode == UNIFORM_RELATIVE or aggregate_over == METRIC_AXIS
    return [
        benchmark
        for benchmark in benchmarks
        if benchmark[0] in frames and not (skip_uniform and benchmark[0] == uniform_key)
    ]


def compute_robustness_scores(  # noqa: PLR0913
    metrics_df: pd.DataFrame,
    benchmarks: list[tuple[str, str, str, str]],
    metrics: list[str],
    models_to_compare: list[str],
    *,
    reference_mode: str,
    aggregate: str = "mean",
    uniform_key: str = "uniform",
    aggregate_over: str = BENCHMARK_AXIS,
) -> dict[str, pd.DataFrame]:
    """Compute per-model ID/OOD scores and a combined ranking, aggregated over one of the two score dimensions.

    For each benchmark, model and metric two reference-relative MASE-style scores are computed (see the module
    docstring): ``id_score`` (seen) and ``ood_score`` (unseen). The reference is selected by ``reference_mode``. Both
    are then aggregated (NaN-safe) over the ``aggregate_over`` dimension, leaving frames whose columns are the *other*
    dimension -- metrics when collapsing benchmarks, benchmark display names when collapsing metrics -- plus a
    ``Combined`` column holding the per-model mean across those columns. The ``combined`` frame is the per-column
    geometric mean ``sqrt(id_score * ood_score)`` (:func:`_geometric_mean_combined`). The Uniform benchmark is excluded
    under ``uniform_relative`` and under :data:`METRIC_AXIS` -- see :func:`_scored_benchmarks`.

    Args:
        metrics_df: Combined results frame with a ``Name`` (``<dataset>_<model>``) column.
        benchmarks: ``(key, name, seen_split, unseen_split)`` tuples in display order.
        metrics: Metric names, in column order.
        models_to_compare: Raw model identifiers to include.
        reference_mode: ``"naive_relative"`` or ``"uniform_relative"``.
        aggregate: ``"mean"`` or ``"median"`` across the collapsed dimension.
        uniform_key: Benchmark key used as the ``uniform_relative`` reference.
        aggregate_over: :data:`BENCHMARK_AXIS` (columns are metrics) or :data:`METRIC_AXIS` (columns are benchmarks).

    Returns:
        Dict keyed by :data:`ID_TERM`, :data:`OOD_TERM` and :data:`COMBINED_TERM`; each value is a DataFrame indexed by
        ``Model`` with one score column per surviving entry of the non-collapsed dimension plus a ``Combined`` column.
        All are higher-is-better scores (``1.0`` == on par with the reference).
    """
    frames: dict[str, pd.DataFrame] = {}
    splits: dict[str, tuple[str, str]] = {}
    for key, _name, seen, unseen in benchmarks:
        benchmark_df = build_benchmark_df(metrics_df, (seen, unseen), metrics, models_to_compare, show_run_id=False)
        if benchmark_df.empty:
            continue
        frames[key] = benchmark_df.set_index("Model")
        splits[key] = (seen, unseen)

    scored = _scored_benchmarks(benchmarks, frames, reference_mode, uniform_key, aggregate_over)

    # accum[term][model][column] -> list of scores over the collapsed dimension
    accum: dict[str, dict[str, dict[str, list[float]]]] = {ID_TERM: {}, OOD_TERM: {}}
    for key, name, seen, unseen in scored:
        frame = frames[key]
        for model in frame.index:
            for metric in metrics:
                column = metric if aggregate_over == BENCHMARK_AXIS else name
                model_seen = float(frame.loc[model, f"{seen}/{metric}"])
                model_unseen = float(frame.loc[model, f"{unseen}/{metric}"])
                ref_seen, ref_unseen = _select_reference(
                    reference_mode,
                    frames=frames,
                    splits=splits,
                    benchmark_key=key,
                    model=model,
                    metric=metric,
                    uniform_key=uniform_key,
                )
                score_id = _score(model_seen, ref_seen)
                score_ood = _score(model_unseen, ref_unseen)
                for term, value in ((ID_TERM, score_id), (OOD_TERM, score_ood)):
                    accum[term].setdefault(model, {}).setdefault(column, []).append(value)

    columns = metrics if aggregate_over == BENCHMARK_AXIS else [name for _key, name, _seen, _unseen in scored]
    agg_fn = np.nanmedian if aggregate == "median" else np.nanmean
    ordered_models = [MODEL_NAME_MAP.get(model, model) for model in models_to_compare]
    id_df = _aggregate_term(accum[ID_TERM], columns, ordered_models, agg_fn)
    ood_df = _aggregate_term(accum[OOD_TERM], columns, ordered_models, agg_fn)
    combined_df = _geometric_mean_combined(id_df, ood_df, columns)
    return {ID_TERM: id_df, OOD_TERM: ood_df, COMBINED_TERM: combined_df}


def _geometric_mean(left: pd.Series, right: pd.Series) -> pd.Series:
    """Element-wise ``sqrt(left * right)``, NaN where either term is missing or non-positive.

    Mirrors the :func:`_score` guard: the geometric mean is only defined for positive reference-relative scores, so a
    degenerate cell drops out of the NaN-safe aggregation.
    """
    valid = (left > 0) & (right > 0)
    return np.sqrt((left * right).where(valid))


def _geometric_mean_combined(id_df: pd.DataFrame, ood_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Combined ranking score: per-column geometric mean of the ID and OOD scores, then mean across columns.

    Per column ``combined_column = sqrt(id_score * ood_score)``; ``Combined`` is the mean across columns (NaN-safe). The
    absolute level rewards quality and, because the geometric mean punishes ID/OOD imbalance, a model that degrades
    under shift is penalized -- so a single reference frame captures both. ``1.0`` means on par with the reference.

    Args:
        id_df: Aggregated ID score frame (indexed by ``Model``, score columns plus ``Combined``).
        ood_df: Aggregated OOD score frame, same shape/index as ``id_df``.
        columns: Score-column names (metrics or benchmarks) to combine.

    Returns:
        DataFrame indexed by ``Model`` with one combined column per entry of ``columns`` plus a ``Combined`` column.
    """
    combined = pd.DataFrame(index=id_df.index)
    for column in columns:
        combined[column] = _geometric_mean(id_df[column], ood_df[column])
    combined[COMBINED_COLUMN] = combined[columns].mean(axis=1, skipna=True)
    return combined


def _write_scores_csv(scores_df: pd.DataFrame, output_path: Path, filename: str, *, label: str) -> Path:
    """Write the per-model scores to CSV and print a formatted summary."""
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{filename}.csv"
    scores_df.to_csv(output_file)
    print(f"{label} (higher = better; 1.0 = on par with the reference):")
    print(scores_df.to_string(float_format="{:.3f}".format))
    print(f"✓ Scores saved as '{output_file}'")
    return output_file


def _write_scores_tex(scores_df: pd.DataFrame, output_path: Path, filename: str, *, caption: str) -> Path:
    r"""Write a LaTeX ``tabular`` of the per-model scores, bolding the best (highest) per column."""
    columns = list(scores_df.columns)
    best = {column: scores_df[column].max() for column in columns}

    body_rows: list[str] = []
    for model, row in scores_df.iterrows():
        cells = [str(model)]
        cells.extend(format_value(row[column], best[column]) for column in columns)
        body_rows.append(" & ".join(cells) + " \\\\")

    col_spec = "l " + "c" * len(columns)
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        " & ".join(["\\textbf{Model}", *(f"\\textbf{{{column}}}" for column in columns)]) + " \\\\",
        "\\midrule",
        *body_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    latex_str = "\n".join(latex_lines)

    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{filename}.tex"
    output_file.write_text(latex_str)
    print(f"✓ LaTeX table saved as '{output_file}'")
    return output_file


def _shared_radial(scores: dict[str, pd.DataFrame]) -> tuple[float, float, float, NDArray]:
    """One radial scale spanning both score axes, so the Seen/Unseen radars share rings and bounds.

    The ``Combined`` column is excluded: it is a summary across the score columns, not one of the radar's axes.
    """
    shared_values = [
        value
        for term in RADAR_TERM_STEMS
        for value in scores[term].drop(columns=COMBINED_COLUMN, errors="ignore").to_numpy().ravel()
        if not pd.isna(value)
    ]
    return radial_ticks(shared_values)


def _write_mode_artifacts(  # noqa: PLR0913
    scores: dict[str, pd.DataFrame],
    mode_output: Path,
    colormap: str,
    *,
    reference_mode: str,
    reference_label: str,
    axis_label: str,
    shared_radial: tuple[float, float, float, NDArray],
    abbrev: dict[str, str] | None,
) -> None:
    """Write one reference mode's artifacts: a radar/CSV/LaTeX trio per score axis, the decomposition scatter, and
    the combined ranking with its own CSV and LaTeX table.

    Args:
        scores: The frames from :func:`compute_robustness_scores`, keyed by score term.
        mode_output: Directory for this mode's artifacts.
        colormap: Seaborn/matplotlib palette name.
        reference_mode: Raw mode string, used in the LaTeX captions.
        reference_label: Human-readable mode, used as the plot subtitle.
        axis_label: Human-readable aggregation axis (from :data:`AXIS_LABELS`), appended to subtitles and captions.
        shared_radial: Radial scale shared by both radars.
        abbrev: Score column -> short radar rim label; ``None`` when the columns are metrics (the radar's own map).
    """
    subtitle = f"{reference_label} · {axis_label}"
    for term, stem in RADAR_TERM_STEMS.items():
        term_df = scores[term]
        term_title = f"{RADAR_TERM_TITLES[term]} ({reference_mode}, {axis_label})"
        plot_score_radar(
            term_df,
            mode_output,
            colormap,
            RADAR_TERM_TITLES[term],
            subtitle,
            f"{stem}_radar",
            radial=shared_radial,
            abbrev=abbrev,
        )
        _write_scores_csv(term_df, mode_output, f"{stem}_scores", label=RADAR_TERM_TITLES[term])
        _write_scores_tex(term_df, mode_output, f"{stem}_scores", caption=term_title)

    plot_robustness_decomposition(
        scores[ID_TERM],
        scores[OOD_TERM],
        mode_output,
        colormap,
        "ID/OOD Score Decomposition",
        subtitle,
        "score_decomposition",
    )

    # Combined ranking: sorted (best first) bar chart + CSV + LaTeX.
    combined_title = f"{COMBINED_TITLE} ({reference_mode}, {axis_label})"
    combined_sorted = scores[COMBINED_TERM].sort_values(COMBINED_COLUMN, ascending=False)
    plot_combined_ranking(
        scores[COMBINED_TERM],
        mode_output,
        colormap,
        COMBINED_TITLE,
        subtitle,
        f"{COMBINED_FILE_STEM}_ranking",
    )
    _write_scores_csv(combined_sorted, mode_output, f"{COMBINED_FILE_STEM}_scores", label=COMBINED_TITLE)
    _write_scores_tex(combined_sorted, mode_output, f"{COMBINED_FILE_STEM}_scores", caption=combined_title)


def run_robustness_scores_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Run score analysis for each configured aggregation axis and reference mode.

    For each ``config.score.aggregate_over`` axis (folder from :data:`AXIS_FOLDERS`, e.g. ``per_metric``) and each
    ``config.score.reference_modes`` entry, computes per-model ID/OOD scores from the combined results file and, under
    ``output_path/<axis>/<folder>/`` (folder from :data:`SUMMARY_FOLDERS`, e.g. ``quality_naive``), writes a radar plot,
    CSV table and LaTeX table for each of the two score terms (stems from :data:`RADAR_TERM_STEMS`, e.g.
    ``seen_score``/``unseen_score``); an ID-vs-OOD decomposition scatter; and the combined robustness ranking
    (per-column ``sqrt(id * ood)``) as a sorted bar chart with its CSV and LaTeX table. Each axis folder also gets a
    ``robustness_summary.png`` combining every reference mode as a column (Seen radar / Unseen radar / Combined
    ranking) with a single shared model legend.

    Args:
        config: Analysis configuration (``benchmarks_filepath``, ``benchmarks``, ``models_to_compare``,
            ``trajectory_forecasting_metrics``, ``score_colormap`` and the ``score`` block).
        log: Logger.
        output_path: Directory to save the generated artifacts.
    """
    set_analysis_theme(log=log)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_df = load_results_csv(Path(config.benchmarks_filepath), log)
    if metrics_df is None:
        return

    metrics = list(config.trajectory_forecasting_metrics)
    models_to_compare = list(config.models_to_compare)
    colormap = config.score_colormap

    benchmarks: list[tuple[str, str, str, str]] = [
        (key, spec.name, spec.seen, spec.unseen) for key, spec in iter_benchmarks(config)
    ]
    # Short rim labels for the benchmark-column radars, keyed by the display name the score columns carry.
    benchmark_abbrevs = {str(spec.name): str(spec.abbrev) for _key, spec in iter_benchmarks(config) if "abbrev" in spec}

    score_cfg = config.score

    for axis in score_cfg.aggregate_over:
        aggregate_over = str(axis)
        axis_output = output_path / AXIS_FOLDERS.get(aggregate_over, aggregate_over)
        axis_label = AXIS_LABELS.get(aggregate_over, aggregate_over)
        abbrev = benchmark_abbrevs if aggregate_over == METRIC_AXIS else None

        # Collected per mode (in config order) to render this axis' combined Quality-vs-Stability summary.
        mode_results: list[
            tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[float, float, float, NDArray]]
        ] = []
        for reference_mode in score_cfg.reference_modes:
            log.info("Computing scores for reference mode '%s' aggregated over %s", reference_mode, aggregate_over)
            scores = compute_robustness_scores(
                metrics_df,
                benchmarks,
                metrics,
                models_to_compare,
                reference_mode=reference_mode,
                aggregate=str(score_cfg.aggregate),
                uniform_key=str(score_cfg.uniform_key),
                aggregate_over=aggregate_over,
            )
            if scores[COMBINED_TERM].empty:
                log.warning("No models scored for reference mode '%s'; skipping.", reference_mode)
                continue

            mode_output = axis_output / str(SUMMARY_FOLDERS.get(reference_mode, reference_mode))
            reference_label = _reference_label(reference_mode)  # plot subtitle, e.g. "Naive-Relative"
            print(f"\n=== Scores: {reference_mode} / {aggregate_over} ({len(scores[ID_TERM])} models) ===")

            shared_radial = _shared_radial(scores)
            _write_mode_artifacts(
                scores,
                mode_output,
                colormap,
                reference_mode=reference_mode,
                reference_label=reference_label,
                axis_label=axis_label,
                shared_radial=shared_radial,
                abbrev=abbrev,
            )

            # The summary renders finished header text, so resolve the mode's display label here rather than there.
            mode_label = SUMMARY_LABELS.get(reference_mode, reference_label)
            column_title = f"{mode_label} ({reference_label})"
            mode_results.append((column_title, scores[ID_TERM], scores[OOD_TERM], scores[COMBINED_TERM], shared_radial))

        # Single combined summary spanning all reference modes (columns), written at this axis' top level.
        plot_combined_summary(mode_results, axis_output, colormap, SUMMARY_FILE_STEM, abbrev=abbrev)

    print("\n✓ Robustness score analysis complete!")
    log.info("Robustness score analysis complete!")
