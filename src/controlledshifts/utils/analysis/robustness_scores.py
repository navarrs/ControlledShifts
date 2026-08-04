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

To rank models by a single value the two axes are reduced, within the same reference frame, to a ``combined`` score
that is the per-metric geometric mean::

    combined_metric = sqrt(id_score * ood_score)
    Combined        = mean(combined_metric across metrics)

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

import math
from collections.abc import Callable
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Transform
from numpy.typing import NDArray
from omegaconf import DictConfig

from controlledshifts.utils.analysis.common import (
    METRIC_ABBREV_MAP,
    METRIC_NAME_MAP,
    MODEL_NAME_MAP,
    model_colors,
)
from controlledshifts.utils.analysis.distribution_shift import build_benchmark_df
from controlledshifts.utils.constants import EPSILON
from controlledshifts.utils.plotting import set_analysis_theme


COMBINED_COLUMN = "Combined"

# A per-model plot color: a hex string (fixed models) or an RGB tuple (palette fallback), as returned by model_colors.
Color = str | tuple[float, float, float]

# Score-term keys returned by :func:`compute_robustness_scores`. The two per-metric score axes are rendered as radars;
# the combined robustness score is rendered as a sorted bar chart.
ID_TERM = "id"
OOD_TERM = "ood"
COMBINED_TERM = "combined"
# Reference-mode strings (drive the scoring/skip logic and the config ``reference_modes``); do NOT rename these.
NAIVE_RELATIVE = "naive_relative"
UNIFORM_RELATIVE = "uniform_relative"
# Radar terms only (the per-metric score axes), mapped to their output-file stem and display title.
RADAR_TERM_STEMS = {ID_TERM: "seen_score", OOD_TERM: "unseen_score"}
RADAR_TERM_TITLES = {ID_TERM: "Seen Score", OOD_TERM: "Unseen Score"}
COMBINED_FILE_STEM = "combined_robustness"
COMBINED_TITLE = "Combined Robustness Score"

# Output folder + semantic label per reference mode, plus the stacked-summary figure stem and its bar-panel title.
# Only OUTPUT naming lives here -- the reference-mode strings above are unchanged. Edit these to rename outputs.
SUMMARY_FOLDERS = {NAIVE_RELATIVE: "quality_naive", UNIFORM_RELATIVE: "stability_uniform"}
SUMMARY_LABELS = {NAIVE_RELATIVE: "Quality", UNIFORM_RELATIVE: "Stability"}
SUMMARY_FILE_STEM = "robustness_summary"
COMBINED_PANEL_TITLE = "Combined Score"

# Cosine/sine deadband for deciding radar label alignment (center vs left/right, top/bottom).
_LABEL_ALIGN_THRESHOLD = 0.1


def _reference_label(reference_mode: str) -> str:
    """Human-readable form of a reference mode for plot subtitles, e.g. ``naive_relative`` -> ``Naive-Relative``."""
    return reference_mode.replace("_", "-").title()


def _set_titles(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Set a large bold main title with a smaller subtitle just beneath it, centered over the figure.

    Args:
        fig: Figure to title.
        title: Main title text.
        subtitle: Subtitle text.
    """
    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.text(0.5, 0.93, subtitle, ha="center", va="top", fontsize=13, color="dimgray")


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
    metrics: list[str],
    ordered_models: list[str],
    agg_fn: Callable[[list[float]], np.floating],
) -> pd.DataFrame:
    """Aggregate per-benchmark scores into a per-model, per-metric frame (NaN-safe) with a ``Combined`` column.

    Args:
        accum: ``accum[model][metric]`` -> list of per-benchmark scores.
        metrics: Metric names, in column order.
        ordered_models: Display model names, in row order.
        agg_fn: NaN-safe reducer applied across benchmarks (``np.nanmean`` or ``np.nanmedian``).

    Returns:
        DataFrame indexed by ``Model`` with one column per metric plus a ``Combined`` column (per-model mean).
    """
    rows: list[dict[str, float | str]] = []
    for model in ordered_models:
        if model not in accum:
            continue
        record: dict[str, float | str] = {"Model": model}
        metric_scores: list[float] = []
        for metric in metrics:
            values = [value for value in accum[model].get(metric, []) if not pd.isna(value)]
            score = float(agg_fn(values)) if values else float("nan")
            record[metric] = score
            metric_scores.append(score)
        valid = [value for value in metric_scores if not pd.isna(value)]
        record[COMBINED_COLUMN] = float(np.mean(valid)) if valid else float("nan")
        rows.append(record)

    return pd.DataFrame(rows).set_index("Model")


def compute_robustness_scores(  # noqa: PLR0913
    metrics_df: pd.DataFrame,
    benchmarks: list[tuple[str, str, str, str]],
    metrics: list[str],
    models_to_compare: list[str],
    *,
    reference_mode: str,
    aggregate: str = "mean",
    uniform_key: str = "uniform",
) -> dict[str, pd.DataFrame]:
    """Compute per-model, per-metric ID/OOD scores and a combined ranking, aggregated across benchmarks.

    For each benchmark, model and metric two reference-relative MASE-style scores are computed (see the module
    docstring): ``id_score`` (seen) and ``ood_score`` (unseen). The reference is selected by ``reference_mode``. Both
    are aggregated across benchmarks (NaN-safe) into per-metric frames with a ``Combined`` column holding the per-model
    mean across metrics. The ``combined`` frame is the per-metric geometric mean ``sqrt(id_score * ood_score)``
    (:func:`_geometric_mean_combined`). For ``uniform_relative`` the Uniform benchmark is excluded from aggregation
    (its self-reference is degenerate).

    Args:
        metrics_df: Combined results frame with a ``Name`` (``<dataset>_<model>``) column.
        benchmarks: ``(key, name, seen_split, unseen_split)`` tuples in display order.
        metrics: Metric names, in column order.
        models_to_compare: Raw model identifiers to include.
        reference_mode: ``"naive_relative"`` or ``"uniform_relative"``.
        aggregate: ``"mean"`` or ``"median"`` across benchmarks.
        uniform_key: Benchmark key used as the ``uniform_relative`` reference.

    Returns:
        Dict keyed by :data:`ID_TERM`, :data:`OOD_TERM` and :data:`COMBINED_TERM`; each value is a DataFrame indexed by
        ``Model`` with one column per metric plus a ``Combined`` column. All are higher-is-better scores (``1.0`` == on
        par with the reference).
    """
    frames: dict[str, pd.DataFrame] = {}
    splits: dict[str, tuple[str, str]] = {}
    for key, _name, seen, unseen in benchmarks:
        benchmark_df = build_benchmark_df(metrics_df, (seen, unseen), metrics, models_to_compare, show_run_id=False)
        if benchmark_df.empty:
            continue
        frames[key] = benchmark_df.set_index("Model")
        splits[key] = (seen, unseen)

    # accum[term][model][metric] -> list of per-benchmark scores
    accum: dict[str, dict[str, dict[str, list[float]]]] = {ID_TERM: {}, OOD_TERM: {}}
    for key, _name, seen, unseen in benchmarks:
        if key not in frames:
            continue
        # Skip the Uniform benchmark entirely under uniform_relative: a model is referenced against its
        # own Uniform row, so scoring Uniform here would be a degenerate uniform-vs-uniform comparison.
        if reference_mode == UNIFORM_RELATIVE and key == uniform_key:
            continue
        frame = frames[key]
        for model in frame.index:
            for metric in metrics:
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
                    accum[term].setdefault(model, {}).setdefault(metric, []).append(value)

    agg_fn = np.nanmedian if aggregate == "median" else np.nanmean
    ordered_models = [MODEL_NAME_MAP.get(model, model) for model in models_to_compare]
    id_df = _aggregate_term(accum[ID_TERM], metrics, ordered_models, agg_fn)
    ood_df = _aggregate_term(accum[OOD_TERM], metrics, ordered_models, agg_fn)
    combined_df = _geometric_mean_combined(id_df, ood_df, metrics)
    return {ID_TERM: id_df, OOD_TERM: ood_df, COMBINED_TERM: combined_df}


def _geometric_mean(left: pd.Series, right: pd.Series) -> pd.Series:
    """Element-wise ``sqrt(left * right)``, NaN where either term is missing or non-positive.

    Mirrors the :func:`_score` guard: the geometric mean is only defined for positive reference-relative scores, so a
    degenerate cell drops out of the NaN-safe aggregation.
    """
    valid = (left > 0) & (right > 0)
    return np.sqrt((left * right).where(valid))


def _geometric_mean_combined(id_df: pd.DataFrame, ood_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Combined ranking score: per-metric geometric mean of the ID and OOD scores, then mean across metrics.

    Per metric ``combined_metric = sqrt(id_score * ood_score)``; ``Combined`` is the mean across metrics (NaN-safe). The
    absolute level rewards quality and, because the geometric mean punishes ID/OOD imbalance, a model that degrades
    under shift is penalized -- so a single reference frame captures both. ``1.0`` means on par with the reference.

    Args:
        id_df: Aggregated ID score frame (indexed by ``Model``, metric columns plus ``Combined``).
        ood_df: Aggregated OOD score frame, same shape/index as ``id_df``.
        metrics: Metric names (the per-metric columns to combine).

    Returns:
        DataFrame indexed by ``Model`` with one combined column per metric plus a ``Combined`` column.
    """
    combined = pd.DataFrame(index=id_df.index)
    for metric in metrics:
        combined[metric] = _geometric_mean(id_df[metric], ood_df[metric])
    combined[COMBINED_COLUMN] = combined[metrics].mean(axis=1, skipna=True)
    return combined


def _metric_label(metric: str) -> str:
    """Clean radar axis label for a metric (falls back to the raw name when unmapped)."""
    return METRIC_NAME_MAP.get(metric, metric)


def _metric_abbrev(metric: str) -> str:
    """Short radar rim label for a metric (falls back to the clean name when unabbreviated)."""
    return METRIC_ABBREV_MAP.get(metric, _metric_label(metric))


def _bold(text: str) -> str:
    r"""Mathtext-bold form of ``text``.

    The figure font (DM Sans) ships a single regular face, so ``fontweight="bold"`` silently falls back to it;
    ``$\bf{...}$`` renders the token in a real bold face instead.
    """
    return rf"$\bf{{{text}}}$"


def _abbrev_caption(metrics: list[str]) -> str:
    """One-line abbreviation key, e.g. ``BF = BrierFDE  ·  MF = MinFDE`` (only the abbreviations bold)."""
    return "   ·   ".join(
        f"{_bold(_metric_abbrev(metric))} = {_metric_label(metric)}"
        for metric in metrics
        if metric in METRIC_ABBREV_MAP
    )


class _AbbrevKeyHandler(HandlerBase):
    """Render a legend handle as its own (bold) label text instead of a line sample.

    Used for the abbreviation-key row: drawing ``BF`` in the handle slot lines the key up with the model handles above
    it, and leaves the ``= BrierFDE`` half aligned with the model names.
    """

    def create_artists(  # noqa: PLR0913
        self,
        legend: Legend,  # noqa: ARG002 -- signature fixed by HandlerBase
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,  # noqa: ARG002 -- signature fixed by HandlerBase
        height: float,  # noqa: ARG002 -- signature fixed by HandlerBase
        fontsize: float,
        trans: Transform,
    ) -> list[Text]:
        text = Text(-xdescent, -ydescent, _bold(orig_handle.get_label()), fontsize=fontsize, ha="left", va="baseline")
        text.set_transform(trans)
        return [text]


def _abbrev_key_entries(metrics: list[str]) -> tuple[list[Patch], list[str]]:
    """Legend ``(handles, labels)`` for the abbreviation key: bold abbreviation in the handle slot, ``= Name`` after.

    The handles are bare :class:`~matplotlib.patches.Patch` carriers for the abbreviation text -- a type the model
    entries do not use, so :class:`_AbbrevKeyHandler` can be mapped to it without touching them.
    """
    keyed = [metric for metric in metrics if metric in METRIC_ABBREV_MAP]
    return [Patch(label=_metric_abbrev(metric)) for metric in keyed], [f"= {_metric_label(metric)}" for metric in keyed]


def _interleave_legend_key(
    handles: list, labels: list[str], key_handles: list, key_labels: list[str]
) -> tuple[list, list[str], int]:
    """Merge model entries and the abbreviation key into one legend, returning ``(handles, labels, ncol)``.

    Matplotlib fills legend columns top to bottom, so pairing each model with a key entry renders the models as the
    first row and the key as the second. Both lists are padded with blank entries so the rows stay aligned when the
    model and metric counts differ.
    """
    ncol = max(len(labels), len(key_labels))
    blank = Line2D([], [], linestyle="none", marker="none")
    padded_handles = [*handles, *[blank] * (ncol - len(handles))]
    padded_labels = [*labels, *[""] * (ncol - len(labels))]
    padded_key_handles = [*key_handles, *[blank] * (ncol - len(key_handles))]
    padded_key_labels = [*key_labels, *[""] * (ncol - len(key_labels))]

    merged_handles: list = []
    merged_labels: list[str] = []
    for handle, label, key_handle, key_label in zip(
        padded_handles, padded_labels, padded_key_handles, padded_key_labels, strict=True
    ):
        merged_handles.extend((handle, key_handle))
        merged_labels.extend((label, key_label))
    return merged_handles, merged_labels, ncol


# Preferred "nice" mantissas (1-2-5-10) for evenly spaced radial gridlines.
_NICE_MANTISSAS = (1.0, 2.0, 5.0, 10.0)


def _nice_step(raw_step: float) -> float:
    """Round a raw axis step up to the nearest 1/2/5 x 10^k so radial gridlines are evenly spaced."""
    if raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    base = raw_step / 10**exponent
    nice = next((mantissa for mantissa in _NICE_MANTISSAS if base <= mantissa), _NICE_MANTISSAS[-1])
    return nice * 10**exponent


def _radial_ticks(values: list[float], *, n_target: int = 5) -> tuple[float, float, float, NDArray]:
    """Return ``(r_lower, r_upper, step, ticks)`` snapped to a nice step around the data and the ``1.0`` reference.

    The bounds are multiples of ``step`` so every gridline circle is equally separated and the outermost ring sits on
    the rim. A full step of headroom is added when the data lands on a boundary so polygons never touch the rim. Scores
    are higher-is-better with ``1.0`` == the reference, so the range always spans the reference ring.
    """
    data_lower = min(1.0, np.nanmin(values)) if values else 0.0
    data_upper = max(1.0, np.nanmax(values)) if values else 1.0
    step = _nice_step(max(data_upper - data_lower, EPSILON) / n_target)
    r_lower = math.floor(data_lower / step) * step
    r_upper = math.ceil(data_upper / step) * step
    if r_upper - data_upper < EPSILON:
        r_upper += step
    if r_lower < 0 and data_lower - r_lower < EPSILON:
        r_lower -= step
    ticks = np.arange(r_lower, r_upper + step / 2, step)
    ticks[np.abs(ticks) < EPSILON] = 0.0  # avoid a "-0.0" tick label
    return r_lower, r_upper, step, ticks


def _draw_score_radar(  # noqa: PLR0913
    ax: plt.Axes,
    scores_df: pd.DataFrame,
    palette: list[Color],
    radial: tuple[float, float, float, NDArray] | None,
    *,
    show_mean: bool,
    show_reference_text: bool = False,
    ref_fontsize: float = 9,
    tick_fontsize: float = 11,
    metric_fontsize: float = 17,
    rlabel_at_bottom: bool = False,
) -> tuple[list, list[str]]:
    """Draw per-model score polygons onto a polar ``ax`` and return its legend ``(handles, labels)``.

    Shared by the standalone radar and the stacked summary. One closed polygon (light fill) per model spans the metric
    axes; a dashed circle marks the ``score = 1.0`` reference ring. Callers must guarantee ``scores_df`` has at least
    one metric column.

    Args:
        ax: Polar axes to draw on.
        scores_df: Frame indexed by ``Model`` with metric columns plus a ``Combined`` column.
        palette: Per-model colors aligned to ``scores_df``'s row order.
        radial: Precomputed ``(r_lower, r_upper, step, ticks)`` to share one scale across radars; when ``None`` the
            bounds are derived from this frame's own data.
        show_mean: Append the per-model ``Combined`` mean to each legend label (``"Model  (mean=X.XX)"``).
        show_reference_text: Annotate the reference ring with ``reference (1.0)``. Off by default: the text sits inside
            the plot area and can overlap the polygons; the dashed ring itself is always drawn.
        ref_fontsize: Font size of the ``reference (1.0)`` ring annotation.
        tick_fontsize: Font size of the radial (score) tick numbers.
        metric_fontsize: Font size of the abbreviated metric axis labels around the rim.
        rlabel_at_bottom: Place the radial tick numbers at the bottom instead of the upper-right, so they clear the
            ``reference (1.0)`` annotation (which sits near the top).
    """
    metrics = [column for column in scores_df.columns if column != COMBINED_COLUMN]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    closed_angles = [*angles, angles[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    all_values: list[float] = []
    for color, (model, row) in zip(palette, scores_df.iterrows(), strict=False):
        values = [row[metric] for metric in metrics]
        all_values.extend(value for value in values if not pd.isna(value))
        closed_values = [*values, values[0]]
        combined = row[COMBINED_COLUMN]
        label = f"{model}  (mean={combined:.2f})" if show_mean and not pd.isna(combined) else str(model)
        ax.plot(closed_angles, closed_values, color=color, linewidth=2.5, marker="o", markersize=5, label=label)
        ax.fill(closed_angles, closed_values, color=color, alpha=0.05)

    r_lower, r_upper, step, ticks = radial if radial is not None else _radial_ticks(all_values)
    ax.set_ylim(r_lower, r_upper)
    ax.set_yticks(ticks)

    # score = 1.0 reference ring (model as good as the reference).
    ax.plot(closed_angles, [1.0] * len(closed_angles), color="dimgray", linestyle="--", linewidth=1.2, zorder=1)
    if show_reference_text:
        ax.annotate(
            "reference (1.0)",
            xy=(angles[0], 1.0),
            fontsize=ref_fontsize,
            color="dimgray",
            ha="center",
            va="bottom",
            xytext=(0, 2),
            textcoords="offset points",
        )

    # Place metric names manually just outside the rim, aligned by their on-screen position so they
    # never overlap the polygons (default polar tick labels sit on top of the data).
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    label_radius = r_upper + step * 0.35
    for angle, metric in zip(angles, metrics, strict=False):
        screen_angle = np.pi / 2 - angle  # accounts for theta offset (pi/2) and clockwise direction
        cos_a, sin_a = np.cos(screen_angle), np.sin(screen_angle)
        horizontal = (
            "left" if cos_a > _LABEL_ALIGN_THRESHOLD else "right" if cos_a < -_LABEL_ALIGN_THRESHOLD else "center"
        )
        vertical = (
            "bottom" if sin_a > _LABEL_ALIGN_THRESHOLD else "top" if sin_a < -_LABEL_ALIGN_THRESHOLD else "center"
        )
        ax.text(
            angle,
            label_radius,
            _metric_abbrev(metric),
            fontsize=metric_fontsize,
            fontweight="bold",
            ha=horizontal,
            va=vertical,
            clip_on=False,
        )

    # Radial ticks: keep them off the spokes and lightly styled. Optionally move them to the bottom gap (data angle
    # 180, opposite the top reference annotation) so the numbers don't collide with the "reference (1.0)" text.
    ax.set_rlabel_position(180.0 if rlabel_at_bottom else np.degrees(np.mean(angles[:2])))
    ax.tick_params(axis="y", labelsize=tick_fontsize, colors="dimgray")
    ax.grid(color="gray", alpha=0.25, linewidth=0.8)
    ax.spines["polar"].set_alpha(0.3)
    return ax.get_legend_handles_labels()


def _plot_score_radar(  # noqa: PLR0913
    scores_df: pd.DataFrame,
    output_path: Path,
    colormap: str,
    title: str,
    subtitle: str,
    filename: str,
    *,
    radial: tuple[float, float, float, NDArray] | None = None,
) -> None:
    """Render a standalone radar/spider plot of per-model scores across the metric axes (higher is better).

    Thin wrapper around :func:`_draw_score_radar`: one closed polygon (light fill) per model, the per-model
    ``Combined`` score annotated in the legend, and a dashed ``score = 1.0`` reference ring. Rim labels are the metric
    abbreviations, spelled out in a caption above the model legend.

    Args:
        scores_df: Frame indexed by ``Model`` with metric columns plus a ``Combined`` column.
        output_path: Directory to save the plot.
        colormap: Seaborn/matplotlib palette name.
        title: Main title.
        subtitle: Subtitle (the reference mode).
        filename: Output file stem (``.png`` appended).
        radial: Optional precomputed ``(r_lower, r_upper, step, ticks)`` to share one scale and rings across radars;
            when ``None`` the bounds are derived from this frame's own data.
    """
    metrics = [column for column in scores_df.columns if column != COMBINED_COLUMN]
    if not metrics:
        return

    palette = model_colors(scores_df.index, colormap)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, polar=True)
    handles, labels = _draw_score_radar(ax, scores_df, palette, radial, show_mean=True)

    # Compress the polar axes so the title band clears the top rim label and the legend has room at the bottom.
    fig.subplots_adjust(top=0.84, bottom=0.12)
    _set_titles(fig, title, subtitle)
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 3),
        fontsize=12,
        title="Model (mean score)",
        title_fontsize=13,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    # Radar rim abbreviations spelled out under the model legend. The legend hangs below the axes and its height grows
    # with the model count, so measure it (after a draw) instead of guessing a fixed offset.
    caption = _abbrev_caption(metrics)
    if caption:
        fig.canvas.draw()
        legend_bottom = legend.get_window_extent().transformed(fig.transFigure.inverted()).y0
        fig.text(0.5, legend_bottom - 0.015, caption, ha="center", va="top", fontsize=12, color="dimgray")

    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{filename}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


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
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("---")
                continue
            cell = f"{value:.3f}"
            if np.isclose(value, best[column]):
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
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


def _plot_robustness_decomposition(  # noqa: PLR0913
    id_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    output_path: Path,
    colormap: str,
    title: str,
    subtitle: str,
    filename: str,
) -> None:
    """Scatter the ID/OOD score decomposition: one panel per metric (plus ``Combined``).

    Each point is a model at ``(id_score, ood_score)``; the reference sits at ``(1, 1)`` and the gray
    lines split the plane into quadrants -- the upper-right quadrant beats the reference on both ID and OOD. The dashed
    ``y = x`` diagonal is the degradation diagnostic: a model on it degrades like the reference, above it degrades less
    (more shift-robust), below it degrades more.

    Args:
        id_df: ID score frame (indexed by ``Model``, metric columns plus ``Combined``).
        ood_df: OOD score frame, same shape/index as ``id_df``.
        output_path: Directory to save the plot.
        colormap: Seaborn/matplotlib palette name.
        title: Main title.
        subtitle: Subtitle (the reference mode).
        filename: Output file stem (``.png`` appended).
    """
    panels = list(id_df.columns)  # metric columns followed by COMBINED_COLUMN
    if not panels:
        return

    # A single shared bound (spanning the data and the 1.0 reference) keeps every panel comparable.
    finite = [v for v in (*id_df.to_numpy().ravel(), *ood_df.to_numpy().ravel(), 1.0) if not pd.isna(v)]
    lo, hi = min(finite), max(finite)
    margin = max(hi - lo, EPSILON) * 0.1
    lower, upper = max(0.0, lo - margin), hi + margin

    palette = model_colors(id_df.index, colormap)
    n_cols = min(3, len(panels))
    n_rows = math.ceil(len(panels) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows), squeeze=False, sharex=True, sharey=True
    )
    flat_axes = axes.flatten()

    for index, (panel, ax) in enumerate(zip(panels, flat_axes, strict=False)):
        # Reference lines through (1, 1) and the y = x "degrades like the reference" diagonal.
        ax.axhline(1.0, color="dimgray", linewidth=0.9, zorder=1)
        ax.axvline(1.0, color="dimgray", linewidth=0.9, zorder=1)
        ax.plot([lower, upper], [lower, upper], color="dimgray", linestyle="--", linewidth=0.8, zorder=1)

        for color, model in zip(palette, id_df.index, strict=False):
            x, y = id_df[panel][model], ood_df[panel][model]
            if pd.isna(x) or pd.isna(y):
                continue
            label = str(model) if index == 0 else None  # collect legend handles once, from the first panel
            ax.scatter(x, y, color=color, s=80, edgecolor="black", linewidth=0.8, alpha=0.9, zorder=3, label=label)

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        panel_label = COMBINED_COLUMN if panel == COMBINED_COLUMN else _metric_label(panel)
        ax.set_title(panel_label, fontsize=13, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=9, colors="dimgray")
        ax.grid(visible=True, color="gray", alpha=0.18, linewidth=0.6)
        sns.despine(ax=ax, trim=False)
        # With shared axes, only label the outer edges to avoid repetition.
        if index % n_cols == 0:
            ax.set_ylabel("OOD score", fontsize=12, fontweight="bold")
        if index >= len(panels) - n_cols:
            ax.set_xlabel("ID score", fontsize=12, fontweight="bold")

    for ax in flat_axes[len(panels) :]:
        ax.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 5),
        fontsize=12,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    _set_titles(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{filename}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def _draw_combined_ranking(  # noqa: PLR0913
    ax: plt.Axes,
    combined_df: pd.DataFrame,
    colormap: str,
    *,
    palette: dict[str, Color] | None = None,
    ref_fontsize: float = 9,
    top_headroom: float = 0.0,
    label_fontsize: float | None = None,
    value_fontsize: float = 11,
    xlabel_fontsize: float = 14,
    xtick_fontsize: float = 11,
) -> None:
    """Draw the sorted horizontal bar chart of the combined score (``Combined`` column) onto ``ax``, best at the top.

    Shared by the standalone ranking and the stacked summary. The dashed line at ``1.0`` marks the reference. When
    ``palette`` is given, bar colors are looked up by model name so they match a shared radar/legend palette; otherwise
    they come from ``colormap`` via :func:`model_colors`.

    Args:
        ax: Axes to draw on.
        combined_df: Combined frame (indexed by ``Model``) whose ``Combined`` column is the ranking score.
        colormap: Seaborn/matplotlib palette name (fallback when ``palette`` is not given).
        palette: Optional model-name -> color map so bars match a shared radar/legend palette.
        ref_fontsize: Font size of the ``reference (1.0)`` annotation.
        top_headroom: Extra blank space (in bar-width units) above the top bar. The ``reference (1.0)`` label sits in
            it, clear of both the top bar below and the panel title above (used by the stacked summary).
        label_fontsize: When set, the y-tick (model name) font size; ``None`` keeps the theme default.
        value_fontsize: Font size of the per-bar value labels.
        xlabel_fontsize: Font size of the ``Score`` x-axis label.
        xtick_fontsize: Font size of the x-axis tick numbers.
    """
    # Higher is better; sort ascending so the largest (best) bar ends up on top of the horizontal chart.
    ranking = combined_df[COMBINED_COLUMN].dropna().sort_values(ascending=True)
    if ranking.empty:
        return

    if palette is not None:
        colors: list[Color] = [palette[model] for model in ranking.index]
    else:
        colors = model_colors(ranking.index, colormap)
    ax.barh(list(ranking.index), ranking.to_numpy(), color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax.axvline(1.0, color="dimgray", linestyle="--", linewidth=1.0)
    if top_headroom:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + top_headroom)
    # Sit just below the top spine; with top_headroom the top bar is pushed down, so this lands in the blank band
    # above the bars and below the title.
    ax.annotate(
        "reference (1.0)",
        xy=(1.0, 1.0),
        xytext=(3, -3),
        xycoords=("data", "axes fraction"),
        textcoords="offset points",
        fontsize=ref_fontsize,
        color="dimgray",
        ha="left",
        va="top",
    )

    for model, value in ranking.items():
        ax.text(value, model, f"  {value:.2f}", va="center", ha="left", fontsize=value_fontsize)

    # Widen the x-range 10% on both ends so the value labels don't overlap the axes.
    lo, hi = ax.get_xlim()
    pad = 0.1 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)

    if label_fontsize is not None:
        ax.tick_params(axis="y", labelsize=label_fontsize)
    ax.set_xlabel("Score", fontsize=xlabel_fontsize, fontweight="bold")
    ax.tick_params(axis="x", labelsize=xtick_fontsize)
    ax.grid(visible=True, axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def _plot_combined_ranking(  # noqa: PLR0913
    combined_df: pd.DataFrame, output_path: Path, colormap: str, title: str, subtitle: str, filename: str
) -> None:
    """Standalone sorted horizontal bar chart of the combined robustness score, best (highest) at the top.

    Thin wrapper around :func:`_draw_combined_ranking`. The score is OWA quality deflated by an ID->OOD
    percentage-degradation penalty; a bar chart fits a ranking better than a radar. The dashed line at ``1.0`` marks the
    reference; bars to its right beat it, bars to its left are worse.

    Args:
        combined_df: Combined frame (indexed by ``Model``) whose ``Combined`` column is the ranking score.
        output_path: Directory to save the plot.
        colormap: Seaborn/matplotlib palette name (fallback for models without a fixed color).
        title: Main title.
        subtitle: Subtitle (the reference mode).
        filename: Output file stem (``.png`` appended).
    """
    ranking = combined_df[COMBINED_COLUMN].dropna()
    if ranking.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(ranking) + 2))
    _draw_combined_ranking(ax, combined_df, colormap)
    _set_titles(fig, title, subtitle)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{filename}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def _plot_combined_summary(
    mode_results: list[tuple[str, dict[str, pd.DataFrame], tuple[float, float, float, NDArray]]],
    output_path: Path,
    colormap: str,
    filename: str,
) -> None:
    """Render one figure combining every reference mode as a column of the three headline views.

    Columns are reference modes (e.g. Quality / Stability) in the given order; rows are the Seen radar, the Unseen radar
    and the Combined-ranking bar. Each column keeps its **own** radial scale (the modes span very different ranges);
    all columns share one palette and a single bottom model legend. Reuses :func:`_draw_score_radar` and
    :func:`_draw_combined_ranking`.

    Args:
        mode_results: ``(reference_mode, scores, radial)`` per column, where ``scores`` is the dict from
            :func:`compute_robustness_scores` and ``radial`` is that mode's shared radial scale.
        output_path: Directory to save the plot.
        colormap: Seaborn/matplotlib palette name.
        filename: Output file stem (``.png`` appended).
    """
    if not mode_results:
        return
    first_id = mode_results[0][1][ID_TERM]
    metrics = [column for column in first_id.columns if column != COMBINED_COLUMN]
    if not metrics:
        return

    n_cols = len(mode_results)
    palette = model_colors(first_id.index, colormap)  # models identical across modes -> one palette + one legend

    fig = plt.figure(figsize=(7.5 * n_cols, 18))
    gs = fig.add_gridspec(
        3, n_cols, height_ratios=[1.0, 1.0, 0.55], hspace=0.22, wspace=0.3, left=0.17, right=0.97, top=0.87, bottom=0.11
    )
    fig_w, fig_h = fig.get_size_inches()

    handles: list = []
    labels: list[str] = []
    row_axes: list[plt.Axes] = []
    legend_left, legend_right = 0.0, 1.0  # widened to the ranking panels' outer edges inside the loop
    for col, (reference_mode, scores, radial) in enumerate(mode_results):
        id_df, ood_df, combined_df = scores[ID_TERM], scores[OOD_TERM], scores[COMBINED_TERM]
        bar_palette = dict(zip(id_df.index, palette, strict=False))

        ax_seen = fig.add_subplot(gs[0, col], polar=True)
        ax_unseen = fig.add_subplot(gs[1, col], polar=True)
        ax_bar = fig.add_subplot(gs[2, col])

        # Center the bar plot area under the radar circle (the polar axes are height-constrained, so each circle is
        # narrower than, and centered in, its cell).
        radar_pos = ax_seen.get_position()
        circle_w = min(radar_pos.width * fig_w, radar_pos.height * fig_h) / fig_w
        circle_cx = radar_pos.x0 + radar_pos.width / 2
        bar_pos = ax_bar.get_position()
        ax_bar.set_position((circle_cx - circle_w / 2, bar_pos.y0, circle_w, bar_pos.height))
        if col == 0:
            legend_left = circle_cx - circle_w / 2  # left edge of the first ranking panel
        legend_right = circle_cx + circle_w / 2  # right edge of the last ranking panel

        handles, labels = _draw_score_radar(
            ax_seen,
            id_df,
            palette,
            radial,
            show_mean=False,
            ref_fontsize=12,
            tick_fontsize=15,
            metric_fontsize=21,
            rlabel_at_bottom=True,
        )
        _draw_score_radar(
            ax_unseen,
            ood_df,
            palette,
            radial,
            show_mean=False,
            ref_fontsize=12,
            tick_fontsize=15,
            metric_fontsize=21,
            rlabel_at_bottom=True,
        )
        _draw_combined_ranking(
            ax_bar,
            combined_df,
            colormap,
            palette=bar_palette,
            ref_fontsize=14,
            top_headroom=0.5,
            label_fontsize=16,
            value_fontsize=16,
            xlabel_fontsize=17,
            xtick_fontsize=16,
        )

        # Column header (reference mode) centered over the column, above the top radar's rim.
        mode_label = SUMMARY_LABELS.get(reference_mode, _reference_label(reference_mode))
        fig.text(
            circle_cx,
            0.915,
            f"{mode_label} ({_reference_label(reference_mode)})",
            ha="center",
            va="bottom",
            fontsize=22,
            fontweight="bold",
        )
        if col == 0:
            row_axes = [ax_seen, ax_unseen, ax_bar]

    # Row labels once, at the far left, vertically centered on each row (the panel identity is shared by both columns).
    row_titles = (RADAR_TERM_TITLES[ID_TERM], RADAR_TERM_TITLES[OOD_TERM], COMBINED_PANEL_TITLE)
    for row_ax, row_title in zip(row_axes, row_titles, strict=False):
        pos = row_ax.get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(0.03, y_center, row_title, ha="center", va="center", rotation=90, fontsize=25, fontweight="bold")

    fig.suptitle("Robustness Summary", fontsize=26, fontweight="bold", y=0.975)
    # Stretch the legend across the full width of the ranking panels (edges tracked in the loop). mode="expand" fills
    # the bbox width so the entries spread evenly rather than clumping in the center.
    # The radar rim abbreviations ride along as a bold second row of the same legend box.
    key_handles, key_labels = _abbrev_key_entries(metrics)
    legend_handles, legend_labels, ncol = (
        _interleave_legend_key(handles, labels, key_handles, key_labels)
        if key_labels
        else (handles, labels, len(labels))
    )
    # Reach a little past the ranking panels so the two-row entries have room to spread.
    box_left, box_right = max(0.01, legend_left - 0.05), min(0.99, legend_right + 0.05)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower left",
        ncol=ncol,
        mode="expand",
        fontsize=17,
        title="Model",
        title_fontsize=18,
        handler_map={Patch: _AbbrevKeyHandler()},
        frameon=True,
        framealpha=0.9,
        # Below the figure: the two-row box would otherwise cover the ranking panels' "Score" axis label.
        bbox_to_anchor=(box_left, -0.05, box_right - box_left, 0.04),
    )

    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{filename}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def run_robustness_scores_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Run score analysis for each configured reference mode.

    For each ``config.score.reference_modes`` entry, computes per-model per-metric ID/OOD scores from the combined
    results file and, under ``output_path/<folder>/`` (folder from :data:`SUMMARY_FOLDERS`, e.g. ``quality_naive``),
    writes a radar plot, CSV table and LaTeX table for each of the two axes (stems from :data:`RADAR_TERM_STEMS`, e.g.
    ``seen_score``/``unseen_score``); an ID-vs-OOD decomposition scatter; and the combined robustness ranking
    (per-metric ``sqrt(id * ood)``) as a sorted bar chart with its CSV and LaTeX table. Finally, writes a single
    top-level ``robustness_summary.png`` combining every reference mode as a column (Seen radar / Unseen radar /
    Combined ranking) with a single shared model legend.

    Args:
        config: Analysis configuration (``benchmarks_filepath``, ``benchmarks``, ``models_to_compare``,
            ``trajectory_forecasting_metrics``, ``score_colormap`` and the ``score`` block).
        log: Logger.
        output_path: Directory to save the generated artifacts.
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
    colormap = config.score_colormap

    benchmarks: list[tuple[str, str, str, str]] = []
    for benchmark_entry in config.benchmarks:
        key, spec = next(iter(benchmark_entry.items()))
        benchmarks.append((key, spec.name, spec.seen, spec.unseen))

    score_cfg = config.score

    # Collected per mode (in config order) to render the single combined Quality-vs-Stability summary after the loop.
    mode_results: list[tuple[str, dict[str, pd.DataFrame], tuple[float, float, float, NDArray]]] = []
    for reference_mode in score_cfg.reference_modes:
        log.info("Computing scores for reference mode '%s'", reference_mode)
        scores = compute_robustness_scores(
            metrics_df,
            benchmarks,
            metrics,
            models_to_compare,
            reference_mode=reference_mode,
            aggregate=str(score_cfg.aggregate),
            uniform_key=str(score_cfg.uniform_key),
        )
        if scores[COMBINED_TERM].empty:
            log.warning("No models scored for reference mode '%s'; skipping.", reference_mode)
            continue

        mode_output = output_path / str(SUMMARY_FOLDERS.get(reference_mode, reference_mode))
        reference_label = _reference_label(reference_mode)  # plot subtitle, e.g. "Naive-Relative"
        print(f"\n=== Scores: {reference_mode} ({len(scores[ID_TERM])} models) ===")

        # Two score axes: radar + CSV + LaTeX each. Share one radial scale across both radars (excluding the
        # Combined column) so the Seen/Unseen rings and bounds match and can be compared side by side.
        shared_values = [
            value
            for term in RADAR_TERM_STEMS
            for value in scores[term].drop(columns=COMBINED_COLUMN, errors="ignore").to_numpy().ravel()
            if not pd.isna(value)
        ]
        shared_radial = _radial_ticks(shared_values)
        for term, stem in RADAR_TERM_STEMS.items():
            term_df = scores[term]
            term_title = f"{RADAR_TERM_TITLES[term]} ({reference_mode})"
            _plot_score_radar(
                term_df,
                mode_output,
                colormap,
                RADAR_TERM_TITLES[term],
                reference_label,
                f"{stem}_radar",
                radial=shared_radial,
            )
            _write_scores_csv(term_df, mode_output, f"{stem}_scores", label=RADAR_TERM_TITLES[term])
            _write_scores_tex(term_df, mode_output, f"{stem}_scores", caption=term_title)

        _plot_robustness_decomposition(
            scores[ID_TERM],
            scores[OOD_TERM],
            mode_output,
            colormap,
            "ID/OOD Score Decomposition",
            reference_label,
            "score_decomposition",
        )

        # Combined ranking: sorted (best first) bar chart + CSV + LaTeX.
        combined_title = f"{COMBINED_TITLE} ({reference_mode})"
        combined_sorted = scores[COMBINED_TERM].sort_values(COMBINED_COLUMN, ascending=False)
        _plot_combined_ranking(
            scores[COMBINED_TERM],
            mode_output,
            colormap,
            COMBINED_TITLE,
            reference_label,
            f"{COMBINED_FILE_STEM}_ranking",
        )
        _write_scores_csv(combined_sorted, mode_output, f"{COMBINED_FILE_STEM}_scores", label=COMBINED_TITLE)
        _write_scores_tex(combined_sorted, mode_output, f"{COMBINED_FILE_STEM}_scores", caption=combined_title)

        mode_results.append((reference_mode, scores, shared_radial))

    # Single combined summary spanning all reference modes (columns), written at the top level.
    _plot_combined_summary(mode_results, output_path, colormap, SUMMARY_FILE_STEM)

    print("\n✓ Robustness score analysis complete!")
    log.info("Robustness score analysis complete!")
