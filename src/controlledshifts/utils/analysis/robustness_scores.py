"""Distribution-shift robustness scoring (per-model, per-metric) and radar visualization.

Reduces the raw SEEN/UNSEEN benchmark numbers into comparable *scores* per model per metric, measured
against a reference:

* ``naive_relative`` -- each model vs the Naive baseline within the same benchmark.
* ``uniform_relative`` -- each model vs its own performance in the Uniform benchmark.

Following the MASE / OWA framing of the N-BEATS paper (arXiv:1905.10437), every metric is positive and
lower-is-better, and each model is characterized by two reference-relative *score* axes (the reciprocal MASE skill):

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
stability score, not a downstream-performance ranking. See `docs/ANALYSIS.md`.
"""

import math
from collections.abc import Callable
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from omegaconf import DictConfig

from controlledshifts.utils.analysis.common import METRIC_NAME_MAP, MODEL_NAME_MAP, model_colors
from controlledshifts.utils.analysis.distribution_shift import build_benchmark_df
from controlledshifts.utils.constants import EPSILON
from controlledshifts.utils.plotting import set_analysis_theme


COMBINED_COLUMN = "Combined"

# Score-term keys returned by :func:`compute_robustness_scores`. The two per-metric score axes are rendered as radars;
# the combined robustness score is rendered as a sorted bar chart.
ID_TERM = "id"
OOD_TERM = "ood"
COMBINED_TERM = "combined"
UNIFORM_RELATIVE = "uniform_relative"
# Radar terms only (the per-metric score axes), mapped to their output-file stem and display title.
RADAR_TERM_STEMS = {ID_TERM: "id_score", OOD_TERM: "ood_score"}
RADAR_TERM_TITLES = {ID_TERM: "Seen Score", OOD_TERM: "Unseen Score"}
COMBINED_FILE_STEM = "combined_robustness"
COMBINED_TITLE = "Combined Robustness Score"

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
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.93, subtitle, ha="center", va="top", fontsize=11, color="dimgray")


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
    """Render a radar/spider plot of per-model scores across the metric axes (higher is better).

    One closed polygon (with light fill) per model spans the metric axes; the per-model ``Combined`` score is annotated
    in the legend. A dashed circle marks the ``score = 1.0`` reference ring (as good as the reference).

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

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    closed_angles = [*angles, angles[0]]

    palette = model_colors(scores_df.index, colormap)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    all_values: list[float] = []
    for color, (model, row) in zip(palette, scores_df.iterrows(), strict=False):
        values = [row[metric] for metric in metrics]
        all_values.extend(value for value in values if not pd.isna(value))
        closed_values = [*values, values[0]]
        combined = row[COMBINED_COLUMN]
        label = f"{model}  (mean={combined:.2f})" if not pd.isna(combined) else str(model)
        ax.plot(closed_angles, closed_values, color=color, linewidth=2.5, marker="o", markersize=5, label=label)
        ax.fill(closed_angles, closed_values, color=color, alpha=0.08)

    r_lower, r_upper, step, ticks = radial if radial is not None else _radial_ticks(all_values)
    ax.set_ylim(r_lower, r_upper)
    ax.set_yticks(ticks)

    # score = 1.0 reference ring (model as good as the reference).
    ax.plot(closed_angles, [1.0] * len(closed_angles), color="dimgray", linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate(
        "reference (1.0)",
        xy=(angles[0], 1.0),
        fontsize=8,
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
            _metric_label(metric),
            fontsize=12,
            fontweight="bold",
            ha=horizontal,
            va=vertical,
            clip_on=False,
        )

    # Radial ticks: keep them off the spokes and lightly styled.
    ax.set_rlabel_position(np.degrees(np.mean(angles[:2])))
    ax.tick_params(axis="y", labelsize=9, colors="dimgray")
    ax.grid(color="gray", alpha=0.25, linewidth=0.8)
    ax.spines["polar"].set_alpha(0.3)

    # Compress the polar axes so the title band clears the top rim label and the legend has room at the bottom.
    fig.subplots_adjust(top=0.84, bottom=0.12)
    _set_titles(fig, title, subtitle)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 3),
        fontsize=10,
        title="Model (mean score)",
        title_fontsize=11,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

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
        ax.set_title(panel_label, fontsize=12, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=7, colors="dimgray")
        ax.grid(visible=True, color="gray", alpha=0.18, linewidth=0.6)
        sns.despine(ax=ax, trim=False)
        # With shared axes, only label the outer edges to avoid repetition.
        if index % n_cols == 0:
            ax.set_ylabel("OOD score", fontsize=10, fontweight="bold")
        if index >= len(panels) - n_cols:
            ax.set_xlabel("ID score", fontsize=10, fontweight="bold")

    for ax in flat_axes[len(panels) :]:
        ax.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 5),
        fontsize=10,
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


def _plot_combined_ranking(  # noqa: PLR0913
    combined_df: pd.DataFrame, output_path: Path, colormap: str, title: str, subtitle: str, filename: str
) -> None:
    """Sorted horizontal bar chart of the combined robustness score (``Combined`` column), best (highest) at the top.

    The score is OWA quality deflated by an ID->OOD percentage-degradation penalty. A bar chart fits a ranking better
    than a radar. The dashed line at ``1.0`` marks the reference; bars to its right beat the reference, bars to its left
    are worse.

    Args:
        combined_df: Combined frame (indexed by ``Model``) whose ``Combined`` column is the ranking score.
        output_path: Directory to save the plot.
        colormap: Seaborn/matplotlib palette name (fallback for models without a fixed color).
        title: Main title.
        subtitle: Subtitle (the reference mode).
        filename: Output file stem (``.png`` appended).
    """
    # Higher is better; sort ascending so the largest (best) bar ends up on top of the horizontal chart.
    ranking = combined_df[COMBINED_COLUMN].dropna().sort_values(ascending=True)
    if ranking.empty:
        return

    colors = model_colors(ranking.index, colormap)
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(ranking) + 2))
    ax.barh(list(ranking.index), ranking.to_numpy(), color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax.axvline(1.0, color="dimgray", linestyle="--", linewidth=1.0)
    ax.annotate(
        "reference (1.0)",
        xy=(1.0, 1.0),
        xytext=(3, -3),
        xycoords=("data", "axes fraction"),
        textcoords="offset points",
        fontsize=8,
        color="dimgray",
        ha="left",
        va="top",
    )

    for model, value in ranking.items():
        ax.text(value, model, f"  {value:.2f}", va="center", ha="left", fontsize=9)

    # Widen the x-range 10% on both ends so the value labels don't overlap the axes.
    lo, hi = ax.get_xlim()
    pad = 0.1 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)

    ax.set_xlabel("Score", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    _set_titles(fig, title, subtitle)
    ax.grid(visible=True, axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{filename}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def run_robustness_scores_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Run score analysis for each configured reference mode.

    For each ``config.score.reference_modes`` entry, computes per-model per-metric ID/OOD scores from the combined
    results file and, under ``output_path/<mode>/``, writes a radar plot, CSV table and LaTeX table for each of the two
    axes; an ID-vs-OOD decomposition scatter; and the combined robustness ranking (per-metric ``sqrt(id * ood)``) as a
    sorted bar chart with its CSV and LaTeX table.

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

        mode_output = output_path / reference_mode
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

    print("\n✓ Robustness score analysis complete!")
    log.info("Robustness score analysis complete!")
