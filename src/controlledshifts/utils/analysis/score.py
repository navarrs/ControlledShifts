"""Distribution-shift robustness scoring (per-model, per-metric) and radar visualization.

Reduces the raw SEEN/UNSEEN benchmark numbers into comparable *robustness scores* per model per metric, measured
against a reference:

* ``naive_relative`` -- each model vs the Naive baseline within the same benchmark.
* ``uniform_relative`` -- each model vs its own performance in the Uniform benchmark.

Working in log space (metrics are positive, lower-is-better errors), each model is characterized by two
reference-relative robustness axes:

* ``seen_robustness_score = log(ref_seen / model_seen)`` -- ID-level robustness: how much better the model already is on
  the seen split (its starting point).
* ``shift_robustness_score = log((ref_unseen/ref_seen) / (model_unseen/model_seen))`` -- shift robustness: how much less
  the model degrades seen->unseen than the reference (sign-preserving, so improving under shift is rewarded).

Both share natural-log units, are symmetric and unbounded both ways (a 2x improvement and a 2x degradation are
``+-log 2``), are ``0`` for the reference compared against itself, and need no epsilon/clip and no regression. The
convention is the same in both reference modes: **higher == more robust than the reference**, zero == on par, negative
== worse.

To rank models by a single value, the two axes are reduced to a ``combined`` score. Their raw sum is *not* used: it
telescopes to ``seen + shift = log(ref_unseen / model_unseen)``, so it ranks models purely by OOD error (the reference
cancels to an additive constant) and adds nothing beyond the OOD numbers. Instead the combined score uses **standardized
equal-influence**: each axis is z-scored across the model cohort (per metric), the two z-scores are summed, and the
per-model mean across metrics is the ranking score. This gives both axes -- and every metric -- equal say regardless of
their natural spread. The trade-off is that the combined score is **cohort-relative**: ``0`` means "cohort average," not
"on par with the reference," and scores recenter if the set of models changes. See `docs/ANALYSIS.md`.
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

from controlledshifts.utils.analysis.common import METRIC_NAME_MAP, MODEL_COLOR_MAP, MODEL_NAME_MAP
from controlledshifts.utils.analysis.distribution_shift import build_benchmark_df
from controlledshifts.utils.constants import EPSILON
from controlledshifts.utils.plotting import set_analysis_theme


COMBINED_COLUMN = "Combined"

# Score-term keys returned by :func:`compute_robustness_scores`. The two per-metric robustness axes are rendered as
# radars; the combined score is a cohort-relative ranking rendered as a sorted bar chart.
SEEN_TERM = "seen"
SHIFT_TERM = "shift"
COMBINED_TERM = "combined"
# Radar terms only (the per-metric robustness axes), mapped to their output-file stem and display title.
RADAR_TERM_STEMS = {SEEN_TERM: "seen_robustness", SHIFT_TERM: "shift_robustness"}
RADAR_TERM_TITLES = {SEEN_TERM: "Seen (ID-level) Robustness", SHIFT_TERM: "Shift Robustness"}
COMBINED_FILE_STEM = "combined_robustness"
COMBINED_TITLE = "Combined Robustness Score (standardized z-sum)"

# Cosine/sine deadband for deciding radar label alignment (center vs left/right, top/bottom).
_LABEL_ALIGN_THRESHOLD = 0.1


def _model_colors(models: pd.Index, colormap: str) -> list[str | tuple[float, float, float]]:
    """Per-model colors aligned to ``models``, using the fixed :data:`MODEL_COLOR_MAP` where available.

    Models without a fixed color fall back to the configured ``colormap`` palette (assigned in order among the
    unmapped models), so the Naive baseline stays black and every model keeps the same color across plots.
    """
    fallback = iter(sns.color_palette(colormap, sum(model not in MODEL_COLOR_MAP for model in models)))
    return [MODEL_COLOR_MAP[model] if model in MODEL_COLOR_MAP else next(fallback) for model in models]


def _log_ratio(numerator: float, denominator: float) -> float:
    """Natural-log ratio ``log(numerator / denominator)`` for positive error metrics.

    Returns NaN if either argument is missing or non-positive. Symmetric and additive: a 2x improvement and a 2x
    degradation map to ``+-log 2``, and log ratios sum, which is what makes the seen/shift/overall robustness terms
    decompose exactly.
    """
    if pd.isna(numerator) or pd.isna(denominator) or numerator <= 0 or denominator <= 0:
        return float("nan")
    return math.log(numerator / denominator)


def _seen_robustness(model_seen: float, ref_seen: float) -> float:
    """ID-level robustness ``log(ref_seen / model_seen)``.

    Positive when the model's seen error is below the reference's, zero when equal (including the reference compared
    against itself), negative when worse.
    """
    return _log_ratio(ref_seen, model_seen)


def _shift_robustness(model_seen: float, model_unseen: float, ref_seen: float, ref_unseen: float) -> float:
    """Shift robustness ``log((ref_unseen/ref_seen) / (model_unseen/model_seen))``.

    Compares the model's seen->unseen degradation factor to the reference's. Positive when the model degrades less than
    the reference (or improves under shift), zero when they degrade equally, negative when the model degrades more.
    Equivalently ``log(ref_unseen/ref_seen) - log(model_unseen/model_seen)``.
    """
    ref_degradation = _log_ratio(ref_unseen, ref_seen)
    model_degradation = _log_ratio(model_unseen, model_seen)
    if pd.isna(ref_degradation) or pd.isna(model_degradation):
        return float("nan")
    return ref_degradation - model_degradation


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
    if reference_mode == "uniform_relative":
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
    """Compute per-model, per-metric robustness scores (seen, shift, combined) aggregated across benchmarks.

    For each benchmark, model and metric two reference-relative log-ratio axes are computed (see the module docstring):
    ``seen_robustness_score`` (ID-level) and ``shift_robustness_score`` (degradation resistance). The reference is
    selected by ``reference_mode``. Both are aggregated across benchmarks (NaN-safe) into per-metric frames with a
    ``Combined`` column. The ``combined`` frame is then derived by :func:`_standardized_combined` -- a cohort-relative
    ranking that z-scores each axis and sums them. For ``uniform_relative`` the Uniform benchmark is excluded from
    aggregation (its self-reference is degenerate).

    Args:
        metrics_df: Combined results frame with a ``Name`` (``<dataset>_<model>``) column.
        benchmarks: ``(key, name, seen_split, unseen_split)`` tuples in display order.
        metrics: Metric names, in column order.
        models_to_compare: Raw model identifiers to include.
        reference_mode: ``"naive_relative"`` or ``"uniform_relative"``.
        aggregate: ``"mean"`` or ``"median"`` across benchmarks.
        uniform_key: Benchmark key used as the ``uniform_relative`` reference.

    Returns:
        Dict keyed by :data:`SEEN_TERM`, :data:`SHIFT_TERM` and :data:`COMBINED_TERM`; each value is a DataFrame indexed
        by ``Model`` with one column per metric plus a ``Combined`` column.
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
    accum: dict[str, dict[str, dict[str, list[float]]]] = {SEEN_TERM: {}, SHIFT_TERM: {}}
    for key, _name, seen, unseen in benchmarks:
        if key not in frames:
            continue
        # Skip the Uniform benchmark entirely under uniform_relative: a model is referenced against its
        # own Uniform row, so scoring Uniform here would be a degenerate uniform-vs-uniform comparison.
        if reference_mode == "uniform_relative" and key == uniform_key:
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
                seen_score = _seen_robustness(model_seen, ref_seen)
                shift_score = _shift_robustness(model_seen, model_unseen, ref_seen, ref_unseen)
                for term, value in ((SEEN_TERM, seen_score), (SHIFT_TERM, shift_score)):
                    accum[term].setdefault(model, {}).setdefault(metric, []).append(value)

    agg_fn = np.nanmedian if aggregate == "median" else np.nanmean
    ordered_models = [MODEL_NAME_MAP.get(model, model) for model in models_to_compare]
    seen_df = _aggregate_term(accum[SEEN_TERM], metrics, ordered_models, agg_fn)
    shift_df = _aggregate_term(accum[SHIFT_TERM], metrics, ordered_models, agg_fn)
    return {SEEN_TERM: seen_df, SHIFT_TERM: shift_df, COMBINED_TERM: _standardized_combined(seen_df, shift_df, metrics)}


def _standardized_combined(seen_df: pd.DataFrame, shift_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Cohort-relative ranking score: z-score each axis across models (per metric), sum, then mean across metrics.

    Standardizing per metric gives each axis -- and each metric -- equal influence regardless of its natural spread
    (see the module docstring). A metric column with ~zero spread across models contributes 0 (no information). The
    ``Combined`` column is the per-model mean across metrics and is the headline ranking value; ``0`` is the cohort
    average, not the reference.

    Args:
        seen_df: Aggregated ``seen`` frame (indexed by ``Model``, metric columns plus ``Combined``).
        shift_df: Aggregated ``shift`` frame, same shape/index as ``seen_df``.
        metrics: Metric names (the per-metric columns to standardize).

    Returns:
        DataFrame indexed by ``Model`` with one z-sum column per metric plus a ``Combined`` ranking column.
    """

    def _zscore(column: pd.Series) -> pd.Series:
        std = column.std(ddof=0)
        if pd.isna(std) or std < EPSILON:
            return column * 0.0  # constant (or empty) column carries no ranking information
        return (column - column.mean()) / std

    combined = pd.DataFrame(index=seen_df.index)
    for metric in metrics:
        combined[metric] = _zscore(seen_df[metric]) + _zscore(shift_df[metric])
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
    """Return ``(r_lower, r_upper, step, ticks)`` snapped to a nice step around the data (and zero).

    The bounds are multiples of ``step`` so every gridline circle is equally separated and the outermost ring sits on
    the rim. A full step of headroom is added when the data lands on a boundary so polygons never touch the rim.
    """
    data_lower = min(0.0, np.nanmin(values)) if values else 0.0
    data_upper = max(0.0, np.nanmax(values)) if values else 1.0
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


def _plot_score_radar(scores_df: pd.DataFrame, output_path: Path, colormap: str, title: str, filename: str) -> None:
    """Render a radar/spider plot of per-model sensitivity scores across the metric axes.

    One closed polygon (with light fill) per model spans the metric axes; the per-model ``Combined`` score is annotated
    in the legend. A dashed circle marks the ``score = 0`` baseline (as good as the reference).

    Args:
        scores_df: Frame indexed by ``Model`` with metric columns plus a ``Combined`` column.
        output_path: Directory to save the plot.
        colormap: Seaborn/matplotlib palette name.
        title: Figure title.
        filename: Output file stem (``.png`` appended).
    """
    metrics = [column for column in scores_df.columns if column != COMBINED_COLUMN]
    if not metrics:
        return

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    closed_angles = [*angles, angles[0]]

    palette = _model_colors(scores_df.index, colormap)
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
        label = f"{model}  (Σ={combined:+.2f})" if not pd.isna(combined) else str(model)
        ax.plot(closed_angles, closed_values, color=color, linewidth=2.5, marker="o", markersize=5, label=label)
        ax.fill(closed_angles, closed_values, color=color, alpha=0.08)

    r_lower, r_upper, step, ticks = _radial_ticks(all_values)
    ax.set_ylim(r_lower, r_upper)
    ax.set_yticks(ticks)

    # score = 0 baseline ring (model as good as the reference).
    ax.plot(closed_angles, [0.0] * len(closed_angles), color="dimgray", linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate(
        "reference (0)",
        xy=(angles[0], 0.0),
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

    ax.set_title(title, fontsize=16, fontweight="bold", pad=55)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.08, 1.05),
        fontsize=10,
        title="Model (Σ = mean score)",
        title_fontsize=11,
        frameon=True,
        framealpha=0.9,
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
    print(f"{label} (higher = more robust than the reference):")
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
    seen_df: pd.DataFrame, shift_df: pd.DataFrame, output_path: Path, colormap: str, title: str, filename: str
) -> None:
    """Scatter the seen/shift robustness decomposition: one panel per metric (plus ``Combined``).

    Each point is a model at ``(seen_robustness_score, shift_robustness_score)``; the reference sits at the origin and
    the gray axes split the plane into quadrants. The upper-right quadrant is both better in-distribution and more
    shift-robust than the reference.

    Args:
        seen_df: ``seen`` term frame (indexed by ``Model``, metric columns plus ``Combined``).
        shift_df: ``shift`` term frame, same shape/index as ``seen_df``.
        output_path: Directory to save the plot.
        colormap: Seaborn/matplotlib palette name.
        title: Figure title.
        filename: Output file stem (``.png`` appended).
    """
    panels = list(seen_df.columns)  # metric columns followed by COMBINED_COLUMN
    if not panels:
        return

    # A single shared, symmetric bound keeps every panel on the same (shared) scale and comparable.
    finite = [v for v in (*seen_df.to_numpy().ravel(), *shift_df.to_numpy().ravel()) if not pd.isna(v)]
    bound = (max((abs(v) for v in finite), default=1.0) or 1.0) * 1.18

    palette = _model_colors(seen_df.index, colormap)
    n_cols = min(3, len(panels))
    n_rows = math.ceil(len(panels) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows), squeeze=False, sharex=True, sharey=True
    )
    flat_axes = axes.flatten()

    for index, (panel, ax) in enumerate(zip(panels, flat_axes, strict=False)):
        # Reference axes through the origin (seen = 0 and shift = 0).
        ax.axhline(0.0, color="dimgray", linewidth=0.9, zorder=1)
        ax.axvline(0.0, color="dimgray", linewidth=0.9, zorder=1)

        for color, model in zip(palette, seen_df.index, strict=False):
            x, y = seen_df[panel][model], shift_df[panel][model]
            if pd.isna(x) or pd.isna(y):
                continue
            label = str(model) if index == 0 else None  # collect legend handles once, from the first panel
            ax.scatter(x, y, color=color, s=80, edgecolor="black", linewidth=0.8, alpha=0.9, zorder=3, label=label)

        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        panel_label = COMBINED_COLUMN if panel == COMBINED_COLUMN else _metric_label(panel)
        ax.set_title(panel_label, fontsize=12, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=7, colors="dimgray")
        ax.grid(visible=True, color="gray", alpha=0.18, linewidth=0.6)
        sns.despine(ax=ax, trim=False)
        # With shared axes, only label the outer edges to avoid repetition.
        if index % n_cols == 0:
            ax.set_ylabel("Shift Robustness", fontsize=10, fontweight="bold")
        if index >= len(panels) - n_cols:
            ax.set_xlabel("Seen Robustness", fontsize=10, fontweight="bold")

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

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{filename}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def _plot_combined_ranking(
    combined_df: pd.DataFrame, output_path: Path, colormap: str, title: str, filename: str
) -> None:
    """Sorted horizontal bar chart of the combined ranking score (the ``Combined`` column), best at the top.

    A bar chart fits a ranking better than a radar and avoids a "reference (0)" baseline, which is meaningless for the
    cohort-relative combined score. The dashed line at ``0`` marks the cohort average.

    Args:
        combined_df: Combined frame (indexed by ``Model``) whose ``Combined`` column is the ranking score.
        output_path: Directory to save the plot.
        colormap: Seaborn/matplotlib palette name (fallback for models without a fixed color).
        title: Figure title.
        filename: Output file stem (``.png`` appended).
    """
    ranking = combined_df[COMBINED_COLUMN].dropna().sort_values(ascending=True)  # ascending -> best ends up on top
    if ranking.empty:
        return

    colors = _model_colors(ranking.index, colormap)
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(ranking) + 2))
    ax.barh(list(ranking.index), ranking.to_numpy(), color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax.axvline(0.0, color="dimgray", linestyle="--", linewidth=1.0)
    ax.annotate(
        "cohort average (0)",
        xy=(0.0, 1.0),
        xytext=(3, -3),
        xycoords=("data", "axes fraction"),
        textcoords="offset points",
        fontsize=8,
        color="dimgray",
        ha="left",
        va="top",
    )

    for model, value in ranking.items():
        ax.text(value, model, f"  {value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)

    ax.set_xlabel("Combined robustness score (standardized z-sum)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(visible=True, axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    fig.tight_layout()
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{filename}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def run_score_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Run robustness-score analysis for each configured reference mode.

    For each ``config.score.reference_modes`` entry, computes per-model per-metric seen/shift robustness scores from
    the combined results file and, under ``output_path/<mode>/``, writes: a radar plot, CSV table and LaTeX table for
    each of the two axes; a seen-vs-shift decomposition scatter; and the standardized combined ranking as a sorted bar
    chart with its CSV and LaTeX table.

    Args:
        config: Analysis configuration (``benchmarks_filepath``, ``benchmarks``, ``models_to_compare``,
            ``trajectory_forecasting_metrics``, ``score_colormap`` and the ``score`` block).
        log: Logger for analysis information.
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
        log.info("Computing robustness scores for reference mode '%s'", reference_mode)
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
        print(f"\n=== Robustness scores: {reference_mode} ({len(scores[COMBINED_TERM])} models) ===")

        # Two robustness axes: radar + CSV + LaTeX each.
        for term, stem in RADAR_TERM_STEMS.items():
            term_df = scores[term]
            term_title = f"{RADAR_TERM_TITLES[term]} ({reference_mode})"
            _plot_score_radar(term_df, mode_output, colormap, term_title, f"{stem}_radar")
            _write_scores_csv(term_df, mode_output, f"{stem}_scores", label=RADAR_TERM_TITLES[term])
            _write_scores_tex(term_df, mode_output, f"{stem}_scores", caption=term_title)

        _plot_robustness_decomposition(
            scores[SEEN_TERM],
            scores[SHIFT_TERM],
            mode_output,
            colormap,
            f"Robustness Decomposition ({reference_mode})",
            "robustness_decomposition",
        )

        # Combined ranking: sorted (best first) bar chart + CSV + LaTeX.
        combined_title = f"{COMBINED_TITLE} ({reference_mode})"
        combined_sorted = scores[COMBINED_TERM].sort_values(COMBINED_COLUMN, ascending=False)
        _plot_combined_ranking(
            scores[COMBINED_TERM], mode_output, colormap, combined_title, f"{COMBINED_FILE_STEM}_ranking"
        )
        _write_scores_csv(combined_sorted, mode_output, f"{COMBINED_FILE_STEM}_scores", label=COMBINED_TITLE)
        _write_scores_tex(combined_sorted, mode_output, f"{COMBINED_FILE_STEM}_scores", caption=combined_title)

    print("\n✓ Robustness score analysis complete!")
    log.info("Robustness score analysis complete!")
