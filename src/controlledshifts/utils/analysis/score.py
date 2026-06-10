"""Distribution-shift sensitivity scoring (per-model, per-metric) and radar visualization.

Reduces the raw ID/OOD benchmark numbers into a single comparable *sensitivity score* per model per
metric, measured against a reference:

* ``naive_relative`` -- each model vs the Naive baseline within the same benchmark.
* ``uniform_relative`` -- each model vs its own performance in the Uniform benchmark.

Each score combines a performance term (in-distribution quality vs the reference) with a non-negative
gap term (how much the metric worsens seen->unseen, relative to the reference). In both reference modes
the convention is the same: **higher == better** (less sensitive to shift than the reference), zero ==
on par with the reference, negative == worse. See `docs/ANALYSIS.md`.
"""

import math
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from omegaconf import DictConfig

from controlledshifts.utils.analysis.common import METRIC_NAME_MAP, MODEL_NAME_MAP, relative_gap_pct
from controlledshifts.utils.analysis.distribution_shift import build_benchmark_df
from controlledshifts.utils.constants import EPSILON


COMBINED_COLUMN = "Combined"

# Cosine/sine deadband for deciding radar label alignment (center vs left/right, top/bottom).
_LABEL_ALIGN_THRESHOLD = 0.1


@dataclass(frozen=True)
class GapConfig:
    """Configuration for the gap term (a non-negative robustness multiplier).

    Attributes:
        mode: ``"ratio"`` for the floored, clipped ratio ``|ref_gap| / |model_gap|`` (``> 1`` when the
            model degrades less than the reference, ``< 1`` when it degrades more) or ``"bounded"`` for
            the ``(0, 1)`` form ``|ref_gap| / (|ref_gap| + |model_gap|)``.
        epsilon: Percentage-point floor added to both magnitudes. Prevents the near-zero blow-up of
            the raw ratio; a meaningful floor (e.g. 1.0) is required since gaps are in percent.
        clip: Upper clip applied to the ``"ratio"`` mode result.
    """

    mode: str = "ratio"
    epsilon: float = 1.0
    clip: float = 5.0


def _performance(model_seen: float, ref_seen: float) -> float:
    """Performance term ``1 - model_seen / ref_seen`` (lower-is-better metrics, in-distribution).

    Positive when the model's seen-split error is below the reference's, zero when equal (including
    the reference compared against itself), negative when worse. The denominator is sign-stabilized
    to avoid division by zero.
    """
    if pd.isna(model_seen) or pd.isna(ref_seen):
        return float("nan")
    denom = ref_seen + (EPSILON if ref_seen >= 0 else -EPSILON)
    return 1.0 - model_seen / denom


def _gap_ratio(model_gap: float, ref_gap: float, cfg: GapConfig) -> float:
    """Non-negative robustness multiplier comparing a model's seen->unseen degradation to the reference's.

    Always ``>= 0`` so it never flips the sign of ``performance * gap_ratio`` -- this keeps the final
    score monotone (higher == better, with the sign carried by the performance term). ``bounded`` mode
    returns a value in ``(0, 1)`` that grows as the model's gap shrinks. ``ratio`` mode returns the
    floored ``|ref_gap| / |model_gap|`` clipped to ``[0, clip]`` (``> 1`` when the model degrades less
    than the reference, ``< 1`` when it degrades more).
    """
    if pd.isna(model_gap) or pd.isna(ref_gap):
        return float("nan")

    if cfg.mode == "bounded":
        return (abs(ref_gap) + cfg.epsilon) / (abs(ref_gap) + abs(model_gap) + cfg.epsilon)

    ratio = (abs(ref_gap) + cfg.epsilon) / (abs(model_gap) + cfg.epsilon)
    return float(np.clip(ratio, 0.0, cfg.clip))


def _compute_model_gaps(benchmark_df: pd.DataFrame, splits: tuple[str, str], metrics: list[str]) -> pd.DataFrame:
    """Per-model seen->unseen gap (percent) per metric for one benchmark.

    Args:
        benchmark_df: Per-model frame from :func:`build_benchmark_df`, indexed by ``Model``.
        splits: ``(seen_split, unseen_split)`` column prefixes.
        metrics: Metric names.

    Returns:
        DataFrame indexed by ``Model`` with one column per metric holding the relative gap percent.
    """
    seen_split, unseen_split = splits
    gaps = {
        metric: relative_gap_pct(benchmark_df[f"{unseen_split}/{metric}"], benchmark_df[f"{seen_split}/{metric}"])
        for metric in metrics
    }
    return pd.DataFrame(gaps, index=benchmark_df.index)


def _select_reference(  # noqa: PLR0913
    reference_mode: str,
    *,
    frames: dict[str, pd.DataFrame],
    gaps: dict[str, pd.DataFrame],
    splits: dict[str, tuple[str, str]],
    benchmark_key: str,
    model: str,
    metric: str,
    uniform_key: str,
) -> tuple[float, float]:
    """Return ``(ref_seen, ref_gap)`` for one ``(benchmark, model, metric)`` cell.

    ``naive_relative`` references the Naive row of the same benchmark; ``uniform_relative`` references
    the same model's row in the Uniform benchmark. Returns ``(nan, nan)`` when the reference is
    absent.
    """
    if reference_mode == "uniform_relative":
        ref_key, ref_model = uniform_key, model
    else:
        ref_key, ref_model = benchmark_key, MODEL_NAME_MAP["naive"]

    ref_frame = frames.get(ref_key)
    if ref_frame is None or ref_model not in ref_frame.index:
        return float("nan"), float("nan")

    seen_split, _ = splits[ref_key]
    ref_seen = float(ref_frame.loc[ref_model, f"{seen_split}/{metric}"])
    ref_gap = float(gaps[ref_key].loc[ref_model, metric])
    return ref_seen, ref_gap


def compute_sensitivity_scores(  # noqa: PLR0913
    metrics_df: pd.DataFrame,
    benchmarks: list[tuple[str, str, str, str]],
    metrics: list[str],
    models_to_compare: list[str],
    *,
    reference_mode: str,
    gap_cfg: GapConfig,
    aggregate: str = "mean",
    uniform_key: str = "uniform",
) -> pd.DataFrame:
    """Compute per-model, per-metric sensitivity scores aggregated across benchmarks.

    For each benchmark, model and metric the score is ``performance * gap_ratio`` where the reference
    is selected by ``reference_mode``. Scores are aggregated across benchmarks (NaN-safe), and a
    ``Combined`` column holds the per-model mean across metrics. For ``uniform_relative`` the Uniform
    benchmark is excluded from aggregation (its self-reference is degenerate).

    Args:
        metrics_df: Combined results frame with a ``Name`` (``<dataset>_<model>``) column.
        benchmarks: ``(key, name, seen_split, unseen_split)`` tuples in display order.
        metrics: Metric names, in column order.
        models_to_compare: Raw model identifiers to include.
        reference_mode: ``"naive_relative"`` or ``"uniform_relative"``.
        gap_cfg: Gap-ratio configuration.
        aggregate: ``"mean"`` or ``"median"`` across benchmarks.
        uniform_key: Benchmark key used as the ``uniform_relative`` reference.

    Returns:
        DataFrame indexed by ``Model`` with one column per metric plus a ``Combined`` column.
    """
    frames: dict[str, pd.DataFrame] = {}
    gaps: dict[str, pd.DataFrame] = {}
    splits: dict[str, tuple[str, str]] = {}
    for key, _name, seen, unseen in benchmarks:
        benchmark_df = build_benchmark_df(metrics_df, (seen, unseen), metrics, models_to_compare, show_run_id=False)
        if benchmark_df.empty:
            continue
        benchmark_df = benchmark_df.set_index("Model")
        frames[key] = benchmark_df
        gaps[key] = _compute_model_gaps(benchmark_df, (seen, unseen), metrics)
        splits[key] = (seen, unseen)

    # accum[model][metric] -> list of per-benchmark scores
    accum: dict[str, dict[str, list[float]]] = {}
    for key, _name, seen, _unseen in benchmarks:
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
                model_gap = float(gaps[key].loc[model, metric])
                ref_seen, ref_gap = _select_reference(
                    reference_mode,
                    frames=frames,
                    gaps=gaps,
                    splits=splits,
                    benchmark_key=key,
                    model=model,
                    metric=metric,
                    uniform_key=uniform_key,
                )
                score = _performance(model_seen, ref_seen) * _gap_ratio(model_gap, ref_gap, gap_cfg)
                accum.setdefault(model, {}).setdefault(metric, []).append(score)

    agg_fn = np.nanmedian if aggregate == "median" else np.nanmean
    ordered_models = [MODEL_NAME_MAP.get(model, model) for model in models_to_compare]

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

    The bounds are multiples of ``step`` so every gridline circle is equally separated and the outermost
    ring sits exactly on the rim. A full step of headroom is added when the data lands on a boundary so
    polygons never touch the rim.
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

    One closed polygon (with light fill) per model spans the metric axes; the per-model ``Combined``
    score is annotated in the legend. A dashed circle marks the ``score = 0`` baseline (as good as
    the reference).

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

    palette = sns.color_palette(colormap, len(scores_df))
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


def _write_scores_csv(scores_df: pd.DataFrame, output_path: Path, filename: str) -> Path:
    """Write the per-model scores to CSV and print a formatted summary."""
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{filename}.csv"
    scores_df.to_csv(output_file)
    print("Sensitivity scores (higher = less sensitive to shift than the reference):")
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


def run_score_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Run sensitivity-score analysis for each configured reference mode.

    For each ``config.score.reference_modes`` entry, computes per-model per-metric scores from the
    combined results file and writes a radar plot, a CSV table and a LaTeX table under
    ``output_path/<mode>/``.

    Args:
        config: Analysis configuration (``benchmarks_filepath``, ``benchmarks``, ``models_to_compare``,
            ``trajectory_forecasting_metrics``, ``score_colormap`` and the ``score`` block).
        log: Logger for analysis information.
        output_path: Directory to save the generated artifacts.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        },
    )

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
    gap_cfg = GapConfig(
        mode=str(score_cfg.gap_mode),
        epsilon=float(score_cfg.gap_epsilon),
        clip=float(score_cfg.gap_clip),
    )

    for reference_mode in score_cfg.reference_modes:
        log.info("Computing sensitivity scores for reference mode '%s'", reference_mode)
        scores_df = compute_sensitivity_scores(
            metrics_df,
            benchmarks,
            metrics,
            models_to_compare,
            reference_mode=reference_mode,
            gap_cfg=gap_cfg,
            aggregate=str(score_cfg.aggregate),
            uniform_key=str(score_cfg.uniform_key),
        )
        if scores_df.empty:
            log.warning("No models scored for reference mode '%s'; skipping.", reference_mode)
            continue

        mode_output = output_path / reference_mode
        print(f"\n=== Sensitivity scores: {reference_mode} ({len(scores_df)} models) ===")
        title = f"Distribution-Shift Sensitivity ({reference_mode})"
        _plot_score_radar(scores_df, mode_output, colormap, title, "sensitivity_radar")
        _write_scores_csv(scores_df, mode_output, "sensitivity_scores")
        _write_scores_tex(scores_df, mode_output, "sensitivity_scores", caption=title)

    print("\n✓ Sensitivity score analysis complete!")
    log.info("Sensitivity score analysis complete!")
