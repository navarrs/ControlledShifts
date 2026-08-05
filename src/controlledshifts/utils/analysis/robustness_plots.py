"""Rendering for the distribution-shift robustness scores.

Every figure the robustness analysis writes lives here: the per-metric score radars, the ID-vs-OOD decomposition
scatter, the combined-ranking bar chart, and the stacked summary that puts one reference mode per column. The scoring
itself, and the CSV/LaTeX writers, stay in ``robustness_scores.py``; this module only draws what that produces.

Font sizes in this module are deliberately local rather than drawn from the shared tiers in ``common.py``: the radars
and the stacked summary size their text against a hand-tuned canvas, so they scale with the layout rather than with
the rest of the package.

See `docs/ANALYSIS.md` for usage details.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.artist import Artist
from matplotlib.gridspec import GridSpec
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Transform
from numpy.typing import NDArray

from controlledshifts.utils.analysis.common import (
    COMBINED_COLUMN,
    COMPACT_LABEL_FONTSIZE,
    COMPACT_LEGEND_FONTSIZE,
    COMPACT_SUBTITLE_FONTSIZE,
    COMPACT_SUPTITLE_FONTSIZE,
    COMPACT_TICK_FONTSIZE,
    COMPACT_TITLE_FONTSIZE,
    FIGURE_DPI,
    GRAY_TEXT_COLOR,
    METRIC_ABBREV_MAP,
    mathtext_bold,
    metric_abbrev,
    metric_label,
    model_colors,
    save_figure,
)
from controlledshifts.utils.constants import EPSILON


# A per-model plot color: a hex string (fixed models) or an RGB tuple (palette fallback), as returned by model_colors.
Color = str | tuple[float, float, float]

# Display titles for the two per-metric score radars (seen then unseen -- the summary's row order), and the bar
# panel's row label. Keyed by the score-term names ``robustness_scores`` uses for the same two axes.
RADAR_TERM_TITLES = {"id": "Seen Score", "ood": "Unseen Score"}
COMBINED_PANEL_TITLE = "Combined Score"

# Cosine/sine deadband for deciding radar label alignment (center vs left/right, top/bottom).
_LABEL_ALIGN_THRESHOLD = 0.1

# Legend layout shared by every figure in this module. ``alignment`` centers the entries stacked in one column on each
# other (the models over the abbreviation key), and _equalize_legend_columns gives every column the same width so the
# entries land on an even pitch instead of a ragged one.
_LEGEND_ALIGNMENT = "center"
_LEGEND_COLUMNSPACING = 1.6


def _legend_layout(ncol: int) -> dict[str, float | str | int]:
    """Shared ``fig.legend`` layout kwargs: fixed column spacing and entries centered within their column."""
    return {"ncol": ncol, "alignment": _LEGEND_ALIGNMENT, "columnspacing": _LEGEND_COLUMNSPACING}


def _equalize_legend_columns(fig: plt.Figure, legend: Legend) -> None:
    """Widen every legend column to the widest one and center its entries, so they sit on an even pitch.

    Matplotlib sizes each legend column to its own widest entry and left-aligns within it, so a legend whose labels
    differ in length (models over the abbreviation key) reads as a ragged block. A column box reports a fixed width as
    its own, and centers its children inside it, which lines the rows up.

    The widths are display pixels and only measurable once there is a renderer, so the figure is first switched to the
    resolution :func:`save_figure` renders at -- measuring at another dpi would leave the saved columns mis-sized.
    """
    fig.set_dpi(FIGURE_DPI)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    columns = legend._legend_handle_box.get_children()  # noqa: SLF001 -- packed columns have no public accessor
    width = max(column.get_bbox(renderer).width for column in columns)
    for column in columns:
        column.set_width(width)
        column.align = "center"


def _set_titles(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Set a large bold main title with a smaller subtitle just beneath it, centered over the figure.

    Args:
        fig: Figure to title.
        title: Main title text.
        subtitle: Subtitle text.
    """
    fig.suptitle(title, fontsize=COMPACT_SUPTITLE_FONTSIZE, fontweight="bold")
    fig.text(0.5, 0.93, subtitle, ha="center", va="top", fontsize=COMPACT_SUBTITLE_FONTSIZE, color=GRAY_TEXT_COLOR)


def _abbrev_pairs(columns: list[str], abbrev: dict[str, str] | None) -> list[tuple[str, str]]:
    """``(short, full)`` label pairs for the columns that have an abbreviation, in column order.

    ``abbrev`` overrides the metric lookup, for radars whose axes are not metrics (e.g. benchmarks). Columns missing
    from it -- or from :data:`METRIC_ABBREV_MAP` when it is ``None`` -- get no key entry and are labelled in full.
    """
    if abbrev is not None:
        return [(abbrev[column], column) for column in columns if column in abbrev]
    return [(metric_abbrev(column), metric_label(column)) for column in columns if column in METRIC_ABBREV_MAP]


def _axis_label(column: str, abbrev: dict[str, str] | None) -> str:
    """Rim label for one radar axis: the abbreviation when there is one, else the clean name."""
    return abbrev.get(column, column) if abbrev is not None else metric_abbrev(column)


def _abbrev_caption(columns: list[str], abbrev: dict[str, str] | None) -> str:
    """One-line abbreviation key, e.g. ``BF = BrierFDE  ·  MF = MinFDE`` (only the abbreviations bold)."""
    return "   ·   ".join(f"{mathtext_bold(short)} = {full}" for short, full in _abbrev_pairs(columns, abbrev))


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
        text = Text(
            -xdescent, -ydescent, mathtext_bold(orig_handle.get_label()), fontsize=fontsize, ha="left", va="baseline"
        )
        text.set_transform(trans)
        return [text]


def _abbrev_key_entries(columns: list[str], abbrev: dict[str, str] | None) -> tuple[list[Patch], list[str]]:
    """Legend ``(handles, labels)`` for the abbreviation key: bold abbreviation in the handle slot, ``= Name`` after.

    The handles are bare :class:`~matplotlib.patches.Patch` carriers for the abbreviation text -- a type the model
    entries do not use, so :class:`_AbbrevKeyHandler` can be mapped to it without touching them.
    """
    pairs = _abbrev_pairs(columns, abbrev)
    return [Patch(label=short) for short, _full in pairs], [f"= {full}" for _short, full in pairs]


def _interleave_legend_key(
    handles: list, labels: list[str], key_handles: list, key_labels: list[str]
) -> tuple[list, list[str], int]:
    """Merge model entries and the abbreviation key into one legend, returning ``(handles, labels, ncol)``.

    Matplotlib fills legend columns top to bottom, so pairing each model with a key entry renders the models as the
    first row and the key as the second. When the two rows have different lengths the shorter one is padded with blank
    entries split evenly on both sides, so it stays centered under the longer row instead of clumping to the left.
    """
    ncol = max(len(labels), len(key_labels))
    blank = Line2D([], [], linestyle="none", marker="none")

    def centered(entries: list, filler: Line2D | str) -> list:
        lead = (ncol - len(entries)) // 2
        return [filler] * lead + list(entries) + [filler] * (ncol - len(entries) - lead)

    padded_handles = centered(handles, blank)
    padded_labels = centered(labels, "")
    padded_key_handles = centered(key_handles, blank)
    padded_key_labels = centered(key_labels, "")

    merged_handles: list = []
    merged_labels: list[str] = []
    for handle, label, key_handle, key_label in zip(
        padded_handles, padded_labels, padded_key_handles, padded_key_labels, strict=True
    ):
        merged_handles.extend((handle, key_handle))
        merged_labels.extend((label, key_label))
    return merged_handles, merged_labels, ncol


# Text sizes for the stacked summary. Scaled to its hand-tuned figsize=(7.5 * n_cols, 18) canvas, not to the shared
# tiers in common.py -- changing the canvas is what should change these.
_SUMMARY_SUPTITLE_FONTSIZE = 26
_SUMMARY_ROW_LABEL_FONTSIZE = 25
_SUMMARY_HEADER_FONTSIZE = 22
_SUMMARY_METRIC_FONTSIZE = 21
_SUMMARY_LEGEND_TITLE_FONTSIZE = 20
_SUMMARY_BAR_XLABEL_FONTSIZE = 17
_SUMMARY_LEGEND_FONTSIZE = 17
_SUMMARY_BAR_LABEL_FONTSIZE = 16
_SUMMARY_TICK_FONTSIZE = 15
_SUMMARY_BAR_REF_FONTSIZE = 14
_SUMMARY_REF_FONTSIZE = 12

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


def radial_ticks(values: list[float], *, n_target: int = 5) -> tuple[float, float, float, NDArray]:
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
    abbrev: dict[str, str] | None = None,
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
        abbrev: Column -> short rim label, for axes that are not metrics (benchmarks). ``None`` uses the metric map.
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

    r_lower, r_upper, step, ticks = radial if radial is not None else radial_ticks(all_values)
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
            _axis_label(metric, abbrev),
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


def plot_score_radar(  # noqa: PLR0913
    scores_df: pd.DataFrame,
    output_path: Path,
    colormap: str,
    title: str,
    subtitle: str,
    filename: str,
    *,
    radial: tuple[float, float, float, NDArray] | None = None,
    abbrev: dict[str, str] | None = None,
) -> None:
    """Render a standalone radar/spider plot of per-model scores across the score-column axes (higher is better).

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
        abbrev: Column -> short rim label, for axes that are not metrics (benchmarks). ``None`` uses the metric map.
    """
    metrics = [column for column in scores_df.columns if column != COMBINED_COLUMN]
    if not metrics:
        return

    palette = model_colors(scores_df.index, colormap)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, polar=True)
    handles, labels = _draw_score_radar(ax, scores_df, palette, radial, show_mean=True, abbrev=abbrev)

    # Compress the polar axes so the title band clears the top rim label and the legend has room at the bottom.
    fig.subplots_adjust(top=0.84, bottom=0.12)
    _set_titles(fig, title, subtitle)
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        **_legend_layout(min(len(labels), 3)),
        fontsize=COMPACT_LEGEND_FONTSIZE,
        title="Model (mean score)",
        title_fontsize=COMPACT_SUBTITLE_FONTSIZE,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )
    _equalize_legend_columns(fig, legend)

    # Radar rim abbreviations spelled out under the model legend. The legend hangs below the axes and its height grows
    # with the model count, so measure it (after a draw) instead of guessing a fixed offset.
    caption = _abbrev_caption(metrics, abbrev)
    if caption:
        fig.canvas.draw()
        legend_bottom = legend.get_window_extent().transformed(fig.transFigure.inverted()).y0
        fig.text(
            0.5,
            legend_bottom - 0.015,
            caption,
            ha="center",
            va="top",
            fontsize=COMPACT_LEGEND_FONTSIZE,
            color=GRAY_TEXT_COLOR,
        )

    save_figure(fig, output_path / f"{filename}.png")


def plot_robustness_decomposition(  # noqa: PLR0913
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
        panel_label = COMBINED_COLUMN if panel == COMBINED_COLUMN else metric_label(panel)
        ax.set_title(panel_label, fontsize=COMPACT_TITLE_FONTSIZE, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=COMPACT_TICK_FONTSIZE, colors=GRAY_TEXT_COLOR)
        ax.grid(visible=True, color="gray", alpha=0.18, linewidth=0.6)
        sns.despine(ax=ax, trim=False)
        # With shared axes, only label the outer edges to avoid repetition.
        if index % n_cols == 0:
            ax.set_ylabel("OOD score", fontsize=COMPACT_LABEL_FONTSIZE, fontweight="bold")
        if index >= len(panels) - n_cols:
            ax.set_xlabel("ID score", fontsize=COMPACT_LABEL_FONTSIZE, fontweight="bold")

    for ax in flat_axes[len(panels) :]:
        ax.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        **_legend_layout(min(len(labels), 5)),
        fontsize=COMPACT_LEGEND_FONTSIZE,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )
    _equalize_legend_columns(fig, legend)

    _set_titles(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_figure(fig, output_path / f"{filename}.png")


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


def plot_combined_ranking(  # noqa: PLR0913
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
    save_figure(fig, output_path / f"{filename}.png")


def _center_bar_under_radar(fig: plt.Figure, ax_radar: plt.Axes, ax_bar: plt.Axes) -> tuple[float, float]:
    """Resize the ranking panel to sit centered under the radar circle, returning its ``(left, right)`` figure-x edges.

    The polar axes are height-constrained, so each circle is narrower than -- and centered in -- its gridspec cell. The
    bar panel would otherwise span the full cell and look offset against the circle above it.
    """
    fig_w, fig_h = fig.get_size_inches()
    radar_pos = ax_radar.get_position()
    circle_w = min(radar_pos.width * fig_w, radar_pos.height * fig_h) / fig_w
    circle_cx = radar_pos.x0 + radar_pos.width / 2
    bar_pos = ax_bar.get_position()
    ax_bar.set_position((circle_cx - circle_w / 2, bar_pos.y0, circle_w, bar_pos.height))
    return circle_cx - circle_w / 2, circle_cx + circle_w / 2


def _draw_summary_column(  # noqa: PLR0913
    fig: plt.Figure,
    gs: GridSpec,
    col: int,
    mode_result: tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[float, float, float, NDArray]],
    palette: list[Color],
    colormap: str,
    abbrev: dict[str, str] | None,
) -> tuple[list, list[str], list[plt.Axes], tuple[float, float]]:
    """Draw one reference mode's column: the Seen radar, the Unseen radar and the combined-ranking bar beneath them.

    Returns the legend ``(handles, labels)`` harvested from the Seen radar, the column's three axes, and the ranking
    panel's ``(left, right)`` figure-x edges so the caller can size the shared legend to span every column.
    """
    column_title, id_df, ood_df, combined_df, radial = mode_result
    bar_palette = dict(zip(id_df.index, palette, strict=False))

    ax_seen = fig.add_subplot(gs[0, col], polar=True)
    ax_unseen = fig.add_subplot(gs[1, col], polar=True)
    ax_bar = fig.add_subplot(gs[2, col])
    panel_left, panel_right = _center_bar_under_radar(fig, ax_seen, ax_bar)

    def draw_radar(ax: plt.Axes, scores_df: pd.DataFrame) -> tuple[list, list[str]]:
        return _draw_score_radar(
            ax,
            scores_df,
            palette,
            radial,
            show_mean=False,
            abbrev=abbrev,
            ref_fontsize=_SUMMARY_REF_FONTSIZE,
            tick_fontsize=_SUMMARY_TICK_FONTSIZE,
            metric_fontsize=_SUMMARY_METRIC_FONTSIZE,
            rlabel_at_bottom=True,
        )

    handles, labels = draw_radar(ax_seen, id_df)
    draw_radar(ax_unseen, ood_df)
    _draw_combined_ranking(
        ax_bar,
        combined_df,
        colormap,
        palette=bar_palette,
        ref_fontsize=_SUMMARY_BAR_REF_FONTSIZE,
        top_headroom=0.5,
        label_fontsize=_SUMMARY_BAR_LABEL_FONTSIZE,
        value_fontsize=_SUMMARY_BAR_LABEL_FONTSIZE,
        xlabel_fontsize=_SUMMARY_BAR_XLABEL_FONTSIZE,
        xtick_fontsize=_SUMMARY_BAR_LABEL_FONTSIZE,
    )

    # Column header centered over the column, above the top radar's rim.
    circle_cx = (panel_left + panel_right) / 2
    fig.text(
        circle_cx, 0.915, column_title, ha="center", va="bottom", fontsize=_SUMMARY_HEADER_FONTSIZE, fontweight="bold"
    )
    return handles, labels, [ax_seen, ax_unseen, ax_bar], (panel_left, panel_right)


def _add_row_labels(fig: plt.Figure, row_axes: list[plt.Axes]) -> None:
    """Label each row once at the far left, rotated and vertically centered (the identity is shared by all columns)."""
    row_titles = (*RADAR_TERM_TITLES.values(), COMBINED_PANEL_TITLE)
    for row_ax, row_title in zip(row_axes, row_titles, strict=False):
        pos = row_ax.get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(
            0.03,
            y_center,
            row_title,
            ha="center",
            va="center",
            rotation=90,
            fontsize=_SUMMARY_ROW_LABEL_FONTSIZE,
            fontweight="bold",
        )


def _add_summary_legend(  # noqa: PLR0913
    fig: plt.Figure,
    handles: list,
    labels: list[str],
    metrics: list[str],
    legend_left: float,
    legend_right: float,
    abbrev: dict[str, str] | None,
) -> None:
    """Attach the shared model legend, with the radar rim abbreviations as a bold second row of the same box.

    Centered on the ranking panels (edges tracked by the caller). The box is sized by its own equalized columns rather
    than stretched to the panel width: stretching fixes the total, which leaves no room to widen the columns to a
    common size, and it is that common size that lines the two rows up.
    """
    key_handles, key_labels = _abbrev_key_entries(metrics, abbrev)
    legend_handles, legend_labels, ncol = (
        _interleave_legend_key(handles, labels, key_handles, key_labels)
        if key_labels
        else (handles, labels, len(labels))
    )
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        **_legend_layout(ncol),
        fontsize=_SUMMARY_LEGEND_FONTSIZE,
        title="Model",
        title_fontsize=_SUMMARY_LEGEND_TITLE_FONTSIZE,
        handler_map={Patch: _AbbrevKeyHandler()},
        frameon=True,
        framealpha=0.9,
        # Below the figure: the two-row box would otherwise cover the ranking panels' "Score" axis label.
        bbox_to_anchor=((legend_left + legend_right) / 2, -0.05),
    )
    _equalize_legend_columns(fig, legend)


def plot_combined_summary(
    mode_results: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[float, float, float, NDArray]]],
    output_path: Path,
    colormap: str,
    filename: str,
    *,
    abbrev: dict[str, str] | None = None,
) -> None:
    """Render one figure combining every reference mode as a column of the three headline views.

    Columns are reference modes (e.g. Quality / Stability) in the given order; rows are the Seen radar, the Unseen radar
    and the Combined-ranking bar. Each column keeps its **own** radial scale (the modes span very different ranges);
    all columns share one palette and a single bottom model legend. Reuses :func:`_draw_score_radar` and
    :func:`_draw_combined_ranking`.

    Args:
        mode_results: ``(column_title, id_df, ood_df, combined_df, radial)`` per column. The caller supplies the
            finished header text and the frames individually, so this module needs no knowledge of reference modes
            or of the scoring dict's term keys.
        output_path: Directory to save the plot.
        colormap: Seaborn/matplotlib palette name.
        filename: Output file stem (``.png`` appended).
        abbrev: Column -> short rim label, for axes that are not metrics (benchmarks). ``None`` uses the metric map.
    """
    if not mode_results:
        return
    first_id = mode_results[0][1]
    metrics = [column for column in first_id.columns if column != COMBINED_COLUMN]
    if not metrics:
        return

    n_cols = len(mode_results)
    palette = model_colors(first_id.index, colormap)  # models identical across modes -> one palette + one legend

    fig = plt.figure(figsize=(7.5 * n_cols, 18))
    gs = fig.add_gridspec(
        3, n_cols, height_ratios=[1.0, 1.0, 0.55], hspace=0.28, wspace=0.3, left=0.17, right=0.97, top=0.87, bottom=0.11
    )

    handles: list = []
    labels: list[str] = []
    row_axes: list[plt.Axes] = []
    legend_left, legend_right = 0.0, 1.0  # widened to the ranking panels' outer edges inside the loop
    for col, mode_result in enumerate(mode_results):
        handles, labels, column_axes, (panel_left, panel_right) = _draw_summary_column(
            fig, gs, col, mode_result, palette, colormap, abbrev
        )
        if col == 0:
            legend_left = panel_left  # left edge of the first ranking panel
            row_axes = column_axes
        legend_right = panel_right  # right edge of the last ranking panel

    _add_row_labels(fig, row_axes)
    fig.suptitle("Robustness Summary", fontsize=_SUMMARY_SUPTITLE_FONTSIZE, fontweight="bold", y=0.975)
    _add_summary_legend(fig, handles, labels, metrics, legend_left, legend_right, abbrev)

    save_figure(fig, output_path / f"{filename}.png")
