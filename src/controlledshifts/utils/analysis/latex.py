r"""LaTeX cell primitives shared by the analysis table writers.

Only individual *cells* live here — the surrounding table scaffolding stays with each writer, because the three tables
differ in environment, sizing, header structure and column layout. Tables using the colored gap annotation need
``\usepackage[table]{xcolor}`` in the document preamble.

See `docs/ANALYSIS.md` for usage details.
"""

import numpy as np
import pandas as pd

from controlledshifts.utils.constants import EPSILON


# Lowest color intensity for a gap annotation. Higher makes small gaps more vibrant in the LaTeX tables.
GAP_MIN_COLOR_VALUE = 20.0


def format_value(value: float, best_value: float) -> str:
    """Render a metric value, bolding it when it ties ``best_value``.

    Whether "best" means the smallest or the largest value is the caller's choice, made when computing ``best_value``.

    Args:
        value: Value to render; ``---`` when missing.
        best_value: The block's best value for this column.

    Returns:
        The formatted cell.
    """
    if pd.isna(value):
        return "---"
    value_str = f"{value:.3f}"
    if pd.notna(best_value) and np.isclose(value, best_value):
        value_str = f"\\textbf{{{value_str}}}"
    return value_str


def format_gap(gap: float, best_gap: float, worst_gap: float) -> str:
    r"""Render a colored relative-gap annotation, e.g. ``\textcolor{OrangeRed!63}{+12.46\%}``.

    Severity is scaled to the block's own ``[best_gap, worst_gap]`` range, so coloring is comparable within a block
    rather than across the whole table. Positive gaps (worse than the reference) are red, negative gaps green.

    Args:
        gap: Relative gap, in percent.
        best_gap: The block's smallest gap.
        worst_gap: The block's largest gap.

    Returns:
        The formatted annotation, without surrounding parentheses.
    """
    denom = max(abs(worst_gap - best_gap), EPSILON)
    severity = float(np.clip(abs(gap - best_gap) / denom, 0, 1))
    intensity = int(GAP_MIN_COLOR_VALUE + severity * (100 - GAP_MIN_COLOR_VALUE))
    color = "OrangeRed" if gap > 0 else "ForestGreen"
    return f"\\textcolor{{{color}!{intensity}}}{{{gap:+.2f}\\%}}"
