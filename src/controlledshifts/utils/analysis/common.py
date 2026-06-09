"""Shared, general-purpose utilities for model analysis.

See `docs/ANALYSIS.md` for usage details.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from numpy.typing import NDArray

from controlledshifts.utils.constants import EPSILON


MODEL_NAME_MAP = {
    "autobot": "AutoBot",
    "scenetransformer": "SceneTransformer",
    "wayformer": "Wayformer",
    "mtr": "MTR",
    "safe-wayformer": "Safe-Wayformer",
    "naive": "Naive",
}

MODEL_SIZE_MAP = {
    "Naive": "624k",
    "AutoBot": "1.5M",
    "SceneTransformer": "7.6M",
    "Wayformer": "15.1M",
    "Safe-Wayformer": "15.2M",
    "MTR": "27.2M",  # This is the size with d_model=256. The original MTR with d_model=512 has 65M parameters.
}

BENCHMARK_NAME_MAP = {
    "causal-benchmark-labeled": "CausalAgents",
    "ego-safeshift-causal-benchmark": "EgoSafeShift",
    "environments-benchmark": "Environments",
}

SPLIT_NAME_MAP = {
    "test/waymo-mini-causal-testing": "CausalAgents/ID",
    "test/waymo-remove-noncausal-testing": "CausalAgents/OOD",
}


def relative_gap_pct(value: float | NDArray, reference: float | NDArray) -> float | NDArray:
    """Compute ``(value - reference) / |reference| * 100``. Works for scalars and numpy arrays."""
    return ((value - reference) / (np.abs(reference) + EPSILON)) * 100


def set_yaxis_limits(
    ax: Axes,
    values: list[float],
    *,
    padding_factor: float = 0.5,
    lower_factor: float = 1.0,
    min_padding: float = 0.05,
) -> None:
    """Set y-axis limits with padding around the data range.

    Args:
        ax: Matplotlib axis to modify.
        values: Data values used to compute the range.
        padding_factor: Fraction of the data range to use as padding.
        lower_factor: Multiplier applied to padding on the lower end (useful for bar charts).
        min_padding: Minimum padding when the data range is zero.
    """
    if not values:
        return
    ymin, ymax = np.nanmin(values), np.nanmax(values)
    padding = padding_factor * (ymax - ymin) if ymax > ymin else min_padding
    ax.set_ylim(ymin - padding * lower_factor, ymax + padding)


def symmetric_vrange(values: list[float]) -> tuple[float, float]:
    """Return ``(-vabs, +vabs)`` where ``vabs = max(|min|, |max|)`` of *values*."""
    vabs = max(abs(np.nanmin(values)), abs(np.nanmax(values)))
    return -vabs, vabs


def flatten_metrics(data: dict, prefix: str = "") -> dict[str, float | int | str | bool | None]:
    """Recursively flatten a nested dict, joining keys with dots."""
    flat: dict[str, float | int | str | bool | None] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name))
        else:
            flat[name] = value
    return flat


def plot_heatmap(  # noqa: PLR0913
    heatmap: npt.NDArray[np.float64],
    title: str,
    x_label: str,
    y_label: str,
    cbar_label: str,
    output_filepath: Path,
    colormap: str = "viridis",
) -> None:
    """Visualizes a heatmap matrix.

    Args:
        heatmap (npt.NDArray[np.float64]): a heatmap matrix to plot.
        title (str): the title of the heatmap.
        x_label (str): the label of the x-axis.
        y_label (str): the label of the y-axis.
        cbar_label (str): the label of the heatmap's colorbar.
        colormap (str): the colormap to use for the heatmap.
        output_filepath (Path): filepath to save the visualization.
    """
    plt.figure(figsize=(35, 30))

    plt.imshow(heatmap, cmap=colormap, aspect="auto")
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=40)
    cbar.set_label(cbar_label, size=40)

    plt.title(title, fontsize=50)
    plt.xlabel(x_label, fontsize=40)
    plt.ylabel(y_label, fontsize=40)
    plt.xticks(range(heatmap.shape[0]))
    plt.yticks(range(heatmap.shape[1]))
    plt.grid(visible=False)

    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()
    print(f"Heatmap saved to {output_filepath}")
