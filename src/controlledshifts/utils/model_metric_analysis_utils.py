import math
from collections.abc import Iterable
from itertools import product
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from omegaconf import DictConfig

from controlledshifts.utils.constants import EPSILON


MODEL_NAME_MAP = {
    "autobot": "AutoBot",
    "scenetransformer": "SceneTransformer",
    "wayformer": "Wayformer",
    "mtr": "MTR",
    "safe-wayformer": "Safe-Wayformer",
}

MODEL_SIZE_MAP = {
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

STRATEGY_NAME_MAP = {
    "random_drop": "Random",
    "kmeans_random_drop": "KMeans-R",
    "simple_kmeans_cosine_drop": "KMeans-SC",
    "gumbel_kmeans_cosine_drop": "KMeans-GC",
    "den_tp": "DenTP",
    "vocab_cluster_jaccard_drop": "DenTP-KJ",
    "vocab_cluster_hamming_drop": "DenTP-KH",
}


# Maps generator model directory names to short labels appended to column headers when
# multiple generator models cover the same strategy.
GENERATOR_MODEL_ABBREV = {
    "wayformer": "WF",
    "mtr": "MTR",
    "baselines": None,  # no tag for files under the baselines/ parent
}

# Maps each strategy name to the CSV file stem (strategy group) it belongs to.
STRATEGY_FILE_GROUP = {
    "random_drop": "random",
    "kmeans_random_drop": "kmeans",
    "simple_kmeans_cosine_drop": "kmeans",
    "gumbel_kmeans_cosine_drop": "kmeans",
    "vocab_cluster_jaccard_drop": "kmeans",
    "vocab_cluster_hamming_drop": "kmeans",
    "den_tp": "dentp",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _relative_gap_pct(value: float | NDArray, reference: float | NDArray) -> float | NDArray:
    """Compute ``(value - reference) / |reference| * 100``. Works for scalars and numpy arrays."""
    return ((value - reference) / (np.abs(reference) + EPSILON)) * 100


def _build_strategy_colormap(config: DictConfig, items: Iterable) -> dict:
    """Return a ``{item: color}`` mapping using the configured lineplot colormap."""
    cmap = plt.cm.get_cmap(config.get("lineplot_colormap", "tab10"))
    colors = [cmap(i) for i in range(cmap.N)]
    return {s: colors[i % len(colors)] for i, s in enumerate(items)}


def _collect_sweep_y_values(
    metrics_df: pd.DataFrame,
    sweep_prefix: str,
    retention_pcts: list[float],
    column: str,
    metric: str,
) -> tuple[NDArray, list[float]]:
    """Collect metric values across retention percentages for one sweep prefix.

    Args:
        metrics_df: DataFrame containing all experiment metrics.
        sweep_prefix: Experiment name prefix identifying this sweep (e.g. ``benchmark_model_strategy``).
        retention_pcts: Ordered list of retention-percentage floats.
        column: DataFrame column to read from.
        metric: Metric name (used to detect the ``Runtime`` special case).

    Returns:
        A tuple ``(y_arr, collected)`` where ``y_arr`` is a float array aligned to ``retention_pcts``
        (NaN for missing entries) and ``collected`` contains only the valid values.
    """
    y: list[float] = []
    collected: list[float] = []
    for pct in retention_pcts:
        row = metrics_df[metrics_df["Name"].str.startswith(f"{sweep_prefix}_{pct}")]
        if not row.empty and column in row.columns:
            val = float(row.iloc[0][column])
            y.append(val)
            collected.append(val)
        else:
            y.append(float("nan"))
    y_arr = np.array(y, dtype=float)
    if metric == "Runtime":
        y_arr = y_arr / 3600.0
        collected = [v / 3600.0 for v in collected]
    return y_arr, collected


def _style_sweep_ax(ax: Axes, metric: str, retention_pcts: list[float]) -> None:
    """Apply standard styling to a sample-selection sweep lineplot axis."""
    ax.set_title((metric[0].upper() + metric[1:]).replace("_", " "), pad=10)
    ax.set_xlabel("Data Retention (%)")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(retention_pcts)
    ax.set_xticklabels([f"{int(p * 100)}%" for p in retention_pcts])
    ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend().remove()


def _set_yaxis_limits(
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


def _load_sample_selection_dataframes(config: DictConfig, log: Logger) -> dict[str, pd.DataFrame] | None:
    """Load all sample-selection metrics CSVs listed in ``config.sample_selection_files``.

    Returns a ``{suffix: DataFrame}`` mapping, or ``None`` if any file is missing.
    The suffix is derived from the first ``_``-separated token of each filename stem.
    """
    metrics_dataframes: dict[str, pd.DataFrame] = {}
    for file in config.sample_selection_files:
        log.info("Loading sample selection file: %s", file)
        metrics_filepath = Path(file)
        if not metrics_filepath.exists():
            log.error("Sample selection CSV not found at %s", metrics_filepath)
            return None
        metrics_dataframes[f"{metrics_filepath.parent.name}/{metrics_filepath.stem}"] = pd.read_csv(metrics_filepath)
    return metrics_dataframes


def _symmetric_vrange(values: list[float]) -> tuple[float, float]:
    """Return ``(-vabs, +vabs)`` where ``vabs = max(|min|, |max|)`` of *values*."""
    vabs = max(abs(np.nanmin(values)), abs(np.nanmax(values)))
    return -vabs, vabs


def _flatten_metrics(data: dict, prefix: str = "") -> dict[str, float | int | str | bool | None]:
    """Recursively flatten a nested dict, joining keys with dots."""
    flat: dict[str, float | int | str | bool | None] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, name))
        else:
            flat[name] = value
    return flat


# ---------------------------------------------------------------------------
# Lineplot functions
# ---------------------------------------------------------------------------


def _plot_joint_sample_selection_sweep_lineplot(
    config: DictConfig, log: Logger, output_path: Path, metrics_dataframes: dict[str, pd.DataFrame]
) -> None:
    """For each (model, split), create a figure with one subplot per metric.

    Each subplot shows one line per ``(strategy, file_key)`` column from all loaded files, using the
    same disambiguating labels as the heatmaps (e.g. ``KMeans-SC (WF)`` vs ``KMeans-SC (SafeST)``).
    A dashed baseline reference line is drawn when baseline data is available.

    Args:
        config: Model analysis configuration.
        log: Logger.
        output_path: Directory for generated plots.
        metrics_dataframes: ``{"parent/stem": DataFrame}`` mapping.
    """
    strategy_columns, col_labels = _build_strategy_column_index(config, metrics_dataframes)
    if not strategy_columns:
        return

    metrics = config.trajectory_forecasting_metrics + config.other_metrics
    log.info("Plotting joint sample selection sweep lineplots for metrics: %s", metrics)
    retention_pcts = list(map(float, config.sample_retention_percentages))
    colormap = _build_strategy_colormap(config, [col_labels[col] for col in strategy_columns])

    for model, split in product(config.models_to_compare, config.sample_selection_splits_to_compare):
        subsplit = split.split("/")[-1]
        log.info("Creating joint sweep plot for model=%s, split=%s", model, split)

        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
        for i, metric in enumerate(metrics):
            ax = axes[0, i]
            column = f"{split}/{metric}" if metric in config.trajectory_forecasting_metrics else metric
            all_y_values: list[float] = []

            for strategy, file_key in strategy_columns:
                sweep_prefix = f"{config.sample_selection_benchmark}_{model}_{strategy}"
                label = col_labels[(strategy, file_key)]
                y, collected = _collect_sweep_y_values(
                    metrics_dataframes[file_key], sweep_prefix, retention_pcts, column, metric
                )
                all_y_values.extend(collected)
                ax.plot(retention_pcts, y, marker="o", ms=6, lw=2.5, c=colormap[label], alpha=0.9, label=label)

            # Baseline reference: search all dataframes
            base_name = f"{config.sample_selection_benchmark}_{model}"
            for df in metrics_dataframes.values():
                base_df = df[df["Name"] == base_name]
                if not base_df.empty and column in base_df.columns:
                    base_value = base_df[column].min()
                    if metric == "Runtime":
                        base_value = base_value / 3600.0
                    ax.axhline(base_value, ls="--", lw=2, c="black", alpha=0.7, label="Base model")
                    all_y_values.append(base_value)
                    break

            _set_yaxis_limits(ax, all_y_values)
            _style_sweep_ax(ax, metric, retention_pcts)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False)
        plt.tight_layout(rect=(0, 0.1, 1, 1))

        output_filepath = output_path / f"{model}_{subsplit}_joint.png"
        fig.savefig(output_filepath, dpi=200)
        plt.close(fig)
        log.info("Saved joint sweep plot to %s", output_filepath)


def plot_sample_selection_sweep_lineplot(config: DictConfig, log: Logger, output_path: Path) -> None:
    """For each (model, subsplit), creates a figure with one subplot per metric. Each subplot shows metric values across
    retention percentages for all sample selection strategies, plus a horizontal base-model reference line. Highlights
    best strategy, auto-scales y-axis, and adds confidence bands when available.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    output_path = output_path / "sample_selection_lineplots"
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_dataframes = _load_sample_selection_dataframes(config, log)
    if metrics_dataframes is None:
        return

    _plot_joint_sample_selection_sweep_lineplot(config, log, output_path, metrics_dataframes)


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------


def _build_strategy_column_index(
    config: DictConfig,
    metrics_dfs: dict[str, pd.DataFrame],
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], str]]:
    """Return ``(column_list, label_dict)`` where each column is a ``(strategy, file_key)`` pair.

    When multiple generator-model files cover the same strategy group, each gets its own column
    and the generator model abbreviation is appended to the label (e.g. ``KMeans-SC (WF)``).
    When only one file covers a strategy the label is just the base strategy name (e.g. ``DenTP``).
    """
    strategies = list(config.sample_selection_strategies_to_compare)

    strategy_to_files: dict[str, list[str]] = {}
    for strategy in strategies:
        group = STRATEGY_FILE_GROUP.get(strategy, strategy)
        strategy_to_files[strategy] = [k for k in metrics_dfs if k.split("/")[-1] == group]

    columns: list[tuple[str, str]] = []
    labels: dict[tuple[str, str], str] = {}

    for strategy in strategies:
        file_keys = strategy_to_files[strategy]
        base_label = str(STRATEGY_NAME_MAP.get(strategy, strategy))
        multi_source = len(file_keys) > 1
        for file_key in file_keys:
            col = (strategy, file_key)
            columns.append(col)
            if multi_source:
                parent = file_key.split("/")[0]
                abbrev = GENERATOR_MODEL_ABBREV.get(parent, parent)
                if abbrev is not None:
                    labels[col] = f"{base_label} ({abbrev})"
                else:
                    labels[col] = base_label
            else:
                labels[col] = base_label

    return columns, labels


def _lookup_strategy_value(df: pd.DataFrame, run_name_prefix: str, column: str) -> float | None:
    """Return the first non-NaN value for ``column`` in rows whose Name starts with ``run_name_prefix``."""
    matches = df[df["Name"].str.startswith(run_name_prefix)]
    if not matches.empty and column in matches.columns:
        val = matches.iloc[0][column]
        if pd.notna(val):
            return float(val)
    return None


# ---------------------------------------------------------------------------
# Heatmap functions
# ---------------------------------------------------------------------------


def _plot_sample_selection_sweep_heatmap(  # noqa: PLR0912, PLR0915
    config: DictConfig, log: Logger, output_path: Path, metrics_dfs: dict[str, pd.DataFrame], suffix: str = ""
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    """Plot heatmaps comparing sample selection strategies across retention percentages for each (model, split, metric).

    Columns are ``(strategy, generator_model)`` pairs derived from the file paths in
    ``config.sample_selection_files``. When multiple generator-model files cover the same strategy
    (e.g. ``wayformer/kmeans.csv`` and ``safe-controlledshifts/kmeans.csv``), each gets its own column
    with a disambiguating label such as ``KMeans-SC (WF)`` and ``KMeans-SC (SafeST)``.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
        metrics_dfs (dict[str, pd.DataFrame]): DataFrames keyed by ``"parent/stem"`` (e.g. ``"wayformer/kmeans"``).
        suffix (str): Suffix to append to output filenames.

    Returns:
        Nested dict ``metric → pct → strategy_label → {lightgray, black, blue}`` with star occurrence counts.
    """
    output_path = output_path / "sample_selection_heatmaps"
    output_path.mkdir(parents=True, exist_ok=True)

    cmap = sns.color_palette(config.get("heatmap_colormap", "mako_r"), as_cmap=True)
    metrics = config.trajectory_forecasting_metrics + config.other_metrics

    log.info("Plotting sample selection sweep heatmaps for metrics: %s", metrics)
    retention_pcts = list(map(float, config.sample_retention_percentages))
    models = list(config.models_to_compare)
    highlight_color = config.get("highlight_color", "dodgerblue")

    strategy_columns, col_labels = _build_strategy_column_index(config, metrics_dfs)
    n_cols = len(strategy_columns)
    num_rows = len(models)

    star_counts: dict[str, dict[str, dict[str, dict[str, int]]]] = {}

    for split in config.sample_selection_splits_to_compare:
        subsplit = split.split("/")[-1]
        log.info("Creating heatmap sweep plots for split=%s", split)

        for metric in metrics:
            column = f"{split}/{metric}" if metric in config.trajectory_forecasting_metrics else metric

            if metric not in star_counts:
                star_counts[metric] = {}

            heatmap_data = {}
            all_values = []

            for pct in retention_pcts:
                data = np.full((num_rows, n_cols), np.nan)
                for i, model in enumerate(models):
                    for j, (strategy, file_key) in enumerate(strategy_columns):
                        run_name_prefix = f"{config.sample_selection_benchmark}_{model}_{strategy}_{pct}"
                        val = _lookup_strategy_value(metrics_dfs[file_key], run_name_prefix, column)
                        if val is not None:
                            data[i, j] = val
                            all_values.append(val)
                heatmap_data[pct] = data

            base_data = np.full((num_rows, 1), np.nan)
            base_values: dict[int, float | None] = {}
            for i, model in enumerate(models):
                base_name = f"{config.sample_selection_benchmark}_{model}"
                for df in metrics_dfs.values():
                    row = df[df["Name"] == base_name]
                    if not row.empty and column in row.columns:
                        val = float(row[column].min())
                        base_data[i, 0] = val
                        base_values[i] = val
                        all_values.append(val)
                        break
                if i not in base_values:
                    base_values[i] = None

            if not all_values:
                log.warning("No data found for metric=%s, split=%s", metric, split)
                continue

            vmin = np.nanmin(all_values)
            vmax = np.nanmax(all_values)

            num_pcts = len(retention_pcts)
            fig_w = (
                config.heatmap_cell_size * (num_pcts * n_cols + 1)
                + config.heatmap_label_margin
                + config.heatmap_cbar_margin
            )
            fig_h = config.heatmap_cell_size * num_rows + config.heatmap_xtick_margin + config.heatmap_title_margin
            marker_size = 25
            fig, axes = plt.subplots(
                1,
                num_pcts + 1,
                figsize=(fig_w, fig_h),
                squeeze=False,
                gridspec_kw={"width_ratios": [n_cols] * num_pcts + [1]},
                layout="constrained",
            )
            fig.get_layout_engine().set(wspace=0.02, w_pad=0.01)  # type: ignore[union-attr]
            axes = axes[0]

            row_labels = [MODEL_NAME_MAP.get(m, m) for m in models]

            for k, (ax, pct) in enumerate(zip(axes[:num_pcts], retention_pcts, strict=False)):
                pct_label = f"{int(pct * 100)}%"
                if pct_label not in star_counts[metric]:
                    star_counts[metric][pct_label] = {
                        col_labels[col]: {"lightgray": 0, "black": 0, "blue": 0} for col in strategy_columns
                    }
                data = heatmap_data[pct]
                masked_data = np.ma.masked_invalid(data)
                im = ax.imshow(masked_data, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
                im.cmap.set_bad(color="#eeeeee")

                ax.set_title(pct_label, pad=6)
                ax.set_xticks(range(n_cols))
                ax.set_xticklabels(
                    [col_labels[col] for col in strategy_columns],
                    rotation=35,
                    ha="right",
                    rotation_mode="anchor",
                )

                if k == 0:
                    ax.set_yticks(range(num_rows))
                    ax.set_yticklabels(row_labels)
                    ax.tick_params(axis="y", pad=6)
                else:
                    ax.set_yticks([])

                ax.tick_params(which="minor", bottom=False, left=False)

                for i in range(data.shape[0]):
                    base_val = base_values.get(i)
                    if base_val is None:
                        continue
                    for j in range(data.shape[1]):
                        if not np.isnan(data[i, j]) and data[i, j] <= base_val:
                            ax.plot(j, i, marker="*", ms=marker_size, mec="lightgray", mew=1.5, c="none", zorder=5)
                            star_counts[metric][pct_label][col_labels[strategy_columns[j]]]["lightgray"] += 1

                for i in range(data.shape[0]):
                    row_data = data[i]
                    if np.all(np.isnan(row_data)):
                        continue

                    j = int(np.nanargmin(row_data))
                    best_val = row_data[j]
                    base_val = base_values.get(i)
                    edge_color, marker_color = "black", "black"
                    if base_val is not None and best_val <= base_val:
                        edge_color = highlight_color
                        marker_color = highlight_color

                    _key = "blue" if marker_color == highlight_color else "black"
                    star_counts[metric][pct_label][col_labels[strategy_columns[j]]][_key] += 1

                    if config.add_rectangle_annotation:
                        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=edge_color, linewidth=3))
                    ax.plot(j, i, marker="*", ms=marker_size, mec=marker_color, mew=1, c=marker_color, zorder=10)

            ax_base = axes[-1]
            masked_base = np.ma.masked_invalid(base_data)
            im = ax_base.imshow(masked_base, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
            im.cmap.set_bad(color="#eeeeee")
            ax_base.set_title("Baseline", pad=6)
            ax_base.set_xticks([])
            ax_base.set_yticks([])

            cbar = fig.colorbar(im, ax=list(axes), pad=0.02, shrink=0.7, fraction=0.08, aspect=8)
            cbar.ax.tick_params(labelsize=9)

            legend = [
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color=highlight_color,
                    markersize=12,
                    label="Best strategy in group ≥ baseline",
                ),
                Line2D([0], [0], marker="*", color="black", markersize=12, label="Best strategy < baseline"),
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="none",
                    mec="lightgray",
                    mew=1.5,
                    markersize=12,
                    label="Equals or beats baseline",
                ),
            ]

            output_file = output_path / f"{metric}_{subsplit}{suffix}.png"
            fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=9)
            fig.suptitle(f"{(metric[0].upper() + metric[1:]).replace('_', ' ')} — {SPLIT_NAME_MAP.get(split, split)}")
            fig.savefig(output_file, dpi=200, bbox_inches="tight")
            plt.close(fig)

            log.info("Saved heatmaps to %s", output_file)

    return star_counts


def _plot_sample_selection_sweep_heatmap_baseline_gap(  # noqa: PLR0912, PLR0915
    config: DictConfig,
    log: Logger,
    output_path: Path,
    metrics_dfs: dict[str, pd.DataFrame],
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    """Plot heatmaps showing % gap to baseline for each (model, split, metric, retention_pct, strategy).

    Columns are ``(strategy, generator_model)`` pairs; when multiple generator-model files cover the
    same strategy each gets its own column with a disambiguating label.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
        metrics_dfs (dict[str, pd.DataFrame]): Dictionary of DataFrames keyed by ``"parent/stem"``.

    Returns:
        Nested dict ``metric → pct → strategy_label → {black, blue}`` with star occurrence counts.
    """
    output_path = output_path / "sample_selection_heatmaps_baseline_gap"
    output_path.mkdir(parents=True, exist_ok=True)

    cmap = sns.color_palette(config.get("heatmap_colormap", "RdYlGn_r"), as_cmap=True)
    highlight_color = config.get("highlight_color", "dodgerblue")

    metrics = config.trajectory_forecasting_metrics + config.other_metrics
    retention_pcts = list(map(float, config.sample_retention_percentages))
    models = list(config.models_to_compare)
    splits = config.sample_selection_splits_to_compare
    log.info("Plotting sample selection sweep heatmaps (baseline gap) for metrics: %s", metrics)

    strategy_columns, col_labels = _build_strategy_column_index(config, metrics_dfs)
    num_rows = len(models)
    num_retention_pcts = len(retention_pcts)
    n_cols = len(strategy_columns)

    star_counts: dict[str, dict[str, dict[str, dict[str, int]]]] = {}

    for split, metric in product(splits, metrics):
        subsplit = split.split("/")[-1]
        column = f"{split}/{metric}" if metric in config.trajectory_forecasting_metrics else metric
        log.info("Creating baseline-gap heatmap plots for split=%s metric=%s", split, metric)

        if metric not in star_counts:
            star_counts[metric] = {}

        base_vals_per_model: dict[str, float | None] = {}
        for model in models:
            base_name = f"{config.sample_selection_benchmark}_{model}"
            for df in metrics_dfs.values():
                row = df[df["Name"] == base_name]
                if not row.empty and column in row.columns:
                    base_vals_per_model[model] = float(row[column].min())
                    break
            if model not in base_vals_per_model:
                base_vals_per_model[model] = None

        heatmap_data = {}
        all_gaps: list[float] = []

        for pct in retention_pcts:
            data = np.full((num_rows, n_cols), np.nan)
            for i, model in enumerate(models):
                base_val = base_vals_per_model.get(model)
                if base_val is None:
                    continue
                for j, (strategy, file_key) in enumerate(strategy_columns):
                    run_name_prefix = f"{config.sample_selection_benchmark}_{model}_{strategy}_{pct}"
                    val = _lookup_strategy_value(metrics_dfs[file_key], run_name_prefix, column)
                    if val is not None:
                        gap = float(_relative_gap_pct(val, base_val))
                        data[i, j] = gap
                        all_gaps.append(gap)
            heatmap_data[pct] = data

        if not all_gaps:
            log.warning("No data found for metric=%s, split=%s", metric, split)
            continue

        vmin, vmax = _symmetric_vrange(all_gaps)

        fig_w = (
            config.heatmap_cell_size * num_retention_pcts * n_cols
            + config.heatmap_label_margin
            + config.heatmap_cbar_margin
        )
        fig_h = config.heatmap_cell_size * num_rows + config.heatmap_xtick_margin + config.heatmap_title_margin
        fig, axes = plt.subplots(1, num_retention_pcts, figsize=(fig_w, fig_h), squeeze=False, layout="constrained")
        fig.get_layout_engine().set(wspace=0.02, w_pad=0.01)  # type: ignore[union-attr]
        axes = axes[0]
        marker_size = 25
        row_labels = [MODEL_NAME_MAP.get(m, m) for m in models]

        im = None
        for k, (ax, pct) in enumerate(zip(axes, retention_pcts, strict=False)):
            pct_label = f"{int(pct * 100)}%"
            if pct_label not in star_counts[metric]:
                star_counts[metric][pct_label] = {col_labels[col]: {"black": 0, "blue": 0} for col in strategy_columns}
            data = heatmap_data[pct]
            masked_data = np.ma.masked_invalid(data)
            im = ax.imshow(masked_data, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
            im.cmap.set_bad(color="#eeeeee")

            ax.set_title(pct_label, pad=6)
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(
                [col_labels[col] for col in strategy_columns], rotation=35, ha="right", rotation_mode="anchor"
            )
            if k == 0:
                ax.set_yticks(range(num_rows))
                ax.set_yticklabels(row_labels)
                ax.tick_params(axis="y", pad=6)
            else:
                ax.set_yticks([])
            ax.tick_params(which="minor", bottom=False, left=False)

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if not np.isnan(data[i, j]) and data[i, j] <= 0:
                        ax.plot(j, i, marker="*", ms=marker_size, mec="black", alpha=0.5, mew=1, c="none", zorder=5)

            for i in range(data.shape[0]):
                row_data = data[i]
                if np.all(np.isnan(row_data)):
                    continue
                j = int(np.nanargmin(row_data))
                gap_val = row_data[j]
                marker_color = highlight_color if gap_val < 0 else "black"
                _key = "blue" if marker_color == highlight_color else "black"
                star_counts[metric][pct_label][col_labels[strategy_columns[j]]][_key] += 1
                if config.add_rectangle_annotation:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=marker_color, linewidth=3))
                ax.plot(j, i, marker="*", ms=marker_size, mec=marker_color, mew=1, c=marker_color, zorder=10)

        if im is not None:
            cbar = fig.colorbar(im, ax=list(axes), pad=0.02, shrink=0.7, fraction=0.08, aspect=8)
            cbar.ax.tick_params(labelsize=9)
            cbar.set_label("Gap to Baseline (%)", fontsize=9)

        legend = [
            Line2D([0], [0], marker="*", color=highlight_color, markersize=10, label="Best strategy beats baseline"),
            Line2D([0], [0], marker="*", color="black", markersize=10, label="Best strategy in group"),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                mec="black",
                alpha=0.1,
                mew=1.5,
                markersize=8,
                label="Equals or beats baseline",
            ),
        ]

        output_file = output_path / f"{metric}_{subsplit}.png"
        fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=9)
        metric_label = (metric[0].upper() + metric[1:]).replace("_", " ")
        fig.suptitle(f"Gap to Baseline — {metric_label} ({SPLIT_NAME_MAP.get(split, split)})")
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)

        log.info("Saved heatmaps to %s", output_file)

    return star_counts


def _plot_sample_selection_sweep_distribution_gap(  # noqa: PLR0912, PLR0915
    config: DictConfig,
    log: Logger,
    output_path: Path,
    metrics_dfs: dict[str, pd.DataFrame],
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    """Plot heatmaps showing the OOD-ID split gap for each (model, metric, retention_pct, strategy).

    Columns are ``(strategy, generator_model)`` pairs; when multiple generator-model files cover the
    same strategy each gets its own column with a disambiguating label.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
        metrics_dfs (dict[str, pd.DataFrame]): Dictionary of DataFrames keyed by ``"parent/stem"``.

    Returns:
        Nested dict ``metric → pct → strategy_label → {black, blue, magenta}`` with star occurrence counts.
    """
    splits = config.sample_selection_splits_to_compare
    if len(splits) < 2:  # noqa: PLR2004
        log.warning("Need at least two splits to compute distribution gap, got: %s", splits)
        return {}

    id_split, ood_split = splits[0], splits[1]
    id_subsplit = id_split.split("/")[-1]
    ood_subsplit = ood_split.split("/")[-1]

    output_path = output_path / "sample_selection_heatmaps_distribution_gap"
    output_path.mkdir(parents=True, exist_ok=True)

    cmap = sns.color_palette(config.get("heatmap_colormap", "RdYlGn_r"), as_cmap=True)
    highlight_color = config.get("highlight_color", "dodgerblue")

    metrics = config.trajectory_forecasting_metrics
    retention_pcts = list(map(float, config.sample_retention_percentages))
    models = list(config.models_to_compare)
    log.info("Plotting sample selection sweep heatmaps (distribution gap) for metrics: %s", metrics)

    strategy_columns, col_labels = _build_strategy_column_index(config, metrics_dfs)
    num_rows = len(models)
    num_retention_pcts = len(retention_pcts)
    n_cols = len(strategy_columns)

    star_counts: dict[str, dict[str, dict[str, dict[str, int]]]] = {metric: {} for metric in metrics}

    for metric in metrics:
        id_column = f"{id_split}/{metric}"
        ood_column = f"{ood_split}/{metric}"
        log.info("Creating distribution gap heatmaps for metric=%s (%s vs %s)", metric, id_split, ood_split)

        baseline_gaps: dict[str, float | None] = {}
        baseline_id_values: dict[str, float | None] = {}
        for model in models:
            base_name = f"{config.sample_selection_benchmark}_{model}"
            for df in metrics_dfs.values():
                base_row = df[df["Name"] == base_name]
                if not base_row.empty and id_column in base_row.columns and ood_column in base_row.columns:
                    id_val = float(base_row.iloc[0][id_column])
                    ood_val = float(base_row.iloc[0][ood_column])
                    baseline_gaps[model] = float(_relative_gap_pct(ood_val, id_val))
                    baseline_id_values[model] = id_val
                    break
            if model not in baseline_gaps:
                baseline_gaps[model] = None
                baseline_id_values[model] = None

        heatmap_data: dict[float, np.ndarray] = {}
        all_gaps: list[float] = []

        for pct in retention_pcts:
            data = np.full((num_rows, n_cols), np.nan)
            for i, model in enumerate(models):
                for j, (strategy, file_key) in enumerate(strategy_columns):
                    run_name_prefix = f"{config.sample_selection_benchmark}_{model}_{strategy}_{pct}"
                    id_val = _lookup_strategy_value(metrics_dfs[file_key], run_name_prefix, id_column)
                    ood_val = _lookup_strategy_value(metrics_dfs[file_key], run_name_prefix, ood_column)
                    if id_val is not None and ood_val is not None:
                        gap = float(_relative_gap_pct(ood_val, id_val))
                        data[i, j] = gap
                        all_gaps.append(gap)
            heatmap_data[pct] = data

        if not all_gaps:
            log.warning("No data found for metric=%s (%s vs %s)", metric, id_split, ood_split)
            continue

        vmin, vmax = _symmetric_vrange(all_gaps)

        fig_w = (
            config.heatmap_cell_size * num_retention_pcts * n_cols
            + config.heatmap_label_margin
            + config.heatmap_cbar_margin
        )
        fig_h = config.heatmap_cell_size * num_rows + config.heatmap_xtick_margin + config.heatmap_title_margin
        fig, axes = plt.subplots(1, num_retention_pcts, figsize=(fig_w, fig_h), squeeze=False, layout="constrained")
        fig.get_layout_engine().set(wspace=0.02, w_pad=0.01)  # type: ignore[union-attr]
        axes = axes[0]

        row_labels = [MODEL_NAME_MAP.get(m, m) for m in models]

        im = None
        for k, (ax, pct) in enumerate(zip(axes, retention_pcts, strict=False)):
            pct_label = f"{int(pct * 100)}%"
            if pct_label not in star_counts[metric]:
                star_counts[metric][pct_label] = {
                    col_labels[col]: {"black": 0, "blue": 0, "magenta": 0} for col in strategy_columns
                }
            data = heatmap_data[pct]
            masked_data = np.ma.masked_invalid(data)
            im = ax.imshow(masked_data, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
            im.cmap.set_bad(color="#eeeeee")

            ax.set_title(pct_label, pad=6)
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(
                [col_labels[col] for col in strategy_columns], rotation=35, ha="right", rotation_mode="anchor"
            )
            if k == 0:
                ax.set_yticks(range(num_rows))
                ax.set_yticklabels(row_labels)
                ax.tick_params(axis="y", pad=6)
            else:
                ax.set_yticks([])
            ax.tick_params(which="minor", bottom=False, left=False)

            for i, model in enumerate(models):
                baseline_gap = baseline_gaps.get(model)
                if baseline_gap is None:
                    continue
                for j in range(data.shape[1]):
                    if not np.isnan(data[i, j]) and data[i, j] <= baseline_gap:
                        ax.plot(j, i, marker="*", ms=18, mec="black", alpha=0.5, mew=1, c="none", zorder=5)

            for i, model in enumerate(models):
                row_data = data[i]
                if np.all(np.isnan(row_data)):
                    continue

                baseline_gap = baseline_gaps.get(model)
                baseline_id_val = baseline_id_values.get(model)
                j = int(np.nanargmin(np.abs(row_data)))
                gap_val = row_data[j]

                best_strategy, best_file_key = strategy_columns[j]
                run_name_prefix = f"{config.sample_selection_benchmark}_{model}_{best_strategy}_{pct}"
                strategy_id_val = _lookup_strategy_value(metrics_dfs[best_file_key], run_name_prefix, id_column)

                if (
                    baseline_gap is not None
                    and baseline_id_val is not None
                    and strategy_id_val is not None
                    and gap_val < baseline_gap
                    and strategy_id_val < baseline_id_val
                ):
                    marker_color = "magenta"
                elif baseline_gap is not None and gap_val < baseline_gap:
                    marker_color = highlight_color
                else:
                    marker_color = "black"

                if marker_color == "magenta":
                    _star_key = "magenta"
                elif marker_color == highlight_color:
                    _star_key = "blue"
                else:
                    _star_key = "black"
                star_counts[metric][pct_label][col_labels[strategy_columns[j]]][_star_key] += 1

                if config.add_rectangle_annotation:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=marker_color, linewidth=3))
                ax.plot(j, i, marker="*", ms=18, mec=marker_color, mew=1, c=marker_color, zorder=10)

        if im is not None:
            cbar = fig.colorbar(im, ax=list(axes), pad=0.02, shrink=0.7, fraction=0.08, aspect=8)
            cbar.ax.tick_params(labelsize=9)
            cbar.set_label(
                f"Gap ({SPLIT_NAME_MAP.get(ood_split, ood_subsplit)} - {SPLIT_NAME_MAP.get(id_split, id_subsplit)}) %",
                fontsize=9,
            )

        magenta_label = "Better performance and gap than baseline"
        highlight_label = "Better gap than baseline"
        black_label = "Best in group, not better than baseline"
        equal_tag = "Equals or beats baseline gap"
        legend = [
            Line2D([0], [0], marker="*", color="magenta", linestyle="None", markersize=10, label=magenta_label),
            Line2D([0], [0], marker="*", color=highlight_color, linestyle="None", markersize=10, label=highlight_label),
            Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=10, label=black_label),
            Line2D([0], [0], marker="*", color="none", mec="black", alpha=0.1, mew=1.5, markersize=9, label=equal_tag),
        ]

        output_file = output_path / f"{metric}_{id_subsplit}_vs_{ood_subsplit}.png"
        fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=7)
        fig.suptitle(
            f"Split Gap — {(metric[0].upper() + metric[1:]).replace('_', ' ')}"
            f" ({SPLIT_NAME_MAP.get(ood_split, ood_subsplit)} - {SPLIT_NAME_MAP.get(id_split, id_subsplit)})"
        )
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)

        log.info("Saved distribution gap heatmaps to %s", output_file)

    return star_counts


def _aggregate_star_counts(
    all_counts: list[dict[str, dict[str, dict[str, dict[str, int]]]]],
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    """Merge ``metric → pct → strategy → counts`` dicts from multiple heatmap functions.

    Args:
        all_counts: List of per-function star-count dicts.

    Returns:
        Single merged dict with the same structure, counts summed across sources.
    """
    result: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for counts in all_counts:
        for metric, pct_dict in counts.items():
            if metric not in result:
                result[metric] = {}
            for pct, strategy_dict in pct_dict.items():
                if pct not in result[metric]:
                    result[metric][pct] = {}
                for strategy, star_dict in strategy_dict.items():
                    if strategy not in result[metric][pct]:
                        result[metric][pct][strategy] = {
                            "lightgray": 0,
                            "black": 0,
                            "blue": 0,
                            "magenta": 0,
                            "best_in_group": 0,
                        }
                    for star_type, n in star_dict.items():
                        result[metric][pct][strategy][star_type] = result[metric][pct][strategy].get(star_type, 0) + n
    for pct_dict in result.values():
        for strategy_dict in pct_dict.values():
            for star_dict in strategy_dict.values():
                star_dict["best_in_group"] = (
                    star_dict.get("black", 0) + star_dict.get("blue", 0) + star_dict.get("magenta", 0)
                )
    return result


def _sum_star_counts(
    groups: Iterable[dict[str, dict[str, int]]],
) -> dict[str, dict[str, int]]:
    """Sum ``strategy → counts`` dicts from an iterable of groups."""
    result: dict[str, dict[str, int]] = {}
    for strategy_dict in groups:
        for strategy, star_dict in strategy_dict.items():
            if strategy not in result:
                result[strategy] = {}
            for star_type, n in star_dict.items():
                result[strategy][star_type] = result[strategy].get(star_type, 0) + n
    return result


def _sum_over_pcts(
    counts: dict[str, dict[str, dict[str, dict[str, int]]]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Collapse the pct dimension → ``metric → strategy → counts``."""
    return {metric: _sum_star_counts(pct_dict.values()) for metric, pct_dict in counts.items()}


def _sum_over_metrics(
    counts: dict[str, dict[str, dict[str, dict[str, int]]]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Collapse the metric dimension → ``pct → strategy → counts``."""
    result: dict[str, dict[str, dict[str, int]]] = {}
    for pct_dict in counts.values():
        for pct, strategy_dict in pct_dict.items():
            if pct not in result:
                result[pct] = {}
            for strategy, star_dict in strategy_dict.items():
                if strategy not in result[pct]:
                    result[pct][strategy] = {}
                for star_type, n in star_dict.items():
                    result[pct][strategy][star_type] = result[pct][strategy].get(star_type, 0) + n
    return result


def _sum_over_all(
    counts: dict[str, dict[str, dict[str, dict[str, int]]]],
) -> dict[str, dict[str, int]]:
    """Collapse both metric and pct dimensions → ``strategy → counts`` (holistic)."""
    return _sum_star_counts(strategy_dict for pct_dict in counts.values() for strategy_dict in pct_dict.values())


def _format_strategy_summary_text(
    counts: dict[str, dict[str, dict[str, dict[str, int]]]],
) -> str:
    """Format star-count summaries as a fixed-width text table.

    Generates one section for holistic totals, one per metric, and one per retention percentage.

    Args:
        counts: ``metric → pct → strategy → star_counts`` (from :func:`_aggregate_star_counts`).

    Returns:
        Multi-section table string.
    """
    cols = ["best_in_group", "black", "lightgray", "blue", "magenta"]
    headers = ["Best", "Black", "Lightgray", "Blue", "Magenta"]

    def _table(summary: dict[str, dict[str, int]], label: str) -> str:
        sorted_strats = sorted(summary, key=lambda s: -summary[s].get("best_in_group", 0))
        max_name = max((len(s) for s in sorted_strats), default=8)
        col_w = [max(len(h), 5) for h in headers]
        header_row = f"{'Strategy':<{max_name}} | " + " | ".join(
            f"{h:>{w}}" for h, w in zip(headers, col_w, strict=True)
        )
        sep = "-" * len(header_row)
        rows = [f"=== Strategy Summary [{label}] ===", header_row, sep]
        for strat in sorted_strats:
            vals = summary[strat]
            row = f"{strat:<{max_name}} | " + " | ".join(
                f"{vals.get(c, 0):>{w}}" for c, w in zip(cols, col_w, strict=True)
            )
            rows.append(row)
        return "\n".join(rows)

    holistic = _sum_over_all(counts)
    per_metric = _sum_over_pcts(counts)
    per_pct = _sum_over_metrics(counts)
    sorted_pcts = sorted(per_pct, key=lambda p: int(p.rstrip("%")))

    sections = [_table(holistic, "all")]
    sections.extend(_table(per_metric[m], m) for m in sorted(per_metric))
    sections.extend(_table(per_pct[p], p) for p in sorted_pcts)
    return "\n\n".join(sections)


def _plot_strategy_summary(
    counts: dict[str, dict[str, dict[str, dict[str, int]]]],
    output_path: Path,
    benchmark_name: str = "",
) -> None:
    """Save two grouped horizontal bar charts — one per-metric figure and one per-retention-pct figure.

    Args:
        counts: ``metric → pct → strategy → star_counts`` (from :func:`_aggregate_star_counts`).
        output_path: Directory in which to save the summary PNGs.
        benchmark_name: Short benchmark label included in filenames and figure titles.
    """
    star_types = ["best_in_group", "black", "lightgray", "blue", "magenta"]
    star_colors = {
        "best_in_group": "#888888",
        "black": "#222222",
        "lightgray": "#cccccc",
        "blue": "dodgerblue",
        "magenta": "magenta",
    }
    star_display = {
        "best_in_group": "Best",
        "black": "Black",
        "lightgray": "Lightgray",
        "blue": "Blue",
        "magenta": "Magenta",
    }

    holistic = _sum_over_all(counts)
    per_metric = _sum_over_pcts(counts)
    per_pct = _sum_over_metrics(counts)
    sorted_pcts = sorted(per_pct, key=lambda p: int(p.rstrip("%")))

    strategies = sorted(holistic, key=lambda s: holistic[s].get("best_in_group", 0))
    stem = f"strategy_summary_{benchmark_name}" if benchmark_name else "strategy_summary"
    output_path.mkdir(parents=True, exist_ok=True)

    def _save_panels(
        panels: list[tuple[str, dict[str, dict[str, int]]]],
        filename: str,
        title: str,
    ) -> None:
        n_panels = len(panels)
        n_strats = len(strategies)
        bar_height = 0.12
        group_gap = 0.15
        group_h = len(star_types) * bar_height + group_gap
        fig_h = max(4.0, n_strats * group_h + 1.5)
        fig_w = max(6.0, n_panels * 3.5)

        fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), sharey=True, layout="constrained")
        if n_panels == 1:
            axes = [axes]  # type: ignore[assignment]

        y_centers = np.arange(n_strats) * group_h
        offsets = np.linspace(-(len(star_types) - 1) / 2, (len(star_types) - 1) / 2, len(star_types)) * bar_height

        for ax, (label, panel_data) in zip(axes, panels, strict=True):
            for st_idx, star_type in enumerate(star_types):
                values = [panel_data.get(s, {}).get(star_type, 0) for s in strategies]
                ax.barh(
                    y_centers + offsets[st_idx],
                    values,
                    height=bar_height,
                    color=star_colors[star_type],
                    edgecolor="white",
                    linewidth=0.3,
                )
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("Count", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

        axes[0].set_yticks(y_centers)
        axes[0].set_yticklabels(strategies, fontsize=8)

        handles = [plt.Rectangle((0, 0), 1, 1, color=star_colors[st], label=star_display[st]) for st in star_types]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
        fig.suptitle(title, fontsize=11)
        fig.savefig(output_path / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    bn_suffix = f" — {benchmark_name}" if benchmark_name else ""
    _save_panels(
        [("all", holistic), *[(m, per_metric[m]) for m in sorted(per_metric)]],
        f"{stem}_metrics.png",
        f"Strategy Star Summary (per metric){bn_suffix}",
    )
    _save_panels(
        [("all", holistic), *[(p, per_pct[p]) for p in sorted_pcts]],
        f"{stem}_pcts.png",
        f"Strategy Star Summary (per retention %){bn_suffix}",
    )


def plot_sample_selection_sweep_heatmap(
    config: DictConfig, log: Logger, output_path: Path
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    """Creates heatmaps comparing sample selection sweeps for each (model, split, retention_percentage, metric).

    For each split and metric, generates P heatmaps (one per retention percentage) with rows as models, columns as
    strategies, and color representing metric values. Also generates baseline gap heatmaps when multiple dataframes are
    available. Saves text tables and bar charts summarizing per-strategy star counts per metric, per retention %, and
    holistically.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.

    Returns:
        Combined ``metric → pct → strategy_label → {best_in_group, black, lightgray, blue, magenta}`` star counts.
    """
    plt.rcParams.update(
        {
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 14,
            "axes.grid": False,
        }
    )

    metrics_dataframes = _load_sample_selection_dataframes(config, log)
    if metrics_dataframes is None:
        return {}

    main_counts = _plot_sample_selection_sweep_heatmap(config, log, output_path, metrics_dataframes)
    all_raw_counts = [main_counts]

    gap_counts = None
    dist_counts = None
    # If multiple metrics files are available, create heatmaps showing gap to baseline across selectors
    if len(metrics_dataframes) > 1:
        gap_counts = _plot_sample_selection_sweep_heatmap_baseline_gap(config, log, output_path, metrics_dataframes)
        dist_counts = _plot_sample_selection_sweep_distribution_gap(config, log, output_path, metrics_dataframes)
        all_raw_counts += [gap_counts, dist_counts]

    benchmark_name = BENCHMARK_NAME_MAP.get(
        config.get("sample_selection_benchmark", ""), config.get("sample_selection_benchmark", "")
    )

    def _save_summary(
        raw_counts: list[dict[str, dict[str, dict[str, dict[str, int]]]]], out_dir: Path, label: str
    ) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
        agg = _aggregate_star_counts(raw_counts)
        stem = f"strategy_summary_{benchmark_name}" if benchmark_name else "strategy_summary"
        text = _format_strategy_summary_text(agg)
        log.info("Strategy summary [%s]:\n%s", label, text)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{stem}.txt").write_text(text)
        _plot_strategy_summary(agg, out_dir, benchmark_name or "")
        log.info("Saved strategy summary [%s] to %s", label, out_dir)
        return agg

    combined = _save_summary(all_raw_counts, output_path / "sample_selection_heatmaps", "combined")
    if len(metrics_dataframes) > 1 and gap_counts is not None and dist_counts is not None:
        _save_summary([gap_counts], output_path / "sample_selection_heatmaps_baseline_gap", "baseline_gap")
        _save_summary([dist_counts], output_path / "sample_selection_heatmaps_distribution_gap", "distribution_gap")

    return combined


# ---------------------------------------------------------------------------
# Benchmark analysis
# ---------------------------------------------------------------------------


def _plot_distribution_shift_comparison(
    summary_df: pd.DataFrame, output_path: Path, colormap: str, id_metric: str, ood_metric: str
) -> None:
    """Plots a comparison of In-Distribution (ID) vs Out-of-Distribution (OOD) performance for different models,
    highlighting the performance gaps.

    Args:
        summary_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        output_path (Path): Directory to save the generated plot.
        colormap (str): Name of the matplotlib colormap to use for consistent coloring.
        id_metric (str): Name of the ID metric.
        ood_metric (str): Name of the OOD metric.
    """
    assert id_metric in summary_df.columns, f"ID metric '{id_metric}' not found in summary_df columns"
    assert ood_metric in summary_df.columns, f"OOD metric '{ood_metric}' not found in summary_df columns"

    palette = sns.color_palette(colormap, len(summary_df))
    models = summary_df["Model"].to_numpy()

    def _plot_bars(ax: Axes, metric: str, title: str) -> None:
        values = summary_df[metric].to_numpy()
        bars = ax.bar(models, values, color=palette, alpha=0.8, edgecolor="black", linewidth=1.5)

        ax.set_ylabel(metric, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9, rotation=30)

        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                x = bar.get_x() + bar.get_width() / 2.0
                ax.text(x, height, f"{height:.3f}", ha="center", va="bottom", fontsize=10)
        ax.yaxis.grid(visible=True, alpha=0.3)

        _set_yaxis_limits(ax, list(values), padding_factor=0.15, lower_factor=0.4, min_padding=0.1)

    n_models = models.shape[0]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.5 * n_models * 3, 6))
    fig.suptitle("Distribution Shift Analysis", fontsize=14, fontweight="bold")

    _plot_bars(ax1, id_metric, "In-Distribution (ID) Performance")
    _plot_bars(ax2, ood_metric, "Out-of-Distribution (OOD) Performance")

    # Performance Gap (OOD - ID, relative %)
    id_values = summary_df[id_metric].to_numpy()
    ood_values = summary_df[ood_metric].to_numpy()
    gap_values: NDArray = _relative_gap_pct(ood_values, id_values)  # pyright: ignore[reportArgumentType, reportAssignmentType]

    gap_colors = ["#f07569" if gap > 0 else "#7cbf7c" for gap in gap_values]
    bars = ax3.bar(models, gap_values, color=gap_colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
    ax3.set_ylabel("Performance Gap (OOD - ID)", fontsize=11, fontweight="bold")
    ax3.set_title("Generalization Gap", fontsize=12, fontweight="bold")
    ax3.tick_params(axis="x", labelsize=12, rotation=30)

    for bar, gap in zip(bars, gap_values, strict=False):
        height = bar.get_height()
        if not np.isnan(height):
            va = "bottom" if height > 0 else "top"
            x = bar.get_x() + bar.get_width() / 2.0
            ax3.text(x, height, f"{gap:.3f}", ha="center", va=va, fontsize=8, fontweight="bold")
    ax3.yaxis.grid(visible=True, alpha=0.3)

    plt.tight_layout()
    output_file = output_path / "distribution_shift_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Plot saved as '{output_file}'")


def _plot_benchmark_comparison(
    summary_df: pd.DataFrame, metrics: dict[str, str], output_path: Path, colormap: str
) -> None:
    """Plots a benchmark comparison across different models for specified metrics.

    Args:
        summary_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        metrics (dict[str, str]): Dictionary mapping metric column names to display names.
        output_path (Path): Directory to save the generated plot.
        colormap (str): Name of the matplotlib colormap to use for consistent coloring.
    """
    num_metrics = len(metrics)
    n_models = summary_df["Model"].shape[0]
    n_cols = min(2, num_metrics)
    n_rows = math.ceil(num_metrics / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_models * n_cols, 4.0 * n_rows), constrained_layout=True)
    fig.suptitle("Model Performance Comparison", fontsize=20, fontweight="bold")

    axes = np.atleast_1d(axes).flatten()

    palette = sns.color_palette(colormap, len(summary_df))
    model_order = summary_df["Model"].to_numpy()

    for idx, metric_name in enumerate(metrics.values()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        values = summary_df[metric_name].to_numpy()
        bars = ax.bar(model_order, values, color=palette, edgecolor="black", linewidth=1.0, alpha=0.8)

        ax.set_title(metric_name, pad=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_axisbelow(True)

        _set_yaxis_limits(ax, list(values), padding_factor=0.15, lower_factor=0.4, min_padding=0.1)

        # Value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="medium",
                )

        # Highlight best model
        best_idx = np.nanargmin(values) if "↓" in metric_name else np.nanargmax(values)
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(4)

    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    output_file = output_path / "benchmark_comparison.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"\n✓ Plot saved as '{output_file}'")


def _plot_performance_gaps(
    summary_df: pd.DataFrame, output_path: Path, metric_pairs: list[tuple[str, str, str]]
) -> None:
    """Plots comprehensive performance gaps (absolute and percentage) between OOD and ID metrics for multiple metrics.

    Args:
        summary_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        output_path (Path): Directory to save the generated plot.
        metric_pairs (list[tuple[str, str, str]]): List of tuples containing (ID metric column name, OOD metric column
            name, metric display name).
    """
    gap_data = {}
    for id_col, ood_col, metric_name in metric_pairs:
        if id_col in summary_df.columns and ood_col in summary_df.columns:
            id_vals = summary_df[id_col].to_numpy()
            ood_vals = summary_df[ood_col].to_numpy()
            ood_id_diff = ood_vals - id_vals
            gap_data[metric_name] = {
                "absolute": ood_id_diff,
                "percent": _relative_gap_pct(ood_vals, id_vals),
            }

    if gap_data:
        num_metrics = len(gap_data)
        num_models = summary_df["Model"].shape[0]
        horizontal_size = num_models * num_metrics * 1.5
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(horizontal_size, 6))
        fig.suptitle("Performance Gaps (OOD - ID)", fontsize=14, fontweight="bold")
        x = np.arange(len(summary_df))
        width = 0.25
        for i, (metric_name, gaps) in enumerate(gap_data.items()):
            offset = width * i
            ax1.bar(x + offset, gaps["absolute"], width, label=metric_name, alpha=0.8, edgecolor="black", linewidth=1)
            ax2.bar(x + offset, gaps["percent"], width, label=metric_name, alpha=0.8, edgecolor="black", linewidth=1)

        ax1.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
        ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Absolute Gap (OOD - ID)", fontsize=12, fontweight="bold")
        ax1.set_title("Absolute Performance Gaps\n(Positive = OOD performs worse)", fontsize=12, fontweight="bold")
        ax1.set_xticks(x + (num_metrics - 1) * width)
        ax1.set_xticklabels(summary_df["Model"].values, ha="right", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.yaxis.grid(visible=True, alpha=0.3)
        ax1.set_axisbelow(True)

        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
        ax2.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Percentage Gap (%)", fontsize=12, fontweight="bold")
        ax2.set_title(
            "Percentage Performance Gaps\n(Positive = OOD worse, % relative to ID)", fontsize=12, fontweight="bold"
        )
        ax2.set_xticks(x + (num_metrics - 1) * width)
        ax2.set_xticklabels(summary_df["Model"].values, ha="right", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.yaxis.grid(visible=True, alpha=0.3)
        ax2.set_axisbelow(True)

        plt.tight_layout()
        output_file = output_path / "performance_gaps.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved as '{output_file}'")

        # Print gap statistics
        print("\n" + "=" * 80)
        print("Performance Gap Analysis (OOD - ID):")
        print("=" * 80)
        for metric_name, gaps in gap_data.items():
            abs_gaps = gaps["absolute"]
            pct_gaps = gaps["percent"]
            print(f"\n{metric_name}:")
            for i, model in enumerate(summary_df["Model"].values):
                print(f"  {model:30s}: {abs_gaps[i]:+.4f} (Absolute) | {pct_gaps[i]:+.2f}% (Relative)")
            print(f"  Average Gap: {np.mean(abs_gaps):+.4f} | {np.mean(pct_gaps):+.2f}%")
            print(f"  Max Gap:     {np.max(abs_gaps):+.4f} | {np.max(pct_gaps):+.2f}%")


def _plot_grouped_bar_chart(
    summary_df: pd.DataFrame, metrics: dict[str, str], output_path: Path, key_metrics_display: list[str]
) -> None:
    """Plots a grouped bar chart comparing multiple key metrics across different models.

    Args:
        summary_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        metrics (dict[str, str]): Dictionary mapping metric column names to display names.
        output_path (Path): Directory to save the generated plot.
        key_metrics_display (list[str]): List of key metric column names to include in the grouped bar chart.
    """
    _fig, ax = plt.subplots(figsize=(14, 7))

    available_metrics = [m for m in key_metrics_display if m in summary_df.columns]

    if available_metrics:
        x = np.arange(len(summary_df))
        width = 0.2
        all_values: list[float] = []
        for i, metric in enumerate(available_metrics):
            values = summary_df[metric].to_numpy()
            all_values.extend(values)
            ax.bar(x + width * i, values, width, label=metric, alpha=0.8, edgecolor="black", linewidth=1)

        _set_yaxis_limits(ax, all_values, padding_factor=0.15, lower_factor=0.4, min_padding=0.1)

        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Metric Value", fontsize=12, fontweight="bold")
        ax.set_title("Multi-Metric Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
        ax.set_xticklabels(summary_df["Model"].values, rotation=35, ha="right")
        ax.legend(loc="upper left", fontsize=10)
        ax.yaxis.grid(visible=True, alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        output_file = output_path / "grouped_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved as '{output_file}'")

    # Print best performing model for each metric
    print("\n" + "=" * 80)
    print("Best Performing Models (Lower is Better):")
    print("=" * 80)
    for metric_name in metrics.values():
        if metric_name in summary_df.columns:
            best_idx = summary_df[metric_name].idxmin()
            if pd.notna(best_idx):
                best_model = summary_df.loc[best_idx, "Model"]
                best_value = summary_df.loc[best_idx, metric_name]
                print(f"{metric_name:30s}: {best_model:30s} ({best_value:.4f})")


def run_benchmark_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Plots multiple In-Distribution (ID) vs Out-of-Distribution (OOD) benchmark analyses based on a CSV file
    containing model metrics.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    output_path = output_path / config.benchmark
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metrics CSV
    metrics_filepath = Path(config.benchmark_filepath)
    if not metrics_filepath.exists():
        log.error("Metrics file not found at %s", metrics_filepath)
        return
    metrics_df = pd.read_csv(metrics_filepath)

    if "Name" not in metrics_df.columns:
        log.error("CSV must contain a 'Name' column")
        return
    if "ID" not in metrics_df.columns:
        metrics_df["ID"] = np.arange(len(metrics_df))

    benchmark_df = metrics_df[metrics_df["Name"].str.contains(config.benchmark, na=False)].copy()
    print(f"Experiments on {config.benchmark}:")
    print(benchmark_df[["Name", "State"]].to_string(index=False))
    print(f"\nTotal experiments found: {len(benchmark_df)}")

    # Extract model names
    benchmark_df["model_name"] = benchmark_df["Name"].str.replace(f"{config.benchmark}_", "")
    benchmark_df["model_name"] = benchmark_df["model_name"].map(lambda x: MODEL_NAME_MAP.get(str(x), str(x)))  # pyright: ignore[reportUnknownLambdaType]
    if config.show_run_id:
        benchmark_df["Model"] = benchmark_df["model_name"].astype(str) + "[" + benchmark_df["ID"].astype(str) + "]"
    else:
        benchmark_df["Model"] = benchmark_df["model_name"].astype(str)

    # Key metrics to compare
    id_split, ood_split = config.benchmark_splits_to_compare
    id_split_name = id_split.split("/")[-1]
    ood_split_name = ood_split.split("/")[-1]
    metrics = {
        f"{split}/{metric}": f"{metric} ({split.split('/')[-1]},↓)"
        for metric, split in product(
            config.trajectory_forecasting_metrics,
            config.benchmark_splits_to_compare,
        )
    }
    log.info("Comparing splits: %s vs %s", id_split, ood_split)
    log.info("Metrics: %s", metrics)

    # Create a summary dataframe
    summary_data = []
    for _, row in benchmark_df.iterrows():
        model_metrics = {"Model": row["Model"]}
        for metric_col, metric_name in metrics.items():
            if metric_col in benchmark_df.columns:
                model_metrics[metric_name] = row[metric_col]
        summary_data.append(model_metrics)

    summary_df = pd.DataFrame(summary_data)
    print("Metrics Summary:")
    print(summary_df.to_string(index=False, float_format="{:.3f}".format))

    colormap = config.get(f"{config.benchmark_colormap}", "tab10")

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

    _plot_benchmark_comparison(summary_df, metrics, output_path, colormap)

    _plot_distribution_shift_comparison(
        summary_df,
        output_path,
        colormap,
        id_metric=f"brierFDE ({id_split_name},↓)",
        ood_metric=f"brierFDE ({ood_split_name},↓)",
    )

    metric_pairs = [
        (f"{metric} ({id_split_name},↓)", f"{metric} ({ood_split_name},↓)", metric)
        for metric in config.trajectory_forecasting_metrics
    ]
    _plot_performance_gaps(summary_df, output_path, metric_pairs)

    key_metrics_display = [
        f"{config.trajectory_forecasting_metrics[0]} ({id_split_name},↓)",
        f"{config.trajectory_forecasting_metrics[0]} ({ood_split_name},↓)",
    ]
    _plot_grouped_bar_chart(summary_df, metrics, output_path, key_metrics_display=key_metrics_display)

    _distribution_shift_to_tex_table(
        benchmark_df,
        BENCHMARK_NAME_MAP.get(config.benchmark) or config.benchmark,
        id_split,
        ood_split,
        config.trajectory_forecasting_metrics,
        output_path,
    )

    print("\n✓ Analysis complete!")


def _distribution_shift_to_tex_table(  # noqa: PLR0912, PLR0913, PLR0915
    benchmark_df: pd.DataFrame,
    benchmark_name: str,
    id_split: str,
    ood_split: str,
    metrics: list[str],
    output_path: Path | None,
    min_color_value: float = 20.0,
) -> str:
    """Converts the distribution shift benchmark DataFrame into a LaTeX table with performance gap annotations/coloring.

    Args:
        benchmark_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        benchmark_name (str): Display name of the benchmark for the table caption.
        id_split (str): Name of the In-Distribution split used in the metrics.
        ood_split (str): Name of the Out-of-Distribution split used in the metrics.
        metrics (list[str]): List of metric column names to include in the table.
        output_path (Path | None): Directory to save the generated LaTeX file. If None, the LaTeX string will be
            returned but not saved to a file.
        min_color_value (float): Minimum color intensity percentage for the gap coloring (0-100). Higher values will
            make the colors more vibrant even for smaller gaps.
    """
    # Precompute best ID/OOD and gap severity per metric
    best_id, best_ood, gap_stats = {}, {}, {}
    for metric in metrics:
        id_col = f"{id_split}/{metric}"
        id_vals = benchmark_df[id_col]
        best_id[metric] = id_vals.min()

        ood_col = f"{ood_split}/{metric}"
        ood_vals = benchmark_df[ood_col]
        best_ood[metric] = ood_vals.min()

        gaps: pd.Series = _relative_gap_pct(ood_vals, id_vals)  # pyright: ignore[reportAssignmentType, reportArgumentType]
        gap_stats[metric] = (gaps.min(), gaps.max())  # best, worst

    # Build rows
    table_rows = []
    first_row = True

    for _, row in benchmark_df.iterrows():
        row_parts = []
        if first_row:
            row_parts.append(f"\\multirow{{{len(benchmark_df)}}}{{*}}{{\\texttt{{{benchmark_name}}}}}")
            first_row = False
        else:
            row_parts.append("")
        row_parts.append(str(row["Model"]))

        # Model size
        if "model/params/total" in row and pd.notna(row["model/params/total"]):
            size_val = row["model/params/total"]
            size_str = f"{size_val:.2e}" if isinstance(size_val, (int, float)) else str(size_val)
        else:
            size_str = MODEL_SIZE_MAP.get(row["Model"], "---")
        row_parts.append(size_str)

        id_values, ood_values = [], []
        for metric in metrics:
            id_col = f"{id_split}/{metric}"
            ood_col = f"{ood_split}/{metric}"
            id_val = row[id_col]
            ood_val = row[ood_col]

            # In-distribution value
            if pd.notna(id_val):
                id_str = f"{id_val:.3f}"
                if np.isclose(id_val, best_id[metric]):
                    id_str = f"\\textbf{{{id_str}}}"
            else:
                id_str = "---"
            id_values.append(id_str)

            # Out-of-distribution value with gap annotation and coloring
            if pd.notna(id_val) and pd.notna(ood_val):
                gap = _relative_gap_pct(ood_val, id_val)

                best_gap, worst_gap = gap_stats[metric]
                denom = max(abs(worst_gap - best_gap), EPSILON)
                severity = np.clip(abs(gap - best_gap) / denom, 0, 1)
                intensity = int(min_color_value + severity * (100 - min_color_value))

                color = "OrangeRed" if gap > 0 else "ForestGreen"
                gap_str = f"\\textcolor{{{color}!{intensity}}}{{{gap:+.2f}\\%}}"

                ood_str = f"{ood_val:.3f}"
                if np.isclose(ood_val, best_ood[metric]):
                    ood_str = f"\\textbf{{{ood_str}}}"
                ood_str = f"{ood_str} ({gap_str})"
            else:
                ood_str = "---"

            ood_values.append(ood_str)

        id_values.append("")  # spacer column
        row_parts.extend(id_values)
        ood_values = ["", *ood_values]  # spacer column
        row_parts.extend(ood_values)
        table_rows.append(" & ".join(row_parts) + " \\\\")

    # Build LaTeX
    n_metrics = len(metrics)
    col_spec = "l l c " + "c" * (2 * n_metrics) + "cc"

    latex_lines: list[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Distribution Shift Results}",
        "\\label{tab:distribution_shift_results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        (
            f"\\multirow{{2}}{{*}}{{\\textbf{{Benchmark}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & "
            f"\\multirow{{2}}{{*}}{{\\textbf{{Model Size}}}} & "
            f"\\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{In Distribution (Validation)}}}} & "
            f"\\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{Out of Distribution (Test)}}}} \\\\"
        ),
        " & & & " + " & ".join([*metrics, "", "", *metrics]) + " \\\\",
        "\\midrule",
        *table_rows,
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ]
    latex_table_str = "\n".join(latex_lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "results.tex").write_text(latex_table_str)

    return latex_table_str
