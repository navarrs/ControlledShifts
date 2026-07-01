"""Environments benchmark clustering analysis.

Visualizes the NetLSD-descriptor clustering that defines the environments benchmark: a TSNE embedding of the
descriptors coloured by assigned cluster, the same embedding coloured by train/validation/testing split, and a
silhouette plot of cluster quality. All inputs are artifacts written by benchmark creation (``descriptors_cache.pkl``,
``scaler.pkl`` and ``{clustering_algorithm}/environment_benchmark.csv`` under ``cache_path``); the descriptors are
scaled with the saved ``StandardScaler`` before TSNE and silhouette, matching benchmark creation.

The TSNE embedding and per-sample silhouette scores are cached to ``environments_embedding.csv`` so re-rendering does
not recompute them unless ``overwrite`` is set.

See `docs/ANALYSIS.md` for usage details.
"""

import pickle  # nosec B403
from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

from controlledshifts.benchmarks.environments import load_descriptor_cache
from controlledshifts.utils.analysis.common import SPLIT_COLOR_MAP, SPLIT_LABELS, SPLIT_ORDER, TEXT_COLOR
from controlledshifts.utils.plotting import set_analysis_theme


def _align_descriptors(scenario_ids: list[str], cache_path: Path, log: Logger) -> tuple[list[str], NDArray[np.float64]]:
    """Builds the descriptor matrix aligned to ``scenario_ids``, dropping scenarios with no cached descriptor.

    Args:
        scenario_ids: Scenario IDs (benchmark CSV order) to look up descriptors for.
        cache_path: Path to ``descriptors_cache.pkl``.
        log: Logger.

    Returns:
        Tuple of (kept_scenario_ids, descriptor_matrix) with one descriptor row per kept scenario.
    """
    cache = load_descriptor_cache(cache_path)
    descriptor_by_id = {scenario_id: descriptor for scenario_id, _, descriptor in cache.values()}

    kept_ids, descriptors = [], []
    for scenario_id in scenario_ids:
        descriptor = descriptor_by_id.get(scenario_id)
        if descriptor is not None:
            kept_ids.append(scenario_id)
            descriptors.append(descriptor)

    missing = len(scenario_ids) - len(kept_ids)
    if missing:
        log.warning("Skipped %d scenarios missing a cached descriptor", missing)
    return kept_ids, np.stack(descriptors)


def _build_embedding_frame(config: DictConfig, log: Logger, output_path: Path) -> pd.DataFrame | None:
    """Builds (and caches) the TSNE embedding frame from the benchmark's descriptor and clustering artifacts.

    Reads the benchmark CSV for cluster and split labels, aligns the cached NetLSD descriptors, scales them with the
    saved ``StandardScaler``, then computes a 2-D TSNE embedding and per-sample silhouette scores. The result is
    written to ``environments_embedding.csv``.

    Args:
        config: Analysis configuration (``cache_path``, ``clustering_algorithm``, ``seed``).
        log: Logger.
        output_path: Directory receiving the cached CSV.

    Returns:
        Frame with columns ``scenario_id``, ``tsne_1``, ``tsne_2``, ``cluster_label``, ``output_set``,
        ``silhouette``, or None when a required artifact is missing.
    """
    cache_path = Path(config.cache_path)
    algorithm_path = cache_path / config.clustering_algorithm
    benchmark_csv = algorithm_path / "environment_benchmark.csv"
    scaler_path = cache_path / "scaler.pkl"
    descriptor_cache_path = cache_path / "descriptors_cache.pkl"

    for required in (benchmark_csv, scaler_path, descriptor_cache_path):
        if not required.exists():
            log.error("Required artifact not found at %s; run the environments benchmark first.", required)
            return None

    benchmark_df = pd.read_csv(benchmark_csv)
    benchmark_df["scenario_id"] = benchmark_df["scenario_id"].astype(str)
    log.info("Loaded %d scenarios from %s", len(benchmark_df), benchmark_csv)

    kept_ids, descriptor_matrix = _align_descriptors(benchmark_df["scenario_id"].tolist(), descriptor_cache_path, log)
    frame = benchmark_df.set_index("scenario_id").loc[kept_ids].reset_index()

    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)  # nosec B301
    scaled = scaler.transform(descriptor_matrix)

    log.info("Computing TSNE embedding for %d scenarios...", len(kept_ids))
    coords = TSNE(n_components=2, random_state=config.seed, init="pca", learning_rate="auto").fit_transform(scaled)
    silhouette = np.asarray(silhouette_samples(scaled, frame["cluster_label"].to_numpy()))

    embedding_df = pd.DataFrame(
        {
            "scenario_id": frame["scenario_id"],
            "tsne_1": coords[:, 0],
            "tsne_2": coords[:, 1],
            "cluster_label": frame["cluster_label"],
            "output_set": frame["output_set"],
            "silhouette": silhouette,
        }
    )
    embedding_df.to_csv(output_path / "environments_embedding.csv", index=False)
    return embedding_df


def _plot_tsne(frame: pd.DataFrame, output_path: Path, *, show_axes: bool) -> None:
    """Saves the TSNE embedding coloured by cluster (left) and by split (right), side by side, to ``tsne.png``.

    The two panels share the y-axis and each carries its own legend below the panel. When ``show_axes`` is false the
    axis ticks, spines and labels are hidden.
    """
    labels = frame["cluster_label"].to_numpy()
    n_clusters = int(labels.max()) + 1
    cmap = get_cmap("tab20", n_clusters)

    fig, (ax_cluster, ax_split) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    fig.suptitle("t-SNE of NetLSD Descriptors", color=TEXT_COLOR)

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if mask.any():
            ax_cluster.scatter(
                frame.loc[mask, "tsne_1"],
                frame.loc[mask, "tsne_2"],
                color=cmap(cluster_id),
                s=12,
                alpha=0.6,
                linewidths=0,
                label=f"C{cluster_id}",
            )
    ax_cluster.set_xlabel("t-SNE 1", color=TEXT_COLOR)
    ax_cluster.set_ylabel("t-SNE 2", color=TEXT_COLOR)
    ax_cluster.set_title("Environment clusters", color=TEXT_COLOR)
    ax_cluster.tick_params(colors=TEXT_COLOR)
    ax_cluster.grid(visible=False)
    ax_cluster.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(n_clusters, 10),
        fontsize=12,
        framealpha=0.8,
        markerscale=3,
        labelcolor=TEXT_COLOR,
    )

    for split in SPLIT_ORDER:
        mask = frame["output_set"] == split
        if mask.any():
            ax_split.scatter(
                frame.loc[mask, "tsne_1"],
                frame.loc[mask, "tsne_2"],
                color=SPLIT_COLOR_MAP[split],
                s=12,
                alpha=0.6,
                linewidths=0,
                label=SPLIT_LABELS[split],
            )
    ax_split.set_xlabel("t-SNE 1", color=TEXT_COLOR)
    ax_split.set_title("Benchmark splits", color=TEXT_COLOR)
    ax_split.tick_params(colors=TEXT_COLOR)
    ax_split.grid(visible=False)
    ax_split.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(SPLIT_ORDER),
        framealpha=0.8,
        markerscale=3,
        labelcolor=TEXT_COLOR,
    )

    ax_cluster.set_title("Environment clusters (t-SNE of NetLSD descriptors)")
    ax_split.set_title("Benchmark splits (t-SNE of NetLSD descriptors)")
    if show_axes:
        ax_cluster.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
        ax_split.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    else:
        for ax in (ax_cluster, ax_split):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(visible=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    output_file = output_path / "tsne.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def _plot_silhouette(frame: pd.DataFrame, output_path: Path) -> None:
    """Saves a per-cluster silhouette plot with the overall mean to ``silhouette.png``."""
    labels = frame["cluster_label"].to_numpy()
    n_clusters = int(labels.max()) + 1
    cmap = get_cmap("tab20", n_clusters)
    mean_silhouette = float(frame["silhouette"].mean())

    fig, ax = plt.subplots(figsize=(8, 10))
    y_lower = 10
    for cluster_id in range(n_clusters):
        values = np.sort(frame.loc[labels == cluster_id, "silhouette"].to_numpy())
        if values.size == 0:
            continue
        y_upper = y_lower + values.size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, values, facecolor=cmap(cluster_id), alpha=0.8, linewidth=0)
        ax.text(-0.02, y_lower + 0.5 * values.size, f"C{cluster_id}", va="center", ha="right", fontsize=10)
        y_lower = y_upper + 10

    ax.axvline(mean_silhouette, color="red", linestyle="--", label=f"mean = {mean_silhouette:.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Scenarios grouped by cluster")
    ax.set_title("Silhouette analysis of environment clusters")
    ax.set_yticks([])
    ax.legend(loc="lower right", framealpha=0.8)

    fig.tight_layout()
    output_file = output_path / "silhouette.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved as '{output_file}'")


def run_environments_distribution_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Visualizes the environments benchmark clustering: TSNE by cluster, TSNE by split, and silhouette scores.

    The plots are driven by the cached ``environments_embedding.csv``: when it already exists (and ``overwrite`` is
    false) it is loaded directly, so re-rendering recomputes neither the TSNE embedding nor the silhouette scores.
    Otherwise the frame is rebuilt from the benchmark's descriptor cache, scaler and split CSV.

    Args:
        config: Analysis configuration (``cache_path``, ``clustering_algorithm``, ``seed``, ``overwrite``,
            ``show_axes``).
        log: Logger.
        output_path: Directory to save the cached embedding and plots.
    """
    set_analysis_theme(log=log)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    embedding_cache = output_path / "environments_embedding.csv"
    if embedding_cache.exists() and not config.overwrite:
        log.info("Regenerating plots from cached %s (set overwrite=true to recompute the embedding)", embedding_cache)
        frame = pd.read_csv(embedding_cache)
        frame["scenario_id"] = frame["scenario_id"].astype(str)
    else:
        frame = _build_embedding_frame(config, log, output_path)
        if frame is None:
            return

    _plot_tsne(frame, output_path, show_axes=config.show_axes)
    _plot_silhouette(frame, output_path)

    print("\n✓ Analysis complete!")
    log.info("Environments analysis complete!")
