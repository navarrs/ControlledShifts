r"""Benchmark creation for the Environments benchmark.

Clusters scenarios by road topology using NetLSD graph descriptors and a configurable clustering algorithm, then assigns
train/validation/testing splits based on cluster hardness (silhouette score or Davies-Bouldin Index). Supported
algorithms: ``kmeans``, ``hdbscan``, ``agglomerative``, ``ward``, ``spectral``. Results are written under a subfolder
named after the chosen algorithm inside ``cache_path``.

Example usage:

    uv run -m controlledshifts.create_benchmark benchmark=environments \\
        input_data_path=/datasets/waymo/processed/mini_causal \\
        output_data_path=/datasets/waymo/processed/environment_benchmark

See configs/benchmark/environments.yaml for all available options.
"""

import functools
import multiprocessing
import pickle  # nosec B403
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import netlsd
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from controlledshifts.benchmarks import common
from controlledshifts.datasets.waymo.repacker import load_scenario
from controlledshifts.utils.map_utils import build_positioned_graph, visualize_scenario_graph
from controlledshifts.utils.pylogger import get_pylogger
from controlledshifts.utils.scenario_visualizers.scenario import ScenarioVisualizer


_LOGGER = get_pylogger(__name__)


def _compute_graph_descriptor(
    filepath: Path,
    *,
    ego_centered: bool = False,
    num_map_elements: int = 100,
    map_range: float = 100.0,
    simplify: bool = True,
) -> tuple[str, str, NDArray[np.float64]] | None:
    """Loads a scenario pickle file, builds its map graph, and computes a NetLSD descriptor.

    When ego_centered is True the graph is filtered to match base_dataset.py's pipeline: polyline points are
    transformed to the ego-centric frame (translate + rotate by heading), filtered by the L∞ range box, and the
    top-K elements by average point distance are kept.

    Args:
        filepath: Path to the scenario pickle file.
        ego_centered: If True, restrict the graph to map elements within range of the ego agent.
        num_map_elements: Maximum number of map elements to retain when ego_centered is True. Defaults to 100.
        map_range: L∞ half-width of the ego-centric range box in metres. Defaults to 100.0.
        simplify: If True, runs simplify_graph on the raw graph before computing the descriptor. Defaults to True.

    Returns:
        A tuple of (scenario_id, split, descriptor), where split is the name of the scenario's parent directory,
        or None if the file does not exist, cannot be unpickled, or has no ``scenario_id`` field.
    """
    if not filepath.exists():
        return None

    try:
        with filepath.open("rb") as f:
            scenario = pickle.load(f)  # nosec B301
    except (OSError, pickle.UnpicklingError):
        return None

    scenario_id = scenario.get("scenario_id")
    if scenario_id is None:
        return None

    ref_xy: NDArray[np.float64] | None = None
    ref_heading: float = 0.0
    if ego_centered:
        sdc_track_index = scenario["sdc_track_index"]
        curr_time_index = scenario["current_time_index"]
        trajs = scenario["track_infos"]["trajs"][sdc_track_index, curr_time_index]
        ref_xy = trajs[:2]
        ref_heading = float(trajs[6])

    map_infos = scenario.get("map_infos", {})
    graph, _ = build_positioned_graph(
        map_infos,
        ref_xy=ref_xy,
        ref_heading=ref_heading,
        num_map_elements=num_map_elements if ego_centered else None,
        map_range=map_range,
        simplify=simplify,
    )

    descriptor: NDArray[np.float64] = netlsd.heat(graph)
    split = filepath.parent.name
    return scenario_id, split, descriptor


def load_descriptor_cache(cache_path: Path) -> dict[str, tuple[str, str, NDArray[np.float64]]]:
    """Loads the descriptor cache from disk.

    The cache maps filepath strings to (scenario_id, split, descriptor) tuples. Returns an empty dict if the cache
    file does not exist or cannot be read.

    Args:
        cache_path: Path to the cache pickle file.
    """
    if not cache_path.exists():
        return {}

    try:
        with cache_path.open("rb") as f:
            return pickle.load(f)  # nosec B301
    except (OSError, pickle.UnpicklingError):
        return {}


def _save_descriptor_cache(cache: dict[str, tuple[str, str, NDArray[np.float64]]], cache_path: Path) -> None:
    """Saves the descriptor cache to disk.

    Args:
        cache: Dict mapping filepath string to (scenario_id, split, descriptor).
        cache_path: Destination path for the cache pickle file.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(cache, f)


def _compute_descriptors_with_cache(  # noqa: PLR0913
    filepaths: list[Path],
    cache_path: Path,
    ego_centered: bool,  # noqa: FBT001
    num_map_elements: int,
    num_workers: int,
    root_path: Path | None = None,
    overwrite: bool = False,  # noqa: FBT001, FBT002
    map_range: float = 100.0,
    simplify: bool = True,  # noqa: FBT001, FBT002
) -> tuple[list[str], list[str], NDArray[np.float64]]:
    """Computes NetLSD descriptors for a list of scenario files, using a disk cache for already-computed results.

    Descriptors for filepaths not present in the cache are computed (in parallel if num_workers > 0) and added to the
    cache. The updated cache is saved to disk before returning.

    Each scenario's split label is derived from its parent directory path. When ``root_path`` is provided the label
    is the relative path from root to the parent directory (e.g. ``training/shard_0``); otherwise the immediate parent
    directory name is used.

    Args:
        filepaths: List of scenario file paths.
        cache_path: Path to the descriptor cache pickle file.
        ego_centered: If True, restrict each map graph to the ego-centric L∞ range box before encoding.
        num_map_elements: Maximum number of map elements to retain per scenario when ego_centered is True.
        num_workers: Number of parallel workers for descriptor computation (0 = single process).
        root_path: Root of the input dataset tree used to derive split labels. When None, labels fall back to the
            immediate parent directory name. Defaults to None.
        overwrite: If True, evict all entries for the given filepaths from the cache before computing, forcing
            recomputation even if descriptors were previously cached. Defaults to False.
        map_range: L∞ half-width of the ego-centric range box in metres. Defaults to 100.0.
        simplify: If True, simplify the map graph before computing the NetLSD descriptor. Defaults to True.

    Returns:
        Tuple of (scenario_ids, split_labels, descriptor_matrix) for all filepaths with a valid cached descriptor.

    Raises:
        ValueError: If no valid descriptors could be computed or retrieved.
    """
    cache = load_descriptor_cache(cache_path)

    if overwrite:
        for fp in filepaths:
            cache.pop(str(fp), None)

    uncached = [fp for fp in filepaths if str(fp) not in cache]
    if uncached:
        num_cached = len(filepaths) - len(uncached)
        _LOGGER.info("Computing descriptors for %d scenarios (%d cached)...", len(uncached), num_cached)
        worker_fn = functools.partial(
            _compute_graph_descriptor,
            ego_centered=ego_centered,
            num_map_elements=num_map_elements,
            map_range=map_range,
            simplify=simplify,
        )

        if num_workers == 0:
            results = [worker_fn(fp) for fp in tqdm(uncached, desc="Encoding map graphs")]
        else:
            with multiprocessing.Pool(num_workers) as pool:
                results = list(tqdm(pool.imap(worker_fn, uncached), total=len(uncached), desc="Encoding map graphs"))

        for fp, result in zip(uncached, results, strict=False):
            if result is not None:
                cache[str(fp)] = result
        _save_descriptor_cache(cache, cache_path)
        _LOGGER.info("Descriptor cache updated at %s", cache_path)
    else:
        _LOGGER.info("All %d descriptors loaded from cache.", len(filepaths))

    valid_entries = [(fp, cache[str(fp)]) for fp in filepaths if str(fp) in cache]
    if not valid_entries:
        error_message = "No valid scenario descriptors were computed. Check the input data path."
        raise ValueError(error_message)

    def _split_label(fp: Path) -> str:
        if root_path is not None:
            try:
                return str(fp.parent.relative_to(root_path))
            except ValueError:
                pass
        return fp.parent.name

    scenario_ids = [entry[0] for _, entry in valid_entries]
    splits = [_split_label(fp) for fp, _ in valid_entries]
    descriptors = [entry[2] for _, entry in valid_entries]
    return scenario_ids, splits, np.stack(descriptors)


def _per_cluster_dbi(
    features: NDArray[np.float64],
    labels: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Computes per-sample Davies-Bouldin Index scores by assigning each sample its cluster's DBI value.

    DBI measures the ratio of within-cluster scatter to between-cluster separation. For each cluster i:

        DBI_i = max_{j ≠ i} (S_i + S_j) / d(c_i, c_j)

    where S_i is the mean L2 distance of cluster-i samples to their centroid (within-cluster scatter) and
    d(c_i, c_j) is the distance between centroids i and j (between-cluster separation).
    Higher DBI_i means the cluster has high intra-scatter relative to centroid separation — i.e. harder to separate.

    Args:
        features: Scaled feature matrix of shape (N, D).
        labels: Integer cluster label per sample, shape (N,).

    Returns:
        Per-sample DBI scores of shape (N,) where each entry is the DBI of the sample's cluster.
    """
    cluster_ids = np.unique(labels)
    centroids = np.array([features[labels == c].mean(axis=0) for c in cluster_ids])
    intra_dists = np.array(
        [np.linalg.norm(features[labels == c] - centroids[i], axis=1).mean() for i, c in enumerate(cluster_ids)]
    )

    n = len(cluster_ids)
    per_cluster = np.zeros(n)
    for i in range(n):
        ratios = []
        for j in range(n):
            if i == j:
                continue
            centroid_dist = np.linalg.norm(centroids[i] - centroids[j])
            if centroid_dist > 0:
                ratios.append((intra_dists[i] + intra_dists[j]) / centroid_dist)
        per_cluster[i] = max(ratios) if ratios else 0.0

    cluster_to_idx = {int(c): i for i, c in enumerate(cluster_ids)}
    return np.array([per_cluster[cluster_to_idx[int(lbl)]] for lbl in labels])


def select_train_test_splits(
    clusters_df: pd.DataFrame,
    hardness_scores: NDArray[np.float64],
    split_ratios: tuple[float, float, float] = (0.70, 0.10, 0.20),
    rng: np.random.Generator | None = None,
    *,
    hardness_ascending: bool = True,
) -> pd.DataFrame:
    """Assigns each scenario to an output split (training/validation/testing) based on cluster hardness.

    Clusters are ranked by their mean hardness score and greedily added to the test set in hardness order. Whole
    clusters are added until the next cluster would exceed ``int(total * split_ratios[2])`` scenarios; at that point
    only the hardest scenarios within that boundary cluster are taken to fill up to the target exactly. When
    ``hardness_ascending=True`` (silhouette), the **lowest** individual score is hardest; when
    ``hardness_ascending=False`` (DBI), the **highest** individual score is hardest. This keeps the test set at
    exactly the target count with a fully deterministic selection.

    The remaining (non-test) scenarios are shuffled and then split: the first ``int(total * split_ratios[1])`` become
    validation and the rest become training.

    Args:
        clusters_df: DataFrame with columns ``scenario_id``, ``split``, and ``cluster``.
        hardness_scores: Per-sample hardness scores aligned with the rows of ``clusters_df``.
        split_ratios: Desired ``(train, val, test)`` fractions of the full dataset. Elements should sum to 1.0.
            Defaults to ``(0.70, 0.10, 0.20)``.
        rng: Used for reproducible shuffling of non-test scenarios. If None, an unseeded (non-reproducible)
            generator is used.
        hardness_ascending: If True, clusters with lower mean scores are harder (silhouette convention). If False,
            clusters with higher mean scores are harder (DBI convention). Defaults to True.

    Returns:
        Copy of clusters_df with columns added: ``hardness_score``, ``input_set`` (renamed from ``split``), and
        ``output_set`` (one of ``"training"``, ``"validation"``, ``"testing"``).
    """
    df = clusters_df.copy()
    df["hardness_score"] = hardness_scores
    df = df.rename(columns={"split": "input_set"})

    rng_to_use = rng if rng is not None else np.random.default_rng()
    total = len(df)
    target_test_count = int(total * split_ratios[2])

    cluster_stats = (
        df.groupby("cluster")
        .agg(mean_hardness=("hardness_score", "mean"), size=("hardness_score", "count"))
        .reset_index()
        .sort_values("mean_hardness", ascending=hardness_ascending)
    )

    test_indices: set[int] = set()
    accumulated = 0
    for _, row in cluster_stats.iterrows():
        if accumulated >= target_test_count:
            break
        cluster_id = int(row["cluster"])
        cluster_indices = df.index[df["cluster"] == cluster_id].tolist()
        remaining = target_test_count - accumulated
        if len(cluster_indices) <= remaining:
            test_indices.update(cluster_indices)
            accumulated += len(cluster_indices)
        else:
            cluster_rows = df.loc[cluster_indices].sort_values("hardness_score", ascending=hardness_ascending)
            test_indices.update(cluster_rows.index[:remaining].tolist())
            accumulated += remaining
            break

    non_test_indices = [i for i in df.index if i not in test_indices]
    rng_to_use.shuffle(non_test_indices)

    num_val = int(total * split_ratios[1])
    val_indices: set[int] = set(non_test_indices[:num_val])

    def _assign_output_set(row: pd.Series) -> str:  # pyright: ignore[reportMissingTypeArgument]
        if row.name in test_indices:
            return "testing"

        if row.name in val_indices:
            return "validation"

        return "training"

    df["output_set"] = df.apply(_assign_output_set, axis=1)
    return df


def visualize_cluster_graphs(  # noqa: PLR0913
    clusters_df: pd.DataFrame,
    input_data_path: Path,
    output_path: Path,
    *,
    n_examples: int = 30,
    seed: int = 42,
    ego_centered: bool = False,
    num_map_elements: int = 100,
    map_range: float = 100.0,
    simplify: bool = True,
) -> None:
    """Renders per-cluster scenario graph PNGs.

    For each cluster, up to ``n_examples`` scenario graphs are rendered and saved under
    ``output_path/cluster_<id>/``.

    Args:
        clusters_df: DataFrame with columns ``scenario_id`` and ``cluster``.
        input_data_path: Root of the input dataset tree; scenario pickle files are discovered recursively from here.
        output_path: Directory under which per-cluster subdirectories are created.
        n_examples: Maximum number of example graphs to render per cluster. Defaults to 30.
        seed: Random seed for example sampling. Defaults to 42.
        ego_centered: If True, each scenario graph is filtered to the ego-centric L∞ range box before rendering,
            reproducing the encoded graph. Defaults to False.
        num_map_elements: Maximum number of map elements to retain per scenario when ego_centered is True.
            Defaults to 100.
        map_range: L∞ half-width of the ego-centric range box in metres. Defaults to 100.0.
        simplify: If True, simplify the map graph before rendering. Defaults to True.
    """
    rng = np.random.default_rng(seed)
    filepath_map = {fp.stem: fp for fp in input_data_path.rglob("*.pkl") if "infos" not in fp.stem}

    for cluster_id, group in clusters_df.groupby("cluster"):
        cluster_dir = output_path / f"cluster_{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        scenario_ids = group["scenario_id"].tolist()
        n_chosen = min(n_examples, len(scenario_ids))
        chosen = rng.choice(scenario_ids, size=n_chosen, replace=False).tolist()

        for scenario_id in tqdm(chosen, desc=f"Visualizing cluster {cluster_id}", leave=False):
            filepath = filepath_map.get(str(scenario_id))
            if filepath is not None:
                visualize_scenario_graph(
                    filepath,
                    cluster_dir,
                    ego_centered=ego_centered,
                    num_map_elements=num_map_elements,
                    map_range=map_range,
                    simplify=simplify,
                )

    _LOGGER.info("Saved per-cluster graph examples to %s", output_path)


def visualize_cluster_scenarios(  # noqa: PLR0913
    clusters_df: pd.DataFrame,
    input_data_path: Path,
    output_path: Path,
    visualizer: ScenarioVisualizer,
    *,
    n_examples: int = 30,
    seed: int = 42,
) -> None:
    """Renders per-cluster scenario PNGs using ScenarioVisualizer.

    For each cluster, up to ``n_examples`` scenarios are loaded and visualized, saved under
    ``output_path/cluster_<id>/``.

    Args:
        clusters_df: DataFrame with columns ``scenario_id`` and ``cluster``.
        input_data_path: Root of the input dataset tree; scenario pickle files are discovered recursively from here.
        output_path: Directory under which per-cluster subdirectories are created.
        visualizer: Instantiated ScenarioVisualizer used to render each scenario.
        n_examples: Maximum number of example scenarios to render per cluster. Defaults to 30.
        seed: Random seed for example sampling. Defaults to 42.
    """
    rng = np.random.default_rng(seed)
    filepath_map = {fp.stem: fp for fp in input_data_path.rglob("*.pkl") if "infos" not in fp.stem}

    for cluster_id, group in clusters_df.groupby("cluster"):
        cluster_dir = output_path / f"cluster_{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        scenario_ids = group["scenario_id"].tolist()
        n_chosen = min(n_examples, len(scenario_ids))
        chosen = rng.choice(scenario_ids, size=n_chosen, replace=False).tolist()

        for scenario_id in tqdm(chosen, desc=f"Visualizing cluster {cluster_id} scenarios", leave=False):
            filepath = filepath_map.get(str(scenario_id))
            if filepath is None:
                continue
            try:
                scenario = load_scenario(filepath)
                visualizer.visualize_scenario(scenario, output_dir=str(cluster_dir))
            except Exception:  # noqa: BLE001
                _LOGGER.warning("Failed to visualize scenario %s, skipping.", scenario_id, exc_info=True)

    _LOGGER.info("Saved per-cluster scenario visualizations to %s", output_path)


def visualize_descriptor_scatter(  # noqa: PLR0913
    clusters_df: pd.DataFrame,
    descriptor_matrix: NDArray[np.float64],
    output_path: Path,
    *,
    seed: int = 42,
    test_clusters: set[int] | None = None,
    reduction: str = "pca",
) -> None:
    """Saves a 2-D scatter plot of NetLSD descriptors coloured by cluster label.

    The scatter is saved as ``output_path/cluster_scatter.png``. When ``clusters_df`` contains a ``hardness_score``
    column, per-cluster mean hardness scores are annotated in the legend. Test clusters are marked with a distinct
    marker style (``"x"`` vs ``"o"`` for train/val).

    Args:
        clusters_df: DataFrame with columns ``cluster``, and optionally ``hardness_score``.
        descriptor_matrix: Array of shape (N, D) — one descriptor row per scenario, aligned with ``clusters_df`` rows.
        output_path: Directory in which to save ``cluster_scatter.png``.
        seed: Random seed for dimensionality reduction. Defaults to 42.
        test_clusters: Set of cluster IDs designated as the test set. When provided, test clusters are drawn with a
            distinct marker. Defaults to None.
        reduction: Dimensionality reduction method: ``"pca"`` (default) or ``"tsne"``.
    """
    cluster_mean_sil: dict[Any, float] = {}
    if "hardness_score" in clusters_df.columns:
        cluster_mean_sil = clusters_df.groupby("cluster")["hardness_score"].mean().to_dict()

    labels = clusters_df["cluster"].to_numpy()
    n_clusters = int(labels.max()) + 1
    cmap = get_cmap("tab20", n_clusters)

    if reduction == "tsne":
        coords = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(
            descriptor_matrix
        )
        xlabel, ylabel, title = "t-SNE 1", "t-SNE 2", "Environment clusters (t-SNE of NetLSD descriptors)"
    else:
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(descriptor_matrix)
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)"
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)"
        title = "Environment clusters (PCA of NetLSD descriptors)"

    fig, ax = plt.subplots(figsize=(10, 8))

    if test_clusters:
        train_mask = np.array([lbl not in test_clusters for lbl in labels])
        test_mask = ~train_mask
        if train_mask.any():
            ax.scatter(
                coords[train_mask, 0],
                coords[train_mask, 1],
                c=labels[train_mask],
                cmap=cmap,
                vmin=0,
                vmax=n_clusters - 1,
                s=12,
                alpha=0.6,
                linewidths=0,
                marker=MarkerStyle("o"),
                label="train/val",
            )
        if test_mask.any():
            ax.scatter(
                coords[test_mask, 0],
                coords[test_mask, 1],
                c=labels[test_mask],
                cmap=cmap,
                vmin=0,
                vmax=n_clusters - 1,
                s=20,
                alpha=0.8,
                linewidths=0.5,
                edgecolors="black",
                marker=MarkerStyle("x"),
                label="test",
            )
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, s=10, alpha=0.6, linewidths=0)
        cbar = fig.colorbar(scatter, ax=ax, ticks=range(n_clusters))
        cbar.set_label("Cluster", fontsize=10)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)

    if cluster_mean_sil:
        legend_handles = []
        for c, mean_sil in sorted(cluster_mean_sil.items()):
            color = cmap(int(c) / max(n_clusters - 1, 1))
            is_test = test_clusters is not None and c in test_clusters
            marker = "x" if is_test else "o"
            tag = " [test]" if is_test else ""
            handle = Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=7,
                label=f"C{c}{tag}  h={mean_sil:.3f}",
            )
            legend_handles.append(handle)
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, framealpha=0.8)

    fig.tight_layout()
    output_path.mkdir(parents=True, exist_ok=True)
    scatter_path = output_path / "cluster_scatter.png"
    fig.savefig(str(scatter_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    _LOGGER.info("Saved scatter plot to %s", scatter_path)


def _fit_clustering_model(
    scaled_data: NDArray[np.float64],
    config: DictConfig,
) -> tuple[Any, NDArray[np.int32]]:
    """Fits a clustering model on scaled data, dispatching by config.clustering_algorithm.

    Supported algorithms: ``'kmeans'``, ``'hdbscan'``, ``'agglomerative'``, ``'ward'``, ``'spectral'``.

    Args:
        scaled_data: Scaled feature matrix of shape (N, D).
        config: Must contain ``clustering_algorithm`` and ``seed``. ``n_clusters`` is used by all algorithms except
            ``'hdbscan'``. For ``'hdbscan'``, ``min_cluster_size`` (default 5) and optionally ``min_samples`` are read.

    Returns:
        ``(model, labels)`` where *model* is the fitted estimator and *labels* are the cluster assignments,
        shape (N,).

    Raises:
        ValueError: If an unsupported clustering algorithm is specified.
    """
    algorithm: str = config.get("clustering_algorithm", "kmeans")
    n_clusters: int = config.n_clusters
    seed: int = config.seed

    match algorithm:
        case "kmeans":
            model: Any = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
        case "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        case "ward":
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        case "spectral":
            model = SpectralClustering(n_clusters=n_clusters, random_state=seed, assign_labels="kmeans")
        case _:
            error_message = f"Unsupported clustering algorithm: {algorithm!r}"
            raise ValueError(error_message)

    labels: NDArray[np.int32] = model.fit_predict(scaled_data).astype(np.int32)
    return model, labels


def _assign_clusters(
    model: Any,  # noqa: ANN401
    sample_scaled: NDArray[np.float64],
    sample_labels: NDArray[np.int32],
    new_data: NDArray[np.float64],
) -> NDArray[np.int32]:
    """Assigns cluster labels to new data points using the fitted model.

    For models that expose a ``predict()`` method (e.g. KMeans), delegates to it directly. For transductive algorithms
    (HDBSCAN, AgglomerativeClustering, SpectralClustering) that lack ``predict()``, computes per-cluster centroids from
    the training data and assigns each new point to the nearest centroid. HDBSCAN noise labels (-1) are excluded when
    computing centroids.

    Args:
        model: Fitted clustering estimator.
        sample_scaled: Training data used to fit the model, shape (N, D).
        sample_labels: Cluster labels for *sample_scaled*, shape (N,).
        new_data: New data to assign, shape (M, D).

    Returns:
        Integer cluster labels for *new_data*, shape (M,).
    """
    if hasattr(model, "predict") and callable(model.predict):
        return model.predict(new_data).astype(np.int32)  # pyright: ignore[reportAttributeAccessIssue]

    valid_ids = np.unique(sample_labels)
    valid_ids = valid_ids[valid_ids >= 0]  # exclude HDBSCAN noise label -1
    centroids = np.stack([sample_scaled[sample_labels == c].mean(axis=0) for c in valid_ids])
    dists = np.linalg.norm(new_data[:, None, :] - centroids[None, :, :], axis=2)
    return valid_ids[np.argmin(dists, axis=1)].astype(np.int32)


def create_environments_benchmark(config: DictConfig) -> common.BenchmarkSplit:  # noqa: PLR0912, PLR0915
    """Clusters scenarios by road topology using NetLSD graph descriptors and a configurable clustering algorithm.

    The pipeline runs in two phases:

    1. **Fit phase** — A random sample of P% of all scenarios (or `num_scenarios` if specified) is used to fit a
       StandardScaler and the selected clustering model. Hardness scores on this sample drive cluster-hardness ranking.
    2. **Assign phase** — The remaining scenarios are embedded using the cached descriptors and assigned to the
       nearest cluster. For algorithms with a ``predict()`` method (KMeans), it is used directly; others fall back to
       nearest-centroid assignment.

    After clustering, `select_train_test_splits` designates the hardest clusters as the test set so that the test
    split is more challenging. The hardness metric is controlled by ``hardness_metric``: ``"silhouette"`` selects the
    clusters with the **lowest** mean silhouette (most ambiguous), and ``"dbi"`` selects clusters with the **highest**
    mean Davies-Bouldin Index (highest within-cluster scatter relative to between-cluster separation). Results are
    saved under ``cache_path/<clustering_algorithm>/``.

    Args:
        config: Hydra config.
            Expected keys: input_data_path, output_data_path, cache_path, clustering_algorithm, n_clusters,
            n_examples, sample_percentage, num_scenarios, num_workers, parallel, ego_centered, num_map_elements, seed,
            overwrite, map_range, reduction, simplify, split_ratios, hardness_metric.

    Returns:
        The cluster-hardness BenchmarkSplit (from ``benchmark_df["output_set"]``).

    Raises:
        ValueError: If no valid scenario descriptors could be computed.
    """
    input_data_path = Path(config.input_data_path)
    cache_path = Path(config.cache_path)

    rng = np.random.default_rng(config.seed)
    output_path = Path(config.output_data_path)
    num_workers = config.num_workers if config.parallel else 0
    algorithm: str = config.get("clustering_algorithm", "kmeans")

    if config.overwrite:
        if output_path.exists():
            shutil.rmtree(output_path)
        if cache_path.exists():
            shutil.rmtree(cache_path)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)
    descriptor_filepath = Path(cache_path) / "descriptors_cache.pkl"

    algorithm_path = cache_path / algorithm
    algorithm_path.mkdir(parents=True, exist_ok=True)
    benchmark_csv_path = algorithm_path / "environment_benchmark.csv"
    model_path = algorithm_path / "clustering_model.pkl"
    scaler_path = cache_path / "scaler.pkl"

    all_scaled: list[NDArray[np.float64]] | None = None
    sample_ids: list[str] = []

    if not config.overwrite and benchmark_csv_path.exists() and model_path.exists() and scaler_path.exists():
        _LOGGER.info("Loading precomputed benchmark from %s", benchmark_csv_path)
        output_df = pd.read_csv(benchmark_csv_path)
        benchmark_df = output_df.rename(columns={"cluster_label": "cluster"})
        test_cluster_ids = set(benchmark_df[benchmark_df["output_set"] == "testing"]["cluster"].unique())
    else:
        all_filepaths: list[Path] = [fp for fp in input_data_path.rglob("*.pkl") if "infos" not in fp.stem]
        _LOGGER.info("Found %d scenario files.", len(all_filepaths))

        num_samples = (
            config.num_scenarios
            if config.num_scenarios is not None
            else max(1, round(len(all_filepaths) * config.sample_percentage))
        )
        num_samples = min(num_samples, len(all_filepaths))
        _LOGGER.info(
            "Using %d scenarios (%.1f%%) to fit the model.", num_samples, 100 * num_samples / len(all_filepaths)
        )

        sample_indices = rng.choice(len(all_filepaths), size=num_samples, replace=False)
        sample_mask = np.zeros(len(all_filepaths), dtype=bool)
        sample_mask[sample_indices] = True
        sample_filepaths = [fp for fp, m in zip(all_filepaths, sample_mask, strict=False) if m]
        remaining_filepaths = [fp for fp, m in zip(all_filepaths, sample_mask, strict=False) if not m]

        _LOGGER.info("[Phase 1] Computing descriptors for %d sample scenarios...", len(sample_filepaths))
        sample_ids, sample_splits, sample_descriptors = _compute_descriptors_with_cache(
            sample_filepaths,
            descriptor_filepath,
            config.ego_centered,
            config.num_map_elements,
            num_workers,
            root_path=input_data_path,
            overwrite=config.overwrite,
            map_range=config.map_range,
            simplify=config.simplify,
        )

        _LOGGER.info("Fitting StandardScaler and %s(%d) on sample...", algorithm, config.n_clusters)
        scaler = StandardScaler()
        sample_scaled = scaler.fit_transform(sample_descriptors)
        clustering_model, sample_labels = _fit_clustering_model(sample_scaled, config)

        all_ids = list(sample_ids)
        all_splits = list(sample_splits)
        all_scaled = list(sample_scaled)
        all_labels = list(sample_labels)

        if remaining_filepaths:
            _LOGGER.info("[Phase 2] Computing descriptors for %d remaining scenarios...", len(remaining_filepaths))
            rem_ids, rem_splits, rem_descriptors = _compute_descriptors_with_cache(
                remaining_filepaths,
                descriptor_filepath,
                config.ego_centered,
                config.num_map_elements,
                num_workers,
                root_path=input_data_path,
                overwrite=config.overwrite,
                map_range=config.map_range,
                simplify=config.simplify,
            )
            rem_scaled = scaler.transform(rem_descriptors)
            rem_labels = _assign_clusters(clustering_model, sample_scaled, sample_labels, rem_scaled)

            all_ids += list(rem_ids)
            all_splits += list(rem_splits)
            all_scaled += list(rem_scaled)
            all_labels += list(rem_labels)

        all_scaled_matrix = np.stack(all_scaled)
        all_labels_array = np.array(all_labels)

        hardness_metric: str = config.get("hardness_metric", "silhouette")
        if hardness_metric == "dbi":
            _LOGGER.info("Computing per-cluster DBI scores for %d scenarios...", len(all_ids))
            hardness_scores: NDArray[np.float64] = _per_cluster_dbi(all_scaled_matrix, all_labels_array)
            hardness_ascending = True
        else:
            _LOGGER.info("Computing silhouette scores for %d scenarios...", len(all_ids))
            hardness_scores = np.asarray(silhouette_samples(all_scaled_matrix, all_labels_array))
            hardness_ascending = False

        clusters_df = pd.DataFrame({"scenario_id": all_ids, "split": all_splits, "cluster": all_labels_array})
        benchmark_df = select_train_test_splits(
            clusters_df,
            hardness_scores,
            split_ratios=tuple(config.split_ratios),
            rng=rng,
            hardness_ascending=hardness_ascending,
        )

        output_df = benchmark_df[["scenario_id", "cluster", "hardness_score", "input_set", "output_set"]].rename(
            columns={"cluster": "cluster_label"}
        )

        output_df.to_csv(benchmark_csv_path, index=False)
        _LOGGER.info("Saved benchmark split assignments to %s", benchmark_csv_path)

        with model_path.open("wb") as f:
            pickle.dump(clustering_model, f)
        _LOGGER.info("Saved clustering model to %s", model_path)

        with scaler_path.open("wb") as f:
            pickle.dump(scaler, f)
        _LOGGER.info("Saved StandardScaler to %s", scaler_path)

        cluster_stats = (
            benchmark_df.groupby("cluster")
            .agg(size=("scenario_id", "count"), mean_hardness=("hardness_score", "mean"))
            .sort_index()
        )
        test_cluster_ids = set(benchmark_df[benchmark_df["output_set"] == "testing"]["cluster"].unique())
        cluster_lines = "\n".join(
            f"  Cluster {cluster_id}: {int(row['size'])} scenarios, mean_hardness={row['mean_hardness']:.3f}"
            + (" [TEST]" if cluster_id in test_cluster_ids else "")
            for cluster_id, row in cluster_stats.iterrows()
        )
        _LOGGER.info("Cluster summary (hardness_metric=%s):\n%s", hardness_metric, cluster_lines)

        split_counts = benchmark_df["output_set"].value_counts()
        split_lines = "\n".join(
            f"  {name}: {count} ({count / len(benchmark_df):.1%})" for name, count in split_counts.items()
        )
        _LOGGER.info("Output split distribution:\n%s", split_lines)

    cluster_graphs_path = algorithm_path / "cluster_graphs"
    if cluster_graphs_path.exists():
        shutil.rmtree(cluster_graphs_path)

    sample_cluster_df = benchmark_df.iloc[: len(sample_ids)].copy() if sample_ids else benchmark_df.copy()
    if config.get("visualize_descriptor_scatter", False) and all_scaled is not None:
        visualize_descriptor_scatter(
            benchmark_df,
            np.stack(all_scaled),  # pyright: ignore[reportArgumentType, reportCallIssue]
            cluster_graphs_path,
            seed=config.seed,
            test_clusters=test_cluster_ids,
            reduction=config.reduction,
        )

    if config.get("visualize_cluster_graphs", False):
        visualize_cluster_graphs(
            sample_cluster_df,
            input_data_path,
            cluster_graphs_path,
            n_examples=config.n_examples,
            seed=config.seed,
            ego_centered=config.ego_centered,
            num_map_elements=config.num_map_elements,
            map_range=config.map_range,
            simplify=config.simplify,
        )

    if config.get("visualize_scenarios", False):
        cluster_scenarios_path = algorithm_path / "cluster_scenarios"
        if cluster_scenarios_path.exists():
            shutil.rmtree(cluster_scenarios_path)

        visualizer = ScenarioVisualizer(config.visualization.visualizer.config)

        visualize_cluster_scenarios(
            sample_cluster_df,
            input_data_path,
            cluster_scenarios_path,
            visualizer,
            n_examples=config.n_examples,
            seed=config.seed,
        )

    placed_ids = set(benchmark_df["scenario_id"].tolist())
    invalid = [fp.stem for fp in common.collect_scenario_filepaths(input_data_path) if fp.stem not in placed_ids]
    return common.split_mapping_to_lists(
        dict(zip(benchmark_df["scenario_id"], benchmark_df["output_set"], strict=False)), invalid=invalid
    )
