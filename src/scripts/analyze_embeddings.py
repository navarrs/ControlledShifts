"""Script to load cached model outputs, cluster their decoder embeddings, and visualize the results."""

import argparse
import pickle  # nosec B403
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from controlledshifts.schemas import output_schemas as output
from controlledshifts.utils.plotting import configure_fonts


def _load_model_outputs(outputs_path: Path, tag: str, num_scenarios: int | None) -> dict[str, output.ModelOutput]:
    """Loads cached model outputs (pickle files) from a directory.

    Args:
        outputs_path: root cache directory holding one per-split subdirectory per tag.
        tag: split subdirectory to load from (e.g. 'val', 'train').
        num_scenarios: if set, limits the number of scenarios loaded.

    Returns:
        A dict mapping scenario_id -> ModelOutput.
    """
    model_outputs: dict[str, output.ModelOutput] = {}
    search_dir = outputs_path / tag if tag else outputs_path
    files = sorted(search_dir.glob("*.pkl"))
    if not files:
        error_message = f"No .pkl files found in {search_dir}"
        raise ValueError(error_message)

    print(f"Found {len(files)} file(s) in {search_dir}")
    for scenario_file in files:
        with scenario_file.open("rb") as f:
            scenario_output: output.ModelOutput = pickle.load(f)  # nosec B301
        model_outputs[scenario_output.scenario_id[0]] = scenario_output
        if num_scenarios is not None and len(model_outputs) >= num_scenarios:
            break

    if num_scenarios is not None:
        scenario_ids = list(model_outputs.keys())[:num_scenarios]
        model_outputs = {sid: model_outputs[sid] for sid in scenario_ids}

    return model_outputs


def _get_best_mode_embeddings(
    model_outputs: dict[str, output.ModelOutput],
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts the input embedding of the best predicted mode for each scenario.

    The best mode is determined by the argmax of mode_probabilities. The embedding
    for that mode is taken from scenario_embedding.scenario_dec, which has shape
    (M, Q) per scenario, yielding a (Q,) vector per scenario.

    Args:
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        scenario_ids: array of scenario IDs in insertion order.
        embeddings: array of shape (num_scenarios, Q).
    """
    scenario_ids = []
    embeddings = []
    for scenario_id, model_output in model_outputs.items():
        selected_mode = model_output.trajectory_decoder_output.mode_probabilities.value.argmax(dim=-1).item()
        emb = model_output.scenario_embedding.scenario_dec.value[selected_mode].detach().cpu().numpy()
        scenario_ids.append(scenario_id)
        embeddings.append(emb)
    return np.asarray(scenario_ids), np.stack(embeddings).astype(np.float64)


def _cluster_embeddings(
    embeddings: np.ndarray,
    num_clusters: int,
    seed: int,
) -> tuple[KMeans, np.ndarray]:
    """Fits KMeans and returns the model and cluster labels."""
    if len(embeddings) <= num_clusters:
        error_message = f"num_scenarios ({len(embeddings)}) must be greater than num_clusters ({num_clusters})"
        raise ValueError(error_message)
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init="auto")
    cluster_labels: np.ndarray = kmeans.fit_predict(embeddings)
    return kmeans, cluster_labels


def _reduce_dimensions(embeddings: np.ndarray, algorithm: str, seed: int, num_components: int = 2) -> np.ndarray:
    """Reduces embedding dimensionality for visualization.

    Applies PCA as a pre-processing step when using t-SNE to speed up computation.

    Args:
        embeddings: array of shape (N, D).
        algorithm: one of 'tsne' or 'pca'.
        seed: random seed.
        num_components: number of output dimensions (default 2).

    Returns:
        Reduced embeddings of shape (N, num_components).
    """
    match algorithm:
        case "tsne":
            pca_components = min(50, embeddings.shape[1], len(embeddings) - 1)
            if pca_components < num_components:
                error_message = (
                    f"PCA pre-reduction to {pca_components} components is less than num_components={num_components}"
                )
                raise ValueError(error_message)
            reduced = PCA(n_components=pca_components, random_state=seed).fit_transform(embeddings)
            return TSNE(
                n_components=num_components, random_state=seed, perplexity=min(30, len(embeddings) - 1)
            ).fit_transform(reduced)  # type: ignore[return-value]
        case "pca":
            return PCA(n_components=num_components, random_state=seed).fit_transform(embeddings)  # type: ignore[return-value]
        case _:
            error_message = f"Unsupported dim reduction algorithm: {algorithm}. Choose 'tsne' or 'pca'."
            raise ValueError(error_message)


def _plot_clusters(
    reduced: np.ndarray,
    cluster_labels: np.ndarray,
    num_clusters: int,
    algorithm: str,
    output_path: Path,
) -> None:
    """Scatter-plots the reduced embeddings colored by cluster assignment.

    Args:
        reduced: array of shape (N, 2) after dimensionality reduction.
        cluster_labels: integer cluster label per scenario.
        num_clusters: total number of clusters (used for colormap).
        algorithm: dim reduction algorithm name, used in plot title and filename.
        output_path: directory where the PNG will be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.get_cmap("nipy_spectral", num_clusters)
    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=cluster_labels,
        cmap=cmap,
        s=10,
        alpha=0.7,
        linewidths=0,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster ID")

    ax.set_title(f"Scenario embeddings clustered with KMeans (k={num_clusters}) — {algorithm.upper()} projection")
    ax.set_xlabel(f"{algorithm.upper()} dim 1")
    ax.set_ylabel(f"{algorithm.upper()} dim 2")
    ax.grid(visible=True, alpha=0.3)

    plt.tight_layout()
    output_filepath = output_path / f"embeddings_clusters_{algorithm}_k{num_clusters}.png"
    plt.savefig(output_filepath, dpi=150)
    plt.close()
    print(f"Saved visualization to {output_filepath}")


def analyze_embeddings(  # noqa: PLR0913
    outputs_path: Path,
    output_path: Path,
    num_clusters: int,
    dim_reduction_algorithm: str,
    tag: str,
    num_scenarios: int | None,
    seed: int,
) -> None:
    """Loads cached model outputs, clusters their embeddings, and saves a cluster visualization.

    Args:
        outputs_path: path to the directory with cached model output pickle files.
        output_path: directory to save the visualization.
        num_clusters: number of KMeans clusters.
        dim_reduction_algorithm: 'tsne' or 'pca' for 2D projection.
        tag: filename tag to filter cached files (e.g. 'val').
        num_scenarios: optional cap on the number of scenarios to load.
        seed: random seed for reproducibility.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading model outputs...")
    model_outputs = _load_model_outputs(outputs_path, tag, num_scenarios)
    print(f"Loaded {len(model_outputs)} scenarios")

    print("Extracting best-mode embeddings...")
    scenario_ids, embeddings = _get_best_mode_embeddings(model_outputs)
    print(f"Embeddings shape: {embeddings.shape}")

    print(f"Clustering into {num_clusters} clusters...")
    _, cluster_labels = _cluster_embeddings(embeddings, num_clusters, seed)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Cluster sizes — min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}")

    print(f"Reducing dimensions with {dim_reduction_algorithm.upper()}...")
    reduced = _reduce_dimensions(embeddings, dim_reduction_algorithm, seed)

    print("Saving visualization...")
    _plot_clusters(reduced, cluster_labels, num_clusters, dim_reduction_algorithm, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster and visualize cached model output embeddings.")
    parser.add_argument("--outputs_path", type=Path, required=True, help="Path to directory with cached model outputs.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("./out/analyze_embeddings/"),
        help="Directory to save the visualization.",
    )
    parser.add_argument("--num_clusters", type=int, default=100, help="Number of KMeans clusters.")
    parser.add_argument(
        "--dim_reduction_algorithm",
        type=str,
        default="tsne",
        choices=["tsne", "pca"],
        help="Dimensionality reduction algorithm for visualization.",
    )
    parser.add_argument(
        "--tag", type=str, default="train", help="Filename tag to filter cached output files (e.g. 'val')."
    )
    parser.add_argument("--num_scenarios", type=int, default=None, help="Maximum number of scenarios to load.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    configure_fonts()
    analyze_embeddings(
        outputs_path=args.outputs_path,
        output_path=args.output_path,
        num_clusters=args.num_clusters,
        dim_reduction_algorithm=args.dim_reduction_algorithm,
        tag=args.tag,
        num_scenarios=args.num_scenarios,
        seed=args.seed,
    )
