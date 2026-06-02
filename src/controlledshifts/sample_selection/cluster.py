"""Clustering-based sample selection strategies."""

import random
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.cluster import KMeans

from controlledshifts.sample_selection.common import (
    aggregate_selected_samples,
    allocate_removal_budget,
    compute_proportional_number_to_drop,
    greedy_select_from_sim_matrix,
    make_group_result,
    sort_ids_by_score,
)
from controlledshifts.schemas import output_schemas as output
from controlledshifts.utils import metrics as metrics_utils
from controlledshifts.utils.model_analysis_utils import get_scenario_dec_embeddings


def _get_mode_embeddings(
    model_outputs: dict[str, output.ModelOutput],
) -> tuple[NDArray[np.str_], NDArray[np.float64]]:
    """Extracts per-mode scenario_dec embeddings without flattening, sorted by descending mode probability.

    Modes are reordered so that index 0 is always the highest-probability (best) mode, index 1 the second-best, etc.

    Args:
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        scenario_ids: array of shape (N,).
        all_embeddings: array of shape (N, M, Q) — mode embeddings sorted by descending probability.
    """
    scenario_ids = []
    all_embeddings = []
    for scenario_id, model_output in model_outputs.items():
        emb = model_output.scenario_embedding.scenario_dec.value.detach().cpu().numpy().astype(np.float64)
        probs = model_output.trajectory_decoder_output.mode_probabilities.value.detach().cpu().numpy()
        sort_order = np.argsort(probs)[::-1]  # descending probability
        scenario_ids.append(scenario_id)
        all_embeddings.append(emb[sort_order])
    return np.asarray(scenario_ids), np.stack(all_embeddings)


def _fit_kmeans(
    embeddings: NDArray[np.float64],
    num_clusters: int,
    seed: int,
) -> tuple[KMeans, NDArray[np.int32]]:
    """Fits a KMeans model and returns both the fitted model and the cluster labels.

    The fitted model is returned so callers can access cluster_centers_ (centroids).
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init="auto")
    cluster_labels: NDArray[np.int32] = kmeans.fit_predict(embeddings)
    return kmeans, cluster_labels


def _cluster_embeddings(config: DictConfig, embeddings: NDArray[np.float64]) -> tuple[Any, NDArray[np.int32]]:
    """Resolves the clustering strategy from config, validates inputs, and fits the model.

    Args:
        config: must contain `clustering_strategy` and `seed`; optionally `num_clusters` (default 100).
        embeddings: array of shape (num_embeddings, embedding_dim).

    Returns:
        A (kmeans, cluster_labels) tuple from the fitted model.

    Raises:
        ValueError: if num_embeddings <= num_clusters or the clustering strategy is unsupported.
    """
    match config.clustering_strategy:
        case "kmeans":
            num_clusters = config.get("num_clusters", 100)
            if len(embeddings) <= num_clusters:
                error_message = f"num_embeddings ({len(embeddings)}) must be greater than num_clusters ({num_clusters})"
                raise ValueError(error_message)
            return _fit_kmeans(embeddings, num_clusters, config.seed)
        case _:
            error_message = f"Unsupported clustering strategy: {config.clustering_strategy}"
            raise ValueError(error_message)


def random_selection_per_cluster(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """A sample selection strategy that clusters scenario_dec embeddings using a clustering algorithm (currently only
    K-Means is supported) and randomly drops samples per cluster proportional to each cluster's size, mirroring the
    logic of random_selection_per_token.

    Args:
        config: encapsulates model analysis configuration parameters.
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        A dictionary containing the IDs of the samples to keep or drop.
    """
    scenario_ids, embeddings = get_scenario_dec_embeddings(model_outputs)
    num_scenarios = len(scenario_ids)

    _, cluster_labels = _cluster_embeddings(config, embeddings)

    clusters_df = pd.DataFrame({"scenario_id": scenario_ids, "cluster": cluster_labels})
    percentage_per_cluster = (clusters_df["cluster"].value_counts() / num_scenarios).to_frame(name="percentage")

    num_scenarios_to_drop = int((1 - config.percentage_to_keep) * num_scenarios)
    min_percentage_per_class = config.min_percentage_per_class
    valid_percentages = percentage_per_cluster[percentage_per_cluster["percentage"] > min_percentage_per_class]
    total_valid_percentage = valid_percentages["percentage"].sum()

    selected_samples = {}
    for _, row in percentage_per_cluster.iterrows():
        cluster_id = row.name
        cluster_scenario_ids = clusters_df["scenario_id"][clusters_df["cluster"] == cluster_id].tolist()

        num_to_drop = compute_proportional_number_to_drop(
            num_scenarios_to_drop, row.percentage, min_percentage_per_class, total_valid_percentage
        )

        random.seed(config.seed)
        random.shuffle(cluster_scenario_ids)
        if num_to_drop > 0:
            selected_samples[cluster_id] = make_group_result(
                keep=cluster_scenario_ids[num_to_drop:],
                drop=cluster_scenario_ids[:num_to_drop],
            )
        else:
            selected_samples[cluster_id] = make_group_result(keep=cluster_scenario_ids, drop=[])

    aggregate_selected_samples(selected_samples)
    return selected_samples


def cosine_selection_per_cluster(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """A sample selection strategy that clusters scenario_dec embeddings using K-Means and drops samples based on
    cosine similarity to the cluster centroid, mirroring the logic of alignment_based_selection_per_token.

    Samples with high cosine similarity to their cluster centroid (most typical/redundant) are prioritized for
    dropping. Supports both simple (deterministic) and Gumbel-weighted (stochastic) sorting strategies.

    Args:
        config: encapsulates model analysis configuration parameters. Requires:
            num_clusters (int), percentage_to_keep (float), min_percentage_per_class (float), seed (int),
            sorting_strategy (str, "simple" or "gumbel").
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        selected_samples: A dictionary containing the IDs of the samples to keep or drop.
    """
    scenario_ids_list, embeddings = get_scenario_dec_embeddings(model_outputs)
    scenario_ids_arr = np.array(scenario_ids_list)
    num_scenarios = len(scenario_ids_list)

    kmeans, cluster_labels = _cluster_embeddings(config, embeddings)

    centroids = kmeans.cluster_centers_
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    cluster_percentages = {
        int(c): count / num_scenarios for c, count in zip(unique_clusters, cluster_counts, strict=True)
    }
    min_percentage_per_class = config.min_percentage_per_class
    valid_percentages = {k: v for k, v in cluster_percentages.items() if v > min_percentage_per_class}
    total_valid_percentage = sum(valid_percentages.values())

    num_scenarios_to_drop = int((1 - config.percentage_to_keep) * num_scenarios)

    selected_samples = {}
    for cluster_id in unique_clusters.tolist():
        cluster_mask = cluster_labels == cluster_id
        cluster_scenario_ids = scenario_ids_arr[cluster_mask]
        cluster_embeddings = embeddings[cluster_mask]
        cluster_percentage = cluster_percentages[cluster_id]

        num_to_drop = compute_proportional_number_to_drop(
            num_scenarios_to_drop, cluster_percentage, min_percentage_per_class, total_valid_percentage
        )

        if num_to_drop > 0:
            scores = metrics_utils.compute_cosine_similarity(cluster_embeddings, centroids[cluster_id])
            sorted_ids, _ = sort_ids_by_score(cluster_scenario_ids, scores, config.sorting_strategy, config.seed)
            selected_samples[cluster_id] = make_group_result(
                keep=sorted_ids[num_to_drop:].tolist(),
                drop=sorted_ids[:num_to_drop].tolist(),
            )
        else:
            selected_samples[cluster_id] = make_group_result(keep=cluster_scenario_ids.tolist(), drop=[])

    aggregate_selected_samples(selected_samples)
    return selected_samples


def vocab_cluster_selection(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """Sample selection using a multi-mode pseudo-vocabulary and greedy diversity-based selection.

    Algorithm:
        1. Extract per-mode decoder embeddings of shape (M, Q) per scenario without flattening.
        2. Fit KMeans on the best-mode embedding of each scenario to produce K cluster centroids.
        3. Group scenarios by their best-mode cluster label.
        4. Label all M mode embeddings per scenario using the fitted centroids to build a pseudo-vocabulary of length M
            (one cluster label per mode).
        5. Allocate a removal budget per cluster proportional to cluster size, so larger clusters are pruned more
            aggressively.
        6. Within each cluster, compute a pairwise similarity matrix over pseudo-vocabularies using the configured
            alignment strategy ('hamming' or 'jaccard') and apply greedy submodular selection to retain the most
            diverse scenarios.

    Args:
        config: encapsulates model analysis configuration parameters. Requires: percentage_to_keep (float),
            alignment_strategy (str, 'hamming' or 'jaccard'), num_clusters (int, default 100), seed (int).
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        selected_samples: a dictionary containing the IDs of the samples to keep or drop.
    """
    scenario_ids, all_embeddings = _get_mode_embeddings(model_outputs)
    num_scenarios = len(scenario_ids)

    if num_scenarios == 0:
        error_message = (
            "No valid scenarios found. Check that model_outputs is not empty and have valid scenario embeddings."
        )
        raise ValueError(error_message)

    # Step 2: fit KMeans on best-mode (index 0) embeddings only — modes are sorted by descending probability.
    best_mode_embeddings = all_embeddings[:, 0]  # (N, Q)
    kmeans, best_mode_labels = _fit_kmeans(best_mode_embeddings, config.get("num_clusters", 100), config.seed)

    # Step 4: label ALL M mode embeddings per scenario using the fitted centroids.
    num_modes = all_embeddings.shape[1]
    # Reshape to (N*M, Q), predict, then reshape back to (N, M).
    all_emb_flat = all_embeddings.reshape(num_scenarios * num_modes, -1)
    all_labels = kmeans.predict(all_emb_flat).reshape(num_scenarios, num_modes).astype(np.int32)  # (N, M)

    # Step 5: allocate removal budget proportional to cluster size.
    unique_clusters, cluster_counts = np.unique(best_mode_labels, return_counts=True)
    cluster_sizes = {int(c): int(cnt) for c, cnt in zip(unique_clusters, cluster_counts, strict=True)}
    total_removal = int((1 - config.percentage_to_keep) * num_scenarios)
    removal_budgets = allocate_removal_budget(cluster_sizes, total_removal)

    alignment_strategy = config.get("alignment_strategy", "hamming")
    match alignment_strategy:
        case "hamming":
            compute_sim_matrix = metrics_utils.compute_pairwise_hamming_similarity
        case "jaccard":
            compute_sim_matrix = metrics_utils.compute_pairwise_jaccard_similarity
        case _:
            error_message = f"Unsupported alignment_strategy '{alignment_strategy}'. Choose 'hamming' or 'jaccard'."
            raise ValueError(error_message)

    selected_samples: dict[Any, Any] = {}

    for cluster_id in unique_clusters.tolist():
        cluster_mask = best_mode_labels == cluster_id
        cluster_scenario_ids = scenario_ids[cluster_mask]
        cluster_vocab = all_labels[cluster_mask]  # (cluster_size, M)

        num_to_remove = removal_budgets.get(cluster_id, 0)
        num_to_keep = len(cluster_scenario_ids) - num_to_remove

        if num_to_remove == 0:
            selected_samples[cluster_id] = make_group_result(keep=cluster_scenario_ids.tolist(), drop=[])
            continue

        # Step 6: pairwise similarity matrix over pseudo-vocabularies, then greedy submodular selection.
        sim_matrix = compute_sim_matrix(cluster_vocab)
        keep, drop = greedy_select_from_sim_matrix(cluster_scenario_ids, sim_matrix, num_to_keep)
        selected_samples[cluster_id] = make_group_result(keep=keep, drop=drop)

    aggregate_selected_samples(selected_samples)
    return selected_samples
