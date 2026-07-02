"""Clustering-based sample selection strategies."""

import random
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.cluster import KMeans

from controlledshifts.schemas import ModelOutput
from controlledshifts.utils.metrics import compute_cosine_similarity

from .common import (
    aggregate_selected_samples,
    compute_proportional_number_to_drop,
    make_group_result,
    sort_ids_by_score,
)
from .embeddings import get_scenario_dec_embeddings


def _fit_kmeans(
    embeddings: NDArray[np.float64],
    num_clusters: int,
    seed: int,
) -> tuple[KMeans, NDArray[np.int32]]:
    """Fits a KMeans model and returns both the fitted model and the cluster labels.

    The fitted model is returned so callers can access ``cluster_centers_`` (centroids).
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init="auto")
    cluster_labels = kmeans.fit_predict(embeddings).astype(np.int32)
    return kmeans, cluster_labels


def _cluster_embeddings(config: DictConfig, embeddings: NDArray[np.float64]) -> tuple[KMeans, NDArray[np.int32]]:
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


def random_selection_per_cluster(config: DictConfig, model_outputs: dict[str, ModelOutput]) -> dict[Any, Any]:
    """Clusters ``scenario_dec`` embeddings (currently only K-Means) and randomly drops samples per cluster
    proportional to each cluster's size.

    Args:
        config: encapsulates the sample selection configuration parameters.
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

    selected_samples: dict[Any, Any] = {}
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


def cosine_selection_per_cluster(config: DictConfig, model_outputs: dict[str, ModelOutput]) -> dict[Any, Any]:
    """A sample selection strategy that clusters ``scenario_dec`` embeddings using K-Means and drops samples based on
    cosine similarity to the cluster centroid.

    Samples with high cosine similarity to their cluster centroid (most typical/redundant) are prioritized for
    dropping. Supports both simple (deterministic) and Gumbel-weighted (stochastic) sorting strategies.

    Args:
        config: encapsulates the sample selection configuration parameters. Requires: num_clusters (int),
            percentage_to_keep (float), min_percentage_per_class (float), seed (int), sorting_strategy (str, "simple" or
            "gumbel").
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

    selected_samples: dict[Any, Any] = {}
    for cluster_id in unique_clusters.tolist():
        cluster_mask = cluster_labels == cluster_id
        cluster_scenario_ids = scenario_ids_arr[cluster_mask]
        cluster_embeddings = embeddings[cluster_mask]
        cluster_percentage = cluster_percentages[cluster_id]

        num_to_drop = compute_proportional_number_to_drop(
            num_scenarios_to_drop, cluster_percentage, min_percentage_per_class, total_valid_percentage
        )

        if num_to_drop > 0:
            scores = compute_cosine_similarity(cluster_embeddings, centroids[cluster_id])
            sorted_ids, _ = sort_ids_by_score(cluster_scenario_ids, scores, config.sorting_strategy, config.seed)
            selected_samples[cluster_id] = make_group_result(
                keep=sorted_ids[num_to_drop:].tolist(),
                drop=sorted_ids[:num_to_drop].tolist(),
            )
        else:
            selected_samples[cluster_id] = make_group_result(keep=cluster_scenario_ids.tolist(), drop=[])

    aggregate_selected_samples(selected_samples)
    return selected_samples
