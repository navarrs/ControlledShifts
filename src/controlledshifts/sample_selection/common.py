"""Shared helpers for sample selection strategies."""

from typing import Any

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray


# Exponent used in Gumbel sorting for zero-weight samples — large enough to strongly deprioritize
# them without completely zeroing out their probability via uniform**inf.
_GUMBEL_LARGE_EXPONENT: float = 8.0


def aggregate_selected_samples(selected_samples: dict[Any, Any]) -> None:
    """Helper function to aggregate the sample IDs to keep and drop across groups (tokens or clusters) into a single
    list of samples to keep and drop. Mutates the input dictionary in place.

    Args:
        selected_samples: a dictionary containing the sample selection results per group.
    """
    keep = []
    drop = []
    for samples in selected_samples.values():
        keep += samples["keep"]
        drop += samples["drop"]
    selected_samples["keep"] = keep
    selected_samples["drop"] = drop
    selected_samples["num_to_keep"] = len(keep)
    selected_samples["num_to_drop"] = len(drop)


def make_group_result(keep: list[Any], drop: list[Any]) -> dict[str, Any]:
    """Constructs a per-group selection result dict."""
    return {"keep": keep, "num_to_keep": len(keep), "drop": drop, "num_to_drop": len(drop)}


def allocate_removal_budget(
    group_sizes: dict[int, int],
    total_removal: int,
) -> dict[int, int]:
    """Allocates the removal budget across groups proportional to their size.

    Larger groups receive more of the removal budget. Any deficit from integer rounding is
    distributed to the largest groups first; any surplus is trimmed from the largest groups first.

    Args:
        group_sizes: mapping from group label to the number of scenarios in that group.
        total_removal: total number of scenarios to remove across all groups.

    Returns:
        Dict mapping each group label to the number of scenarios to remove from it.
    """
    total = sum(group_sizes.values())
    if total == 0:
        return dict.fromkeys(group_sizes, 0)

    allocations: dict[int, int] = {k: int(total_removal * size / total) for k, size in group_sizes.items()}
    remaining = total_removal - sum(allocations.values())

    # Distribute leftover removals to the largest groups first.
    for k in sorted(group_sizes, key=lambda x: group_sizes[x], reverse=True):
        if remaining == 0:
            break
        available = group_sizes[k] - allocations[k]
        if available > 0:
            add = min(available, remaining)
            allocations[k] += add
            remaining -= add

    # Trim any accidental over-allocation from the largest groups first.
    if remaining < 0:
        excess = -remaining
        for k in sorted(group_sizes, key=lambda x: group_sizes[x], reverse=True):
            if excess == 0:
                break
            trim = min(allocations[k], excess)
            allocations[k] -= trim
            excess -= trim

    return allocations


def compute_proportional_number_to_drop(
    total_number_to_drop: int, percentage: float, min_percentage: float, total_valid_percentage: float
) -> int:
    """Computes the proportional number of samples to drop for a group.

    Args:
        total_number_to_drop: the total number of samples to drop across all groups.
        percentage: the percentage of samples in the group.
        min_percentage: the min percentage threshold for a group to be considered valid for dropping samples.
        total_valid_percentage: the total percentage of samples across all valid groups.

    Returns:
        0 if percentage does not exceed min_percentage, favoring underrepresented groups by flooring rather than
        ceiling the per-group drop count.
    """
    return int(percentage * total_number_to_drop / total_valid_percentage) if percentage > min_percentage else 0


def weighted_sorting(
    samples: NDArray[object], weights: NDArray[np.float64], *, sort_ascending: bool = True
) -> tuple[NDArray[object], NDArray[np.float64]]:
    """Sorts the samples of an array using based on their weight values.

    Args:
        samples: a numpy array containing samples.
        weights: weights values in [0.0, 1.0] corresponding to each sample.
        sort_ascending: if 'True' it sorts the samples in ascending order so the lowest weight values appear first.

    Returns:
        samples: the sorted samples.
        weights: the sorted weights.
    """
    if len(samples) != len(weights):
        error_message = f"Size of samples {len(samples)} and weights {len(weights)} must be the same."
        raise ValueError(error_message)

    # Sort the sample indices based on the key values. When sort_ascending=True, the lowest weight values appear first.
    sorted_indices = np.argsort(weights) if sort_ascending else np.argsort(weights)[::-1]

    return samples[sorted_indices], weights[sorted_indices]


def weighted_sorting_gumbel(
    samples: NDArray[object],
    weights: NDArray[np.float64],
    generator: Generator,
    *,
    sort_ascending: bool = True,
    large_exponent: float = np.inf,
) -> tuple[NDArray[object], NDArray[np.float64]]:
    """Sorts the samples of an array using the Gumbel Max weighted sampling trick:
        https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/. Weights are assumed to be in [0, 1].

    Args:
        samples: a numpy array containing samples.
        weights: weights values in [0.0, 1.0] corresponding to each sample.
        generator: a random generator instance.
        sort_ascending: if 'True' it sorts the samples in ascending order, based on the key values.
        large_exponent: exponent value to use for samples whose weights are zero.

    Returns:
        samples: the sorted samples.
        weights: the sorted weights.
    """
    if len(samples) != len(weights):
        error_message = f"Size of samples {len(samples)} and weights {len(weights)} must be the same."
        raise ValueError(error_message)

    # Generate random numbers in [0, 1]
    uniform = generator.random(len(samples))

    # Calculate the exponent term (1 / W_i), if the weight of a sample is low its exponent to will be high.
    exponent = np.where(weights > 0.0, 1.0 / weights, large_exponent)

    # Calculate the priority values (uniform ** (1 / W_i)). Elements in 'uniform' raised to a large power (inf) will
    # result in 0.0.
    priority = uniform**exponent

    # Sort the sample indices based on the key values. If 'sort_ascending=False' higher priority values will show first.
    sorted_indices = np.argsort(priority) if sort_ascending else np.argsort(priority)[::-1]

    return samples[sorted_indices], weights[sorted_indices]


def greedy_select_from_sim_matrix(
    scenario_ids: NDArray[np.str_],
    sim_matrix: NDArray[np.float64],
    num_to_keep: int,
) -> tuple[list[str], list[str]]:
    """Greedy submodular selection given a precomputed (N x N) similarity matrix.

    Iteratively selects samples that minimise:
        P(S_j) = Σ_{i ∈ C_k} sim(i, j)  -  Σ_{i ∈ D_k/C_k} sim(i, j)
                      selected similarity    -    unselected similarity

    Minimising P(S_j) simultaneously rewards diversity w.r.t. already-selected samples (low similarity to C_k) and
    coverage of the unselected pool (high similarity to D_k/C_k).

    Args:
        scenario_ids: array of scenario IDs.
        sim_matrix: precomputed (N, N) pairwise similarity matrix with values in [0, 1].
        num_to_keep: number of samples to keep.

    Returns:
        (keep, drop): lists of scenario IDs.
    """
    num_scenarios = len(scenario_ids)
    if num_to_keep >= num_scenarios:
        return scenario_ids.tolist(), []
    if num_to_keep <= 0:
        return [], scenario_ids.tolist()

    selected_mask = np.zeros(num_scenarios, dtype=bool)

    # selected_sim[j] = Σ_{i ∈ C_k} sim(i, j) — starts at zero (C_k is empty)
    selected_sim = np.zeros(num_scenarios, dtype=np.float64)

    # unselected_sim[j] = Σ_{i ∈ D_k\C_k} sim(i, j) — initially all are unselected
    unselected_sim = sim_matrix.sum(axis=0).copy()

    for _ in range(num_to_keep):
        # P(S_j) = selected_sim[j] - unselected_sim[j]. We want to select the sample with the lowest P(S_j) to
        # maximize coverage and diversity.
        p_scores = selected_sim - unselected_sim

        # Exclude already selected samples by setting their P(S_j) to infinity so they won't be selected again.
        p_scores[selected_mask] = np.inf

        # Select the sample with the lowest P(S_j) score.
        best_j = int(np.argmin(p_scores))
        selected_mask[best_j] = True

        # Incremental update: best_j leaves D_k\C_k and joins C_k
        selected_sim += sim_matrix[best_j]
        unselected_sim -= sim_matrix[best_j]

    return scenario_ids[selected_mask].tolist(), scenario_ids[~selected_mask].tolist()


def sort_ids_by_score(
    ids: NDArray[object],
    scores: NDArray[np.float64],
    sorting_strategy: str,
    seed: int,
) -> tuple[NDArray[object], NDArray[np.float64]]:
    """Sorts IDs by score so that the lowest-priority candidates (to drop) appear first.

    For 'gumbel': uses (1 - scores) with the Gumbel Max trick so high-scoring (typical) samples are softly
    deprioritized. For other strategies: sorts by raw scores descending so the highest-scoring IDs appear last.
    """
    if sorting_strategy == "gumbel":
        generator = default_rng(seed)
        return weighted_sorting_gumbel(
            ids, 1.0 - scores, generator, sort_ascending=True, large_exponent=_GUMBEL_LARGE_EXPONENT
        )
    return weighted_sorting(ids, scores, sort_ascending=False)
