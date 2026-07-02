"""Den-TP density-aware sample selection strategy."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from controlledshifts.schemas import ModelOutput
from controlledshifts.utils.constants import INVALID_AGENT_ID
from controlledshifts.utils.metrics import compute_pairwise_cosine_similarity

from .common import aggregate_selected_samples, greedy_select_from_sim_matrix, make_group_result
from .embeddings import get_scenario_dec_embeddings


def _get_agent_count(model_output: ModelOutput) -> int:
    """Returns the number of valid agent slots for a scenario, used as the Den-TP density proxy."""
    agent_ids = model_output.agent_ids.value.detach().cpu().numpy()
    return int(np.count_nonzero(agent_ids != INVALID_AGENT_ID))


def _greedy_submodular_select(
    scenario_ids: NDArray[np.str_],
    embeddings: NDArray[np.float64],
    num_to_keep: int,
) -> tuple[list[str], list[str]]:
    """Greedy submodular selection using Den-TP's P(S_j) objective with cosine similarity.

    NOTE: This implementation uses the model embeddings as a proxy for the gradient features described in the paper,
    since gradient computation requires live model access.

    Args:
        scenario_ids: array of scenario IDs.
        embeddings: array of shape (N, d) - one embedding per scenario.
        num_to_keep: number of samples to keep.

    Returns:
        (keep, drop): lists of scenario IDs.
    """
    sim_matrix = compute_pairwise_cosine_similarity(embeddings)
    return greedy_select_from_sim_matrix(scenario_ids, sim_matrix, num_to_keep)


def _dentp_allocate_budget(
    partitions: dict[int, NDArray[np.intp]],
    total_budget: int,
    agent_density_weight: float,
) -> dict[int, int]:
    """Pre-allocates the keep budget across Den-TP density bins.

    NOTE: This is an adjustment from the original Den-TP paper's DynamicSelect rule (n_k = min(|D_k|, floor(budget/k))),
    which processes bins sequentially and implicitly assumes low agent-density scenes are common. In datasets where that
    assumption does not hold - e.g. high agent-density scenes are over-represented in terms of scenario count, or the
    range of agents per scenario is small and skews the bin distribution - that rule gives such scenes an unfair share
    of the budget.
    This function instead pre-allocates the full budget across all bins simultaneously using a weight that combines both
    agent density (k, number of agents per scene) and scenario density complement (|D| - |D_k|, number of scenarios not
    in the kth bin).

    Weight formula: w_k = k^agent_density_weight / (|D| - |D_k|)
        - agent_density_weight=0    -> pure inverse-scenario-density (combat over-represented bins only)
        - agent_density_weight=1    -> agent density and scenario density balanced (default)
        - agent_density_weight-> inf -> approaches agent-density-only (original paper's implicit behaviour)

    After proportional allocation, surplus from bins capped at |D_k| is redistributed agent-density-descending;
    rounding deficits are trimmed agent-density-ascending to preserve high agent-density allocations.

    Args:
        partitions: mapping from 1-indexed density bin k to array of scenario positions.
        total_budget: total number of scenarios to keep across all bins.
        agent_density_weight: non-negative float controlling the density/frequency trade-off.

    Returns:
        Dict mapping each bin k to the number of scenarios to keep from that bin.
    """
    if not partitions:
        return {}

    bin_keys = sorted(partitions.keys())
    sizes = {k: len(partitions[k]) for k in bin_keys}

    # Check for the edge case where there's only one bin, to avoid division by zero in weight calculation.
    if len(partitions) == 1:
        (k,) = partitions
        if total_budget >= sizes[k]:
            error_message = (
                f"Total_budget ({total_budget}) exceeds the number of scenarios in the only density bin ({sizes[k]}). "
                "Reduce percentage_to_keep or check density_interval - all scenarios fall into a single bin."
            )
            raise ValueError(error_message)
        return {k: total_budget}

    # Calculate the allocation weights for each bin.
    total_num_scenarios = sum(sizes.values())
    raw_weights = {k: (k**agent_density_weight) / (total_num_scenarios - sizes[k]) for k in bin_keys}
    total_weight = sum(raw_weights.values())
    percentages = {k: round(raw_weights[k] / total_weight, 3) for k in bin_keys}

    # Proportional allocation: n_k = total_budget * w_k / total_weight, capped at |D_k| to avoid overallocation.
    allocations: dict[int, int] = {k: min(int(percentages[k] * total_budget), sizes[k]) for k in bin_keys}
    remaining = total_budget - sum(allocations.values())

    # If we still have budget left after capping, distribute the remaining budget to bins with available capacity in
    # descending order of percentage.
    if remaining > 0:
        for k in sorted(bin_keys, key=lambda x: percentages[x], reverse=True):
            available_capacity = sizes[k] - allocations[k]
            if available_capacity > 0:
                num_to_add = min(available_capacity, remaining)
                allocations[k] += num_to_add
                remaining -= num_to_add
                if remaining == 0:
                    break

    # If we over-allocated due to rounding, trim the excess from bins in ascending order of percentage to preserve
    # more data from low-density bins.
    elif remaining < 0:
        excess = -remaining
        for k in sorted(bin_keys, key=lambda x: percentages[x]):
            if allocations[k] > 0:
                num_to_remove = min(allocations[k], excess)
                allocations[k] -= num_to_remove
                excess -= num_to_remove
                if excess == 0:
                    break

    return allocations


def dentp_selection(config: DictConfig, model_outputs: dict[str, ModelOutput]) -> dict[Any, Any]:
    """Sample selection implementing the Den-TP algorithm: https://arxiv.org/pdf/2409.17385.

    Partitions scenarios into density bins by agent count and applies greedy submodular selection within each bin using
    decoder embeddings as feature representations. Budget is pre-allocated across all bins using
    w_k = k^agent_density_weight / (|D| - |D_k|), which jointly accounts for agent density and over-representation per
    category. Surplus freed by capping is redistributed density-descending; rounding deficits are trimmed
    density-ascending.

    Args:
        config: encapsulates the sample selection configuration parameters. Requires: percentage_to_keep (float),
            agent_density_interval (int, default 4), agent_density_weight (float, default 1.0).
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        selected_samples: a dictionary containing the IDs of the samples to keep or drop.
    """
    scenario_ids, embeddings = get_scenario_dec_embeddings(model_outputs)
    num_scenarios = len(scenario_ids)

    if num_scenarios == 0:
        error_message = (
            "No valid scenarios found. Check that model_outputs is not empty and has valid scenario embeddings."
        )
        raise ValueError(error_message)

    agent_counts = np.array([_get_agent_count(model_outputs[sid]) for sid in scenario_ids], dtype=np.int64)

    tau = int(config.get("agent_density_interval", 4))
    rho_min = agent_counts.min().item()

    # Assign each scenario to a 1-indexed density bin.
    k_indices = ((agent_counts - rho_min) // tau + 1).astype(int)
    max_k = k_indices.max().item()

    # Build mapping: bin index -> array of positions into scenario_ids.
    scenario_partitions: dict[int, NDArray[np.intp]] = {}
    for k in range(1, max_k + 1):
        scenario_idxs = np.where(k_indices == k)[0]
        if len(scenario_idxs) > 0:
            scenario_partitions[k] = scenario_idxs

    keep_budget = int(config.percentage_to_keep * num_scenarios)
    agent_density_weight = float(config.get("agent_density_weight", 1.0))
    bin_allocations = _dentp_allocate_budget(scenario_partitions, keep_budget, agent_density_weight)

    selected_samples: dict[Any, Any] = {}

    # Process partitions in descending order (high density first).
    for k in range(max_k, 0, -1):
        if k not in scenario_partitions:
            continue

        partition_idxs = scenario_partitions[k]
        partition_scenario_ids = scenario_ids[partition_idxs]
        partition_embeddings = embeddings[partition_idxs]

        num_to_keep = bin_allocations.get(k, 0)

        keep, drop = _greedy_submodular_select(partition_scenario_ids, partition_embeddings, num_to_keep)
        selected_samples[k] = make_group_result(keep=keep, drop=drop)

    aggregate_selected_samples(selected_samples)
    return selected_samples
