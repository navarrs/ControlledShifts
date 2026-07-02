"""Random sample selection strategy."""

import random
from typing import Any

from omegaconf import DictConfig


def random_selection(config: DictConfig, scenario_ids: list[str]) -> dict[str, Any]:
    """A sample selection strategy that randomly keeps a specified percentage of all scenarios.

    Args:
        config: encapsulates the sample selection configuration parameters.
        scenario_ids: the training scenario IDs to partition.

    Returns:
        A dictionary containing the IDs of the samples (scenarios) to keep or drop for training.
    """
    # Sort before shuffling so the partition depends only on (seed, set of scenario IDs), not on the incoming ordering
    # (which derives from an unsorted glob / dict upstream). sorted() also returns a fresh list, so we do not mutate the
    # caller's list across sweep iterations.
    scenario_ids = sorted(scenario_ids)
    num_scenarios = len(scenario_ids)
    random.seed(config.seed)
    random.shuffle(scenario_ids)

    min_scenarios_to_keep = int(config.percentage_to_keep * num_scenarios)
    keep = scenario_ids[:min_scenarios_to_keep]
    drop = scenario_ids[min_scenarios_to_keep:]
    return {"keep": keep, "num_to_keep": len(keep), "drop": drop, "num_to_drop": len(drop)}
