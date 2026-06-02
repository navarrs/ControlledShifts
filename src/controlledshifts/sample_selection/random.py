"""Random sample selection strategies."""

import random
from typing import Any

from omegaconf import DictConfig

from controlledshifts.schemas import output_schemas as output


def random_selection(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """A sample selection strategy that randomly keeps a specified percentage of all scenarios.

    Args:
        config: encapsulates model analysis configuration parameters.
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        A dictionary containing the IDs of the samples (scenarios) to keep or drop for training.
    """
    scenario_ids = list(model_outputs.keys())
    num_scenarios = len(scenario_ids)
    random.seed(config.seed)
    random.shuffle(scenario_ids)

    min_scenarios_to_keep = int(config.percentage_to_keep * num_scenarios)
    keep = scenario_ids[:min_scenarios_to_keep]
    drop = scenario_ids[min_scenarios_to_keep:]
    return {"keep": keep, "num_to_keep": len(keep), "drop": drop, "num_to_drop": len(drop)}
