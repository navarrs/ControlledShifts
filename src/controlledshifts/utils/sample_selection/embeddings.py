"""Scenario-embedding extraction shared by the embedding-based sample selection strategies.

NOTE: this mirrors ``utils.analysis.model.get_scenario_dec_embeddings`` but is kept self-contained here so the strategy
subpackage does not import the heavy ``utils.analysis`` package (matplotlib/seaborn). Consider consolidating both into a
single light shared util if a third caller appears.
"""

import numpy as np
from numpy.typing import NDArray

from controlledshifts.schemas import ModelOutput


def get_scenario_dec_embeddings(
    model_outputs: dict[str, ModelOutput],
) -> tuple[NDArray[np.str_], NDArray[np.float64]]:
    """Extracts the flattened decoder scenario embedding (``scenario_dec``) for each scenario.

    Args:
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        scenario_ids: array of scenario IDs, shape (N,).
        embeddings: array of shape (N, D) where D is the flattened ``scenario_dec`` dimension.
    """
    scenario_ids = []
    embeddings = []
    for scenario_id, model_output in model_outputs.items():
        emb = model_output.scenario_embedding.scenario_dec.value.detach().cpu().numpy().flatten()
        scenario_ids.append(scenario_id)
        embeddings.append(emb)
    return np.asarray(scenario_ids), np.stack(embeddings).astype(np.float64)
