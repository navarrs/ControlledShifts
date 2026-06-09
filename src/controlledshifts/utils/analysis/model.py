"""Utility functions for model / scenario-embedding analysis.

See `docs/ANALYSIS.md` for usage details.
"""

import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
from sklearn.manifold import TSNE

from controlledshifts.schemas import output_schemas as output


def get_scenario_dec_embeddings(
    model_outputs: dict[str, output.ModelOutput],
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.float64]]:
    """Extracts and flattens scenario_dec embeddings from model outputs.

    Args:
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.

    Returns:
        scenario_ids (npt.NDArray[np.str_]): array of scenario IDs in insertion order.
        embeddings (npt.NDArray[np.float64]): array of shape (num_scenarios, embedding_dim).
    """
    scenario_ids = []
    embeddings = []
    for scenario_id, model_output in model_outputs.items():
        emb = model_output.scenario_embedding.scenario_dec.value.detach().cpu().numpy().flatten()
        scenario_ids.append(scenario_id)
        embeddings.append(emb)
    return np.asarray(scenario_ids), np.stack(embeddings).astype(np.float64)


def compute_dimensionality_reduction(
    config: DictConfig, model_outputs: dict[str, output.ModelOutput], output_path: Path
) -> npt.NDArray[np.float64]:
    """Uses a manifold learning algorithm (TSNE, UMAP) to reduce the dimensionality of the scenario embeddings.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.
        output_path (Path): output path where visualization will be saved to.

    Returns:
        model_results (npt.NDArray[np.float64]): a numpy array of shape (num_scenarios, config.num_components)
            encapsulating the dimensionality reduction results.
    """
    algorithm = config.dim_reduction_algorithm
    match algorithm:
        case "tsne":
            model = TSNE(n_components=config.num_components, random_state=config.seed)
        case "umap":
            from umap import UMAP  # noqa: PLC0415

            model = UMAP(n_components=config.num_components, transform_seed=config.seed)
        case _:
            error_message = f"Algorithm: {config.dim_reduction_algorithm} not supported."
            raise ValueError(error_message)

    print(f"Computing {algorithm} analysis...")
    print(f"\tNumber of components {config.num_components}")
    _, embeddings = get_scenario_dec_embeddings(model_outputs)
    result = model.fit_transform(embeddings)

    if config.save_result:
        output_filepath = Path(f"{output_path}/{config.dim_reduction_algorithm}.pkl")
        with output_filepath.open("wb") as f:
            pickle.dump(result, f)

    print("\tDone")
    return result  # pyright: ignore[reportReturnType]
