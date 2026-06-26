"""This file includes relevant model output schemas.
NOTE: The output values don't have a specific dimension yet, since the exact output values are still in development.
"""

from typing import Any

from pydantic import BaseModel
from pydantic_tensor import Tensor
from pydantic_tensor.backend.torch import TorchTensor
from pydantic_tensor.types import Float, Int


class TrajectoryDecoderOutput(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates the output values of the MotionDecoder defined in 'models/scene/decoder.py'.
    NOTE: The output values don't have a specific dimension yet, since the exact output values are still in development.

    Attributes:
        decoded_trajectories (TorchTensor(Float)): a set M decoded trajectories data.
        mode_probabilities (TorchTensor(Int)): the probability of each mode's (M) decoded trajectory.
    """

    decoded_trajectories: Tensor[TorchTensor, Any, Float]
    mode_probabilities: Tensor[TorchTensor, Any, Float]
    mode_logits: Tensor[TorchTensor, Any, Float] | None = None


class CausalOutput(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates causal output values.

    Attributes:
        causal_gt (TorchTensor(Float)): contains values in {0, 1} where 0 means not causal and 1 means causal.
            Invalid or padded entries are indicated by causal_mask.
        causal_pred (TorchTensor(Float)): contains values in {0, 1} indicating the predicted causal class.
        causal_pred_probs (TorchTensor(Float)): contains values in [0, 1] representing probability of being causal.
        causal_logits (TorchTensor(Float)): contains the causal logit values before applying a Sigmoid activation.
        causal_mask (TorchTensor(Float)): boolean-valued float mask (1.0 = valid, 0.0 = invalid/padded).
            Invalid entries have no ground-truth causal label available or are padding agents.
    """

    causal_gt: Tensor[TorchTensor, Any, Float]
    causal_pred: Tensor[TorchTensor, Any, Float]
    causal_pred_probs: Tensor[TorchTensor, Any, Float]
    causal_logits: Tensor[TorchTensor, Any, Float] | None = None
    causal_mask: Tensor[TorchTensor, Any, Float] | None = None


class ScenarioEmbedding(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates the output values of the ScenarioEmbedded defined in 'models/scene/embedder.py'.

    Attributes:
        scenario_enc (TorchTensor(Float)): an encoded embedding of a scenario if pre-processing is performed.
        scenario_dec (TorchTensor(Int)): a decoded embedding of a scenario
    """

    scenario_enc: Tensor[TorchTensor, Any, Float] | None = None
    scenario_dec: Tensor[TorchTensor, Any, Float]


class ScenarioScores(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates the output values of the ScenarioScores

    Attributes:
        individual_agent_scores (TorchTensor(Float)): the scores of each agent in the scenario.
        individual_scenario_score (TorchTensor(Float)): the overall score of the scenario based on individual agents.
        interaction_agent_scores (TorchTensor(Float)): the scores of each agent in the scenario based on interactions.
        interaction_scenario_score (TorchTensor(Float)): the overall score of the scenario based on interactions.
    """

    individual_agent_scores: Tensor[TorchTensor, Any, Float]
    individual_scenario_score: Tensor[TorchTensor, Any, Float]
    interaction_agent_scores: Tensor[TorchTensor, Any, Float]
    interaction_scenario_score: Tensor[TorchTensor, Any, Float]


class ModelOutput(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates all model outputs decoded values of a scenario defined in 'models'.

    Attributes:
        scenario_embedding (ScenarioEmbedding): the scenario embedding output values.
        trajectory_decoder_output (TrajectoryDecoderOutput | None): the trajectory decoder output values.
        causal_output (CausalOutput | None): the causal output values.
        history_ground_truth (TorchTensor(Float)): the ground truth history trajectories.
        future_ground_truth (TorchTensor(Float)): the ground truth future trajectories.
        dataset_name (list[str]): the name of the dataset for each scenario in the batch.
        scenario_id (list[str]): the unique identifier of each scenario in the batch.
        agent_ids (TorchTensor(Int)): the unique identifiers of each agent in the batch.
        agent_scores (TorchTensor(Float) | None): the scores of each agent in the batch if available.
        scene_score (TorchTensor(Float) | None): the score of each scene in the batch if available.
    """

    scenario_embedding: ScenarioEmbedding
    trajectory_decoder_output: TrajectoryDecoderOutput | None = None
    causal_output: CausalOutput | None = None

    # Meta Information
    history_ground_truth: Tensor[TorchTensor, Any, Float]
    future_ground_truth: Tensor[TorchTensor, Any, Float]
    dataset_name: list[str]
    scenario_id: list[str]
    agent_ids: Tensor[TorchTensor, Any, Int]
    scenario_scores: ScenarioScores | None = None
