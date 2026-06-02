"""This file includes relevant model output schemas.
NOTE: The output values don't have a specific dimension yet, since the exact output values are still in development.
"""

from typing import Any

from pydantic import BaseModel
from pydantic_tensor import Tensor
from pydantic_tensor.backend.torch import TorchTensor
from pydantic_tensor.types import Float, Int


class TokenizationOutput(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates the output values of the Tokenizers defined in 'models/scene/tokenizers'.
    NOTE: The output values don't have a specific dimension yet, since the exact output values are still in development.

    Attributes:
        token_probabilities (TorchTensor(Float)): the probability mass over the tokens classes.
        token_indices (TorchTensor(Int)): the selected token classes.
        input_embedding (TorchTensor(Float)): the embedding values before encoding and/or quantizing.
        reconstructed_embedding (TorchTensor(Float) | None): the reconstructed embedding.
        quantized_embedding (TorchTensor(Float) | None): the discretized embedding if using quantization.
        loss (TorchTensor(Float) | None): the loss value if tokenizer is using a dedicated loss function.
    """

    num_tokens: int = 0
    token_probabilities: Tensor[TorchTensor, Any, Float] | None = None
    token_indices: Tensor[TorchTensor, Any, Int]
    input_embedding: Tensor[TorchTensor, Any, Float]
    reconstructed_embedding: Tensor[TorchTensor, Any, Float] | None = None
    quantized_embedding: Tensor[TorchTensor, Any, Float] | None = None
    loss: Tensor[TorchTensor, Any, Float] | None = None


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


class SafetyOutput(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates safety classification output values.

    Attributes:
        individual_safety_gt (TorchTensor(Float)): contains categories in {0-N}, where 1 to N are the different
            safety levels. Values of 0 indicate a score below the lowest threshold; values above N indicate a score
            above the highest threshold. Invalid or padded entries are indicated by individual_safety_mask.
        individual_safety_pred (TorchTensor(Float)): contains the predicted tensor of safety levels.
        individual_safety_pred_probs (TorchTensor(Float)): contains the predicted tensor as a probability vector.
        individual_safety_logits (TorchTensor(Float)): contains the logit values before applying a Softmax.
        individual_safety_mask (TorchTensor(Float)): boolean-valued float mask (1.0 = valid, 0.0 = invalid/padded).
            Invalid entries have no valid scorer output or are padding agents.
        interaction_safety_gt (TorchTensor(Float)): contains categories in {0-N}, where 1 to N are the different
            safety levels. Values of 0 indicate a score below the lowest threshold; values above N indicate a score
            above the highest threshold. Invalid or padded entries are indicated by interaction_safety_mask.
        interaction_safety_pred (TorchTensor(Float)): contains the predicted tensor of safety levels.
        interaction_safety_pred_probs (TorchTensor(Float)): contains the predicted tensor as a probability vector.
        interaction_safety_logits (TorchTensor(Float)): contains the logit values before applying a Softmax.
        interaction_safety_mask (TorchTensor(Float)): boolean-valued float mask (1.0 = valid, 0.0 = invalid/padded).
            Invalid entries have no valid scorer output or are padding agents.
    """

    individual_safety_gt: Tensor[TorchTensor, Any, Float]
    individual_safety_pred: Tensor[TorchTensor, Any, Float]
    individual_safety_pred_probs: Tensor[TorchTensor, Any, Float]
    individual_safety_logits: Tensor[TorchTensor, Any, Float] | None = None
    individual_safety_mask: Tensor[TorchTensor, Any, Float] | None = None

    interaction_safety_gt: Tensor[TorchTensor, Any, Float]
    interaction_safety_pred: Tensor[TorchTensor, Any, Float]
    interaction_safety_pred_probs: Tensor[TorchTensor, Any, Float]
    interaction_safety_logits: Tensor[TorchTensor, Any, Float] | None = None
    interaction_safety_mask: Tensor[TorchTensor, Any, Float] | None = None


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
        tokenization_output (TokenizationOutput | None): the tokenization output values.
        safety_output (SafetyOutput | None): the safety output values.
        causal_output (CausalOutput | None): the causal output values.
        causal_tokenization_output (TokenizationOutput | None): the causal tokenization output values.
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
    tokenization_output: TokenizationOutput | None = None
    safety_output: SafetyOutput | None = None
    causal_output: CausalOutput | None = None
    causal_tokenization_output: TokenizationOutput | None = None

    # Meta Information
    history_ground_truth: Tensor[TorchTensor, Any, Float]
    future_ground_truth: Tensor[TorchTensor, Any, Float]
    dataset_name: list[str]
    scenario_id: list[str]
    agent_ids: Tensor[TorchTensor, Any, Int]
    scenario_scores: ScenarioScores | None = None
