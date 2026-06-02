"""Code for the Reconstruction criterion."""

import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig

from controlledshifts.models.criterion.base_criterion import Criterion
from controlledshifts.schemas.output_schemas import ModelOutput, TokenizationOutput


class Reconstruction(Criterion):
    """Criterion for reconstruction and tokenization losses."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.reduction = config.get("reduction", "mean")
        self.reconstruction_weight = config.get("reconstruction_weight", 1.0)
        self.tokenization_weight = config.get("tokenization_weight", 1.0)

    def compute_tokenization_reconstruction(self, tokenization: TokenizationOutput) -> torch.Tensor:
        """Compute tokenization reconstruction loss as MSE plus tokenization loss.

        Args:
            tokenization (TokenizationOutput): Tokenization outputs and auxiliary loss.

        Returns:
            torch.Tensor: Tokenization reconstruction loss.
        """
        pre_ae_embedding = tokenization.input_embedding.value
        post_ae_embedding = tokenization.reconstructed_embedding.value

        # Encourage the decoder to reconstruct the pre-encoded embedding.
        loss = self.reconstruction_weight * F.mse_loss(
            pre_ae_embedding,
            post_ae_embedding,
            reduction=self.reduction,
        )

        tokenization_loss = tokenization.loss
        if tokenization_loss is not None:
            return loss + self.tokenization_weight * tokenization_loss.value
        return loss

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Compute reconstruction and tokenization losses.

        Notation:
            L_scene: scenario-tokenization reconstruction loss
            L_causal: causal-tokenization reconstruction loss

        Args:
            model_output (ModelOutput): Structured model outputs.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        scenario_tokenization_loss = None
        if model_output.tokenization_output is not None:
            scenario_tokenization_loss = self.compute_tokenization_reconstruction(model_output.tokenization_output)

        # If auxiliary guidance is enabled, include causal tokenization reconstruction.
        causal_tokenization_loss = None
        if model_output.causal_tokenization_output is not None:
            causal_tokenization_loss = self.compute_tokenization_reconstruction(model_output.causal_tokenization_output)

        assert (scenario_tokenization_loss is not None) or (causal_tokenization_loss is not None), (
            "Disable reconstruction loss when neither tokenization output is available."
        )

        # Return whichever branch is available.
        if scenario_tokenization_loss is None:
            return causal_tokenization_loss.mean()
        if causal_tokenization_loss is None:
            return scenario_tokenization_loss.mean()

        # Final scalar loss.
        return (scenario_tokenization_loss + causal_tokenization_loss).mean()
