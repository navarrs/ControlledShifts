"""Perceiver embedder for scene representation.

This module defines `PerceiverEmbedder`, which maps mixed scene features to encoded
and decoded scenario embeddings with a Perceiver IO encoder-decoder stack.
"""

import torch
from torch import nn

from controlledshifts.models.components import common
from controlledshifts.models.components.perceiver_io import PerceiverDecoder, PerceiverEncoder, TrainableQueryProvider
from controlledshifts.schemas import ScenarioEmbedding


class PerceiverEmbedder(nn.Module):
    """Encode mixed scene features and decode fixed-size scenario embeddings."""

    def __init__(
        self, num_encoder_queries: int, num_decoder_queries: int, hidden_size: int, query_init_scale: float
    ) -> None:
        """Initialize the Perceiver-based scenario embedder.

        Args:
            num_encoder_queries (int): Number of latent queries used by the
                `PerceiverEncoder`.
            num_decoder_queries (int): Number of learned output queries used by
                the `PerceiverDecoder`.
            hidden_size (int): Channel dimension for inputs, latents, and outputs.
            query_init_scale (float): Standard deviation used to initialize
                decoder queries.
        """
        super().__init__()
        # Perceiver encoder over mixed scene tokens.
        self.perceiver_encoder = PerceiverEncoder(
            num_encoder_queries,
            hidden_size,
            num_cross_attention_qk_channels=hidden_size,
            num_cross_attention_v_channels=hidden_size,
            num_self_attention_qk_channels=hidden_size,
            num_self_attention_v_channels=hidden_size,
        )
        # Perceiver decoder over trainable output queries.
        scenario_decoder_query = TrainableQueryProvider(
            num_queries=num_decoder_queries,
            num_query_channels=hidden_size,
            init_scale=query_init_scale,
        )
        self.perceiver_decoder = PerceiverDecoder(scenario_decoder_query, hidden_size)
        self.apply(common.initialize_weights_with_xavier)

    def forward(self, features: torch.Tensor, masks: torch.Tensor) -> ScenarioEmbedding:
        """Embed scene features with a Perceiver IO encoder-decoder stack.

        Notation:
            B: Batch size.
            M: Number of mixed scene tokens.
            H: Hidden size (feature channels).
            Qe: Number of encoder latent queries.
            Qd: Number of decoder output queries.

        Args:
            features (torch.Tensor): Mixed scene features with shape `(B, M, H)`.
            masks (torch.Tensor): Padding mask for `features` with shape `(B, M)`.
                `True` entries indicate padded tokens ignored by cross-attention.

        Returns:
            ScenarioEmbedding: A container with:
                scenario_enc (torch.Tensor): Encoded representation of shape `(B, Qe, H)`.
                scenario_dec (torch.Tensor): Decoded representation of shape `(B, Qd, H)`.
        """
        scenario_enc = self.perceiver_encoder(features, masks)
        scenario_dec = self.perceiver_decoder(scenario_enc)
        return ScenarioEmbedding(scenario_enc=scenario_enc, scenario_dec=scenario_dec)
