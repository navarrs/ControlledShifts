import torch
from torch import nn as nn
from einops import rearrange

from controlledshifts.models.components.common import initialize_weights_with_normal
from controlledshifts.models.components.cross_attention import CrossAttentionLayer
from controlledshifts.models.components.self_attention import SelfAttentionBlock, SelfAttentionLayer

class QueryProvider:
    """Provider of cross-attention query input."""

    @property
    def num_query_channels(self):
        raise NotImplementedError

    def __call__(self, x=None):
        raise NotImplementedError

class TrainableQueryProvider(nn.Module, QueryProvider):
    """Provider of learnable cross-attention query input.

    This is the latent array in Perceiver IO encoders and the output query array in most Perceiver IO decoders.
    """
    def __init__(self, num_queries: int, num_query_channels: int, init_scale: float = 0.02):
        super().__init__()
        self._query = nn.Parameter(torch.empty(num_queries, num_query_channels))
        self._initialize_weights_with_normal(init_scale)

    def _initialize_weights_with_normal(self, init_scale: float):
        with torch.no_grad():
            self._query.normal_(0.0, init_scale)

    @property
    def num_query_channels(self):
        return self._query.shape[-1]

    def forward(self, x=None):
        return rearrange(self._query, "... -> 1 ...")

class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: int | None = None,
        num_cross_attention_v_channels: int | None = None,
        num_cross_attention_layers: int = 1,
        first_cross_attention_layer_shared: bool = False,
        cross_attention_widening_factor: int = 4,
        num_self_attention_heads: int = 4,
        num_self_attention_qk_channels: int | None = None,
        num_self_attention_v_channels: int | None = None,
        num_self_attention_layers_per_block: int = 1,
        num_self_attention_blocks: int = 2,
        first_self_attention_block_shared: bool = False,
        self_attention_widening_factor: int = 4,
        dropout: float = 0.1,
        residual_dropout: float = 0.0,
        init_scale: float = 0.02,
    ):
        """Generic Perceiver IO encoder.

        :param input_adapter: Transforms and position-encodes task-specific input to generic encoder input of shape (B,
                M, C) where B is the batch size, M the input sequence length and C the number of key/value input
                channels. C is determined by the `num_input_channels` property of the `input_adapter`.
        :param num_latents: Number of latent variables (N).
        :param num_latent_channels: Number of latent channels (D).
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention
                (see`MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention (see
                `MultiHeadAttention.num_v_channels` for details).
        :param num_cross_attention_layers: Number of cross-attention layers (alternating with self-attention blocks).
        :param first_cross_attention_layer_shared: Whether the first cross-attention layer should share its weights with
                subsequent cross-attention layers (if any).
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_qk_channels: Number of query and key channels for self-attention (see
                `MultiHeadAttention.num_qk_channels` for details).
        :param num_self_attention_v_channels: Number of value channels for self-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_self_attention_layers_per_block: Number of self-attention layers per self-attention block.
        :param num_self_attention_blocks: Number of self-attention blocks, with weights shared between corresponding
            self-attention layers.
        :param first_self_attention_block_shared: Whether the first self-attention block should share its weights with
                subsequent self-attention blocks (if any).
        :param dropout: Dropout probability for self- and cross-attention layers.
        :param residual_dropout: Dropout probability for residual connections.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for each self-attention layer and
                each cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        self.latent_provider = TrainableQueryProvider(num_latents, num_latent_channels, init_scale=0.02)

        if num_cross_attention_layers <= 0:
            raise ValueError("num_cross_attention_layers must be > 0")

        if num_self_attention_blocks <= 0:
            raise ValueError("num_self_attention_blocks must be > 0")

        if num_cross_attention_layers > num_self_attention_blocks:
            raise ValueError("num_cross_attention_layers must be <= num_self_attention_blocks")

        self.num_cross_attention_layers = num_cross_attention_layers
        self.num_self_attention_blocks = num_self_attention_blocks

        self.first_cross_attention_layer_shared = first_cross_attention_layer_shared
        self.first_self_attention_block_shared = first_self_attention_block_shared

        def cross_attn():
            layer = CrossAttentionLayer(
                num_heads=num_cross_attention_heads,
                num_q_input_channels=num_latent_channels,
                num_kv_input_channels=num_latent_channels,
                num_qk_channels=num_cross_attention_qk_channels,
                num_v_channels=num_cross_attention_v_channels,
                widening_factor=cross_attention_widening_factor,
                dropout=dropout,
                residual_dropout=residual_dropout,
            )
            return layer

        def self_attn():
            return SelfAttentionBlock(
                num_layers=num_self_attention_layers_per_block,
                num_heads=num_self_attention_heads,
                num_channels=num_latent_channels,
                num_qk_channels=num_self_attention_qk_channels,
                num_v_channels=num_self_attention_v_channels,
                widening_factor=self_attention_widening_factor,
                dropout=dropout,
                residual_dropout=residual_dropout,
            )

        self.cross_attn_1 = cross_attn()
        self.self_attn_1 = self_attn()

        if self.extra_cross_attention_layer:
            self.cross_attn_n = cross_attn()

        if self.extra_self_attention_block:
            self.self_attn_n = self_attn()

        self._initialize_weights_with_normal(init_scale)

    def _initialize_weights_with_normal(self, init_scale: float):
        with torch.no_grad():
            initialize_weights_with_normal(self, init_scale)

    @property
    def extra_cross_attention_layer(self):
        return self.num_cross_attention_layers > 1 and not self.first_cross_attention_layer_shared

    @property
    def extra_self_attention_block(self):
        return self.num_self_attention_blocks > 1 and not self.first_self_attention_block_shared

    def forward(self, x, pad_mask=None, return_adapted_input=False):
        b, *_ = x.shape

        x_adapted = x
        x_latent = self.latent_provider()

        x_latent = self.cross_attn_1(x_latent, x_adapted, pad_mask=pad_mask).last_hidden_state
        x_latent = self.self_attn_1(x_latent).last_hidden_state

        cross_attn_n = self.cross_attn_n if self.extra_cross_attention_layer else self.cross_attn_1
        self_attn_n = self.self_attn_n if self.extra_self_attention_block else self.self_attn_1

        for i in range(1, self.num_self_attention_blocks):
            if i < self.num_cross_attention_layers:
                x_latent = cross_attn_n(x_latent, x_adapted, pad_mask=pad_mask).last_hidden_state
            x_latent = self_attn_n(x_latent).last_hidden_state

        if return_adapted_input:
            return x_latent, x_adapted
        return x_latent


class PerceiverDecoder(nn.Module):
    def __init__(
        self,
        output_query_provider: QueryProvider,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_layers: int = 8,
        num_cross_attention_qk_channels: int | None = None,
        num_cross_attention_v_channels: int | None = None,
        cross_attention_widening_factor: int = 4,
        cross_attention_residual: bool = True,
        dropout: float = 0.1,
        init_scale: float = 0.02,
    ):
        """Generic Perceiver IO decoder.

        :param output_adapter: Transforms generic decoder cross-attention output of shape (B, O, F) to task-specific
                output. B is the batch size, O the output sequence length and F the number of cross-attention output
                channels.
        :param output_query_provider: Provides the decoder's output query. Abstracts over output query details e.g. can
                be a learned query, a deterministic function of the model's input, etc. Configured by `PerceiverIO`
                subclasses.
        :param num_latent_channels: Number of latent channels of the Perceiver IO encoder output.
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention             (see
                `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param dropout: Dropout probability for cross-attention layer.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for the decoder's
            cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        self.output_query_provider = output_query_provider

        self.num_cross_attention_layers = num_cross_attention_layers
        self.self_attn = nn.ModuleList(
            [
                SelfAttentionLayer(
                    num_heads=num_cross_attention_heads,
                    num_channels=num_latent_channels,
                    num_qk_channels=num_latent_channels,
                    num_v_channels=num_latent_channels,
                    causal_attention=False,
                    widening_factor=cross_attention_widening_factor,
                    dropout=dropout,
                )
                for _ in range(num_cross_attention_layers)
            ],
        )
        self.cross_attn = nn.ModuleList(
            [
                CrossAttentionLayer(
                    num_heads=num_cross_attention_heads,
                    num_q_input_channels=output_query_provider.num_query_channels,
                    num_kv_input_channels=num_latent_channels,
                    num_qk_channels=num_cross_attention_qk_channels,
                    num_v_channels=num_cross_attention_v_channels,
                    widening_factor=cross_attention_widening_factor,
                    attention_residual=cross_attention_residual,
                    dropout=dropout,
                )
                for _ in range(num_cross_attention_layers)
            ],
        )

        self._initialize_weights_with_normal(init_scale)

    def _initialize_weights_with_normal(self, init_scale: float):
        with torch.no_grad():
            initialize_weights_with_normal(self, init_scale)

    def forward(self, x_latent, x_adapted=None, **kwargs):
        output_query = self.output_query_provider(x_adapted)

        output = self.cross_attn[0](output_query, x_latent).last_hidden_state

        for i in range(1, len(self.cross_attn)):
            output = self.self_attn[i - 1](output).last_hidden_state
            output = self.cross_attn[i](output, x_latent).last_hidden_state

        output = self.self_attn[-1](output).last_hidden_state
        return output
