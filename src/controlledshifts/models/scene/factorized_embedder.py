"""Factorized scene embedder for scenario tokenization."""

import torch
from torch import nn

from controlledshifts.models.components import common
from controlledshifts.models.components.cross_attention import CrossAttentionBlock
from controlledshifts.models.components.self_attention import FactorizedSelfAttentionBlock
from controlledshifts.schemas.output_schemas import ScenarioEmbedding


class FactorizedEmbedder(nn.Module):
    """Encode scene features with factorized attention over time, agents, and map."""

    def __init__(  # noqa: PLR0913
        self,
        num_pre_self_attention_blocks: int,
        num_pre_cross_attention_blocks: int,
        num_mid_self_attention_blocks: int,
        num_mid_cross_attention_blocks: int,
        num_post_self_attention_blocks: int,
        hidden_size: int,
        num_timesteps: int,
        num_agents: int,
        num_queries: int,
        num_heads: int,
        widening_factor: int,
        dropout: float = 0.0,
        *,
        bias: bool = True,
    ) -> None:
        """Initialize the factorized scenario embedder.

        Args:
            num_pre_self_attention_blocks (int): Number of self-attention
                pairs (time then agents) before pre cross-attention.
            num_pre_cross_attention_blocks (int): Number of pre
                cross-attention blocks between agents and map features.
            num_mid_self_attention_blocks (int): Number of self-attention
                pairs (time then agents) in the middle stage.
            num_mid_cross_attention_blocks (int): Number of middle
                cross-attention blocks between agents and map features.
            num_post_self_attention_blocks (int): Number of self-attention
                pairs (time then agents) in the post stage.
            hidden_size (int): Feature-channel dimension.
            num_timesteps (int): Number of time steps in the agent history.
            num_agents (int): Maximum number of non-ego agents.
            num_queries (int): Number of output scenario queries.
            num_heads (int): Number of attention heads.
            widening_factor (int): Expansion ratio for MLP layers.
            dropout (float): Dropout probability.
            bias (bool): If `True`, enables bias terms in linear layers.
        """
        super().__init__()

        def get_self_attention_block(across: str) -> FactorizedSelfAttentionBlock:
            return FactorizedSelfAttentionBlock(
                input_size=hidden_size,
                hidden_size=hidden_size,
                widening_factor=widening_factor,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                across=across,
            )

        def get_cross_attention_block() -> CrossAttentionBlock:
            return CrossAttentionBlock(
                input_size=hidden_size,
                hidden_size=hidden_size,
                widening_factor=widening_factor,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
            )

        self.attention_blocks = []

        # Social-temporal encoding based on factorized attention: https://arxiv.org/pdf/2106.08417.pdf
        # Each stage applies self-attention across time, then across agents.
        for _ in range(num_pre_self_attention_blocks):
            self.attention_blocks.append(get_self_attention_block(across="time"))
            self.attention_blocks.append(get_self_attention_block(across="agents"))

        # Cross-attention block for agents and map
        for _ in range(num_pre_cross_attention_blocks):
            self.attention_blocks.append(get_cross_attention_block())

        # Intermediate self-attention and cross-attention blocks
        for _ in range(num_mid_self_attention_blocks):
            self.attention_blocks.append(get_self_attention_block(across="time"))
            self.attention_blocks.append(get_self_attention_block(across="agents"))

        # Cross-attention block for agents and map
        for _ in range(num_mid_cross_attention_blocks):
            self.attention_blocks.append(get_cross_attention_block())

        # Post self-attention blocks
        for _ in range(num_post_self_attention_blocks):
            self.attention_blocks.append(get_self_attention_block(across="time"))
            self.attention_blocks.append(get_self_attention_block(across="agents"))

        # Combine all blocks
        self.attention_blocks = nn.ModuleList(self.attention_blocks)

        # Refiner layers that map factorized features into decoder-compatible scene tokens.
        # Temporal refiner summarizes time: (B, A, T, H) -> (B, A, H).
        self.temporal_refiner = nn.Sequential(nn.Linear(num_timesteps * hidden_size, hidden_size), nn.GELU())

        # Social refiner summarizes agents: (B, A, H) -> (B, H).
        self.social_refiner = nn.Sequential(nn.Linear((1 + num_agents) * hidden_size, hidden_size), nn.GELU())

        # Scenario decoder produces Q scenario prototypes.
        self.num_queries = num_queries
        self.scenario_decoder = nn.Linear(hidden_size, self.num_queries * hidden_size)

        self.apply(common.initialize_weights_with_xavier)

    def forward(
        self,
        agent_features: torch.Tensor,
        agent_masks: torch.Tensor,
        road_features: torch.Tensor,
        road_masks: torch.Tensor,
    ) -> ScenarioEmbedding:
        """Embed scene features with factorized self- and cross-attention.

        Notation:
            B: Batch size.
            A: Number of agents, including ego (`A = 1 + N`).
            T: Number of timesteps.
            P: Number of map tokens.
            H: Hidden size (feature channels).
            Q: Number of output scenario queries.

        Args:
            agent_features (torch.Tensor): Agent features with shape `(B, A, T, H)`.
            agent_masks (torch.Tensor): Padding mask for `agent_features` with shape `(B, A, T)`. `True` entries are
                masked.
            road_features (torch.Tensor): Map features with shape `(B, P, H)`.
            road_masks (torch.Tensor): Padding mask for `road_features` with shape `(B, P)`. `True` entries are masked.

        Returns:
            ScenarioEmbedding: A container with:
                scenario_enc (torch.Tensor): Encoded scene features of shape `(B, A, T, H)`.
                scenario_dec (torch.Tensor): Decoded scene features of shape `(B, Q, H)`.
        """
        # Encode scene context with factorized attention across time, agents, and map.
        scenario_enc = agent_features
        for block in self.attention_blocks:
            scenario_enc = block(scenario_enc, road_features, agent_masks, road_masks)

        # Summarize the time axis.
        batch_size, num_agents, _, _ = scenario_enc.shape
        # Reshape for temporal summary: (B, A, T, H) -> (B, A, T * H).
        # After temporal summary: (B, A, H).
        scenario_dec = self.temporal_refiner(scenario_enc.view(batch_size, num_agents, -1))
        # Reshape for social summary: (B, A, H) -> (B, A * H).
        # After social summary: (B, H).
        scenario_dec = self.social_refiner(scenario_dec.view(batch_size, -1))
        # Decode into scenario tokens: (B, H) -> (B, Q, H).
        scenario_dec = self.scenario_decoder(scenario_dec).reshape(batch_size, self.num_queries, -1)
        return ScenarioEmbedding(scenario_enc=scenario_enc, scenario_dec=scenario_dec)
