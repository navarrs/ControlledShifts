"""Code for the AutoBot model."""

import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig
from torch import nn

from controlledshifts.models.base_model import BaseModel
from controlledshifts.models.components import common
from controlledshifts.schemas.output_schemas import ModelOutput, ScenarioEmbedding, TrajectoryDecoderOutput


class AutoBot(BaseModel):
    """AutoBot trajectory forecasting model.

    Reference: https://arxiv.org/abs/2104.00563.
    Adapted from the UniTraj framework: https://github.com/vita-epfl/UniTraj/blob/main/unitraj/models/autobot/autobot.py
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the AutoBot model.

        Args:
            config (DictConfig): Configuration for the model.
        """
        super().__init__(config=config)

        # Ego agent encoding
        self.agents_dynamic_encoder = nn.Sequential(nn.Linear(self.config.agents_input_size, self.config.hidden_size))

        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for _ in range(self.config.num_encoder_layers):
            tx_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size,
                nhead=self.config.tx_num_heads,
                dropout=self.config.dropout,
                dim_feedforward=self.config.tx_hidden_size,
            )
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

            tx_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size,
                nhead=self.config.tx_num_heads,
                dropout=self.config.dropout,
                dim_feedforward=self.config.tx_hidden_size,
            )
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

        # Map encoding
        self.map_encoder = common.MapEncoderPts(
            d_k=self.config.hidden_size, map_attr=self.config.map_input_size, dropout=self.config.dropout
        )
        self.map_attn_layers = nn.MultiheadAttention(
            self.config.hidden_size, num_heads=self.config.tx_num_heads, dropout=self.config.dropout
        )

        # Trajectory decoding
        self.trajectory_output_query = nn.Parameter(
            torch.Tensor(self.config.future_len, 1, self.config.num_modes, self.config.hidden_size), requires_grad=True
        )
        nn.init.xavier_uniform_(self.trajectory_output_query)

        self.tx_decoder = []
        for _ in range(self.config.num_decoder_layers):
            self.tx_decoder.append(
                nn.TransformerDecoderLayer(
                    d_model=self.config.hidden_size,
                    nhead=self.config.tx_num_heads,
                    dropout=self.config.dropout,
                    dim_feedforward=self.config.tx_hidden_size,
                )
            )
        self.tx_decoder = nn.ModuleList(self.tx_decoder)

        # Positional encodings
        self.pos_encoder = common.PositionalEncoding(self.config.hidden_size, dropout=0.0, max_len=self.config.past_len)

        self.output_model = common.OutputModel(d_k=self.config.hidden_size)

        self.trajectory_modes_query = nn.Parameter(
            torch.Tensor(self.config.num_modes, 1, self.config.hidden_size), requires_grad=True
        )  # Appendix C.2.
        nn.init.xavier_uniform_(self.trajectory_modes_query)

        self.mode_map_attn = nn.MultiheadAttention(self.config.hidden_size, num_heads=self.config.tx_num_heads)

        self.prob_decoder = nn.MultiheadAttention(
            self.config.hidden_size, num_heads=self.config.tx_num_heads, dropout=self.config.dropout
        )
        self.prob_predictor = nn.Linear(self.config.hidden_size, 1)

        self.criterion = self.config.criterion

        self.apply(common.initialize_weights_with_xavier)
        self.print_and_get_num_params()

    def forward(self, batch: dict) -> ModelOutput:
        """Run the model forward pass.

        Notation:
            B: batch size
            N: max number of non-ego agents in the scene
            H: history length
            F: future length
            M: number of predicted modes
            P: number of map polylines
            L: number of points per polyline
            Da: number of agent features
            Dr: number of road features
            Qe: number of encoded queries
            Qd: number of decoded queries
            E: hidden size

        Args:
            batch (dict): dictionary containing the following input data batch:
                batch_size (int)
                input_dict (dict): dictionary containing the scenario information

        Returns:
            ModelOutput: structured outputs including scenario embeddings, trajectory decoder outputs, ground-truth
                tensors, metadata, and optional scenario scores.
        """
        inputs = batch["input_dict"]
        # history_ground_truth shape: (B, N, H, Da + 1); future_ground_truth shape: (B, F, 3) = (x, y, mask).
        history_ground_truth, future_ground_truth = BaseModel.gather_ground_truth(inputs)

        # Gathered input shapes
        #   ego_agent: (B, H, Da + 1)
        #   other_agents: (B, H, N, Da + 1)
        #   roads: (B, P, L, Dr + 1)
        ego_agent, other_agents, roads = BaseModel.gather_input(inputs)

        # Get scenario scores if available
        scenario_scores = BaseModel.gather_scores(inputs)

        batch_size = ego_agent.size(0)

        # Encode inputs and return only the ego-agent embedding, with shape (B, H, E).
        agents_tensor, ego_mask, other_agents_masks = self.process_agents_tensor(ego_agent, other_agents)
        ego_soctemp_emb = self.embed_agents(agents_tensor, other_agents_masks, return_ego_only=True)

        # Map features
        #   orig_map_features shape (B, P, E)
        #   orig_road_segs_masks shape (B, P)
        #   map_features shape (B * M, P, E) after repeating for each mode
        #   road_segs_masks shape (B * M, P) after repeating for each mode
        orig_map_features, orig_road_segs_masks = self.map_encoder(roads, ego_soctemp_emb)
        map_features = (
            orig_map_features.unsqueeze(2)
            .repeat(1, 1, self.num_modes, 1)
            .view(-1, batch_size * self.num_modes, self.config.hidden_size)
        )
        road_segs_masks = (
            orig_road_segs_masks.unsqueeze(1).repeat(1, self.num_modes, 1).view(batch_size * self.num_modes, -1)
        )

        # Repeat the tensors for the number of modes for efficient forward pass.
        context = ego_soctemp_emb.unsqueeze(2).repeat(1, 1, self.num_modes, 1)
        context = context.view(-1, batch_size * self.num_modes, self.config.hidden_size)

        # AutoBot-Ego Decoding
        #  out_seq shape (F, B * M, E) after decoding and permuting for output layer
        out_seq = self.trajectory_output_query.repeat(1, batch_size, 1, 1).view(
            self.config.future_len, batch_size * self.num_modes, -1
        )
        time_masks = self.generate_causal_mask(seq_len=self.config.future_len, device=ego_agent.device)
        for d in range(self.config.num_decoder_layers):
            ego_dec_emb_map = self.map_attn_layers(
                query=out_seq, key=map_features, value=map_features, key_padding_mask=road_segs_masks
            )[0]
            out_seq = out_seq + ego_dec_emb_map
            out_seq = self.tx_decoder[d](out_seq, context, tgt_mask=time_masks, memory_key_padding_mask=ego_mask)

        # Output distances shape (M, F, B, 5)
        out_dists = (
            self.output_model(out_seq)
            .reshape(self.config.future_len, batch_size, self.num_modes, -1)
            .permute(2, 0, 1, 3)
        )

        # Mode prediction
        mode_params_emb = self.trajectory_modes_query.repeat(1, batch_size, 1)
        mode_params_emb = self.prob_decoder(query=mode_params_emb, key=ego_soctemp_emb, value=ego_soctemp_emb)[0]
        mode_params_emb = (
            self.mode_map_attn(
                query=mode_params_emb,
                key=orig_map_features,
                value=orig_map_features,
                key_padding_mask=orig_road_segs_masks,
            )[0]
            + mode_params_emb
        )

        mode_logits = self.prob_predictor(mode_params_emb).squeeze(-1).transpose(0, 1)  # shape (B, M)
        mode_probs = F.softmax(mode_logits, dim=1)
        trajectory_decoder_output = TrajectoryDecoderOutput(
            decoded_trajectories=out_dists.permute(2, 0, 1, 3),  # reshape to (B, M, F, 5)
            mode_probabilities=mode_probs,  # pyright: ignore[reportArgumentType]
            mode_logits=mode_logits,
        )

        return ModelOutput(
            scenario_embedding=ScenarioEmbedding(scenario_dec=out_seq),  # pyright: ignore[reportArgumentType]
            trajectory_decoder_output=trajectory_decoder_output,
            history_ground_truth=history_ground_truth,
            future_ground_truth=future_ground_truth,
            dataset_name=inputs["dataset_name"],
            scenario_id=inputs["scenario_id"],
            agent_ids=inputs["obj_ids"].squeeze(-1).squeeze(-1),
            scenario_scores=scenario_scores,
        )

    def embed_agents(
        self, agents_tensor: torch.Tensor, other_agents_masks: torch.Tensor, *, return_ego_only: bool = True
    ) -> torch.Tensor:
        """Embed agents in the scenario using a factorized attention encoder.

        Notation:
            B: batch size
            N: max number of non-ego agents in the scene
            Na: number of returned agents (1 if `return_ego_only` is True, otherwise N + 1)
            H: history length
            Da: number of agent features
            E: hidden size

        Args:
            agents_tensor (torch.Tensor): tensor with shape (B, H, N + 1, Da), containing agent features.
            other_agents_masks (torch.Tensor): tensor with shape (B, H, N + 1), containing mask information for ego and
                other agents.
            return_ego_only (bool): whether to return only the ego-agent embedding or embeddings for all agents.

        Returns:
            torch.Tensor: tensor containing embedded agent features across time and social dimensions. Shape is
                (H, B, N_a, E), or (H, B, E) when `return_ego_only` is True.
        """
        # Agents emb shape: (B, H, N+1, D) -> (H, B, N+1, E) after encoding and permuting for attention layers
        agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)
        # Process through AutoBot's encoder
        for i in range(self.config.num_encoder_layers):
            agents_emb = self.compute_temporal_attention(
                agents_emb, other_agents_masks, layer=self.temporal_attn_layers[i]
            )
            agents_emb = self.compute_social_attention(agents_emb, other_agents_masks, layer=self.social_attn_layers[i])

        if return_ego_only:
            return agents_emb[:, :, 0]  # Keep only ego-agent encodings.
        return agents_emb

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate a causal decoder mask for future time steps.

        Args:
            seq_len (int): length of the sequence.
            device (torch.device): device on which to create the mask tensor.

        Returns:
            torch.Tensor: boolean tensor with shape (seq_len, seq_len), where True indicates masked (future) positions.
        """
        return (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()

    def process_agents_tensor(
        self, ego_agent: torch.Tensor, other_agents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process input observations into agent states and masks.

        Notation:
            B: batch size
            N: max number of non-ego agents in the scene
            H: history length
            Da: number of agent features
            M: number of predicted modes

        Args:
            ego_agent (torch.Tensor): tensor with shape (B, H, Da + 1), containing ego-agent features and mask.
            other_agents (torch.Tensor): tensor with shape (B, H, N, Da + 1), containing other-agent features and masks.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - agents_tensor: shape (B, H, N + 1, Da), without mask values.
                - env_masks: shape (B * M, H), repeated masks for decoding.
                - other_agents_masks: shape (B, H, N + 1), where True indicates masked agent entries.
        """
        # Ego information
        #   ego tensor shape: (B, H, D+1)
        #   ego mask shape: (B, H)
        ego_tensor = ego_agent[:, :, : self.config.agents_input_size]
        env_masks_orig = ego_agent[:, :, -1]
        env_masks = (1.0 - env_masks_orig).to(torch.bool)
        env_masks = (
            env_masks.unsqueeze(1)
            .repeat(1, self.config.num_modes, 1)
            .view(ego_agent.shape[0] * self.config.num_modes, -1)
        )

        # Other agents information
        #   agent tensor shape: (B, H, N, D+1)
        #   agent masks shape: (B, H, N)
        other_agents_tensor = other_agents[:, :, :, : self.config.agents_input_size]  # only opponent states
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), other_agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # Agent-level masks.

        # Combined agents information
        #   agents tensor shape: (B, H, 1+N, D)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), other_agents_tensor), dim=2)
        return agents_tensor, env_masks, opps_masks

    def compute_temporal_attention(
        self, agents_emb: torch.Tensor, agent_masks: torch.Tensor, layer: nn.Module
    ) -> torch.Tensor:
        """Compute temporal attention over each agent's history.

        Notation:
            H: history length
            B: batch size
            N: max number of agents in the scene
            E: embedding size

        Args:
            agents_emb (torch.Tensor): tensor with shape (H, B, N, E), containing embedded agent features.
            agent_masks (torch.Tensor): tensor with shape (B, H, N), where 1 corresponds to valid steps and 0
                corresponds to invalid steps.
            layer (nn.Module): transformer encoder layer used for temporal attention.

        Returns:
            torch.Tensor: tensor with shape (H, B, N, E) after temporal attention.
        """
        hist_len, batch_size, num_agents, _ = agents_emb.size()

        # Temporal mask shape (B * N, H)
        temp_masks = agent_masks.permute(0, 2, 1).reshape(-1, hist_len)
        temp_masks = temp_masks.masked_fill((temp_masks.sum(-1) == hist_len).unsqueeze(-1), value=False)

        # Positional encoding shape (H, B * N, E)
        pos_enc = self.pos_encoder(agents_emb.reshape(hist_len, batch_size * num_agents, -1))

        # Apply transformer encoder for temporal attention.
        agents_temp_emb = layer(pos_enc, src_key_padding_mask=temp_masks)
        return agents_temp_emb.view(hist_len, batch_size, num_agents, -1)

    def compute_social_attention(
        self, agents_emb: torch.Tensor, agent_masks: torch.Tensor, layer: nn.Module
    ) -> torch.Tensor:
        """Compute social attention across agents.

        Notation:
            H: history length
            B: batch size
            N: max number of agents in the scene
            E: embedding size

        Args:
            agents_emb (torch.Tensor): tensor with shape (H, B, N, E), containing embedded agent features.
            agent_masks (torch.Tensor): tensor with shape (B, H, N), where 1 corresponds to valid steps and 0
                corresponds to invalid steps.
            layer (nn.Module): transformer encoder layer used for social attention.

        Returns:
            torch.Tensor: tensor with shape (H, B, N, E) after social attention.
        """
        hist_len, batch_size, num_agents, _ = agents_emb.size()
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(num_agents, batch_size * hist_len, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, num_agents))
        return agents_soc_emb.view(num_agents, batch_size, hist_len, -1).permute(2, 1, 0, 3)
