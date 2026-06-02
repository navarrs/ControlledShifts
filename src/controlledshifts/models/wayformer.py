"""Code for the Wayformer model."""

import torch
from omegaconf import DictConfig
from torch import nn

from controlledshifts.models.base_model import BaseModel
from controlledshifts.models.components import common
from controlledshifts.schemas.output_schemas import ModelOutput, ScenarioEmbedding, TrajectoryDecoderOutput


class Wayformer(BaseModel):
    """Wayformer trajectory forecasting model."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the Wayformer model.

        Args:
            config (DictConfig): Configuration for the model.
        """
        super().__init__(config=config)
        # Agent and Road Encoders
        self.road_encoder = nn.Sequential(nn.Linear(self.config.map_input_size, self.config.hidden_size))
        self.agent_encoder = nn.Sequential(nn.Linear(self.config.agents_input_size, self.config.hidden_size))
        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self.config.max_num_agents + 1), self.config.hidden_size)),
            requires_grad=True,
        )
        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.past_len, 1, self.config.hidden_size)),
            requires_grad=True,
        )
        self.selu = nn.SELU(inplace=True)

        # NOTE: The scenario embedding, motion decoder and criterion and their respective hyper-parameters are directly
        # specified in the configuration (see `configs/model`) and instantiated through Hydra.

        # Scenario Encoder
        self.scenario_embedder = self.config.scenario_embedder

        # Trajectory decoder
        self.motion_decoder = self.config.motion_decoder

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
            batch (dict): dictionary containing the input data batch:
                batch_size (int)
                input_dict (dict): dictionary containing the scenario information

        Returns:
            ModelOutput: model outputs containing scenario embeddings, trajectory decoder outputs, ground-truth tensors,
                metadata, and optional scenario scores.
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

        # Process and embed historical information using `self.embed()`.
        scenario_embedding: ScenarioEmbedding = self.embed(ego_agent, other_agents, roads)

        # Decode the scenario trajectories
        context = scenario_embedding.scenario_dec.value
        decoded_trajectories: TrajectoryDecoderOutput = self.motion_decoder(context)

        return ModelOutput(
            scenario_embedding=scenario_embedding,
            trajectory_decoder_output=decoded_trajectories,
            history_ground_truth=history_ground_truth,
            future_ground_truth=future_ground_truth,
            dataset_name=inputs["dataset_name"],
            scenario_id=inputs["scenario_id"],
            agent_ids=inputs["obj_ids"].squeeze(-1).squeeze(-1),
            scenario_scores=scenario_scores,
        )

    def embed(self, ego_agent: torch.Tensor, other_agents: torch.Tensor, roads: torch.Tensor) -> ScenarioEmbedding:
        """Encode scenario context from agents and map features.

        Notation:
            B: batch size
            N: max number of non-ego agents in the scene
            H: history length
            P: number of map polylines
            L: number of points per polyline
            Da: number of agent features
            Dr: number of road features
            E: hidden size

        Args:
            ego_agent (torch.Tensor): tensor with shape (B, H, Da + 1), containing ego-agent features and mask.
            other_agents (torch.Tensor): tensor with shape (B, H, N, Da + 1), containing other-agent features and masks.
            roads (torch.Tensor): tensor with shape (B, P, L, Dr + 1), containing map features and mask.

        Returns:
            ScenarioEmbedding: encoded and decoded scenario context used by the trajectory decoder.
        """
        # Ego information
        #   ego tensor shape: (B, H, D+1)
        #   ego mask shape: (B, H)
        ego_tensor = ego_agent[:, :, : self.config.agents_input_size]
        ego_masks = ego_agent[:, :, -1]
        # Other agents information
        #   agent tensor shape: (B, H, N, D+1)
        #   agent masks shape: (B, H, N)
        other_agents_tensor = other_agents[:, :, :, : self.config.agents_input_size]  # only opponent states
        other_agents_masks = other_agents[:, :, :, -1]
        # Combined agents information
        #   agents tensor shape: (B, H, 1+N, D)
        #   agents mask shape: (B, H, 1+N)
        agent_masks = torch.cat((torch.ones_like(ego_masks.unsqueeze(-1)), other_agents_masks), dim=-1)
        agent_masks_inv = (1.0 - agent_masks).to(torch.bool)  # Masked agent entries.
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), other_agents_tensor), dim=2)

        batch_size, _, num_agents, _ = agents_tensor.shape
        # Encode agent information
        #   agents_emb shape: (B, H, 1+N, hidden_size)
        #   agent_posemb shape: (1, 1, 1+N, hidden_size)
        #   temporal_posemb shape: (1, H, 1, hidden_size)
        #   pos_emb shape: (1, H, 1+N, hidden_size)
        #   agents_emb shape: (B, H * (1+N), hidden_size)
        agents_emb = self.agent_encoder(agents_tensor)
        agents_emb = self.selu(agents_emb)
        pos_emb = self.agents_positional_embedding[:, :, :num_agents] + self.temporal_positional_embedding
        agents_emb = (agents_emb + pos_emb).view(batch_size, -1, self.config.hidden_size)

        # Encode map information
        #   roads tensor shape: (B, P, L, Dr)
        #   roads mask shape: (B, P, L)
        #   roads emb shape: (B, P * L, hidden_size)
        roads_tensor = roads[:, : self.max_num_roads, :, : self.config.map_input_size]
        roads_mask = roads[:, : self.max_num_roads, :, -1]
        roads_inv = (1.0 - roads_mask).to(torch.bool)
        road_emb = self.road_encoder(roads_tensor).view(batch_size, -1, self.config.hidden_size)
        road_emb = self.selu(road_emb)

        # Process mixed information using the Perceiver encoder.
        #   mixed_input_features shape: (B, H * (1+N) + P * L, hidden_size)
        #   mixed_input_masks shape: (B, H * (1+N) + P * L)
        mixed_input_features = torch.concat([agents_emb, road_emb], dim=1)
        mixed_input_masks = torch.concat([agent_masks_inv.view(batch_size, -1), roads_inv.view(batch_size, -1)], dim=1)
        return self.scenario_embedder(mixed_input_features, mixed_input_masks)
