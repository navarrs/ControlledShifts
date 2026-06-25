"""Code for the Naive learned baseline."""

import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig
from torch import nn

from controlledshifts.models.base_model import BaseModel
from controlledshifts.models.components import common
from controlledshifts.schemas import ModelOutput, ScenarioEmbedding, TrajectoryDecoderOutput


class NaiveModel(BaseModel):
    """Minimal learned trajectory-forecasting baseline.

    Predicts the ego agent's future trajectory from its own (x, y) history alone — no map, no other agents, no
    attention. The flattened ego history is passed through a single-hidden-layer MLP that regresses, for each of the
    ``M`` modes, a per-step bivariate-Gaussian parameterisation of the future. A separate linear head produces the
    per-mode logits. This is intended as a lower bound for the attention-based models: whatever it cannot capture is
    the value added by map and social context.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the Naive baseline.

        Args:
            config (DictConfig): Configuration for the model. Recognises ``agents_input_size`` (number of ego input
                features, expected to be 2 for x/y) and ``mlp_hidden_size`` (the single hidden layer width).
        """
        super().__init__(config=config)

        # Number of ego input features used per step (x, y).
        self.input_size = self.config.agents_input_size
        self.mlp_hidden_size = self.config.mlp_hidden_size

        # Flattened ego history -> single hidden layer. Shared by both output heads (one nonlinearity total).
        in_dim = self.past_len * self.input_size
        self.encoder = nn.Sequential(nn.Linear(in_dim, self.mlp_hidden_size), nn.ReLU())

        # Trajectory head: per-mode bivariate-Gaussian parameters (mu_x, mu_y, log_sigma_x, log_sigma_y, rho) per step.
        self.trajectory_head = nn.Linear(
            self.mlp_hidden_size, self.num_modes * self.future_len * self.num_bivdist_params
        )
        # Mode head: per-mode logits.
        self.mode_head = nn.Linear(self.mlp_hidden_size, self.num_modes)

        self.criterion = self.config.criterion

        self.apply(common.initialize_weights_with_xavier)  # pyright: ignore[reportArgumentType]
        self.print_and_get_num_params()

    def forward(self, batch: dict) -> ModelOutput:
        """Run the Naive forward pass.

        Notation:
            B: batch size
            N: max number of agents in the scene
            H: history length
            F: future length
            M: number of predicted modes
            Da: number of agent features

        Args:
            batch (dict): dictionary containing the input data batch:
                batch_size (int)
                input_dict (dict): dictionary containing the scenario information

        Returns:
            ModelOutput: model outputs with the learned per-mode bivariate-Gaussian trajectories.
        """
        inputs = batch["input_dict"]

        # Agent history tensors.
        obj_trajs = inputs["obj_trajs"]  # shape (B, N, H, Da)
        obj_trajs_mask = inputs["obj_trajs_mask"]  # shape (B, N, H)

        # Ground-truth tensors.
        # history_ground_truth shape: (B, N, H, Da + 1); future_ground_truth shape: (B, F, 3) = (x, y, mask).
        history_ground_truth, future_ground_truth = BaseModel.gather_ground_truth(inputs)

        # Gather the ego agent's history.
        # ego_xy shape: (B, H, 2); ego_mask shape: (B, H).
        ego_xy, ego_mask = BaseModel.gather_ego_history(obj_trajs, obj_trajs_mask, inputs["track_index_to_predict"])

        # Zero out invalid history steps, then flatten the ego history into the MLP input. Shape (B, H * input_size).
        ego_xy = ego_xy * ego_mask.unsqueeze(-1)
        batch_size = ego_xy.shape[0]
        flat_history = ego_xy.reshape(batch_size, -1)

        # Single hidden layer shared by both heads. Shape (B, mlp_hidden_size).
        hidden = self.encoder(flat_history)

        # Per-mode bivariate-Gaussian parameters. The trajectory criterion (TrajectoryPrediction) interprets channels
        # 2-3 as log-sigma and channel 4 as rho, applying its own exp/clip, so the head outputs them raw.
        decoded_trajectories = self.trajectory_head(hidden).reshape(
            batch_size, self.num_modes, self.future_len, self.num_bivdist_params
        )  # shape (B, M, F, 5)

        mode_logits = self.mode_head(hidden)  # shape (B, M)
        mode_probabilities = F.softmax(mode_logits, dim=-1)

        trajectory_decoder_output = TrajectoryDecoderOutput(
            decoded_trajectories=decoded_trajectories,  # pyright: ignore[reportArgumentType]
            mode_probabilities=mode_probabilities,  # pyright: ignore[reportArgumentType]
            mode_logits=mode_logits,  # pyright: ignore[reportArgumentType]
        )

        return ModelOutput(
            scenario_embedding=ScenarioEmbedding(scenario_dec=hidden.unsqueeze(1)),  # pyright: ignore[reportArgumentType]
            trajectory_decoder_output=trajectory_decoder_output,
            history_ground_truth=history_ground_truth,  # pyright: ignore[reportArgumentType]
            future_ground_truth=future_ground_truth,  # pyright: ignore[reportArgumentType]
            dataset_name=inputs["dataset_name"],
            scenario_id=inputs["scenario_id"],
            agent_ids=inputs["obj_ids"].squeeze(-1).squeeze(-1),
            scenario_scores=BaseModel.gather_scores(inputs),
        )
