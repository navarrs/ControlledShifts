"""Code for the Constant Velocity Model (CVM) baseline."""

import torch
from omegaconf import DictConfig
from torch import nn

from controlledshifts.models.base_model import BaseModel
from controlledshifts.schemas.output_schemas import ModelOutput, ScenarioEmbedding, TrajectoryDecoderOutput


class ConstantVelocityModel(BaseModel):
    """Constant velocity trajectory forecasting baseline.

    Predicts the ego agent's future trajectory by linearly extrapolating from its most recent valid position using the
    velocity estimated from the last two valid history steps. The same trajectory is replicated across all ``M`` modes
    with uniform probability, since this baseline is deterministic.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the CVM baseline.

        Args:
            config (DictConfig): Configuration for the model. Recognises ``log_sigma`` as the constant
                log-standard-deviation used for the GMM output parameters.
        """
        super().__init__(config=config)

        # Constant log-sigma used for the bivariate-Gaussian output. Defaults to ~1m (log(1) = 0).
        self.log_sigma = float(config.get("log_sigma", 0.0))

        # Dummy parameter so the optimizer has something to step over. Predictions never depend on it
        # (a `0 * dummy_param` term keeps the loss attached to the autograd graph).
        self.dummy_param = nn.Parameter(torch.zeros(1))

        self.criterion = self.config.criterion
        self.print_and_get_num_params()

    def forward(self, batch: dict) -> ModelOutput:
        """Run the CVM forward pass.

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
            ModelOutput: model outputs with linearly extrapolated trajectories duplicated across all ``M`` modes.
        """
        inputs = batch["input_dict"]

        # Agent history tensors.
        obj_trajs = inputs["obj_trajs"]  # shape (B, N, H, Da)
        obj_trajs_mask = inputs["obj_trajs_mask"]  # shape (B, N, H)

        # Ground-truth tensors.
        # history_ground_truth shape: (B, N, H, Da + 1); future_ground_truth shape: (B, F, 3) = (x, y, mask).
        history_ground_truth, future_ground_truth = BaseModel.gather_ground_truth(inputs)

        # Gather the ego agent's history.
        # ego_xy shape: (B, H, 2)
        # ego_mask shape: (B, H)
        ego_xy, ego_mask = self._gather_ego_history(obj_trajs, obj_trajs_mask, inputs["track_index_to_predict"])

        # Linearly extrapolate the future trajectory from the last observed velocity.
        future_xy = self._extrapolate_future(ego_xy, ego_mask)  # shape (B, F, 2)

        # Wrap the extrapolated trajectory in the framework's GMM output container.
        trajectory_decoder_output: TrajectoryDecoderOutput = self._build_trajectory_output(future_xy)

        # NOTE: Placeholder scenario embedding (CVM has no learned context) — kept to satisfy the ModelOutput schema.
        batch_size = obj_trajs.shape[0]
        scenario_embedding = ScenarioEmbedding(
            scenario_dec=torch.zeros(  # shape (B, 1, 1)
                batch_size, 1, 1, device=obj_trajs.device
            ),  # pyright: ignore[reportArgumentType]
        )

        return ModelOutput(
            scenario_embedding=scenario_embedding,
            trajectory_decoder_output=trajectory_decoder_output,
            history_ground_truth=history_ground_truth,  # pyright: ignore[reportArgumentType]
            future_ground_truth=future_ground_truth,  # pyright: ignore[reportArgumentType]
            dataset_name=inputs["dataset_name"],
            scenario_id=inputs["scenario_id"],
            agent_ids=inputs["obj_ids"].squeeze(-1).squeeze(-1),
            scenario_scores=BaseModel.gather_scores(inputs),
        )

    @staticmethod
    def _gather_ego_history(
        obj_trajs: torch.Tensor, obj_trajs_mask: torch.Tensor, track_index_to_predict: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the ego agent's (x, y) history and validity mask out of the per-scene agent tensors.

        Args:
            obj_trajs (torch.Tensor): agent history features, shape (B, N, H, Da).
            obj_trajs_mask (torch.Tensor): agent history validity mask, shape (B, N, H).
            track_index_to_predict (torch.Tensor): index of the ego agent within each scene, shape (B,).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - ego_xy (torch.Tensor): ego positions, shape (B, H, 2).
                - ego_mask (torch.Tensor): ego validity mask, shape (B, H).
        """
        idx = track_index_to_predict.long()  # shape (B,)
        batch_idx = torch.arange(len(idx), device=obj_trajs.device)  # shape (B,)
        ego_xy = obj_trajs[batch_idx, idx, :, :2]  # shape (B, H, 2)
        ego_mask = obj_trajs_mask[batch_idx, idx]  # shape (B, H)
        return ego_xy, ego_mask

    def _extrapolate_future(self, ego_xy: torch.Tensor, ego_mask: torch.Tensor) -> torch.Tensor:
        """Extrapolate the ego agent's future trajectory from the last observed velocity.

        Estimates a per-step velocity from the last two valid history steps and propagates the most recent valid
        position forward over the prediction horizon.

        Args:
            ego_xy (torch.Tensor): ego positions, shape (B, H, 2).
            ego_mask (torch.Tensor): ego validity mask, shape (B, H), with 1 for valid steps.

        Returns:
            torch.Tensor: extrapolated future positions, shape (B, F, 2), where ``future_xy[:, t]`` corresponds to the
            ``(t + 1)``-th step after the last observed position.
        """
        # Last valid position and per-step velocity. Both shape (B, 2).
        last_pos, velocity = self._compute_last_pos_and_velocity(ego_xy, ego_mask)

        # Time offsets used for extrapolation, shape (F,). The horizon starts at step 1 (the first future step).
        time_steps = torch.arange(1, self.future_len + 1, device=last_pos.device, dtype=last_pos.dtype)

        # Broadcast to (B, F, 2): future_xy[b, t] = last_pos[b] + (t + 1) * velocity[b].
        return last_pos.unsqueeze(1) + time_steps.view(1, -1, 1) * velocity.unsqueeze(1)

    def _build_trajectory_output(self, future_xy: torch.Tensor) -> TrajectoryDecoderOutput:
        """Wrap the extrapolated trajectory in the GMM-style output container expected by the criterion.

        The same trajectory is replicated across all ``M`` modes with uniform probability since the baseline is
        deterministic. A ``0 * dummy_param`` term is folded into the decoded trajectories and mode logits so that
        ``loss.backward()`` has a leaf parameter to terminate at.

        Args:
            future_xy (torch.Tensor): extrapolated future positions, shape (B, F, 2).

        Returns:
            TrajectoryDecoderOutput: container with
                - decoded_trajectories (torch.Tensor): GMM parameters, shape (B, M, F, 5) where the last axis stores
                  ``(mu_x, mu_y, log_sigma_x, log_sigma_y, rho)``.
                - mode_probabilities (torch.Tensor): uniform probabilities, shape (B, M).
                - mode_logits (torch.Tensor): zero logits (uniform after softmax), shape (B, M).
        """
        batch_size = future_xy.shape[0]
        device = future_xy.device

        # Constant log-sigma and zero correlation, both shape (B, F, 1).
        log_sigma = torch.full_like(future_xy[..., :1], self.log_sigma)
        rho = torch.zeros_like(future_xy[..., :1])

        # GMM parameters per timestep: (mu_x, mu_y, log_sigma_x, log_sigma_y, rho). Shape (B, F, 5).
        gmm_params = torch.cat([future_xy, log_sigma, log_sigma, rho], dim=-1)

        # Replicate identical trajectory across all M modes. Shape (B, M, F, 5).
        decoded_trajectories = gmm_params.unsqueeze(1).expand(-1, self.num_modes, -1, -1).contiguous()
        # Tie outputs to the dummy parameter so `loss.backward()` has a leaf to flow into.
        decoded_trajectories = decoded_trajectories + 0.0 * self.dummy_param

        # Uniform mode logits/probabilities (deterministic prediction). Both shape (B, M).
        mode_logits = torch.zeros(batch_size, self.num_modes, device=device) + 0.0 * self.dummy_param
        mode_probabilities = torch.full(
            (batch_size, self.num_modes), 1.0 / self.num_modes, device=device, dtype=future_xy.dtype
        )

        return TrajectoryDecoderOutput(
            decoded_trajectories=decoded_trajectories,  # pyright: ignore[reportArgumentType]
            mode_probabilities=mode_probabilities,  # pyright: ignore[reportArgumentType]
            mode_logits=mode_logits,  # pyright: ignore[reportArgumentType]
        )

    @staticmethod
    def _compute_last_pos_and_velocity(
        ego_xy: torch.Tensor, ego_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the last valid ego position and a per-step velocity estimate.

        The velocity is taken as the displacement between the two most recent valid steps. If only one valid step is
        available, the velocity falls back to zero (stationary agent).

        Notation:
            B: batch size
            H: history length

        Args:
            ego_xy (torch.Tensor): ego positions, shape (B, H, 2).
            ego_mask (torch.Tensor): ego validity mask, shape (B, H), with 1 for valid steps.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - last_pos (torch.Tensor): shape (B, 2), the most recent valid position.
                - velocity (torch.Tensor): shape (B, 2), per-step velocity estimate.
        """
        batch_size, hist_len, _ = ego_xy.shape
        device = ego_xy.device
        mask_bool = ego_mask.bool()  # shape (B, H)

        # Per-step index grid, shape (B, H). Invalid steps get index -1 so they lose the max-reduction.
        idx_grid = torch.arange(hist_len, device=device).unsqueeze(0).expand(batch_size, -1)
        masked_idx = torch.where(mask_bool, idx_grid, torch.full_like(idx_grid, -1))
        # Index of the most recent valid step per sample, shape (B,). Clamped to 0 when nothing is valid so the
        # fallback gather is safe (velocity is forced to zero below in that case).
        last_valid_idx = masked_idx.max(dim=-1).values.clamp(min=0)

        batch_idx = torch.arange(batch_size, device=device)  # shape (B,)
        last_pos = ego_xy[batch_idx, last_valid_idx]  # shape (B, 2)

        prev_idx = (last_valid_idx - 1).clamp(min=0)  # shape (B,)
        prev_pos = ego_xy[batch_idx, prev_idx]  # shape (B, 2)
        # `prev_valid` is False both when the step before the last is masked and when the last valid step is step 0.
        prev_valid = mask_bool[batch_idx, prev_idx] & (last_valid_idx > 0)  # shape (B,)
        velocity = torch.where(  # shape (B, 2)
            prev_valid.unsqueeze(-1), last_pos - prev_pos, torch.zeros_like(last_pos)
        )
        return last_pos, velocity
