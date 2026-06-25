"""Code for the Constant Velocity Model (CVM) baseline."""

import torch
from omegaconf import DictConfig
from torch import nn

from controlledshifts.models.base_model import BaseModel
from controlledshifts.schemas import ModelOutput, ScenarioEmbedding, TrajectoryDecoderOutput
from controlledshifts.utils.data_utils import interpolate_polyline_at_arclength, project_point_to_polylines


# Lane polyline feature layout within ``map_polylines`` (see datasets.base_dataset.get_centered_map_data): the first
# two channels are the (x, y) position and channels 3-4 are the unit direction (lane heading). The 20-way map-type
# one-hot block starts at channel 9, so the lane types (freeway / surface-street / bike-lane = ids 1/2/3) sit at
# absolute channels 10-12.
_LANE_XY_SLICE = slice(0, 2)
_LANE_DIR_SLICE = slice(3, 5)
_LANE_TYPE_ONEHOT_SLICE = slice(10, 13)


class ConstantVelocityModel(BaseModel):
    """Constant velocity trajectory forecasting baseline.

    Predicts the ego agent's future trajectory by linearly extrapolating from its most recent valid position using the
    velocity estimated from the last two valid history steps. The same trajectory is replicated across all ``M`` modes
    with uniform probability, since this baseline is deterministic.

    When ``map_aware`` is enabled the model instead snaps the ego's last position onto nearby lane centerlines and
    advances along each lane's arc-length at the observed speed, yielding one trajectory mode per candidate lane. The
    modes are weighted by how close and well-aligned each lane is to the ego's current motion. Samples with no usable
    lane nearby, or an essentially stationary ego, fall back to the straight-line extrapolation, so the baseline never
    regresses.
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

        # Map-awareness configuration (see configs/model/cvm.yaml). When disabled the model is the plain straight-line
        # constant-velocity baseline and these knobs are ignored.
        self.map_aware = bool(config.get("map_aware", False))
        self.lane_search_radius = float(config.get("lane_search_radius", 5.0))
        self.min_speed = float(config.get("min_speed", 0.5))
        self.score_temperature = float(config.get("score_temperature", 1.0))
        self.alignment_weight = float(config.get("alignment_weight", 1.0))

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
        ego_xy, ego_mask = BaseModel.gather_ego_history(obj_trajs, obj_trajs_mask, inputs["track_index_to_predict"])

        if self.map_aware:
            # Snap onto nearby lanes and advance along their centerlines, one mode per candidate lane.
            future_xy_modes, mode_logits = self._extrapolate_future_map_aware(
                ego_xy, ego_mask, inputs["map_polylines"], inputs["map_polylines_mask"]
            )  # shapes (B, M, F, 2) and (B, M)
            trajectory_decoder_output: TrajectoryDecoderOutput = self._build_multimodal_trajectory_output(
                future_xy_modes, mode_logits
            )
        else:
            # Linearly extrapolate the future trajectory from the last observed velocity.
            future_xy = self._extrapolate_future(ego_xy, ego_mask)  # shape (B, F, 2)
            # Wrap the extrapolated trajectory in the framework's GMM output container.
            trajectory_decoder_output = self._build_trajectory_output(future_xy)

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

    def _extrapolate_future_map_aware(
        self,
        ego_xy: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polylines_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extrapolate the future by advancing the ego along nearby lane centerlines, one mode per candidate lane.

        For each sample the ego's last position is projected onto every lane polyline; the closest, best-aligned lanes
        become the trajectory modes. A mode follows its lane by sampling the centerline at arc-lengths
        ``s0 + t * speed`` (constant speed along the lane), so the prediction tracks curves. Samples with no usable
        lane within ``lane_search_radius``, or with an essentially stationary ego (``speed < min_speed``), fall back to
        the straight-line extrapolation replicated across all modes.

        Notation:
            B: batch size; F: future length; M: number of modes; P: number of polylines; L: points per polyline.

        Args:
            ego_xy (torch.Tensor): ego positions, shape (B, H, 2).
            ego_mask (torch.Tensor): ego validity mask, shape (B, H).
            map_polylines (torch.Tensor): ego-centric map polyline features, shape (B, P, L, Dm).
            map_polylines_mask (torch.Tensor): map polyline validity mask, shape (B, P, L).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - future_xy_modes (torch.Tensor): per-mode future positions, shape (B, M, F, 2).
                - mode_logits (torch.Tensor): per-mode logits, shape (B, M); higher means a closer, better-aligned lane.
        """
        last_pos, velocity = self._compute_last_pos_and_velocity(ego_xy, ego_mask)  # both (B, 2)
        speed = torch.linalg.norm(velocity, dim=-1)  # (B,)

        # Straight-line baseline, reused for stationary / no-lane samples and to pad unused modes. Shape (B, F, 2).
        straight_xy = self._extrapolate_future(ego_xy, ego_mask)

        batch_size = ego_xy.shape[0]
        device = ego_xy.device
        num_polylines = map_polylines.shape[1]

        # Degenerate scene with no polylines (e.g. empty HD map): fall back entirely.
        if num_polylines == 0:
            future_xy_modes = straight_xy.unsqueeze(1).expand(-1, self.num_modes, -1, -1).contiguous()
            return future_xy_modes, torch.zeros(batch_size, self.num_modes, device=device)

        lane_xy, lane_dir, lane_valid = self._gather_lane_polylines(map_polylines, map_polylines_mask)
        scores, nearest_idx, cum_s = self._score_candidate_lanes(last_pos, velocity, lane_xy, lane_dir, lane_valid)

        # Select the top-scoring candidate lanes per sample (at most as many as there are polylines or modes).
        num_cand = min(self.num_modes, num_polylines)
        top_scores, top_idx = torch.topk(scores, num_cand, dim=-1)  # both (B, num_cand)

        # Gather the selected lanes' geometry. batch_idx broadcasts the sample index across the candidate axis.
        batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, num_cand)  # (B, num_cand)
        sel_xy = lane_xy[batch_idx, top_idx]  # (B, num_cand, L, 2)
        sel_mask = lane_valid[batch_idx, top_idx]  # (B, num_cand, L)
        sel_cum_s = cum_s[batch_idx, top_idx]  # (B, num_cand, L)
        sel_nearest = nearest_idx[batch_idx, top_idx]  # (B, num_cand)

        # Arc-length of the projected start point, then query distances s0 + t * speed for t in 1..F.
        s0 = torch.gather(sel_cum_s, -1, sel_nearest.unsqueeze(-1)).squeeze(-1)  # (B, num_cand)
        time_steps = torch.arange(1, self.future_len + 1, device=device, dtype=last_pos.dtype)  # (F,)
        query_s = s0.unsqueeze(-1) + time_steps.view(1, 1, -1) * speed.view(batch_size, 1, 1)  # (B, num_cand, F)

        # Interpolate each candidate lane at its query arc-lengths (flatten the candidate axis into the group axis).
        num_pts = sel_xy.shape[2]
        groups = batch_size * num_cand
        lane_future = interpolate_polyline_at_arclength(
            sel_xy.reshape(groups, num_pts, 2),
            sel_mask.reshape(groups, num_pts),
            sel_cum_s.reshape(groups, num_pts),
            query_s.reshape(groups, self.future_len),
        ).reshape(batch_size, num_cand, self.future_len, 2)  # (B, num_cand, F, 2)

        # Assemble the M modes: lane-following trajectories padded with the straight-line fallback.
        future_xy_modes = straight_xy.unsqueeze(1).expand(-1, self.num_modes, -1, -1).contiguous()  # (B, M, F, 2)
        valid_cand = torch.isfinite(top_scores)  # (B, num_cand): a finite score means a lane within search radius
        future_xy_modes[:, :num_cand] = torch.where(valid_cand[..., None, None], lane_future, straight_xy.unsqueeze(1))

        # Per-mode logits from the candidate scores; padded / invalid modes get a very low (finite) logit.
        neg_inf = torch.finfo(top_scores.dtype).min
        mode_logits = torch.full((batch_size, self.num_modes), neg_inf, device=device, dtype=top_scores.dtype)
        mode_logits[:, :num_cand] = (top_scores / self.score_temperature).clamp(min=neg_inf)

        # Sample-level fallback: no lane in range, or essentially stationary -> straight line with uniform weights.
        fallback = ~(valid_cand.any(dim=-1) & (speed >= self.min_speed))  # (B,)
        if bool(fallback.any()):
            future_xy_modes[fallback] = straight_xy[fallback].unsqueeze(1).expand(-1, self.num_modes, -1, -1)
            mode_logits[fallback] = 0.0
        return future_xy_modes, mode_logits

    @staticmethod
    def _gather_lane_polylines(
        map_polylines: torch.Tensor, map_polylines_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract lane positions, directions and a lane-only validity mask from the map polyline features.

        Args:
            map_polylines (torch.Tensor): ego-centric map polyline features, shape (B, P, L, Dm).
            map_polylines_mask (torch.Tensor): map polyline validity mask, shape (B, P, L).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - lane_xy (torch.Tensor): polyline point positions, shape (B, P, L, 2).
                - lane_dir (torch.Tensor): polyline unit direction (lane heading), shape (B, P, L, 2).
                - lane_valid (torch.Tensor): per-point mask that is True only for valid lane-type points, shape
                  (B, P, L).
        """
        lane_xy = map_polylines[..., _LANE_XY_SLICE]
        lane_dir = map_polylines[..., _LANE_DIR_SLICE]
        is_lane = map_polylines[..., _LANE_TYPE_ONEHOT_SLICE].sum(dim=-1) > 0  # (B, P, L)
        lane_valid = map_polylines_mask.bool() & is_lane
        return lane_xy, lane_dir, lane_valid

    def _score_candidate_lanes(
        self,
        last_pos: torch.Tensor,
        velocity: torch.Tensor,
        lane_xy: torch.Tensor,
        lane_dir: torch.Tensor,
        lane_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score each lane polyline by proximity to and heading-alignment with the ego's current motion.

        Args:
            last_pos (torch.Tensor): last observed ego position, shape (B, 2).
            velocity (torch.Tensor): per-step ego velocity, shape (B, 2).
            lane_xy (torch.Tensor): lane point positions, shape (B, P, L, 2).
            lane_dir (torch.Tensor): lane unit directions, shape (B, P, L, 2).
            lane_valid (torch.Tensor): lane-only validity mask, shape (B, P, L).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - score (torch.Tensor): per-lane score, shape (B, P); lanes with no valid point or beyond
                  ``lane_search_radius`` get ``-inf``.
                - nearest_idx (torch.Tensor): index of the nearest valid lane point, shape (B, P).
                - cum_s (torch.Tensor): cumulative arc-length along each lane, shape (B, P, L).
        """
        nearest_idx, nearest_dist, cum_s = project_point_to_polylines(last_pos, lane_xy, lane_valid)

        # Lane direction at the nearest point, shape (B, P, 2).
        gather_idx = nearest_idx[..., None, None].expand(-1, -1, 1, lane_dir.shape[-1])  # (B, P, 1, 2)
        nearest_dir = torch.gather(lane_dir, 2, gather_idx).squeeze(2)  # (B, P, 2)

        # Cosine alignment between ego heading and lane heading (both unit vectors), shape (B, P).
        speed = torch.linalg.norm(velocity, dim=-1)  # (B,)
        vel_unit = velocity / speed.clamp(min=1e-6).unsqueeze(-1)  # (B, 2)
        alignment = (nearest_dir * vel_unit[:, None, :]).sum(dim=-1)  # (B, P)

        # Closer lanes and better-aligned lanes score higher. Out-of-range / empty lanes are removed.
        score = -nearest_dist + self.alignment_weight * alignment  # (B, P)
        score = score.masked_fill(nearest_dist > self.lane_search_radius, float("-inf"))
        return score, nearest_idx, cum_s

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

    def _build_multimodal_trajectory_output(
        self, future_xy_modes: torch.Tensor, mode_logits: torch.Tensor
    ) -> TrajectoryDecoderOutput:
        """Wrap per-mode trajectories in the GMM-style output container expected by the criterion.

        Unlike :meth:`_build_trajectory_output`, the modes are distinct (one lane-following trajectory each) and carry
        real, non-uniform logits. A ``0 * dummy_param`` term is folded into the decoded trajectories and logits so that
        ``loss.backward()`` has a leaf parameter to terminate at.

        Args:
            future_xy_modes (torch.Tensor): per-mode future positions, shape (B, M, F, 2).
            mode_logits (torch.Tensor): per-mode logits, shape (B, M).

        Returns:
            TrajectoryDecoderOutput: container with
                - decoded_trajectories (torch.Tensor): GMM parameters, shape (B, M, F, 5) where the last axis stores
                  ``(mu_x, mu_y, log_sigma_x, log_sigma_y, rho)``.
                - mode_probabilities (torch.Tensor): softmax over ``mode_logits``, shape (B, M).
                - mode_logits (torch.Tensor): the per-mode logits, shape (B, M).
        """
        # Constant log-sigma and zero correlation, both shape (B, M, F, 1).
        log_sigma = torch.full_like(future_xy_modes[..., :1], self.log_sigma)
        rho = torch.zeros_like(future_xy_modes[..., :1])

        # GMM parameters per mode and timestep: (mu_x, mu_y, log_sigma_x, log_sigma_y, rho). Shape (B, M, F, 5).
        gmm_params = torch.cat([future_xy_modes, log_sigma, log_sigma, rho], dim=-1)
        # Tie outputs to the dummy parameter so `loss.backward()` has a leaf to flow into.
        decoded_trajectories = gmm_params + 0.0 * self.dummy_param
        mode_logits = mode_logits + 0.0 * self.dummy_param
        mode_probabilities = torch.softmax(mode_logits, dim=-1)

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
