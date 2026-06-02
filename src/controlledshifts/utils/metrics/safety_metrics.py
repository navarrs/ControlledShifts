from collections.abc import Sequence

import torch

from controlledshifts.utils.constants import DEFAULT_COLLISION_THRESHOLDS


def compute_collision_rate(  # noqa: PLR0913
    ego_pred_traj: torch.Tensor,
    ego_pred_prob: torch.Tensor,
    ego_index: torch.Tensor,
    others_gt_trajs: torch.Tensor,
    others_gt_trajs_mask: torch.Tensor,
    collision_thresholds: Sequence[float] | None = None,
    *,
    best_mode_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Computes the collision rate between the predicted trajectory and other agents' trajectories.

    Notation:
        B: batch size
        M: number of predicted modes
        T: number of timesteps
        D: trajectory dimensions (usually 2 for x and y)
        N: number of other agents in the scene

    Args:
        ego_pred_traj (torch.Tensor(B, M, T, D)): predicted trajectory of the ego agent.
        ego_pred_prob (torch.Tensor(B, M)): predicted probability of each mode for the ego agent.
        ego_index (torch.Tensor(B)): index of the ego agent in the others_gt_trajs tensor.
        others_gt_trajs (torch.Tensor(B, N, T, D)): ground truth trajectories of other agents.
        others_gt_trajs_mask (torch.Tensor(B, N, T)): mask for valid trajectory points of other agents.
        collision_thresholds (Sequence[float] | None): list of distance thresholds to consider for collision
            calculation. If None, defaults to DEFAULT_COLLISION_THRESHOLDs = (0.1, 0.25, 0.5, 1.0).
        best_mode_only (bool): if True, only considers the best mode for collision calculation. Here, best mode is
            defined as the mode with the highest predicted probability. If False, considers all modes for calculation.

    Returns:
        collision_rate (dict[str, torch.Tensor]): dictionary containing the collision rate for each threshold.
    """
    if collision_thresholds is None:
        collision_thresholds = DEFAULT_COLLISION_THRESHOLDS

    batch_size, _, _, _ = ego_pred_traj.shape

    # Zero out the ego agent's trajectory in the others_gt_trajs tensor to avoid self-collision
    batch_indices = torch.arange(batch_size, device=ego_pred_traj.device)
    other_agents = others_gt_trajs.clone()
    other_agents[batch_indices, ego_index] = 0.0
    other_agents_masks = others_gt_trajs_mask.clone()
    other_agents_masks[batch_indices, ego_index] = False

    # Compute pairwise distances between predicted trajectory and other agents' trajectories
    # distances shape: (B, M, N, T)
    if best_mode_only:
        best_mode_indices = torch.argmax(ego_pred_prob, dim=1)  # shape: (B,)
        ego_pred_traj = ego_pred_traj[batch_indices, best_mode_indices]  # shape: (B, T, D)
        ego_pred_traj = ego_pred_traj.unsqueeze(1)  # shape: (B, 1, T, D)
    ego = ego_pred_traj[:, :, :, :2].unsqueeze(2)  # (B, M, 1, T, D)
    others = other_agents[:, None, :, :, :2]  # (B, 1, N, T, D)
    distances = torch.norm(ego - others, dim=-1)

    # Invalidate masked GT trajectory points
    distances = distances.masked_fill(~other_agents_masks[:, None, :, :], float("inf"))

    # TODO: address issue with vectorized approach (AssertionError: CUDA error: device-side assert triggered)
    # Check for collisions based on distance thresholds
    # thresholds = torch.tensor(collision_thresholds, device=ego_pred_traj.device).view(1, 1, 1, 1, -1)
    # # Calculate collision counts: shape (B, M, N, T, num_thresholds)
    # collision_counts = distances.unsqueeze(-1) < thresholds
    # # Collapse agents and time, each pair of predicted trajectory and other agent is considered a collision if any of
    # # the timesteps is a collision, we dont want to double count.
    # mode_collisions = collision_counts.any(dim=(2, 3))  # shape (B, M, num_thresholds)
    # Return a dictionary with the collision rate for each threshold
    # collision_rate = {}
    # if separate_by_thresholds:
    #     for i, threshold in enumerate(collision_thresholds):
    #         collision_rate[f"{threshold}"] = mode_collisions[:, :, i].float().mean(dim=1)
    # else:
    #     collision_rate = {"all": (mode_collisions.any(dim=-1)).float().mean(dim=1)}

    collision_rate = {}
    for threshold in collision_thresholds:
        # Collision counts: shape (B, M, N, T)
        collision_counts = distances < threshold
        # Mode collisions: shape (B, M)
        mode_collisions = collision_counts.any(dim=(2, 3))
        collision_rate[f"{threshold}"] = mode_collisions.float().mean(dim=1)
    return collision_rate
