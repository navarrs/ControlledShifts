import torch


def compute_displacement_error(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    mask: torch.Tensor,
    valid_idx: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Computes the average error between the valid states of two trajectories.

    Notation:
        B: batch size
        M: number of modes
        T: number of timesteps
        D: trajectory dimensions (usually 2 for x and y)

    Args:
        pred_traj (torch.Tensor(B, M, T, D)): predicted trajectory
        gt_traj (torch.Tensor(B, 1, T, D)): ground truth trajectory
        mask (torch.Tensor(B, 1, T)): valid trajectory datapoints
        valid_idx (torch.Tensor(B)): valid_indeces for computing FDE

    Returns:
        ade (torch.Tensor(B, M)) sum of average errors across the trajectory.
        fde (torch.Tensor(B, M)) final error at the endpoint of the trajectory.
    """
    # ade_dist (B, M, T)
    ade_dist = torch.norm(pred_traj - gt_traj, 2, dim=-1)
    # ade_losses (B, M)
    ade = torch.sum(ade_dist * mask, dim=-1) / torch.sum(mask, dim=-1)
    fde = torch.gather(ade_dist, -1, valid_idx).squeeze(-1)
    return ade, fde


def compute_miss_rate(distances: torch.Tensor, miss_threshold: float = 2.0) -> torch.Tensor:
    """Computes the miss rate of the final distances.

    Notation:
        B: batch size
        M: number of modes

    Args:
        distances (torch.Tensor(B, M)): array of distances
        miss_threshold (float): value for determining of a distances is considered a miss

    Return:
        miss_rate (torch.Tensor(B))
    """
    num_modes = distances.shape[1]
    miss_values = distances > miss_threshold
    return miss_values.sum(axis=-1) / num_modes
