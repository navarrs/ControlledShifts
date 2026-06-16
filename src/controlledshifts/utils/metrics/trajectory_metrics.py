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
        pred_traj: predicted trajectory, shape (B, M, T, D).
        gt_traj: ground truth trajectory, shape (B, 1, T, D).
        mask: valid trajectory datapoints, shape (B, 1, T).
        valid_idx: valid indices for computing FDE, shape (B).

    Returns:
        ade: sum of average errors across the trajectory, shape (B, M).
        fde: final error at the endpoint of the trajectory, shape (B, M).
    """
    # ade_dist (B, M, T)
    ade_dist = torch.norm(pred_traj - gt_traj, 2, dim=-1)
    # ade (B, M)
    ade = torch.sum(ade_dist * mask, dim=-1) / torch.sum(mask, dim=-1)
    fde = torch.gather(ade_dist, -1, valid_idx).squeeze(-1)
    return ade, fde


def compute_miss_rate(distances: torch.Tensor, miss_threshold: float = 2.0) -> torch.Tensor:
    """Computes the miss rate of the final distances.

    Notation:
        B: batch size
        M: number of modes

    Args:
        distances: final distances, shape (B, M).
        miss_threshold: distance above which a prediction is considered a miss.

    Returns:
        miss_rate: fraction of modes that miss, shape (B).
    """
    num_modes = distances.shape[1]
    miss_values = distances > miss_threshold
    return miss_values.sum(axis=-1) / num_modes
