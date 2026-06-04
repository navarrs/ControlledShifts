import json
import math
import os
import pickle
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from torch.utils.data import Sampler

from controlledshifts.schemas import output_schemas as output
from controlledshifts.utils import pylogger
from controlledshifts.utils.constants import TrajectoryType


MAX_NUM_BATCHES = 1_000_000
_LOGGER = pylogger.get_pylogger(__name__)

# Number of coordinate components for 2D (x, y) and 3D (x, y, z) points.
_NUM_2D_COORDS = 2
_NUM_3D_COORDS = 3
# Number of dimensions of a (batch, time, features) tensor.
_TENSOR_NDIM_3D = 3


def minmax_scaler(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalizes an input array using Min-Max normalization.

    Args:
        x (NDArray[np.float64]): input array.

    Returns:
        NDArray[np.float64]: array normalized to the [0, 1] range.
    """
    # compute the distribution range
    value_range = np.max(x) - np.min(x)

    # move the distribution so that it starts from zero by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def classify_track(  # noqa: PLR0913
    start_point: NDArray[np.float64],
    end_point: NDArray[np.float64],
    start_velocity: NDArray[np.float64],
    end_velocity: NDArray[np.float64],
    start_heading: float,
    end_heading: float,
) -> int:
    """Classifies a trajectory into a `TrajectoryType` from its endpoints, velocities, and headings.

    The classification strategy is taken from waymo_open_dataset/metrics/motion_metrics_utils.cc#L28.

    Args:
        start_point (NDArray[np.float64]): start position as `(x, y)`.
        end_point (NDArray[np.float64]): end position as `(x, y)`.
        start_velocity (NDArray[np.float64]): start velocity as `(vx, vy)`.
        end_velocity (NDArray[np.float64]): end velocity as `(vx, vy)`.
        start_heading (float): start heading in radians.
        end_heading (float): end heading in radians.

    Returns:
        int: the value of the matching `TrajectoryType`.
    """
    # Parameters for classification, taken from WOD
    max_speed_for_stationary = 2.0  # (m/s)
    max_displacement_for_stationary = 5.0  # (m)
    max_lateral_displacement_for_straight = 5.0  # (m)
    min_longitudinal_displacement_for_uturn = -5.0  # (m)
    max_abs_heading_diff_for_straight = np.pi / 6.0  # (rad)

    x_delta = end_point[0] - start_point[0]
    y_delta = end_point[1] - start_point[1]

    final_displacement = np.hypot(x_delta, y_delta)
    heading_diff = end_heading - start_heading
    normalized_delta = np.array([x_delta, y_delta])
    # Rotate the displacement into the start-heading frame so that its components become longitudinal (dx,
    # along the initial heading) and lateral (dy), which the thresholds below are expressed in terms of.
    rotation_matrix = np.array(
        [[np.cos(-start_heading), -np.sin(-start_heading)], [np.sin(-start_heading), np.cos(-start_heading)]],
    )
    normalized_delta = np.dot(rotation_matrix, normalized_delta)
    start_speed = np.hypot(start_velocity[0], start_velocity[1])
    end_speed = np.hypot(end_velocity[0], end_velocity[1])
    max_speed = max(start_speed, end_speed)
    dx, dy = normalized_delta

    # Check for different trajectory types based on the computed parameters.
    if max_speed < max_speed_for_stationary and final_displacement < max_displacement_for_stationary:
        return TrajectoryType.STATIONARY.value
    if np.abs(heading_diff) < max_abs_heading_diff_for_straight:
        if np.abs(normalized_delta[1]) < max_lateral_displacement_for_straight:
            return TrajectoryType.STRAIGHT.value
        return TrajectoryType.STRAIGHT_RIGHT.value if dy < 0 else TrajectoryType.STRAIGHT_LEFT.value
    if heading_diff < -max_abs_heading_diff_for_straight and dy < 0:
        return (
            TrajectoryType.RIGHT_U_TURN.value
            if normalized_delta[0] < min_longitudinal_displacement_for_uturn
            else TrajectoryType.RIGHT_TURN.value
        )
    if dx < min_longitudinal_displacement_for_uturn:
        return TrajectoryType.LEFT_U_TURN.value
    return TrajectoryType.LEFT_TURN.value


def get_heading(trajectory: NDArray[np.float64]) -> NDArray[np.float64]:
    """Approximates per-step headings from a sequence of positions.

    Args:
        trajectory (NDArray[np.float64]): array of shape `(Time, 2)` with `(x, y)` positions.

    Returns:
        NDArray[np.float64]: array of shape `(Time - 1,)` with the heading at each step in radians.
    """
    dx = np.diff(trajectory[:, 0])
    dy = np.diff(trajectory[:, 1])
    return np.arctan2(dy, dx)


def get_trajectory_type(output: list[dict[str, Any]]) -> None:
    """Classifies each data sample's trajectory and stores it in place.

    For every sample, the trajectory is classified with `classify_track` and the resulting type is written back under
    the ``trajectory_type`` key. Samples that fail classification are set to -1.

    Args:
        output (list[dict[str, Any]]): data samples with the keys ``center_gt_final_valid_idx``,
            ``obj_trajs_future_state``, ``obj_trajs``, and ``obj_trajs_mask``. Mutated in place.
    """
    for data_sample in output:
        # Get last gt position, velocity and heading
        valid_end_point = int(data_sample["center_gt_final_valid_idx"])
        end_point = data_sample["obj_trajs_future_state"][0, valid_end_point, :2]  # (x,y)
        end_velocity = data_sample["obj_trajs_future_state"][0, valid_end_point, 2:]  # (vx, vy)
        # Get last heading, manually approximate it from the series of future position
        end_heading = get_heading(data_sample["obj_trajs_future_state"][0, : valid_end_point + 1, :2])[-1]

        # Get start position, velocity and heading.
        assert data_sample["obj_trajs_mask"][0, -1]  # Assumes that the start point is always valid
        start_point = data_sample["obj_trajs"][0, -1, :2]  # (x,y)
        start_velocity = data_sample["obj_trajs"][0, -1, -4:-2]  # (vx, vy)
        start_heading = 0.0  # Initial heading is zero

        # Classify the trajectory
        try:
            trajectory_type = classify_track(
                start_point,
                end_point,
                start_velocity,
                end_velocity,
                start_heading,
                end_heading,
            )
        except Exception:  # noqa: BLE001
            trajectory_type = -1
        data_sample["trajectory_type"] = trajectory_type


def set_random_seed(seed: int) -> None:
    """Seeds Python, NumPy, and PyTorch RNGs and enables deterministic cuDNN behavior.

    Args:
        seed (int): the random seed to apply.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def estimate_kalman_filter(history: NDArray[np.float64], prediction_horizon: int) -> tuple[float, float]:  # noqa: PLR0915
    """Predicts a future position by running a Kalman filter over the position history.

    Code taken from "On Exposing the Challenging Long Tail in Future Prediction of Traffic Actors". The `p_*` arrays
    hold the state covariances and the `k_*` arrays hold the Kalman gains.

    Args:
        history (NDArray[np.float64]): array of shape `(length_of_history, 2)` with `(x, y)` positions.
        prediction_horizon (int): number of steps into the future to predict.

    Returns:
        tuple[float, float]: the predicted `(x, y)` position.
    """
    length_history = history.shape[0]
    # Measurements: the observed x/y positions the filter is fed step by step.
    z_x = history[:, 0]
    z_y = history[:, 1]
    # Seed the constant-velocity model with the mean per-step displacement over the history.
    v_x = 0
    v_y = 0
    for index in range(length_history - 1):
        v_x += z_x[index + 1] - z_x[index]
        v_y += z_y[index + 1] - z_y[index]
    v_x = v_x / (length_history - 1)
    v_y = v_y / (length_history - 1)
    # State estimates (x_*) and their covariances (p_* for position, p_v* for velocity).
    x_x = np.zeros(length_history + 1, np.float32)
    x_y = np.zeros(length_history + 1, np.float32)
    p_x = np.zeros(length_history + 1, np.float32)
    p_y = np.zeros(length_history + 1, np.float32)
    p_vx = np.zeros(length_history + 1, np.float32)
    p_vy = np.zeros(length_history + 1, np.float32)

    # we initialize the uncertainty to one (unit gaussian)
    p_x[0] = 1.0
    p_y[0] = 1.0
    p_vx[0] = 1.0
    p_vy[0] = 1.0
    x_x[0] = z_x[0]
    x_y[0] = z_y[0]

    # q: process noise (trust in the motion model); r: measurement noise (trust in observations).
    q = 0.00001
    r = 0.0001
    # Kalman gains, recomputed each step from the current covariances.
    k_x = np.zeros(length_history + 1, np.float32)
    k_y = np.zeros(length_history + 1, np.float32)
    k_vx = np.zeros(length_history + 1, np.float32)
    k_vy = np.zeros(length_history + 1, np.float32)
    k = length_history - 2
    for k in range(length_history - 1):
        # Predict: advance the state by the constant velocity and grow the covariance by the process noise q.
        x_x[k + 1] = x_x[k] + v_x
        x_y[k + 1] = x_y[k] + v_y
        p_x[k + 1] = p_x[k] + p_vx[k] + q
        p_y[k + 1] = p_y[k] + p_vy[k] + q
        p_vx[k + 1] = p_vx[k] + q
        p_vy[k + 1] = p_vy[k] + q
        # Update: the gain weights predicted covariance against measurement noise r, then corrects the
        # state toward the measurement z and shrinks the covariance accordingly.
        k_x[k + 1] = p_x[k + 1] / (p_x[k + 1] + r)
        k_y[k + 1] = p_y[k + 1] / (p_y[k + 1] + r)
        x_x[k + 1] = x_x[k + 1] + k_x[k + 1] * (z_x[k + 1] - x_x[k + 1])
        x_y[k + 1] = x_y[k + 1] + k_y[k + 1] * (z_y[k + 1] - x_y[k + 1])
        p_x[k + 1] = p_x[k + 1] - k_x[k + 1] * p_x[k + 1]
        p_y[k + 1] = p_y[k + 1] - k_y[k + 1] * p_y[k + 1]
        k_vx[k + 1] = p_vx[k + 1] / (p_vx[k + 1] + r)
        k_vy[k + 1] = p_vy[k + 1] / (p_vy[k + 1] + r)
        p_vx[k + 1] = p_vx[k + 1] - k_vx[k + 1] * p_vx[k + 1]
        p_vy[k + 1] = p_vy[k + 1] - k_vy[k + 1] * p_vy[k + 1]

    # Extrapolate the filtered state prediction_horizon steps past the last measurement.
    k = k + 1
    x_x[k + 1] = x_x[k] + v_x * prediction_horizon
    x_y[k + 1] = x_y[k] + v_y * prediction_horizon
    p_x[k + 1] = p_x[k] + p_vx[k] * prediction_horizon * prediction_horizon + q
    p_y[k + 1] = p_y[k] + p_vy[k] * prediction_horizon * prediction_horizon + q
    p_vx[k + 1] = p_vx[k] + q
    p_vy[k + 1] = p_vy[k] + q
    return x_x[k + 1], x_y[k + 1]


def calculate_epe(pred: tuple[float, float], gt: NDArray[np.float64]) -> float:
    """Computes the Euclidean end-point error between a predicted and ground-truth position.

    Args:
        pred (tuple[float, float]): predicted `(x, y)` position.
        gt (NDArray[np.float64]): ground-truth position as `(x, y)`.

    Returns:
        float: the Euclidean distance between the two positions.
    """
    diff_x = (gt[0] - pred[0]) * (gt[0] - pred[0])
    diff_y = (gt[1] - pred[1]) * (gt[1] - pred[1])
    return math.sqrt(diff_x + diff_y)


def count_valid_steps_past(mask: NDArray[np.bool_]) -> int:
    """Counts the number of trailing valid steps in a mask.

    Args:
        mask (NDArray[np.bool_]): 1D mask where True entries are valid.

    Returns:
        int: the number of valid steps counted backwards from the end until the first zero, or the
            full length of the mask if it contains no zeros.
    """
    # Reverse the mask so the most recent step is first, then the position of the first invalid (zero)
    # entry equals the count of contiguous valid steps at the end of the original mask.
    reversed_mask = mask[::-1]
    idx_of_first_zero = np.where(reversed_mask == 0)[0]
    if len(idx_of_first_zero) == 0:
        return len(mask)  # If no zeros, every step is valid
    return idx_of_first_zero[0]


def get_kalman_difficulty(output: list[dict[str, Any]], sampling_freq: int = 10) -> None:
    """Computes the Kalman difficulty at 2s, 4s, and 6s for each data sample, in place.

    If the ground-truth future is not valid up to the considered second, the difficulty for that
    horizon is set to -1. The result is stored under the ``kalman_difficulty`` key.

    Args:
        output (list[dict[str, Any]]): data samples with the keys ``obj_trajs``, ``obj_trajs_mask``,
            ``obj_trajs_future_state``, and ``center_gt_final_valid_idx``. Mutated in place.
        sampling_freq (int): frequency at which the trajectory is sampled. In WOMD, datapoints are sampled at 10hz.
    """
    num_steps_2s = 2 * sampling_freq - 1  # -1 since counting from 0
    num_steps_4s = 4 * sampling_freq - 1
    num_steps_6s = 6 * sampling_freq - 1
    for data_sample in output:
        # past trajectory of agent of interest
        past_trajectory = data_sample["obj_trajs"][0, :, :2]  # Time X (x,y)
        past_mask = data_sample["obj_trajs_mask"][0, :]
        valid_past = count_valid_steps_past(past_mask)
        past_trajectory_valid = past_trajectory[-valid_past:, :]  # Time(valid) X (x,y)

        # future gt trajectory of agent of interest
        gt_future = data_sample["obj_trajs_future_state"][0, :, :2]  # Time x (x, y)
        # Get last valid position
        valid_future = int(data_sample["center_gt_final_valid_idx"])

        kalman_difficulty_2s, kalman_difficulty_4s, kalman_difficulty_6s = -1, -1, -1
        try:
            if valid_future >= num_steps_2s:  # -1 since counting from 0
                # Get kalman future prediction at the horizon length, second argument is horizon length
                kalman_2s = estimate_kalman_filter(past_trajectory_valid, num_steps_2s + 1)  # (x,y)
                gt_future_2s = gt_future[num_steps_2s, :]
                kalman_difficulty_2s = calculate_epe(kalman_2s, gt_future_2s)

                if valid_future >= num_steps_4s:
                    kalman_4s = estimate_kalman_filter(past_trajectory_valid, num_steps_4s + 1)  # (x,y)
                    gt_future_4s = gt_future[num_steps_4s, :]
                    kalman_difficulty_4s = calculate_epe(kalman_4s, gt_future_4s)

                    if valid_future >= num_steps_6s:
                        kalman_6s = estimate_kalman_filter(past_trajectory_valid, num_steps_6s + 1)  # (x,y)
                        gt_future_6s = gt_future[num_steps_6s, :]
                        kalman_difficulty_6s = calculate_epe(kalman_6s, gt_future_6s)
        except Exception:  # noqa: BLE001
            kalman_difficulty_2s, kalman_difficulty_4s, kalman_difficulty_6s = -1, -1, -1
        data_sample["kalman_difficulty"] = np.array([kalman_difficulty_2s, kalman_difficulty_4s, kalman_difficulty_6s])


def is_ddp() -> bool:
    """Returns whether the process is running under Distributed Data Parallel.

    Returns:
        bool: True if the ``WORLD_SIZE`` environment variable is set.
    """
    return "WORLD_SIZE" in os.environ


def generate_mask(current_index: int, total_length: int, interval: int) -> NDArray[np.int_]:
    """Builds a periodic 0/1 mask anchored at a given index.

    Args:
        current_index (int): index that the periodic pattern is anchored to.
        total_length (int): length of the mask to produce.
        interval (int): spacing between consecutive valid (1) positions.

    Returns:
        NDArray[np.int_]: array of length `total_length` with 1 at positions whose offset from
            `current_index` is a multiple of `interval`, and 0 elsewhere.
    """
    mask = []
    for i in range(total_length):
        # Check if the position is a multiple of the frequency starting from current_index
        if (i - current_index) % interval == 0:
            mask.append(1)
        else:
            mask.append(0)

    return np.array(mask)


def get_polyline_dir(polyline: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes the unit direction vector at each point of a polyline.

    Args:
        polyline (NDArray[np.float64]): array of shape `(N, D)` with the polyline points.

    Returns:
        NDArray[np.float64]: array of shape `(N, D)` with the normalized direction at each point.
    """
    # Shift the polyline forward by one so each point can be differenced against its predecessor; the
    # first point is differenced against itself (zero direction). Normalize, clipping to avoid /0.
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    return diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)


def check_numpy_to_torch(x: NDArray[np.float64] | torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Converts a NumPy array to a float tensor, leaving tensors untouched.

    Args:
        x (NDArray[np.float64] | torch.Tensor): the input array or tensor.

    Returns:
        tuple[torch.Tensor, bool]: the value as a tensor and a flag that is True if the input was a NumPy array.
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z_tensor(
    points: NDArray[np.float64] | torch.Tensor,
    angle: NDArray[np.float64] | torch.Tensor,
) -> NDArray[np.float64] | torch.Tensor:
    """Rotates points around the Z-axis using PyTorch, accepting NumPy or tensor inputs.

    Args:
        points (NDArray[np.float64] | torch.Tensor): points of shape `(B, N, 3 + C)`.
        angle (NDArray[np.float64] | torch.Tensor): per-batch angle of shape `(B,)` along the z-axis; the
            angle increases as x rotates towards y.

    Returns:
        NDArray[np.float64] | torch.Tensor: the rotated points, returned as a NumPy array if `points` was a
            NumPy array.
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    if points.shape[-1] == _NUM_2D_COORDS:
        rot_matrix = torch.stack((cosa, sina, -sina, cosa), dim=1).view(-1, 2, 2).float()
        points_rot = torch.matmul(points, rot_matrix)
    else:
        ones = angle.new_ones(points.shape[0])
        rot_matrix = (
            torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1).view(-1, 3, 3).float()
        )
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def rotate_points_along_z(points: NDArray[np.float64], angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate points around the Z-axis using the given angle.

    Args:
        points (NDArray[np.float64]): array of shape `(B, N, 3 + C)` — B batches, N points per batch, 3
            coordinates `(x, y, z)` plus C extra channels.
        angle (NDArray[np.float64]): array of shape `(B,)` with the angle for each batch in radians.

    Returns:
        NDArray[np.float64]: the rotated points.
    """
    # Checking if the input is 2D or 3D points
    is_2d = points.shape[-1] == _NUM_2D_COORDS

    # Cosine and sine of the angles
    cosa = np.cos(angle)
    sina = np.sin(angle)

    if is_2d:
        # Rotation matrix for 2D
        rot_matrix = np.stack((cosa, sina, -sina, cosa), axis=1).reshape(-1, 2, 2)

        # Apply rotation
        points_rot = np.matmul(points, rot_matrix)
    else:
        # Rotation matrix for 3D
        rot_matrix = np.stack(
            (
                cosa,
                sina,
                np.zeros_like(angle),
                -sina,
                cosa,
                np.zeros_like(angle),
                np.zeros_like(angle),
                np.zeros_like(angle),
                np.ones_like(angle),
            ),
            axis=1,
        ).reshape(-1, 3, 3)

        # Apply rotation to the first 3 dimensions
        points_rot = np.matmul(points[:, :, :3], rot_matrix)

        # Concatenate any additional dimensions back
        if points.shape[-1] > _NUM_3D_COORDS:
            points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)

    return points_rot


def find_true_segments(mask: NDArray[np.bool_]) -> list[list[int]]:
    """Returns indices of each contiguous run of True values in mask."""
    # np.diff is non-zero exactly where the value flips; those transition points (plus the array ends)
    # bound the runs. Keep only the runs that start on a True value.
    change_points = np.where(np.diff(mask))[0] + 1
    indices = np.concatenate(([0], change_points, [len(mask)]))
    return [list(range(int(indices[i]), int(indices[i + 1]))) for i in range(len(indices) - 1) if mask[indices[i]]]


def merge_batch_by_padding_2nd_dim(
    tensor_list: list[torch.Tensor],
    *,
    return_pad_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Stacks tensors that differ along their 2nd dimension by zero-padding to the maximum length.

    Args:
        tensor_list (list[torch.Tensor]): tensors of shape `(B, T, F1)` or `(B, T, F1, F2)`, all
            sharing the trailing feature dimensions but possibly differing in `T`.
        return_pad_mask (bool): whether to also return the padding mask.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: the padded, concatenated tensor, and the
            boolean padding mask if `return_pad_mask` is True.
    """
    assert len(tensor_list[0].shape) in [_TENSOR_NDIM_3D, 4]
    only_3d_tensor = False
    if len(tensor_list[0].shape) == _TENSOR_NDIM_3D:
        # Add a trailing singleton feature dim so 3D and 4D inputs share the same padding code path.
        tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]
        only_3d_tensor = True
    # Pad every tensor's 2nd dimension up to the largest one seen across the batch.
    maxt_feat0 = max([x.shape[1] for x in tensor_list])

    _, _, num_feat1, num_feat2 = tensor_list[0].shape

    ret_tensor_list = []
    ret_mask_list = []
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]
        assert cur_tensor.shape[2] == num_feat1
        assert cur_tensor.shape[3] == num_feat2

        # Copy the real values into a zero-padded buffer and mark the copied region True in the mask.
        new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0, num_feat1, num_feat2)
        new_tensor[:, : cur_tensor.shape[1], :, :] = cur_tensor
        ret_tensor_list.append(new_tensor)

        new_mask_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0)
        new_mask_tensor[:, : cur_tensor.shape[1]] = 1
        ret_mask_list.append(new_mask_tensor.bool())

    ret_tensor = torch.cat(ret_tensor_list, dim=0)  # (num_stacked_samples, num_feat0_maxt, num_feat1, num_feat2)
    ret_mask = torch.cat(ret_mask_list, dim=0)

    if only_3d_tensor:
        ret_tensor = ret_tensor.squeeze(dim=-1)

    if return_pad_mask:
        return ret_tensor, ret_mask
    return ret_tensor


def get_batch_offsets(batch_idxs: torch.Tensor, bs: int) -> torch.Tensor:
    """Computes per-batch offsets from a tensor of batch indices.

    Args:
        batch_idxs (torch.Tensor): tensor of shape `(N,)` with the batch index of each element.
        bs (int): the batch size.

    Returns:
        torch.Tensor: tensor of shape `(bs + 1,)` with the cumulative element offsets per batch.
    """
    # Each offset is the running total of elements in the preceding batches, so batch i occupies
    # the slice [batch_offsets[i], batch_offsets[i + 1]) of a flat, batch-sorted tensor.
    batch_offsets = torch.zeros(bs + 1).int()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def interpolate_polyline(polyline: NDArray[np.float64], step: float = 0.5) -> NDArray[np.float64]:
    """Resamples a polyline at a fixed arc-length step using linear interpolation.

    Args:
        polyline (NDArray[np.float64]): array of shape `(N, >= 2)`; only the first two columns are used.
        step (float): arc-length spacing between resampled points.

    Returns:
        NDArray[np.float64]: array of shape `(M, 3)` with the resampled points and a zero z-coordinate, or the
            original polyline if it contains a single point.
    """
    # Calculate the cumulative distance along the polyline
    if polyline.shape[0] == 1:
        return polyline
    polyline = polyline[:, :2]
    distances = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)  # start with a distance of 0

    # Create the new distance array
    max_distance = distances[-1]
    new_distances = np.arange(0, max_distance, step)

    # Interpolate for x, y, z
    new_polyline = []
    for dim in range(polyline.shape[1]):
        interp_func = interp1d(distances, polyline[:, dim], kind="linear")
        new_polyline.append(interp_func(new_distances))

    new_polyline = np.column_stack(new_polyline)
    # add the third dimension back with zeros
    return np.concatenate((new_polyline, np.zeros((new_polyline.shape[0], 1))), axis=1)


def polyline_cumulative_arclength(poly_xy: torch.Tensor, poly_mask: torch.Tensor) -> torch.Tensor:
    """Cumulative arc-length along each polyline, measured from its first point.

    Segment lengths spanning a masked endpoint are treated as zero, so trailing padding points do not advance the
    arc-length (valid points are assumed contiguous from index 0, as produced by the dataset's segment packing).

    Args:
        poly_xy (torch.Tensor): polyline point coordinates, shape ``(..., M, 2)``.
        poly_mask (torch.Tensor): per-point validity mask, shape ``(..., M)``.

    Returns:
        torch.Tensor: cumulative arc-length at each point, shape ``(..., M)``; the first point is always 0.
    """
    # Per-segment displacement length between consecutive points, shape (..., M - 1).
    seg_len = torch.linalg.norm(poly_xy[..., 1:, :] - poly_xy[..., :-1, :], dim=-1)
    # Zero out segments whose either endpoint is invalid so padding does not inflate the length.
    seg_valid = poly_mask[..., 1:].bool() & poly_mask[..., :-1].bool()
    seg_len = seg_len * seg_valid.to(seg_len.dtype)
    cum = torch.cumsum(seg_len, dim=-1)
    # Prepend a leading zero for the first point.
    return torch.cat([torch.zeros_like(cum[..., :1]), cum], dim=-1)


def project_point_to_polylines(
    point: torch.Tensor, poly_xy: torch.Tensor, poly_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a per-sample query point onto each polyline by nearest valid vertex.

    Notation:
        B: batch size
        P: number of polylines per sample
        M: max number of points per polyline

    Args:
        point (torch.Tensor): query points, shape ``(B, 2)``.
        poly_xy (torch.Tensor): polyline point coordinates, shape ``(B, P, M, 2)``.
        poly_mask (torch.Tensor): per-point validity mask, shape ``(B, P, M)``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - nearest_idx (torch.Tensor): index of the nearest valid vertex per polyline, shape ``(B, P)``.
            - nearest_dist (torch.Tensor): distance from ``point`` to that vertex, shape ``(B, P)``; polylines with no
              valid point get ``+inf``.
            - cum_s (torch.Tensor): cumulative arc-length at each point, shape ``(B, P, M)``.
    """
    cum_s = polyline_cumulative_arclength(poly_xy, poly_mask)  # (B, P, M)
    # Distance from the query point to every polyline vertex, shape (B, P, M).
    dist = torch.linalg.norm(poly_xy - point[:, None, None, :], dim=-1)
    # Mask invalid vertices out of the min-reduction.
    dist = dist.masked_fill(~poly_mask.bool(), float("inf"))
    nearest_dist, nearest_idx = dist.min(dim=-1)  # both (B, P)
    return nearest_idx, nearest_dist, cum_s


def interpolate_polyline_at_arclength(
    poly_xy: torch.Tensor, poly_mask: torch.Tensor, cum_s: torch.Tensor, query_s: torch.Tensor
) -> torch.Tensor:
    """Sample polyline positions at arbitrary arc-lengths using linear interpolation.

    Query distances beyond a polyline's total length are extrapolated by continuing straight along the final valid
    segment direction, so a trajectory that outruns its lane keeps moving at constant velocity.

    Notation:
        G: number of polylines (group dimension)
        M: max number of points per polyline
        F: number of query distances per polyline

    Args:
        poly_xy (torch.Tensor): polyline point coordinates, shape ``(G, M, 2)``.
        poly_mask (torch.Tensor): per-point validity mask, shape ``(G, M)``.
        cum_s (torch.Tensor): cumulative arc-length at each point, shape ``(G, M)`` (see
            :func:`polyline_cumulative_arclength`).
        query_s (torch.Tensor): query arc-lengths, shape ``(G, F)``; expected non-negative.

    Returns:
        torch.Tensor: interpolated positions, shape ``(G, F, 2)``.
    """
    num_groups, num_points, _ = poly_xy.shape
    arange_g = torch.arange(num_groups, device=poly_xy.device)
    # Index of the last valid point per polyline, shape (G,). Clamped so single-/zero-point polylines stay in range.
    last_idx = (poly_mask.long().sum(dim=-1) - 1).clamp(min=0)  # (G,)
    total_len = cum_s[arange_g, last_idx]  # (G,)

    # In-range linear interpolation. searchsorted finds the first point whose cum_s reaches the query distance.
    hi = torch.searchsorted(cum_s.contiguous(), query_s.contiguous()).clamp(min=1, max=num_points - 1)  # (G, F)
    lo = hi - 1
    s_lo = torch.gather(cum_s, 1, lo)  # (G, F)
    s_hi = torch.gather(cum_s, 1, hi)
    frac = ((query_s - s_lo) / (s_hi - s_lo).clamp(min=1e-6)).clamp(0.0, 1.0)  # (G, F)
    lo_xy = torch.gather(poly_xy, 1, lo[..., None].expand(-1, -1, 2))  # (G, F, 2)
    hi_xy = torch.gather(poly_xy, 1, hi[..., None].expand(-1, -1, 2))
    interp_xy = lo_xy + frac[..., None] * (hi_xy - lo_xy)

    # Straight extrapolation past the last valid point, along the final segment direction.
    end_xy = poly_xy[arange_g, last_idx]  # (G, 2)
    end_dir = end_xy - poly_xy[arange_g, (last_idx - 1).clamp(min=0)]  # (G, 2)
    end_dir = end_dir / torch.linalg.norm(end_dir, dim=-1, keepdim=True).clamp(min=1e-6)
    extra = (query_s - total_len[:, None]).clamp(min=0.0)  # (G, F)
    extrap_xy = end_xy[:, None, :] + extra[..., None] * end_dir[:, None, :]

    beyond = (query_s > total_len[:, None])[..., None]  # (G, F, 1)
    return torch.where(beyond, extrap_xy, interp_xy)


class _DynamicSamplerDataset(Protocol):
    """Structural type for the dataset collection consumed by `DynamicSampler`."""

    config: dict[str, Any]
    dataset_idx: dict[str, NDArray[np.int_]]


class DynamicSampler(Sampler):
    """Sampler that draws a configurable subset of indices from a collection of datasets."""

    def __init__(self, datasets: _DynamicSamplerDataset) -> None:
        """Initializes the sampler from a dataset collection and its sampling configuration.

        Args:
            datasets (_DynamicSamplerDataset): collection exposing a ``config`` mapping (with
                ``sample_num``, ``sample_mode``, and ``max_data_num``) and a ``dataset_idx`` mapping
                from dataset name to the available indices.
        """
        self.datasets = datasets
        self.config = datasets.config
        all_dataset = self.datasets.dataset_idx.keys()
        self.sample_num = self.config["sample_num"]
        self.sample_mode = self.config["sample_mode"]

        max_data_num = self.config["max_data_num"]
        data_usage_dict = dict(zip(all_dataset, max_data_num, strict=False))
        self.set_sampling_strategy(data_usage_dict)

    def set_sampling_strategy(self, sampleing_dict: dict[str, float]) -> None:
        """Selects the indices to sample from each dataset according to the given usage amounts.

        Args:
            sampleing_dict (dict[str, float]): maps dataset name to the amount of data to use. Values in `[0, 1]` are
                interpreted as a fraction of the dataset; values above 1 are an absolute count.
        """
        all_idx = []
        selected_idx = {}
        for k, v in sampleing_dict.items():
            assert k in self.datasets.dataset_idx
            data_idx = self.datasets.dataset_idx[k]
            data_num = int(len(data_idx) * v) if v <= 1.0 else int(v)
            if data_num == 0:
                continue
            data_num = min(data_num, len(data_idx))
            # randomly select data_idx by data_num
            sampled_data_idx = np.random.choice(data_idx, data_num, replace=False).tolist()  # noqa: NPY002
            all_idx.extend(sampled_data_idx)
            selected_idx[k] = sampled_data_idx

        self.idx = all_idx[: self.sample_num]
        self.selected_idx = selected_idx

    def __iter__(self) -> Iterator[int]:
        """Returns an iterator over the selected indices."""
        return iter(self.idx)

    def __len__(self) -> int:
        """Returns the number of selected indices."""
        return len(self.idx)

    def reset(self) -> None:
        """Restores the index list to the full set of indices selected per dataset."""
        all_index = []
        for v in self.selected_idx.values():
            all_index.extend(v)
        self.idx = all_index

    def set_idx(self, idx: list[int]) -> None:
        """Overrides the current index list.

        Args:
            idx (list[int]): the indices to sample from.
        """
        self.idx = idx


def resplit_batch(batch: output.ModelOutput) -> dict[str, output.ModelOutput]:
    """Splits a batched `ModelOutput` into per-scenario `ModelOutput` objects on CPU.

    Args:
        batch (output.ModelOutput): a batched model output covering several scenarios.

    Returns:
        dict[str, output.ModelOutput]: maps each scenario id to its detached, CPU-resident output.
    """
    batch_resplit = {}

    # Unpack model output
    batch_scenario_embedding = batch.scenario_embedding
    batch_trajectory_output = batch.trajectory_decoder_output
    batch_safety_output = batch.safety_output
    batch_causal_output = batch.causal_output
    batch_history_gt = batch.history_ground_truth.value
    batch_future_gt = batch.future_ground_truth.value
    batch_dataset_name = batch.dataset_name
    batch_agent_ids = batch.agent_ids.value
    batch_scene_score = batch.scenario_scores

    for n, scenario_id in enumerate(batch.scenario_id):
        # Scenario Embedding
        scenario_embedding = output.ScenarioEmbedding(
            scenario_enc=batch_scenario_embedding.scenario_enc.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]
            scenario_dec=batch_scenario_embedding.scenario_dec.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
        )
        trajectory_decoder_output = None
        if batch_trajectory_output is not None:
            trajectory_decoder_output = output.TrajectoryDecoderOutput(
                decoded_trajectories=batch_trajectory_output.decoded_trajectories.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                mode_probabilities=batch_trajectory_output.mode_probabilities.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                mode_logits=batch_trajectory_output.mode_logits.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]
            )
        causal_output = None
        if batch_causal_output is not None:
            causal_output = output.CausalOutput(
                causal_gt=batch_causal_output.causal_gt.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                causal_pred=batch_causal_output.causal_pred.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                causal_pred_probs=batch_causal_output.causal_pred_probs.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                causal_logits=batch_causal_output.causal_logits.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]
            )
        safety_output = None
        if batch_safety_output is not None:
            safety_output = output.SafetyOutput(
                individual_safety_gt=batch_safety_output.individual_safety_gt.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                individual_safety_pred=batch_safety_output.individual_safety_pred.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                individual_safety_pred_probs=batch_safety_output.individual_safety_pred_probs.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                individual_safety_logits=batch_safety_output.individual_safety_logits.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]
                interaction_safety_gt=batch_safety_output.interaction_safety_gt.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                interaction_safety_pred=batch_safety_output.interaction_safety_pred.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                interaction_safety_pred_probs=batch_safety_output.interaction_safety_pred_probs.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                interaction_safety_logits=batch_safety_output.interaction_safety_logits.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]
            )

        scenario_scores = None
        if batch_scene_score is not None:
            scenario_scores = output.ScenarioScores(
                individual_agent_scores=batch_scene_score.individual_agent_scores.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                individual_scenario_score=batch_scene_score.individual_scenario_score.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                interaction_agent_scores=batch_scene_score.interaction_agent_scores.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                interaction_scenario_score=batch_scene_score.interaction_scenario_score.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
            )

        # Output
        batch_resplit[scenario_id] = output.ModelOutput(
            scenario_embedding=scenario_embedding,
            trajectory_decoder_output=trajectory_decoder_output,
            safety_output=safety_output,
            causal_output=causal_output,
            history_ground_truth=batch_history_gt[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
            future_ground_truth=batch_future_gt[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
            dataset_name=[batch_dataset_name[n]],
            scenario_id=[scenario_id],
            agent_ids=batch_agent_ids[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
            scenario_scores=scenario_scores,
        )

    return batch_resplit


def load_batches(
    base_data_path: str | Path,
    num_batches: int | None,
    num_scenarios: int | None,
    seed: int,
    tag: str = "val",
) -> dict[str, output.ModelOutput]:
    """Loads pickled scenario batches from disk and returns a random subset of scenarios.

    Args:
        base_data_path (str | Path): directory containing the pickled batch files.
        num_batches (int | None): maximum number of batch files to load; None loads all (capped at `MAX_NUM_BATCHES`).
        num_scenarios (int | None): number of scenarios to keep; None keeps all loaded scenarios.
        seed (int): random seed used to select batches and scenarios.
        tag (str): substring that batch filenames must contain (e.g. ``val``).

    Returns:
        dict[str, output.ModelOutput]: maps the selected scenario ids to their per-scenario outputs.

    Raises:
        ValueError: if no batch files matching `tag` are found.
    """
    _LOGGER.info("Loading scenario batches from %s", base_data_path)
    num_batches = MAX_NUM_BATCHES if num_batches is None else min(num_batches, MAX_NUM_BATCHES)
    random.seed(seed)

    batches = {}
    for n, batch_file in enumerate(Path(base_data_path).glob(f"*{tag}*")):
        if n >= num_batches:
            break
        with batch_file.open("rb") as f:
            batch: output.ModelOutput = pickle.load(f)
        batch_resplit = resplit_batch(batch)
        batches.update(batch_resplit)

    if not batches:
        msg = f"No batches found in {base_data_path} with tag {tag}"
        raise ValueError(msg)
    # Select scenarios
    scenario_ids = batches.keys()
    total_scenarios = len(scenario_ids)
    num_scenarios = max(1, total_scenarios) if num_scenarios is None else max(1, min(num_scenarios, total_scenarios))
    _LOGGER.info("Selecting %s / %s scenarios", num_scenarios, total_scenarios)
    selected_scenarios = random.sample(list(batches.keys()), num_scenarios)
    return {scenario: batches[scenario] for scenario in selected_scenarios}


def load_scenario_scores(scores_paths: str | Path, scenario_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Loads precomputed SafeShift scene and agent scores for the given scenarios.

    Args:
        scores_paths (str | Path): directory holding one pickle of scores per scenario.
        scenario_ids (Iterable[str]): scenario ids to load scores for.

    Returns:
        dict[str, dict[str, Any]]: a mapping with ``scene_scores`` and ``agents_scores``, each keyed by scenario id.
    """
    _LOGGER.info("Loading scenario scores...")
    scores_files = {str(f).split("/")[-1].split(".")[0]: f for f in Path(scores_paths).iterdir()}
    scene_scores, agents_scores = {}, {}
    for scenario_id in scenario_ids:
        scores_file = scores_files.get(scenario_id)
        if scores_file is None:
            _LOGGER.warning("No score file found for scenario %s", scenario_id)
            continue
        with scores_file.open("rb") as f:
            scenario_scores = pickle.load(f)
        scene_scores[str(scenario_id)] = scenario_scores["safeshift_scene_score"]
        agents_scores[str(scenario_id)] = scenario_scores["safeshift_agent_scores"]
    return {"scene_scores": scene_scores, "agents_scores": agents_scores}


def load_causal_agents_labels(causal_agents_labels_path: str | Path, scenario_ids: list[str]) -> dict[str, Any]:
    """Loads causal-agent labels for the given scenarios from JSON files.

    Args:
        causal_agents_labels_path (str | Path): directory holding one JSON of labels per scenario.
        scenario_ids (list[str]): scenario ids to load labels for.

    Returns:
        dict[str, Any]: maps scenario id to its loaded causal-agent labels.
    """
    _LOGGER.info("Loading causal agents labels...")
    causal_labels_files = {str(f).split("/")[-1].split(".")[0]: f for f in Path(causal_agents_labels_path).iterdir()}
    causal_agents_labels = {}
    for scenario_id in scenario_ids:
        causal_file = causal_labels_files.get(scenario_id)
        if causal_file is None:
            _LOGGER.warning("No label file found for scenario %s", scenario_id)
            continue
        with causal_file.open("r") as f:
            causal_labels = json.load(f)
        causal_agents_labels[str(scenario_id)] = causal_labels
    return causal_agents_labels


def save_cache(cache_infos: output.ModelOutput, filepath: Path) -> None:
    """Pickles a model output to disk.

    Args:
        cache_infos (output.ModelOutput): the model output to cache.
        filepath (Path): destination path for the pickle file.
    """
    with filepath.open("wb") as f:
        pickle.dump(cache_infos, f)
