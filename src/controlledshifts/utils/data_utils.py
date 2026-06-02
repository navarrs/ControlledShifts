import math
import os
import random

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Sampler
from numpy.typing import NDArray
from controlledshifts.utils.constants import TrajectoryType
from controlledshifts.utils import pylogger
import json
import pickle
from pathlib import Path

from typing import Any
from easydict import EasyDict

from controlledshifts.schemas import output_schemas as output


MAX_NUM_BATCHES = 1e6
_LOGGER = pylogger.get_pylogger(__name__)


def minmax_scaler(x: np.ndarray) -> np.ndarray:
    """Normalizes an input value using Min-Max normalizaiton.

    Args:
        x (np.ndarray): input array.
    Returns:
        x_norm (np.ndarray): normalized array.
    """
    # compute the distribution range
    value_range = np.max(x) - np.min(x)

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def classify_track(start_point, end_point, start_velocity, end_velocity, start_heading, end_heading):
    # The classification strategy is taken from
    # waymo_open_dataset/metrics/motion_metrics_utils.cc#L28

    # Parameters for classification, taken from WOD
    kMaxSpeedForStationary = 2.0  # (m/s)
    kMaxDisplacementForStationary = 5.0  # (m)
    kMaxLateralDisplacementForStraight = 5.0  # (m)
    kMinLongitudinalDisplacementForUTurn = -5.0  # (m)
    kMaxAbsHeadingDiffForStraight = np.pi / 6.0  # (rad)

    x_delta = end_point[0] - start_point[0]
    y_delta = end_point[1] - start_point[1]

    final_displacement = np.hypot(x_delta, y_delta)
    heading_diff = end_heading - start_heading
    normalized_delta = np.array([x_delta, y_delta])
    rotation_matrix = np.array(
        [[np.cos(-start_heading), -np.sin(-start_heading)], [np.sin(-start_heading), np.cos(-start_heading)]],
    )
    normalized_delta = np.dot(rotation_matrix, normalized_delta)
    start_speed = np.hypot(start_velocity[0], start_velocity[1])
    end_speed = np.hypot(end_velocity[0], end_velocity[1])
    max_speed = max(start_speed, end_speed)
    dx, dy = normalized_delta

    # Check for different trajectory types based on the computed parameters.
    if max_speed < kMaxSpeedForStationary and final_displacement < kMaxDisplacementForStationary:
        return TrajectoryType.STATIONARY.value
    if np.abs(heading_diff) < kMaxAbsHeadingDiffForStraight:
        if np.abs(normalized_delta[1]) < kMaxLateralDisplacementForStraight:
            return TrajectoryType.STRAIGHT.value
        return TrajectoryType.STRAIGHT_RIGHT.value if dy < 0 else TrajectoryType.STRAIGHT_LEFT.value
    if heading_diff < -kMaxAbsHeadingDiffForStraight and dy < 0:
        return (
            TrajectoryType.RIGHT_U_TURN.value
            if normalized_delta[0] < kMinLongitudinalDisplacementForUTurn
            else TrajectoryType.RIGHT_TURN.value
        )
    if dx < kMinLongitudinalDisplacementForUTurn:
        return TrajectoryType.LEFT_U_TURN.value
    return TrajectoryType.LEFT_TURN.value


def get_heading(trajectory):
    # trajectory has shape (Time X (x,y))

    dx = np.diff(trajectory[:, 0])
    dy = np.diff(trajectory[:, 1])
    heading = np.arctan2(dy, dx)

    return heading


def get_trajectory_type(output):
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
        except:
            trajectory_type = -1
        data_sample["trajectory_type"] = trajectory_type


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def estimate_kalman_filter(history, prediction_horizon):
    """Predict the future position by running the kalman filter.

    :param history: 2d array of shape (length_of_history, 2)
    :param prediction_horizon: how many steps in the future to predict
    :return: the predicted position (x, y)

    Code taken from:
    On Exposing the Challenging Long Tail in Future Prediction of Traffic Actors
    """
    length_history = history.shape[0]
    z_x = history[:, 0]
    z_y = history[:, 1]
    v_x = 0
    v_y = 0
    for index in range(length_history - 1):
        v_x += z_x[index + 1] - z_x[index]
        v_y += z_y[index + 1] - z_y[index]
    v_x = v_x / (length_history - 1)
    v_y = v_y / (length_history - 1)
    x_x = np.zeros(length_history + 1, np.float32)
    x_y = np.zeros(length_history + 1, np.float32)
    P_x = np.zeros(length_history + 1, np.float32)
    P_y = np.zeros(length_history + 1, np.float32)
    P_vx = np.zeros(length_history + 1, np.float32)
    P_vy = np.zeros(length_history + 1, np.float32)

    # we initialize the uncertainty to one (unit gaussian)
    P_x[0] = 1.0
    P_y[0] = 1.0
    P_vx[0] = 1.0
    P_vy[0] = 1.0
    x_x[0] = z_x[0]
    x_y[0] = z_y[0]

    Q = 0.00001
    R = 0.0001
    K_x = np.zeros(length_history + 1, np.float32)
    K_y = np.zeros(length_history + 1, np.float32)
    K_vx = np.zeros(length_history + 1, np.float32)
    K_vy = np.zeros(length_history + 1, np.float32)
    for k in range(length_history - 1):
        x_x[k + 1] = x_x[k] + v_x
        x_y[k + 1] = x_y[k] + v_y
        P_x[k + 1] = P_x[k] + P_vx[k] + Q
        P_y[k + 1] = P_y[k] + P_vy[k] + Q
        P_vx[k + 1] = P_vx[k] + Q
        P_vy[k + 1] = P_vy[k] + Q
        K_x[k + 1] = P_x[k + 1] / (P_x[k + 1] + R)
        K_y[k + 1] = P_y[k + 1] / (P_y[k + 1] + R)
        x_x[k + 1] = x_x[k + 1] + K_x[k + 1] * (z_x[k + 1] - x_x[k + 1])
        x_y[k + 1] = x_y[k + 1] + K_y[k + 1] * (z_y[k + 1] - x_y[k + 1])
        P_x[k + 1] = P_x[k + 1] - K_x[k + 1] * P_x[k + 1]
        P_y[k + 1] = P_y[k + 1] - K_y[k + 1] * P_y[k + 1]
        K_vx[k + 1] = P_vx[k + 1] / (P_vx[k + 1] + R)
        K_vy[k + 1] = P_vy[k + 1] / (P_vy[k + 1] + R)
        P_vx[k + 1] = P_vx[k + 1] - K_vx[k + 1] * P_vx[k + 1]
        P_vy[k + 1] = P_vy[k + 1] - K_vy[k + 1] * P_vy[k + 1]

    k = k + 1
    x_x[k + 1] = x_x[k] + v_x * prediction_horizon
    x_y[k + 1] = x_y[k] + v_y * prediction_horizon
    P_x[k + 1] = P_x[k] + P_vx[k] * prediction_horizon * prediction_horizon + Q
    P_y[k + 1] = P_y[k] + P_vy[k] * prediction_horizon * prediction_horizon + Q
    P_vx[k + 1] = P_vx[k] + Q
    P_vy[k + 1] = P_vy[k] + Q
    return x_x[k + 1], x_y[k + 1]


def calculate_epe(pred, gt):
    diff_x = (gt[0] - pred[0]) * (gt[0] - pred[0])
    diff_y = (gt[1] - pred[1]) * (gt[1] - pred[1])
    epe = math.sqrt(diff_x + diff_y)
    return epe


def count_valid_steps_past(mask):
    reversed_mask = mask[::-1]  # Reverse the mask
    idx_of_first_zero = np.where(reversed_mask == 0)[0]  # Find the index of the first zero
    if len(idx_of_first_zero) == 0:
        return len(mask)  # If no zeros, return the length of the mask
    return idx_of_first_zero[0]  # Return the index of the first zero


def get_kalman_difficulty(output, sampling_freq: int = 10):
    """Return the kalman difficulty at 2s, 4s, and 6s
    if the gt future is not valid up to the considered second, the difficulty is set to -1.

    Args:
        sampling_freq (int): frequency at which the trajectory is sampled. In WOMD, datapoints are sampled at 10hz
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
        except:
            kalman_difficulty_2s, kalman_difficulty_4s, kalman_difficulty_6s = -1, -1, -1
        data_sample["kalman_difficulty"] = np.array([kalman_difficulty_2s, kalman_difficulty_4s, kalman_difficulty_6s])


def is_ddp():
    return "WORLD_SIZE" in os.environ


def generate_mask(current_index: int, total_length: int, interval: int) -> np.ndarray:
    mask = []
    for i in range(total_length):
        # Check if the position is a multiple of the frequency starting from current_index
        if (i - current_index) % interval == 0:
            mask.append(1)
        else:
            mask.append(0)

    return np.array(mask)


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z_tensor(points, angle):
    """Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    if points.shape[-1] == 2:
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


def rotate_points_along_z(points, angle):
    """Rotate points around the Z-axis using the given angle.

    Args:
        points: ndarray of shape (B, N, 3 + C) - B batches, N points per batch, 3 coordinates (x, y, z) + C extra channels
        angle: ndarray of shape (B,) - angles for each batch in radians

    Returns:
        Rotated points as an ndarray.
    """
    # Checking if the input is 2D or 3D points
    is_2d = points.shape[-1] == 2

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
        if points.shape[-1] > 3:
            points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)

    return points_rot

def find_true_segments(mask: NDArray[np.bool_]) -> list[list[int]]:
    """Returns indices of each contiguous run of True values in mask."""
    change_points = np.where(np.diff(mask))[0] + 1
    indices = np.concatenate(([0], change_points, [len(mask)]))
    return [list(range(int(indices[i]), int(indices[i + 1]))) for i in range(len(indices) - 1) if mask[indices[i]]]


def merge_batch_by_padding_2nd_dim(tensor_list, return_pad_mask=False):
    assert len(tensor_list[0].shape) in [3, 4]
    only_3d_tensor = False
    if len(tensor_list[0].shape) == 3:
        tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]
        only_3d_tensor = True
    maxt_feat0 = max([x.shape[1] for x in tensor_list])

    _, _, num_feat1, num_feat2 = tensor_list[0].shape

    ret_tensor_list = []
    ret_mask_list = []
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]
        assert cur_tensor.shape[2] == num_feat1 and cur_tensor.shape[3] == num_feat2, print(cur_tensor.shape)

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


def get_batch_offsets(batch_idxs, bs):
    """:param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    """
    batch_offsets = torch.zeros(bs + 1).int()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def count_valid_steps_past(mask):
    reversed_mask = mask[::-1]  # Reverse the mask
    idx_of_first_zero = np.where(reversed_mask == 0)[0]  # Find the index of the first zero
    if len(idx_of_first_zero) == 0:
        return len(mask)  # If no zeros, return the length of the mask
    return idx_of_first_zero[0]  # Return the index of the first zero


def interpolate_polyline(polyline, step=0.5):
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
    new_polyline = np.concatenate((new_polyline, np.zeros((new_polyline.shape[0], 1))), axis=1)
    return new_polyline


class DynamicSampler(Sampler):
    def __init__(self, datasets):
        """datasets: Dictionary of datasets.
        epoch_to_datasets: A dict where keys are epoch numbers and values are lists of dataset names to be used in that epoch.
        """
        self.datasets = datasets
        self.config = datasets.config
        all_dataset = self.datasets.dataset_idx.keys()
        self.sample_num = self.config["sample_num"]
        self.sample_mode = self.config["sample_mode"]

        data_usage_dict = {}
        max_data_num = self.config["max_data_num"]
        for k, num in zip(all_dataset, max_data_num, strict=False):
            data_usage_dict[k] = num
        # self.selected_idx = self.datasets.dataset_idx
        # self.reset()
        self.set_sampling_strategy(data_usage_dict)

    def set_sampling_strategy(self, sampleing_dict):
        all_idx = []
        selected_idx = {}
        for k, v in sampleing_dict.items():
            assert k in self.datasets.dataset_idx.keys()
            data_idx = self.datasets.dataset_idx[k]
            if v <= 1.0:
                data_num = int(len(data_idx) * v)
            else:
                data_num = int(v)
            if data_num == 0:
                continue
            data_num = min(data_num, len(data_idx))
            # randomly select data_idx by data_num
            sampled_data_idx = np.random.choice(data_idx, data_num, replace=False).tolist()
            all_idx.extend(sampled_data_idx)
            selected_idx[k] = sampled_data_idx

        self.idx = all_idx[: self.sample_num]
        self.selected_idx = selected_idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

    def reset(self):
        all_index = []
        for k, v in self.selected_idx.items():
            all_index.extend(v)
        self.idx = all_index

    def set_idx(self, idx):
        self.idx = idx


def separate_batches_by_group(token_infos):
    dataset_names = token_infos["dataset_name"]
    unique_groups = np.unique(dataset_names)
    token_info_groups = {}
    for group in unique_groups:
        print(f"Separating group: {group}")
        group_idx = np.where(dataset_names == group)[0]
        print(f"Found {group_idx.shape[0]} elements")

        # Get elements for each group
        token_info_groups[group] = {}
        for key, value in token_infos.items():
            if key in ["unique_classes", "classes_counts", "num_classes"]:
                continue
            print(f"Key: {key} shape: {value.shape}")
            token_info_groups[group][key] = value[group_idx]

        # Re-compute number of classee
        scenario_classes = token_info_groups[group]["scenario_classes"]
        unique_classes, class_counts = np.unique(scenario_classes, return_counts=True)
        num_classes = len(unique_classes)
        token_info_groups[group]["unique_classes"] = unique_classes
        token_info_groups[group]["classes_counts"] = class_counts
        token_info_groups[group]["num_classes"] = num_classes

    return token_info_groups


def load_batches_list(base_data_path, num_batches, tag="val"):
    print("Loading scenario batches...")
    num_batches = MAX_NUM_BATCHES if num_batches is None else min(num_batches, MAX_NUM_BATCHES)

    # Batch info to repack
    dataset_names = []
    scenario_ids = []
    # Scenario Tokenization
    scenario_classes = []
    scenario_embedding = []
    scenario_quantized_embedding = []
    # Causal Tokenization
    causal_gt = []
    causal_pred = []
    causal_classes = []
    causal_embedding = []
    causal_quantized_embedding = []

    for n, batch_file in enumerate(Path(base_data_path).glob(f"*{tag}*")):
        if n >= num_batches:
            break
        # print(f"Loading batch from: {batch_file}")
        with batch_file.open("rb") as f:
            batch: output.ModelOutput = pickle.load(f)

        # Meta information
        dataset_names.append(batch.dataset_name)
        scenario_ids.append(batch.scenario_id)

        # TODO: Add Trajectory outputs

        # Scenario Tokenization
        tokenization_output = batch.tokenization_output
        scenario_classes.append(tokenization_output.token_indices.value.detach().cpu().numpy())  # (B, M)
        scenario_embedding.append(tokenization_output.input_embedding.value.detach().cpu().numpy())  # (B, M, D)
        scenario_quantized_embedding.append(
            tokenization_output.quantized_embedding.value.detach().cpu().numpy()
        )  # (B, M, T)

        # Causal Tokenization
        causal_output = batch.causal_output
        causal_gt.append(causal_output.causal_gt.value.detach().cpu().numpy())  # (B, N)
        causal_pred.append(causal_output.causal_pred_probs.value.detach().cpu().numpy())  # (B, N, C)
        causal_tokenization = batch.causal_tokenization_output
        causal_classes.append(causal_tokenization.token_indices.value.detach().cpu().numpy())  # (B, M)
        causal_embedding.append(causal_tokenization.input_embedding.value.detach().cpu().numpy())
        causal_quantized_embedding.append(causal_tokenization.quantized_embedding.value.detach().cpu().numpy())

    # Concatenate all
    return EasyDict(
        {
            "dataset_name": np.concatenate(dataset_names),
            "scenario_ids": np.concatenate(scenario_ids),
            "scenario_classes": np.concatenate(scenario_classes),
            "scenario_embedding": np.concatenate(scenario_embedding),
            "scenario_quantized_embedding": np.concatenate(scenario_quantized_embedding),
            "causal_gt": np.concatenate(causal_gt),
            "causal_pred": np.concatenate(causal_pred),
            "causal_classes": np.concatenate(causal_classes),
            "causal_embedding": np.concatenate(causal_embedding),
            "causal_quantized_embedding": np.concatenate(causal_quantized_embedding),
        }
    )


def resplit_batch(batch: output.ModelOutput) -> list[output.ModelOutput]:
    batch_resplit = {}

    # Unkpack model output
    batch_scenario_embedding = batch.scenario_embedding
    batch_trajectory_output = batch.trajectory_decoder_output
    batch_tokenization_output = batch.tokenization_output
    batch_safety_output = batch.safety_output
    batch_causal_output = batch.causal_output
    batch_causal_tokenization_output = batch.causal_tokenization_output
    batch_history_gt = batch.history_ground_truth.value
    batch_future_gt = batch.future_ground_truth.value
    batch_dataset_name = batch.dataset_name
    batch_agent_ids = batch.agent_ids.value
    batch_scene_score = batch.scenario_scores

    for n, scenario_id in enumerate(batch.scenario_id):
        # Scenario Embedding
        scenario_embedding = output.ScenarioEmbedding(
            scenario_enc=batch_scenario_embedding.scenario_enc.value[n].detach().cpu(),
            scenario_dec=batch_scenario_embedding.scenario_dec.value[n].detach().cpu(),
        )
        trajectory_decoder_output = None
        if batch_trajectory_output is not None:
            trajectory_decoder_output = output.TrajectoryDecoderOutput(
                decoded_trajectories=batch_trajectory_output.decoded_trajectories.value[n].detach().cpu(),
                mode_probabilities=batch_trajectory_output.mode_probabilities.value[n].detach().cpu(),
                mode_logits=batch_trajectory_output.mode_logits.value[n].detach().cpu(),
            )
        tokenization_output = None
        if batch_tokenization_output is not None:
            probs = batch_tokenization_output.token_probabilities
            tokenization_output = output.TokenizationOutput(
                token_probabilities=None if probs is None else probs.value[n].detach().cpu(),
                token_indices=batch_tokenization_output.token_indices.value[n].detach().cpu(),
                input_embedding=batch_tokenization_output.input_embedding.value[n].detach().cpu(),
                reconstructed_embedding=batch_tokenization_output.reconstructed_embedding.value[n].detach().cpu(),
                quantized_embedding=batch_tokenization_output.quantized_embedding.value[n].detach().cpu(),
            )
        causal_output = None
        if batch_causal_output is not None:
            causal_output = output.CausalOutput(
                causal_gt=batch_causal_output.causal_gt.value[n].detach().cpu(),
                causal_pred=batch_causal_output.causal_pred.value[n].detach().cpu(),
                causal_pred_probs=batch_causal_output.causal_pred_probs.value[n].detach().cpu(),
                causal_logits=batch_causal_output.causal_logits.value[n].detach().cpu(),
            )
        safety_output = None
        if batch_safety_output is not None:
            safety_output = output.SafetyOutput(
                individual_safety_gt=batch_safety_output.individual_safety_gt.value[n].detach().cpu(),
                individual_safety_pred=batch_safety_output.individual_safety_pred.value[n].detach().cpu(),
                individual_safety_pred_probs=batch_safety_output.individual_safety_pred_probs.value[n].detach().cpu(),
                individual_safety_logits=batch_safety_output.individual_safety_logits.value[n].detach().cpu(),
                interaction_safety_gt=batch_safety_output.interaction_safety_gt.value[n].detach().cpu(),
                interaction_safety_pred=batch_safety_output.interaction_safety_pred.value[n].detach().cpu(),
                interaction_safety_pred_probs=batch_safety_output.interaction_safety_pred_probs.value[n].detach().cpu(),
                interaction_safety_logits=batch_safety_output.interaction_safety_logits.value[n].detach().cpu(),
            )

        causal_tokenization_output = None
        if batch_causal_tokenization_output is not None:
            probs = batch_causal_tokenization_output.token_probabilities
            causal_tokenization_output = output.TokenizationOutput(
                token_probabilities=None if probs is None else probs.value[n].detach().cpu(),
                token_indices=batch_causal_tokenization_output.token_indices.value[n].detach().cpu(),
                input_embedding=batch_causal_tokenization_output.input_embedding.value[n].detach().cpu(),
                reconstructed_embedding=batch_causal_tokenization_output.reconstructed_embedding.value[n].detach().cpu(),
                quantized_embedding=batch_causal_tokenization_output.quantized_embedding.value[n].detach().cpu(),
            )

        scenario_scores = None
        if batch_scene_score is not None:
            scenario_scores = output.ScenarioScores(
                individual_agent_scores=batch_scene_score.individual_agent_scores.value[n].detach().cpu(),
                individual_scenario_score=batch_scene_score.individual_scenario_score.value[n].detach().cpu(),
                interaction_agent_scores=batch_scene_score.interaction_agent_scores.value[n].detach().cpu(),
                interaction_scenario_score=batch_scene_score.interaction_scenario_score.value[n].detach().cpu(),
            )

        # Output
        batch_resplit[scenario_id] = output.ModelOutput(
            scenario_embedding=scenario_embedding,
            trajectory_decoder_output=trajectory_decoder_output,
            tokenization_output=tokenization_output,
            safety_output=safety_output,
            causal_output=causal_output,
            causal_tokenization_output=causal_tokenization_output,
            history_ground_truth=batch_history_gt[n].detach().cpu(),
            future_ground_truth=batch_future_gt[n].detach().cpu(),
            dataset_name=[batch_dataset_name[n]],
            scenario_id=[scenario_id],
            agent_ids=batch_agent_ids[n].detach().cpu(),
            scenario_scores=scenario_scores,
        )

    return batch_resplit


def load_batches(base_data_path, num_batches, num_scenarios, seed, tag="val"):
    _LOGGER.info(f"Loading scenario batches from {base_data_path}")
    num_batches = MAX_NUM_BATCHES if num_batches is None else min(num_batches, MAX_NUM_BATCHES)
    random.seed(seed)

    batches = {}
    for n, batch_file in enumerate(Path(base_data_path).glob(f"*{tag}*")):
        if n >= num_batches:
            break
        # print(f"Loading batch from: {batch_file}")
        with batch_file.open("rb") as f:
            batch: output.ModelOutput = pickle.load(f)
        batch_resplit = resplit_batch(batch)
        batches.update(batch_resplit)

    if not batches:
        raise ValueError(f"No batches found in {base_data_path} with tag {tag}")
    # Select scenarios
    scenario_ids = batches.keys()
    total_scenarios = len(scenario_ids)
    if num_scenarios is None:
        num_scenarios = max(1, total_scenarios)
    else:
        num_scenarios = max(1, min(num_scenarios, total_scenarios))
    _LOGGER.info(f"Selecting {num_scenarios} / {total_scenarios} scenarios")
    selected_scenarios = random.sample(list(batches.keys()), num_scenarios)
    return {scenario: batches[scenario] for scenario in selected_scenarios}

def load_scenario_scores(scores_paths, scenario_ids):
    print("Loading scenario scores...")
    scores_files = {str(f).split("/")[-1].split(".")[0]: f for f in Path(scores_paths).iterdir()}
    scene_scores, agents_scores = {}, {}
    for scenario_id in scenario_ids:
        scores_file = scores_files.get(scenario_id)
        if scores_file is None:
            print(f"Warning no score file found for scenario {scenario_id}")
            continue
        with scores_file.open("rb") as f:
            scenario_scores = pickle.load(f)
        scene_scores[str(scenario_id)] = scenario_scores["safeshift_scene_score"]
        agents_scores[str(scenario_id)] = scenario_scores["safeshift_agent_scores"]
    return {"scene_scores": scene_scores, "agents_scores": agents_scores}


def load_causal_agents_labels(causal_agents_labels_path, scenario_ids=None):
    print("Loading causal agents labels...")
    causal_labels_files = {str(f).split("/")[-1].split(".")[0]: f for f in Path(causal_agents_labels_path).iterdir()}
    causal_agents_labels = {}
    for scenario_id in scenario_ids:
        causal_file = causal_labels_files.get(scenario_id)
        if causal_file is None:
            print(f"Warning no score file found for scenario {scenario_id}")
            continue
        with causal_file.open("r") as f:
            causal_labels = json.load(f)
        causal_agents_labels[str(scenario_id)] = causal_labels
    return causal_agents_labels


def save_cache(cache_infos: Any, filepath: Path) -> None:
    with filepath.open("wb") as f:
        pickle.dump(cache_infos, f)
