"""Tests for the map-aware (lane-following) mode of the Constant Velocity Model."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from controlledshifts.models.cvm import ConstantVelocityModel


# Geometry constants for the synthetic scenes.
FUTURE_LEN = 5
NUM_MODES = 6
MAX_POINTS_PER_LANE = 20
LANE_TYPE_ID = 1  # surface-street lane; one-hot lands at absolute feature index 9 + 1 = 10
MAP_FEATURE_DIM = 29


def _build_model(tmp_path, **overrides) -> ConstantVelocityModel:
    """Instantiate a CVM with a minimal config, map-awareness on by default."""
    config = OmegaConf.create(
        {
            "future_len": FUTURE_LEN,
            "past_len": 2,
            "num_modes": NUM_MODES,
            "max_num_agents": 32,
            "max_points_per_lane": MAX_POINTS_PER_LANE,
            "max_num_roads": 384,
            "optimizer": None,
            "scheduler": None,
            "monitor": "losses/val",
            "criterion": None,
            "cache_batch": False,
            "cache_every_batch_idx": 100,
            "batch_cache_path": str(tmp_path / "cache"),
            "log_sigma": 0.0,
            "map_aware": True,
            "lane_search_radius": 5.0,
            "min_speed": 0.5,
            "score_temperature": 1.0,
            "alignment_weight": 1.0,
            **overrides,
        }
    )
    return ConstantVelocityModel(config)


def _lane_feature(points: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a single (L, 29) lane polyline feature tensor and its mask from (n, 2) points."""
    feat = torch.zeros(MAX_POINTS_PER_LANE, MAP_FEATURE_DIM)
    mask = torch.zeros(MAX_POINTS_PER_LANE, dtype=torch.bool)
    num = points.shape[0]
    pts = torch.tensor(points, dtype=torch.float32)

    feat[:num, 0:2] = pts
    # Unit directions from consecutive differences (the first point copies the second's direction).
    diff = torch.zeros(num, 2)
    if num > 1:
        diff[1:] = pts[1:] - pts[:-1]
        diff[0] = diff[1]
    feat[:num, 3:5] = diff / diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    # Lane-type one-hot flag (type id 1 -> absolute feature index 10).
    feat[:num, 9 + LANE_TYPE_ID] = 1.0
    mask[:num] = True
    return feat, mask


def _scene(lanes: list[np.ndarray], all_invalid: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack lane polylines into batched (1, P, L, 29) features and (1, P, L) mask."""
    feats, masks = zip(*(_lane_feature(p) for p in lanes), strict=True)
    map_polylines = torch.stack(feats, dim=0).unsqueeze(0)  # (1, P, L, 29)
    map_polylines_mask = torch.stack(masks, dim=0).unsqueeze(0)  # (1, P, L)
    if all_invalid:
        map_polylines_mask = torch.zeros_like(map_polylines_mask)
    return map_polylines, map_polylines_mask


def _ego_moving_x() -> tuple[torch.Tensor, torch.Tensor]:
    """Ego history moving along +x at unit speed, ending at the origin."""
    ego_xy = torch.tensor([[[-1.0, 0.0], [0.0, 0.0]]])  # (1, H, 2)
    ego_mask = torch.ones(1, 2)
    return ego_xy, ego_mask


def test_map_aware_follows_curved_lane(tmp_path):
    """A curving lane should bend the prediction away from the straight-line baseline and track the lane."""
    model = _build_model(tmp_path)
    ego_xy, ego_mask = _ego_moving_x()

    # Quarter-circle lane (radius 10) starting at the origin heading +x and curving toward +y.
    theta = np.linspace(-np.pi / 2, 0.0, MAX_POINTS_PER_LANE)
    radius = 10.0
    lane = np.stack([radius * np.cos(theta), radius + radius * np.sin(theta)], axis=1)
    map_polylines, map_polylines_mask = _scene([lane])

    future_modes, mode_logits = model._extrapolate_future_map_aware(ego_xy, ego_mask, map_polylines, map_polylines_mask)
    straight = model._extrapolate_future(ego_xy, ego_mask)  # (1, F, 2)

    assert future_modes.shape == (1, NUM_MODES, FUTURE_LEN, 2)
    # The straight-line baseline stays on the x-axis; the lane-following mode bends upward.
    assert straight[0, -1, 1].abs() < 1e-3
    assert future_modes[0, 0, -1, 1] > 0.5
    # Every lane-following point stays close to the lane centerline.
    lane_pts = torch.tensor(lane, dtype=torch.float32)
    dists = torch.cdist(future_modes[0, 0], lane_pts).min(dim=-1).values
    assert dists.max() < 0.5
    # The best lane gets the highest logit; padded fallback modes are far lower.
    assert mode_logits[0, 0] > mode_logits[0, 1]


def test_map_aware_multimodal_fork(tmp_path):
    """At a fork the better-aligned lane should be the top mode with the highest probability."""
    model = _build_model(tmp_path)
    ego_xy, ego_mask = _ego_moving_x()  # heading +x

    straight_lane = np.stack([np.linspace(0, 10, MAX_POINTS_PER_LANE), np.zeros(MAX_POINTS_PER_LANE)], axis=1)
    diagonal_lane = np.stack([np.linspace(0, 10, MAX_POINTS_PER_LANE), np.linspace(0, 10, MAX_POINTS_PER_LANE)], axis=1)
    map_polylines, map_polylines_mask = _scene([straight_lane, diagonal_lane])

    future_modes, mode_logits = model._extrapolate_future_map_aware(ego_xy, ego_mask, map_polylines, map_polylines_mask)
    output = model._build_multimodal_trajectory_output(future_modes, mode_logits)
    probs = output.mode_probabilities.value[0]

    # Probabilities form a valid distribution.
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    # The straight (best-aligned) lane is the top mode and beats both the diagonal lane and the padded modes.
    assert probs[0] > probs[1]
    assert probs[1] > probs[2]
    # The top mode follows the straight lane (stays on the x-axis).
    assert future_modes[0, 0, -1, 1].abs() < 0.5
    # Decoded trajectory means match the per-mode positions.
    assert torch.allclose(output.decoded_trajectories.value[0, 0, :, :2], future_modes[0, 0])


def test_map_aware_falls_back_without_lanes(tmp_path):
    """With no valid lanes the prediction must equal the straight-line CVM, replicated with uniform probabilities."""
    model = _build_model(tmp_path)
    ego_xy, ego_mask = _ego_moving_x()

    lane = np.stack([np.linspace(0, 10, MAX_POINTS_PER_LANE), np.zeros(MAX_POINTS_PER_LANE)], axis=1)
    map_polylines, map_polylines_mask = _scene([lane], all_invalid=True)

    future_modes, mode_logits = model._extrapolate_future_map_aware(ego_xy, ego_mask, map_polylines, map_polylines_mask)
    straight = model._extrapolate_future(ego_xy, ego_mask)
    probs = torch.softmax(mode_logits, dim=-1)[0]

    # Every mode equals the straight-line baseline and probabilities are uniform.
    assert torch.allclose(future_modes[0], straight.expand(NUM_MODES, -1, -1))
    assert torch.allclose(probs, torch.full((NUM_MODES,), 1.0 / NUM_MODES), atol=1e-5)


def test_map_aware_falls_back_when_stationary(tmp_path):
    """A stationary ego (speed below min_speed) should fall back even when a valid lane is present."""
    model = _build_model(tmp_path)
    ego_xy = torch.zeros(1, 2, 2)  # both history steps at the origin -> zero velocity
    ego_mask = torch.ones(1, 2)

    lane = np.stack([np.linspace(0, 10, MAX_POINTS_PER_LANE), np.zeros(MAX_POINTS_PER_LANE)], axis=1)
    map_polylines, map_polylines_mask = _scene([lane])

    future_modes, mode_logits = model._extrapolate_future_map_aware(ego_xy, ego_mask, map_polylines, map_polylines_mask)
    straight = model._extrapolate_future(ego_xy, ego_mask)
    probs = torch.softmax(mode_logits, dim=-1)[0]

    assert torch.allclose(future_modes[0], straight.expand(NUM_MODES, -1, -1))
    assert torch.allclose(probs, torch.full((NUM_MODES,), 1.0 / NUM_MODES), atol=1e-5)


def test_disabled_map_aware_is_unchanged(tmp_path):
    """With the flag off the trajectory output replicates the straight-line baseline across modes."""
    model = _build_model(tmp_path, map_aware=False)
    ego_xy, ego_mask = _ego_moving_x()

    straight = model._extrapolate_future(ego_xy, ego_mask)
    output = model._build_trajectory_output(straight)

    assert not model.map_aware
    for mode in range(NUM_MODES):
        assert torch.allclose(output.decoded_trajectories.value[0, mode, :, :2], straight[0])
    assert torch.allclose(output.mode_probabilities.value[0], torch.full((NUM_MODES,), 1.0 / NUM_MODES), atol=1e-5)


@pytest.mark.parametrize("num_lanes", [1, 3, NUM_MODES + 2])
def test_map_aware_output_shapes(tmp_path, num_lanes):
    """The map-aware path always emits exactly num_modes trajectories regardless of the lane count."""
    model = _build_model(tmp_path)
    ego_xy, ego_mask = _ego_moving_x()

    lanes = [
        np.stack([np.linspace(0, 10, MAX_POINTS_PER_LANE), np.full(MAX_POINTS_PER_LANE, float(i))], axis=1)
        for i in range(num_lanes)
    ]
    map_polylines, map_polylines_mask = _scene(lanes)

    future_modes, mode_logits = model._extrapolate_future_map_aware(ego_xy, ego_mask, map_polylines, map_polylines_mask)
    assert future_modes.shape == (1, NUM_MODES, FUTURE_LEN, 2)
    assert mode_logits.shape == (1, NUM_MODES)
