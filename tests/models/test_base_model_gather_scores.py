"""Tests for BaseModel.gather_scores optional-safety-score gating.

Safety scores are optional. After the HDF5->pickle storage switch, absent scores arrive as present-but-None keys that
collate into a Python list of None, so gather_scores must gate on the values being real tensors, not on key presence.
"""

import torch

from controlledshifts.models.base_model import BaseModel
from controlledshifts.schemas.output_schemas import ScenarioScores


def test_gather_scores_returns_none_when_keys_absent():
    # paths=uniform behavior with HDF5 storage: the score keys were never present.
    assert BaseModel.gather_scores({"obj_trajs": torch.zeros(2, 3)}) is None


def test_gather_scores_returns_none_for_present_list_of_none():
    # Post-refactor collate fallback: a batch of [None, None] stays a Python list.
    inputs = {
        "individual_agent_scores": [None, None],
        "interaction_agent_scores": [None, None],
        "individual_scene_scores": [None, None],
        "interaction_scene_scores": [None, None],
    }
    assert BaseModel.gather_scores(inputs) is None


def test_gather_scores_returns_none_for_present_none():
    inputs = {
        "individual_agent_scores": None,
        "interaction_agent_scores": None,
        "individual_scene_scores": None,
        "interaction_scene_scores": None,
    }
    assert BaseModel.gather_scores(inputs) is None


def test_gather_scores_builds_scenario_scores_from_valid_tensors():
    # Autolabel-on shape: collate stacks per-record (N, 1, 1) into (B, N, 1, 1); squeeze(-1) -> (B, N, 1).
    inputs = {
        "individual_agent_scores": torch.rand(2, 4, 1, 1),
        "interaction_agent_scores": torch.rand(2, 4, 1, 1),
        "individual_scene_scores": torch.rand(2),
        "interaction_scene_scores": torch.rand(2),
    }
    scores = BaseModel.gather_scores(inputs)
    assert isinstance(scores, ScenarioScores)
    assert scores.individual_agent_scores.value.shape == (2, 4, 1)
    assert scores.interaction_agent_scores.value.shape == (2, 4, 1)
