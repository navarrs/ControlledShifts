"""Tests for the standalone ScenarioScorer wiring."""

import pytest
from omegaconf import OmegaConf

from controlledshifts.datasets import scenario_scorer


class _FakeProcessor:
    """Records the config it was built with so the test can assert wiring without touching real meta files."""

    def __init__(self, config):
        self.config = config


def _scoring_config():
    return OmegaConf.create(
        {
            "scenario_characterization": {"feature_type": "continuous"},
            "conflict_points": {"intersection_threshold": 0.1},
            "closest_lanes": {"num_lanes": 16},
        }
    )


def test_scenario_scorer_wires_processors(monkeypatch):
    """ScenarioScorer builds the SafeShift processors from scenario_characterization and keeps the map configs."""
    monkeypatch.setattr(scenario_scorer, "SafeShiftFeatures", _FakeProcessor)
    monkeypatch.setattr(scenario_scorer, "SafeShiftScorer", _FakeProcessor)

    config = _scoring_config()
    scorer = scenario_scorer.ScenarioScorer(config)

    assert scorer.features.config is config.scenario_characterization
    assert scorer.scorer.config is config.scenario_characterization
    assert scorer.conflict_points is config.conflict_points
    assert scorer.closest_lanes is config.closest_lanes


@pytest.mark.parametrize("missing", ["scenario_characterization", "conflict_points", "closest_lanes"])
def test_scenario_scorer_requires_all_blocks(monkeypatch, missing):
    """A missing scoring block raises a clear error naming it."""
    monkeypatch.setattr(scenario_scorer, "SafeShiftFeatures", _FakeProcessor)
    monkeypatch.setattr(scenario_scorer, "SafeShiftScorer", _FakeProcessor)

    config = _scoring_config()
    del config[missing]
    with pytest.raises(ValueError, match=missing):
        scenario_scorer.ScenarioScorer(config)
