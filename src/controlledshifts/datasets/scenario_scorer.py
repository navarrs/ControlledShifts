"""Standalone SafeShift scenario scoring.

Turns an open ``Scenario`` into ``ScenarioScores`` using the characterization API (``SafeShiftFeatures`` +
``SafeShiftScorer``). This is deliberately independent of the agent-centric tensor pipeline (``AgentCentricProcessor``):
it depends only on a focused scoring config (the ``scenario_characterization``/``conflict_points``/``closest_lanes``
blocks), so both benchmark creation (scoring scenarios from disk) and the tensor processor (autolabelling agents) can
reuse it without dragging in tensor-shaping config.
"""

from characterization.features.safeshift_features import SafeShiftFeatures
from characterization.schemas import Scenario, ScenarioScores
from characterization.scorer.safeshift_scorer import SafeShiftScorer
from characterization.utils.geometric_utils import find_closest_lanes, find_conflict_points
from omegaconf import DictConfig


def add_scenario_map_metadata(
    scenario: Scenario,
    conflict_points_config: DictConfig,
    closest_lanes_config: DictConfig,
    total_steps: int | None = None,
) -> Scenario:
    """Computes map-related metadata (conflict points and closest lanes) and stores it on the scenario.

    Args:
        scenario: The input scenario for which to compute map metadata (mutated in place).
        conflict_points_config: ``conflict_points`` config block.
        closest_lanes_config: ``closest_lanes`` config block.
        total_steps: If given, agent-indexed metadata is truncated to the first ``total_steps`` timesteps (used by the
            tensor pipeline to match its shaped trajectories); if ``None``, the full-length metadata is kept.

    Returns:
        The scenario with updated map metadata.
    """
    time_slice = slice(None) if total_steps is None else slice(None, total_steps)

    conflict_points_info = find_conflict_points(
        scenario,
        resample_factor=conflict_points_config.get("resample_factor", 1),
        intersection_threshold=conflict_points_config.get("intersection_threshold", 0.5),
        return_static_conflict_points=conflict_points_config.get("return_static_conflict_points", False),
        return_lane_conflict_points=conflict_points_config.get("return_lane_conflict_points", False),
        return_dynamic_conflict_points=conflict_points_config.get("return_dynamic_conflict_points", False),
    )
    if conflict_points_info is not None:
        agent_distances_to_conflict_points = (
            None
            if conflict_points_info["agent_distances_to_conflict_points"] is None
            else conflict_points_info["agent_distances_to_conflict_points"][:, time_slice, :]
        )
        conflict_points = (
            None if conflict_points_info["all_conflict_points"] is None else conflict_points_info["all_conflict_points"]
        )
        scenario.static_map_data.map_conflict_points = conflict_points
        scenario.static_map_data.agent_distances_to_conflict_points = agent_distances_to_conflict_points

    closest_lanes_info = find_closest_lanes(
        scenario,
        k_closest=closest_lanes_config.get("num_lanes", 16),
        threshold_distance=closest_lanes_config.get("threshold_distance", 10.0),
        subsample_factor=closest_lanes_config.get("subsample_factor", 2),
    )
    if closest_lanes_info is not None:
        scenario.static_map_data.agent_closest_lanes = closest_lanes_info["agent_closest_lanes"][:, time_slice, :, :]

    return scenario


class ScenarioScorer:
    """Computes SafeShift ``ScenarioScores`` from a ``Scenario``.

    Depends only on a focused scoring config containing ``scenario_characterization``, ``conflict_points`` and
    ``closest_lanes``; it has no knowledge of trajectory shaping, the agent-centric tensors, or the dataset.
    """

    def __init__(self, config: DictConfig) -> None:
        """Builds the scorer from a scoring config (the three ``scenario_characterization``/map blocks)."""
        self.conflict_points = config.get("conflict_points", None)
        self.closest_lanes = config.get("closest_lanes", None)
        scenario_characterization = config.get("scenario_characterization", None)
        missing = [
            name
            for name, value in (
                ("conflict_points", self.conflict_points),
                ("closest_lanes", self.closest_lanes),
                ("scenario_characterization", scenario_characterization),
            )
            if value is None
        ]
        if missing:
            error_message = f"ScenarioScorer requires the scoring config blocks: {missing} are missing."
            raise ValueError(error_message)
        self.features = SafeShiftFeatures(scenario_characterization)
        self.scorer = SafeShiftScorer(scenario_characterization)

    def score_features(self, scenario: Scenario, *, max_workers: int | None = None) -> ScenarioScores:
        """Computes features and scores; assumes map metadata is already populated on the scenario.

        ``max_workers`` is passed to the SafeShift feature computation: ``None`` lets it use all CPUs; pass ``1`` to run
        in-process (e.g. when this is already inside a worker pool, which cannot spawn child processes).
        """
        features = self.features.compute(scenario, max_workers=max_workers)
        return self.scorer.compute(scenario, features)

    def compute(
        self, scenario: Scenario, *, total_steps: int | None = None, max_workers: int | None = None
    ) -> ScenarioScores:
        """Adds map metadata (optionally truncated to ``total_steps``) then computes features and scores."""
        # TODO: conflict-point / closest-lane metadata (and features) are recomputed from scratch on every call -- the
        # expensive geometry is not cached. Add an on-disk cache (keyed by scenario id + map-config hash) so repeated
        # scoring with different score weightings does not re-pay the geometry/feature cost.
        scenario = add_scenario_map_metadata(scenario, self.conflict_points, self.closest_lanes, total_steps)
        return self.score_features(scenario, max_workers=max_workers)
