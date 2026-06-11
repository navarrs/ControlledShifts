"""Waymo repack: convert a decoded raw scenario dict into the open ``Scenario`` format.

This is the thin, dataset-specific bridge between the raw Waymo store (``variants/<variant>/<id>.pkl``, a decoded dict
produced by ``waymo_preprocessing``) and the dataset-agnostic agent-centric transform
(``controlledshifts.datasets.agent_centric_processor.AgentCentricProcessor``, which consumes ``Scenario`` objects).

The repack is THIN -- format conversion only, no profile-dependent shaping (that is applied per processing profile by
``AgentCentricProcessor.shape_scenario``). The stored raw dict is kept because it carries map detail (e.g. lane
entry/exit connectivity) that the ``Scenario`` schema does not represent and that the environments benchmark needs.

Unlike ``waymo.preprocessor`` (which imports tensorflow + waymo-open-dataset and runs only under Python 3.10), this
module depends only on numpy + ``characterization`` and is safe to import in the main Python 3.12 environment.
"""

import pickle  # nosec B403
from pathlib import Path

import numpy as np
from characterization.schemas import (
    AgentData,
    DynamicMapData,
    Scenario,
    ScenarioMetadata,
    StaticMapData,
    TracksToPredict,
)
from characterization.utils.common import AgentType


def _polyline_ids(polyline: dict, key: str) -> np.ndarray:
    """Extracts polyline IDs from a decoded map dictionary."""
    return np.array([value["id"] for value in polyline[key]], dtype=np.int32)


def _speed_limit_mph(polyline: dict, key: str) -> np.ndarray:
    """Extracts lane speed limits (mph) from a decoded map dictionary."""
    return np.array([value["speed_limit_mph"] for value in polyline[key]], dtype=np.float32)


def _polyline_idxs(polyline: dict, key: str) -> np.ndarray | None:
    """Extracts polyline index ranges from a decoded map dictionary (``None`` when empty)."""
    polyline_idxs = np.array(
        [[value["polyline_index"][0], value["polyline_index"][1]] for value in polyline[key]], dtype=np.int32
    )
    if polyline_idxs.shape[0] == 0:
        return None
    return polyline_idxs


def repack_agent_data(track_infos: dict) -> AgentData:
    """Converts decoded Waymo agent fields into an ``AgentData`` object (full trajectories, no profile masking)."""
    object_types = [AgentType[name] for name in track_infos["object_type"]]
    return AgentData(
        agent_ids=track_infos["object_id"],
        agent_types=object_types,
        agent_trajectories=track_infos["trajs"],
    )


def repack_static_map_data(map_infos: dict | None) -> StaticMapData | None:
    """Converts decoded Waymo static map fields into a ``StaticMapData`` object."""
    if map_infos is None or "all_polylines" not in map_infos:
        return None
    map_polylines = map_infos["all_polylines"].astype(np.float32)
    return StaticMapData(
        map_polylines=map_polylines,
        lane_ids=_polyline_ids(map_infos, "lane") if "lane" in map_infos else None,
        lane_speed_limits_mph=_speed_limit_mph(map_infos, "lane") if "lane" in map_infos else None,
        lane_polyline_idxs=_polyline_idxs(map_infos, "lane") if "lane" in map_infos else None,
        road_line_ids=_polyline_ids(map_infos, "road_line") if "road_line" in map_infos else None,
        road_line_polyline_idxs=_polyline_idxs(map_infos, "road_line") if "road_line" in map_infos else None,
        road_edge_ids=_polyline_ids(map_infos, "road_edge") if "road_edge" in map_infos else None,
        road_edge_polyline_idxs=_polyline_idxs(map_infos, "road_edge") if "road_edge" in map_infos else None,
        crosswalk_ids=_polyline_ids(map_infos, "crosswalk") if "crosswalk" in map_infos else None,
        crosswalk_polyline_idxs=_polyline_idxs(map_infos, "crosswalk") if "crosswalk" in map_infos else None,
        speed_bump_ids=_polyline_ids(map_infos, "speed_bump") if "speed_bump" in map_infos else None,
        speed_bump_polyline_idxs=_polyline_idxs(map_infos, "speed_bump") if "speed_bump" in map_infos else None,
        stop_sign_ids=_polyline_ids(map_infos, "stop_sign") if "stop_sign" in map_infos else None,
        stop_sign_polyline_idxs=_polyline_idxs(map_infos, "stop_sign") if "stop_sign" in map_infos else None,
        stop_sign_lane_ids=[stop_sign["lane_ids"] for stop_sign in map_infos.get("stop_sign", {"lane_ids": []})],
    )


def repack_dynamic_map_data(dynamic_map_infos: dict) -> DynamicMapData:
    """Converts decoded Waymo dynamic map fields into a ``DynamicMapData`` object (full length, no profile slicing)."""
    stop_points = dynamic_map_infos["stop_point"]
    lane_id = dynamic_map_infos["lane_id"]
    states = dynamic_map_infos["state"]
    if len(stop_points) == 0:
        return DynamicMapData(stop_points=None, lane_ids=None, states=None)
    return DynamicMapData(stop_points=stop_points, lane_ids=lane_id, states=states)


def repack_scenario(raw: dict) -> Scenario:
    """Repacks a decoded Waymo scenario dict into an open ``Scenario`` (thin: no profile-dependent shaping)."""
    agent_data = repack_agent_data(raw["track_infos"])
    static_map_data = repack_static_map_data(raw["map_infos"])
    dynamic_map_data = repack_dynamic_map_data(raw["dynamic_map_infos"])

    timestamps = list(raw["timestamps_seconds"])
    diffs = np.diff(timestamps)
    frequency_hz = min(np.round(1 / np.mean(diffs)).item(), 10.0) if len(diffs) else 10.0

    ego_index = raw["sdc_track_index"]
    tracks_to_predict = TracksToPredict(
        track_index=raw["tracks_to_predict"]["track_index"],
        difficulty=raw["tracks_to_predict"]["difficulty"],
        object_type=[AgentType[name] for name in raw["tracks_to_predict"]["object_type"]],
    )
    metadata = ScenarioMetadata(
        scenario_id=raw["scenario_id"],
        timestamps_seconds=timestamps,
        frequency_hz=frequency_hz,
        current_time_index=raw["current_time_index"],
        ego_vehicle_id=agent_data.agent_ids[ego_index],
        ego_vehicle_index=ego_index,
        track_length=len(timestamps),
        objects_of_interest=raw["objects_of_interest"],
        dataset="waymo",
    )
    return Scenario(
        metadata=metadata,
        agent_data=agent_data,
        tracks_to_predict=tracks_to_predict,
        static_map_data=static_map_data,
        dynamic_map_data=dynamic_map_data,
    )


def load_scenario(path: Path) -> Scenario:
    """Loads a decoded raw scenario dict from a variant store and repacks it into an open ``Scenario``."""
    with Path(path).open("rb") as f:
        raw = pickle.load(f)  # nosec B301
    return repack_scenario(raw)
