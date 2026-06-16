"""Waymo preprocessing: decode raw Waymo scenario protos into the canonical raw scenario store (Stage A).

Reads raw Waymo Open Motion scenario protos and writes one decoded scenario dict per scenario into a flat canonical
variant store (``variants/base/<scenario_id>.pkl``), plus a ``_variant_manifest.json`` recording each scenario's raw
Waymo origin split. The decoded dict (``track_infos``, ``map_infos``, ``dynamic_map_infos``, ...) is kept as the
canonical format because it carries map detail (e.g. lane entry/exit connectivity) that downstream consumers such as
the environments benchmark need and that the open ``Scenario`` schema does not represent. The thin repack to
``Scenario`` happens later, in the main environment, via ``controlledshifts.datasets.waymo.repacker``.

Adapted from: https://arxiv.org/abs/2209.13508

NOTE: this script depends on ``tensorflow`` + ``waymo-open-dataset``, which require Python 3.10. Run it in a dedicated
Python 3.10 environment (it has no other project dependency). It is deliberately kept out of the training/build import
graph (``datasets/__init__.py`` and ``datasets/waymo/__init__.py`` stay empty) so the main Python 3.12 environment
never imports tensorflow/waymo.
"""

import argparse
import json
import multiprocessing
import pickle  # nosec B403
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2


object_type = {0: "TYPE_UNSET", 1: "TYPE_VEHICLE", 2: "TYPE_PEDESTRIAN", 3: "TYPE_CYCLIST", 4: "TYPE_OTHER"}

lane_type = {0: "TYPE_UNDEFINED", 1: "TYPE_FREEWAY", 2: "TYPE_SURFACE_STREET", 3: "TYPE_BIKE_LANE"}

road_line_type = {
    0: "TYPE_UNKNOWN",
    1: "TYPE_BROKEN_SINGLE_WHITE",
    2: "TYPE_SOLID_SINGLE_WHITE",
    3: "TYPE_SOLID_DOUBLE_WHITE",
    4: "TYPE_BROKEN_SINGLE_YELLOW",
    5: "TYPE_BROKEN_DOUBLE_YELLOW",
    6: "TYPE_SOLID_SINGLE_YELLOW",
    7: "TYPE_SOLID_DOUBLE_YELLOW",
    8: "TYPE_PASSING_DOUBLE_YELLOW",
}

road_edge_type = {
    0: "TYPE_UNKNOWN",
    # // Physical road boundary that doesn't have traffic on the other side (e.g.,
    # // a curb or the k-rail on the right side of a freeway).
    1: "TYPE_ROAD_EDGE_BOUNDARY",
    # // Physical road boundary that separates the car from other traffic
    # // (e.g. a k-rail or an island).
    2: "TYPE_ROAD_EDGE_MEDIAN",
}

polyline_type = {
    # for lane
    "TYPE_UNDEFINED": -1,
    "TYPE_FREEWAY": 1,
    "TYPE_SURFACE_STREET": 2,
    "TYPE_BIKE_LANE": 3,
    # for roadline
    "TYPE_UNKNOWN": -1,
    "TYPE_BROKEN_SINGLE_WHITE": 6,
    "TYPE_SOLID_SINGLE_WHITE": 7,
    "TYPE_SOLID_DOUBLE_WHITE": 8,
    "TYPE_BROKEN_SINGLE_YELLOW": 9,
    "TYPE_BROKEN_DOUBLE_YELLOW": 10,
    "TYPE_SOLID_SINGLE_YELLOW": 11,
    "TYPE_SOLID_DOUBLE_YELLOW": 12,
    "TYPE_PASSING_DOUBLE_YELLOW": 13,
    # for roadedge
    "TYPE_ROAD_EDGE_BOUNDARY": 15,
    "TYPE_ROAD_EDGE_MEDIAN": 16,
    # for stopsign
    "TYPE_STOP_SIGN": 17,
    # for crosswalk
    "TYPE_CROSSWALK": 18,
    # for speed bump
    "TYPE_SPEED_BUMP": 19,
}


signal_state = {
    0: "LANE_STATE_UNKNOWN",
    # // States for traffic signals with arrows.
    1: "LANE_STATE_ARROW_STOP",
    2: "LANE_STATE_ARROW_CAUTION",
    3: "LANE_STATE_ARROW_GO",
    # // Standard round traffic signals.
    4: "LANE_STATE_STOP",
    5: "LANE_STATE_CAUTION",
    6: "LANE_STATE_GO",
    # // Flashing light signals.
    7: "LANE_STATE_FLASHING_STOP",
    8: "LANE_STATE_FLASHING_CAUTION",
}

signal_state_to_id = {val: key for key, val in signal_state.items()}


class TrackInfos(TypedDict):
    """Decoded agent tracks: per-object ids, types, and stacked trajectories."""

    object_id: list[int]
    object_type: list[str]
    trajs: np.ndarray


class MapInfos(TypedDict):
    """Decoded map features grouped by type, plus the concatenated polyline buffer."""

    lane: list[dict]
    road_line: list[dict]
    road_edge: list[dict]
    stop_sign: list[dict]
    crosswalk: list[dict]
    speed_bump: list[dict]
    all_polylines: NotRequired[np.ndarray]


class DynamicMapInfos(TypedDict):
    """Decoded dynamic map states (e.g. traffic signals) per timestep."""

    lane_id: list[np.ndarray]
    state: list[np.ndarray]
    stop_point: list[np.ndarray]


def decode_tracks_from_proto(tracks: Iterable[scenario_pb2.Track]) -> TrackInfos:
    """Decodes agent tracks from Waymo scenario proto.

    Args:
        tracks: List of scenario_pb2.Track objects.

    Returns:
        dict: Dictionary with keys 'object_id', 'object_type', and 'trajs' containing
            agent IDs, types, and trajectories as numpy arrays.
    """
    object_ids: list[int] = []
    object_types: list[str] = []
    trajs: list[np.ndarray] = []
    for cur_data in tracks:  # number of objects
        cur_traj = [
            np.array(
                [
                    x.center_x,
                    x.center_y,
                    x.center_z,
                    x.length,
                    x.width,
                    x.height,
                    x.heading,
                    x.velocity_x,
                    x.velocity_y,
                    x.valid,
                ],
                dtype=np.float32,
            )
            for x in cur_data.states
        ]

        object_ids.append(cur_data.id)
        object_types.append(object_type[cur_data.object_type])
        trajs.append(np.stack(cur_traj, axis=0))  # (num_timestamp, 10)

    return {
        "object_id": object_ids,
        "object_type": object_types,
        "trajs": np.stack(trajs, axis=0),  # (num_objects, num_timestamp, 10)
    }


def get_polyline_dir(polyline: np.ndarray) -> np.ndarray:
    """Computes direction vectors for each segment of a polyline.

    Args:
        polyline (np.ndarray): Array of polyline points (shape: [N, 3]).

    Returns:
        np.ndarray: Array of direction vectors (shape: [N, 3]).
    """
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    return diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)


def decode_map_features_from_proto(map_features: Iterable[scenario_pb2.MapFeature]) -> MapInfos:  # noqa: PLR0915
    """Decodes map features from Waymo scenario proto.

    Args:
        map_features: List of scenario_pb2.MapFeature objects.

    Returns:
        dict: Dictionary containing map features (lanes, road lines, road edges, stop signs,
            crosswalks, speed bumps) and all polylines as numpy arrays.
    """
    map_infos: MapInfos = {
        "lane": [],
        "road_line": [],
        "road_edge": [],
        "stop_sign": [],
        "crosswalk": [],
        "speed_bump": [],
    }
    polylines_list = []

    point_cnt = 0
    for cur_data in map_features:
        cur_info = {"id": cur_data.id}

        if cur_data.lane.ByteSize() > 0:
            cur_info["speed_limit_mph"] = cur_data.lane.speed_limit_mph
            cur_info["type"] = lane_type[
                cur_data.lane.type
            ]  # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane

            cur_info["interpolating"] = cur_data.lane.interpolating
            cur_info["entry_lanes"] = list(cur_data.lane.entry_lanes)
            cur_info["exit_lanes"] = list(cur_data.lane.exit_lanes)

            cur_info["left_boundary"] = [
                {
                    "start_index": x.lane_start_index,
                    "end_index": x.lane_end_index,
                    "feature_id": x.boundary_feature_id,
                    "boundary_type": x.boundary_type,
                }
                for x in cur_data.lane.left_boundaries
            ]
            cur_info["right_boundary"] = [
                {
                    "start_index": x.lane_start_index,
                    "end_index": x.lane_end_index,
                    "feature_id": x.boundary_feature_id,
                    "boundary_type": road_line_type[x.boundary_type],
                }
                for x in cur_data.lane.right_boundaries
            ]

            global_type = polyline_type[cur_info["type"]]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.lane.polyline], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["lane"].append(cur_info)

        elif cur_data.road_line.ByteSize() > 0:
            cur_info["type"] = road_line_type[cur_data.road_line.type]

            global_type = polyline_type[cur_info["type"]]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_line.polyline], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["road_line"].append(cur_info)

        elif cur_data.road_edge.ByteSize() > 0:
            cur_info["type"] = road_edge_type[cur_data.road_edge.type]

            global_type = polyline_type[cur_info["type"]]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_edge.polyline], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["road_edge"].append(cur_info)

        elif cur_data.stop_sign.ByteSize() > 0:
            cur_info["lane_ids"] = list(cur_data.stop_sign.lane)
            point = cur_data.stop_sign.position
            cur_info["position"] = np.array([point.x, point.y, point.z])

            global_type = polyline_type["TYPE_STOP_SIGN"]
            cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 7)

            map_infos["stop_sign"].append(cur_info)
        elif cur_data.crosswalk.ByteSize() > 0:
            global_type = polyline_type["TYPE_CROSSWALK"]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.crosswalk.polygon], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["crosswalk"].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            global_type = polyline_type["TYPE_SPEED_BUMP"]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.speed_bump.polygon], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["speed_bump"].append(cur_info)

        else:
            continue
            # print(cur_data)
            # raise ValueError

        polylines_list.append(cur_polyline)
        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    polylines = np.zeros((0, 7), dtype=np.float32)
    if len(polylines_list) == 0:
        print("No polylines found in the map features.")
        return map_infos

    polylines = np.concatenate(polylines_list, axis=0).astype(np.float32)
    map_infos["all_polylines"] = polylines
    return map_infos


def decode_dynamic_map_states_from_proto(dynamic_map_states: Iterable[scenario_pb2.DynamicMapState]) -> DynamicMapInfos:
    """Decodes dynamic map states (e.g., traffic signals) from Waymo scenario proto.

    Args:
        dynamic_map_states: List of scenario_pb2.DynamicMapState objects.

    Returns:
        dict: Dictionary with lane IDs, signal states, and stop points for each timestep.
    """
    dynamic_map_infos: DynamicMapInfos = {"lane_id": [], "state": [], "stop_point": []}
    for cur_data in dynamic_map_states:  # (num_timestamp)
        lane_id, state, stop_point = [], [], []
        # Skip over empty ones
        if not len(cur_data.lane_states):
            lane_id.append([])
            state.append([])
            stop_point.append([])
            continue

        for cur_signal in cur_data.lane_states:  # (num_observed_signals)
            lane_id.append(cur_signal.lane)
            state.append(signal_state[cur_signal.state])
            stop_point.append([cur_signal.stop_point.x, cur_signal.stop_point.y, cur_signal.stop_point.z])

        dynamic_map_infos["lane_id"].append(np.array([lane_id]))
        dynamic_map_infos["state"].append(np.array([state]))
        dynamic_map_infos["stop_point"].append(np.array([stop_point]))

    return dynamic_map_infos


def process_waymo_data_with_scenario_proto(
    data_file: Path, output_path: Path, scenario_ids: list[str] | None = None
) -> list[dict]:
    """Decodes one .tfrecord file, writing a decoded scenario dict per scenario; returns per-scenario metadata."""
    dataset = tf.data.TFRecordDataset(str(data_file), compression_type="")
    ret_infos = []
    for data in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data.numpy())
        if scenario_ids is not None and scenario.scenario_id not in scenario_ids:
            continue

        info = {
            "scenario_id": scenario.scenario_id,
            "timestamps_seconds": list(scenario.timestamps_seconds),
            "current_time_index": scenario.current_time_index,
            "sdc_track_index": scenario.sdc_track_index,
            "objects_of_interest": list(scenario.objects_of_interest),
        }
        info["tracks_to_predict"] = {
            "track_index": [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
            "difficulty": [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict],
        }
        track_infos = decode_tracks_from_proto(scenario.tracks)
        info["tracks_to_predict"]["object_type"] = [
            track_infos["object_type"][cur_idx] for cur_idx in info["tracks_to_predict"]["track_index"]
        ]
        map_infos = decode_map_features_from_proto(scenario.map_features)
        dynamic_map_infos = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)

        save_infos: dict[str, object] = {
            "track_infos": track_infos,
            "dynamic_map_infos": dynamic_map_infos,
            "map_infos": map_infos,
        }
        save_infos.update(info)

        with (output_path / f"{scenario.scenario_id}.pkl").open("wb") as f:
            pickle.dump(save_infos, f)
        ret_infos.append(info)
    return ret_infos


def get_infos_from_protos(
    data_path: Path, output_path: Path, scenario_ids: list[str] | None = None, num_workers: int = 8
) -> list[dict]:
    """Decodes all .tfrecord files in a directory in parallel, writing decoded scenario dicts to ``output_path``."""
    output_path.mkdir(parents=True, exist_ok=True)
    src_files = sorted(data_path.glob("*.tfrecord*"))
    func = partial(process_waymo_data_with_scenario_proto, output_path=output_path, scenario_ids=scenario_ids)
    with multiprocessing.Pool(num_workers) as pool:
        data_infos = list(tqdm(pool.imap(func, src_files), total=len(src_files)))
    return [item for infos in data_infos for item in infos]


def run(  # noqa: PLR0913
    *,
    raw_data_path: Path,
    proc_data_path: Path,
    split: str,
    search_safeshift: bool,
    safeshift_data_splits_path: Path,
    safeshift_prefix: str,
    num_workers: int = 8,
) -> None:
    """Decodes every scenario of a raw Waymo split into the flat canonical raw scenario store.

    Args:
        raw_data_path (Path): Root of the raw Waymo scenario protos (expects a ``<split>`` subdirectory).
        proc_data_path (Path): Flat output directory for the canonical base variant store (decoded scenario dicts).
        split (str): Raw Waymo split to process ('training', 'validation', or 'testing').
        search_safeshift (bool): If True, only process scenarios listed in the SafeShift split metadata.
        safeshift_data_splits_path (Path): Directory with the SafeShift ``*_infos.pkl`` metadata files.
        safeshift_prefix (str): Filename prefix for the SafeShift metadata files.
        num_workers (int): Number of parallel worker processes. Defaults to 8.

    Raises:
        ValueError: If the raw split directory does not exist.
    """
    split_raw_data_path = raw_data_path / split
    if not split_raw_data_path.exists():
        error_message = f"Raw data path {split_raw_data_path} does not exist."
        raise ValueError(error_message)

    # Write all scenarios flat into the canonical base variant store (keyed by scenario_id), regardless of which raw
    # Waymo split they came from. Benchmark splits (splits/*.json) are the source of truth for train/val/test; the raw
    # origin split is recorded in _variant_manifest.json only for provenance.
    proc_data_path.mkdir(parents=True, exist_ok=True)

    scenario_ids: list[str] = []
    if search_safeshift:
        print("Only searching for SafeShift scenarios")
        for safeshift_split in ["training", "val", "test"]:
            info_filepath = (
                safeshift_data_splits_path / f"{safeshift_prefix}processed_scenarios_{safeshift_split}_infos.pkl"
            )
            print(f"Loading infos from: {info_filepath}")
            with info_filepath.open("rb") as f:
                scenario_info = pickle.load(f)  # nosec B301
            scenario_ids.extend([scenario["scenario_id"] for scenario in scenario_info])
    selected_ids = None if len(scenario_ids) == 0 else scenario_ids

    sample_infos = get_infos_from_protos(
        data_path=split_raw_data_path, output_path=proc_data_path, scenario_ids=selected_ids, num_workers=num_workers
    )
    print(f"Wrote {len(sample_infos)} scenario dicts to {proc_data_path}")

    # Record each scenario's raw Waymo origin split in the variant manifest (merged across per-split runs).
    manifest_path = proc_data_path / "_variant_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r") as f:
            manifest = json.load(f)
    for info in sample_infos:
        manifest[info["scenario_id"]] = split
    with manifest_path.open("w") as f:
        json.dump(manifest, f)
    print(f"Updated variant manifest at {manifest_path} ({len(manifest)} scenarios)")


def main() -> None:
    """CLI entry point for the Waymo decode preprocessing."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path", type=Path, default=Path("/data/driving/waymo/raw/mini"), help="Root of the raw input data."
    )
    parser.add_argument(
        "--proc_data_path",
        type=Path,
        default=Path("/data/driving/waymo/variants/base"),
        help="Flat output directory for the canonical base variant store (decoded scenario dicts).",
    )
    parser.add_argument("--split", type=str, default="training", choices=["training", "validation", "testing"])
    parser.add_argument(
        "--search_safeshift", action="store_true", help="If set, only process scenarios from the SafeShift splits."
    )
    parser.add_argument(
        "--safeshift_data_splits_path",
        type=Path,
        default=Path("/data/driving/waymo/meta/safeshift/mtr_process_splits"),
        help="Directory with the SafeShift split metadata files.",
    )
    parser.add_argument(
        "--safeshift_prefix",
        type=str,
        default="score_asym_combined_80_",
        help="Prefix for the SafeShift metadata files.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes.")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
