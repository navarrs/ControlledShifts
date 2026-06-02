"""A sanity-check script to validate scenarios in Causal-WOMD match those in WOMD's validation set."""

import json
import os
from glob import glob
from pathlib import Path
from typing import Any

import tensorflow as tf  # pyright: ignore[reportMissingTypeStubs]
from waymo_open_dataset.protos import scenario_pb2


def get_record_info(record_filepath: str) -> dict[str, Any]:
    """Reads TFrecord and summarizes its scenario information.

    Args:
        record_filepath (str): filepath to the scenario TFrecord to read.

    Returns:
        dict[str, Any]: a dictionary containing scenario information contained int he TFrecord.
    """
    record_data = tf.data.TFRecordDataset(record_filepath)

    record_info: dict[str, Any] = {"scenario_id": [], "scenario_num": [], "total_num_scenarios": 0}
    num_total_scenarios = 0
    for num_scenario, data in enumerate(record_data):
        num_total_scenarios += 1
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(data.numpy()))

        record_info["scenario_id"].append(scenario.scenario_id)
        record_info["scenario_num"].append(num_scenario)
    record_info["total_num_scenarios"] = num_total_scenarios
    return record_info


def get_intersection(causal_scenario_ids: list[str], record_scenario_ids: list[str]) -> tuple[list[str], list[int]]:
    """Gets the intersection between the Causal Scenario IDs and the original TFrecord IDs.

    Args:
        causal_scenario_ids (list[str]): list of causal scenario IDs.
        record_scenario_ids (list[str]): list of original scenario IDs.

    Returns:
        tuple[list[str], list[int]]: a tuple:
            - intersecting_scenarios (list[str]): list containing the intresecting scenarios between the two lists.
            - intersecting_idxs (list[int]): list of record indices of intersecting scenarios.
    """
    intersecting_scenarios, intersecting_idxs = [], []
    for n, scenario_id in enumerate(record_scenario_ids):
        if scenario_id in causal_scenario_ids:
            intersecting_scenarios.append(scenario_id)
            intersecting_idxs.append(n)
    return intersecting_scenarios, intersecting_idxs


def run(validation_data_path: Path, summary_filepath: Path, validation_records_output_path: Path) -> None:
    assert validation_data_path.exists(), f"Validation path {validation_data_path} does not exist!"
    assert summary_filepath.exists(), (
        f"Causal summary {summary_filepath} does not exist! Run 'process_causal_agent_labels.py' first."
    )

    print("\n\n---------------- Waymo Causal Agents ----------------")
    print(f"Loading summary file from: {summary_filepath}")
    with summary_filepath.open("r") as f:
        summary = json.load(f)
    causal_agents_scenario_ids = summary["scenario_ids"]

    validation_records_info: dict[str, Any] = {
        "labeled": {},
        "unlabeled": [],
    }
    validation_records = glob(os.path.join(validation_data_path, "*.tfrecord*"))  # noqa: PTH207, PTH118
    validation_scenario_ids = []
    print(f"Reading validation data from {len(validation_records)} records")
    for validation_record in validation_records:
        print(f"\tReading: {validation_record}")
        record_info = get_record_info(validation_record)

        record_scenario_ids = record_info["scenario_id"]
        validation_scenario_ids += record_scenario_ids
        intersecting_scenarios, intersecting_idxs = get_intersection(causal_agents_scenario_ids, record_scenario_ids)
        record_filename = validation_record.split("/")[-1]
        if len(intersecting_scenarios) == 0:
            validation_records_info["unlabeled"].append(record_filename)
        else:
            validation_records_info["labeled"][record_filename] = {
                "intersecting_scenarios": intersecting_scenarios,
                "intersecting_idxs": intersecting_idxs,
            }

    causal_set = set(causal_agents_scenario_ids)
    validation_set = set(validation_scenario_ids)
    not_found = list(causal_set.symmetric_difference(validation_set))
    print(f"Couldn't find {len(not_found)} scenarios")

    validation_records_info["not_found"] = not_found
    validation_records_info["num_not_found"] = len(not_found)

    # Save validation records summary
    validation_records_output_path.mkdir(parents=True, exist_ok=True)
    validation_records_filepath = validation_records_output_path / "validation_records.json"
    with validation_records_filepath.open("w") as f:
        json.dump(validation_records_info, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation_data_path",
        type=Path,
        default="/datasets/waymo/raw/scenario/validation",
        help="Paths to the input data.",
    )
    parser.add_argument(
        "--summary_filepath",
        type=Path,
        default="/datasets/waymo/causal_agents/summary.json",
        help="Paths to the processed data.",
    )
    parser.add_argument(
        "--validation_records_output_path",
        type=Path,
        default="/datasets/waymo/causal_agents/",
        help="Paths to the processed data.",
    )
    args = parser.parse_args()
    run(**vars(args))
