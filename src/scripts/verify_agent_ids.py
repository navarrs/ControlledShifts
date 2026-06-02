"""A script that validates if causal agent IDs exist in the processed version of the dataset."""

import json
import pickle
from pathlib import Path
from typing import Any

from tqdm import tqdm


def run(input_data_path: Path, labels_path: Path, summary_path: Path) -> None:
    assert input_data_path.exists(), f"Input data path {input_data_path} does not exist!"
    assert labels_path.exists(), f"Labels path {labels_path} does not exist!"

    scenario_filepaths = []
    for split in ["training", "validation", "testing"]:
        split_path = input_data_path / split
        files = split_path.glob("*.pkl")
        scenario_filepaths.extend(files)

    scenario_dict = {}
    for file in scenario_filepaths:
        scenario_id = file.split("/")[-1].split(".")[0]
        scenario_dict[scenario_id] = file

    valid_scenarios: dict[str, Any] = {
        "found_valid_ids": [],
        "found_invalid_ids": [],
        "not_found": [],
        "num_found_valid_ids": 0,
        "num_found_invalid_ids": 0,
        "num_not_found": 0,
    }

    labels_filepaths = labels_path.glob("*json")
    for labels_filepath in tqdm(labels_filepaths):
        scenario_id = str(labels_filepath).split("/")[-1].split(".")[0]

        scenario_filepath = scenario_dict.get(scenario_id)
        if scenario_filepath is None:
            valid_scenarios["not_found"].append(scenario_id)
            continue

        with labels_filepath.open("r") as f:
            labels = json.load(f)
        causal_agent_ids = set(map(int, labels["causal_ids"]))

        with scenario_filepath.open("rb") as f:
            scenario = pickle.load(f)
        agent_ids = scenario["track_infos"]["object_id"]

        if causal_agent_ids.issubset(agent_ids):
            valid_scenarios["found_valid_ids"].append(scenario_id)
        else:
            valid_scenarios["found_invalid_ids"].append(scenario_id)

    num_found_valid_ids = len(valid_scenarios["found_valid_ids"])
    num_found_invalid_ids = len(valid_scenarios["found_invalid_ids"])
    num_not_found = len(valid_scenarios["not_found"])
    valid_scenarios["num_found_valid_ids"] = num_found_valid_ids
    valid_scenarios["num_found_invalid_ids"] = num_found_invalid_ids
    valid_scenarios["num_not_found"] = num_not_found

    print(f"Valid agent IDs:\n{json.dumps(valid_scenarios, indent=2)}")
    summary_filepath = summary_path / "valid_labels.json"
    with summary_filepath.open("w") as f:
        json.dump(valid_scenarios, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_path",
        type=Path,
        default="/datasets/waymo/processed/mini_causal",
        help="Paths to the output data.",
    )
    parser.add_argument(
        "--labels_path",
        type=Path,
        default="/datasets/waymo/causal_agents/processed_labels",
        help="Paths to the output data.",
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default="/datasets/waymo/causal_agents/",
        help="Paths to the output data.",
    )
    args = parser.parse_args()

    run(**vars(args))
