import pickle
from pathlib import Path


def _get_scenarios_in_set(base_path: Path, split: str) -> dict[str, Path]:
    split_path = base_path / split
    if not split_path.exists():
        error_message = f"Data path {split_path} does not exist."
        raise ValueError(error_message)
    return {f.stem: f for f in split_path.rglob("*.pkl")}


def _unlink_blacklisted(input_filepaths: dict[str, Path], blacklist: list[str]) -> None:
    """Copies a file from a input path to an destination path and then unlinks the source one.

    Args:
        input_filepaths (dict[str, Path]): Dictionary containing the scenario filepaths
        blacklist (Path): List of scenarios to unlink.
    """
    for scenario in blacklist:
        filepath = input_filepaths[scenario]
        print(f"Removing: {filepath}", end="\r")
        filepath.unlink()


def run(causal_benchmark_path: Path, safeshift_benchmark_path: Path) -> None:
    """Checks if there's data leakage between the training set from the CausalAgents benchmark and the val/test sets
    from the SafeShift benchmark. If there are intersecting files, they'll get removed from the val/test sets.

    Args:
       causal_benchmark_path (Path): path to the causal data.
       safeshift_benchmark_path (Path): path to the safeshift data.
    """
    # Want to check if there is any overlaps between the causal train data and the safeshift val/test data
    causal_training_infos_filepath = causal_benchmark_path / "training_processed_infos.pkl"
    if not causal_training_infos_filepath.exists():
        error_message = f"Training data path {causal_training_infos_filepath} does not exist."
        raise ValueError(error_message)
    with causal_training_infos_filepath.open("rb") as f:
        training_infos = pickle.load(f)
    causal_scenario_ids = [scenario["scenario_id"] for scenario in training_infos]
    print(f"Causal training set has {len(causal_scenario_ids)} scenarios.")

    for split in ["validation", "testing"]:
        print(f"Checking {split}...")
        safeshift_ids = _get_scenarios_in_set(safeshift_benchmark_path, split)
        print(f"\tSafeShift {split} set has {len(safeshift_ids)} scenarios.")

        causal_safeshift_intersection = list(set(causal_scenario_ids) & set(safeshift_ids.keys()))
        print(f"\tCausal and SafeShift set have {len(causal_safeshift_intersection)} intersecting scenarios")
        _unlink_blacklisted(safeshift_ids, causal_safeshift_intersection)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--causal_benchmark_path",
        type=Path,
        default="/datasets/waymo/processed/mini_causal",
        help="Path to the Causal Agents data.",
    )
    parser.add_argument(
        "--safeshift_benchmark_path",
        type=Path,
        default="/datasets/waymo/processed/safeshift_causal",
        help="Path to the SafeShift data.",
    )
    args = parser.parse_args()
    run(**vars(args))
