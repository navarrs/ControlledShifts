"""A script that creates a 'mini' WOMD dataset by selects a random subsample from the full dataset."""

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
from tqdm import tqdm


LOWEST_PERCENTAGE = 0.0
HIGHEST_PERCENTAGE = 1.0
SPLIT_PERCENTAGES = {"training": 0.70, "validation": 0.10, "testing": 0.20}


def copy(source_path: Path, destination_path: Path, filename: str) -> None:
    """Copies file from source to destination.

    Args:
        source_path (Path): path to the source file.
        destination_path (Path): path to the destination file.
        filename (str): name of the file to copy.
    """
    input_filepath = source_path / filename
    output_filepath = destination_path / filename
    shutil.copy(input_filepath, output_filepath)


def run(  # noqa: PLR0913
    input_data_path: Path,
    meta_path: Path,
    output_path: Path,
    output_dir: str,
    percentage: float,
    seed: int,
    n_jobs: int,
    *,
    parallel: bool = True,
) -> None:
    assert input_data_path.exists(), f"Input data path ({input_data_path}) does not exist!"
    assert LOWEST_PERCENTAGE < percentage <= HIGHEST_PERCENTAGE, (
        f"Percentage: {percentage} not in range ({LOWEST_PERCENTAGE}, {HIGHEST_PERCENTAGE}]"
    )

    input_data_filepaths = os.listdir(input_data_path)  # noqa: PTH208
    n_input_files = len(input_data_filepaths)
    n_selected = int(percentage * n_input_files)
    perc_selected = percentage * 100
    print(f"Input data directory has {n_input_files} files. {n_selected} ({perc_selected}%) will be selected.")
    meta: dict[str, Any] = {
        "input_data_dir": str(input_data_path),
        "total_input_files": n_input_files,
        "percentage_selected": int(perc_selected),
        "total_selected_files": n_selected,
        "selection_criteria": "random",
        "seed": seed,
    }
    random.seed(seed)

    # Randomize filelist
    random.shuffle(input_data_filepaths)
    selected_data_filepaths = input_data_filepaths[:n_selected]

    print(f"Meta:\n{json.dumps(meta, indent=2)}")
    n_train = int(n_selected * SPLIT_PERCENTAGES["training"])
    n_val = int(n_selected * SPLIT_PERCENTAGES["validation"])

    training_data_filepaths = selected_data_filepaths[:n_train]
    validation_data_filepaths = selected_data_filepaths[n_train : n_train + n_val]
    testing_data_filepaths = selected_data_filepaths[n_train + n_val :]

    combined = training_data_filepaths + validation_data_filepaths + testing_data_filepaths
    assert len(selected_data_filepaths) == len(set(combined)), "Duplicate items found across lists"
    splits = ["training", "validation", "testing"]
    data_splits = [training_data_filepaths, validation_data_filepaths, testing_data_filepaths]

    for split, split_files in zip(splits, data_splits, strict=False):
        # For each split select x% of files
        print(f"Selecting sample files for {split} split...")
        num_files = len(split_files)

        meta[split] = {
            "split_percentage": SPLIT_PERCENTAGES[split],
            "split_total_files": num_files,
            "split_files": split_files,
        }

        # Copy the files to the output_path
        print("Creating copy of selected files...")
        split_output_path = output_path / output_dir / split
        split_output_path.mkdir(parents=True, exist_ok=True)

        start = time.time()
        if parallel:
            Parallel(n_jobs=n_jobs)(
                delayed(copy)(input_data_path, split_output_path, f) for f in tqdm(split_files, total=len(split_files))
            )
        else:
            for f in tqdm(split_files, total=len(split_files)):
                copy(input_data_path, split_output_path, f)
        print(f"Done with {split} split in {time.time() - start} seconds.")

    # Save meta information for reference

    meta_path.mkdir(parents=True, exist_ok=True)
    output_filepath = meta_path / f"waymo_{output_dir}.json"
    print(f"Saving data selection summary to: {output_filepath}")
    with output_filepath.open("w") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_path", type=Path, default="/datasets/waymo/raw/scenario/training", help="Paths to the input data."
    )
    parser.add_argument("--meta_path", type=Path, default="./meta", help="Paths to the input data.")
    parser.add_argument("--output_path", type=Path, default="/datasets/waymo/raw/", help="Paths to the input data.")
    parser.add_argument("--output_dir", type=str, default="mini", help="Paths to the input data.")
    parser.add_argument("--percentage", type=float, default=0.20, help="Percentage of input data (0, 1] to re-split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs to run")
    parser.add_argument("--parallel", action="store_true", help="Whether to run in parallel or not.")
    args = parser.parse_args()

    run(**vars(args))
