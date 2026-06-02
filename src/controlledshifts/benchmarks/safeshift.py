r"""Benchmark creation for the SafeShift benchmark.

Re-organizes SafeShift scenarios into training/validation/testing splits using per-split metadata pickle files.

Example usage:

    uv run -m controlledshifts.create_benchmark benchmark=safeshift \
        input_data_path=/datasets/scenarios/safeshift_all \
        output_data_path=/datasets/waymo/processed/safeshift \
        scores_path=/datasets/waymo/mtr_process_splits \
        prefix=score_asym_combined_80_

See configs/benchmark/safeshift.yaml for all available options.
"""

import itertools
import multiprocessing
import pickle
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from omegaconf import DictConfig
from tqdm import tqdm

from controlledshifts import utils
from controlledshifts.benchmarks.common import Benchmark, copy_scenario, create_split_dirs


_LOGGER = utils.get_pylogger(__name__)


def _verify_splits(output_path: Path, scores_path: Path, prefix: str) -> None:
    """Verifies splits have no overlapping scenario IDs and saves a score density plot.

    Args:
        output_path: Root directory containing the split subdirectories.
        scores_path: Path to the SafeShift score metadata directory.
        prefix: Filename prefix used to locate the metadata files.
    """
    splits = ["training", "validation", "testing"]
    split_data: dict[str, set[str]] = {
        split: {p.name for p in (output_path / split).rglob("*.pkl") if p.is_file()} for split in splits
    }
    for set1, set2 in itertools.combinations(splits, 2):
        intersection = split_data[set1] & split_data[set2]
        _LOGGER.info("Intersection between (%s, %s): %d", set1, set2, len(intersection))

    colors = {"training": "green", "testing": "red", "validation": "blue"}
    for split in splits:
        split_infos = "test" if split == "testing" else "val" if split == "validation" else "training"
        metadata_filepath = scores_path / f"{prefix}extra_processed_scenarios_{split_infos}_infos.pkl"

        with metadata_filepath.open("rb") as f:
            scenario_metadata = pickle.load(f)

        scores_ac = np.asarray([scenario["traj_scores_asym_combined"].max() for scenario in scenario_metadata])
        scores_fe = np.asarray([scenario["traj_scores_fe"].max() for scenario in scenario_metadata])

        sns.kdeplot(scores_ac, fill=True, color=colors[split], label=f"{split}_ac")
        sns.kdeplot(scores_fe, fill=True, color=colors[split], label=f"{split}_fe", alpha=0.7)

    plt.title("Score density plot")
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig("scores_pdf.png")


def create_safeshift_benchmark(config: DictConfig) -> None:
    """Creates benchmark splits for the SafeShift benchmark.

    Reads per-split metadata from SafeShift pickle files, then copies the corresponding scenario files into
    training/validation/testing subdirectories under output_data_path.

    Args:
        config: Hydra config.
            Expected keys: input_data_path, output_data_path, scores_path, prefix, num_workers.
    """
    input_data_path = Path(config.input_data_path)
    output_data_path = Path(config.output_data_path)
    scores_path = Path(config.scores_path)

    _LOGGER.info("Creating %s benchmark", Benchmark.SAFESHIFT.value)
    create_split_dirs(output_data_path)

    scenario_filepaths: dict[str, Path] = {path.stem: path for path in input_data_path.rglob("*.pkl")}
    _LOGGER.info("Found %d total scenario files in %s", len(scenario_filepaths), input_data_path)

    tasks: list[tuple[str, Path, Path]] = []
    for split in ["training", "validation", "testing"]:
        split_infos = "test" if split == "testing" else "val" if split == "validation" else "training"
        metadata_filepath = scores_path / f"{config.prefix}processed_scenarios_{split_infos}_infos.pkl"

        if not metadata_filepath.exists():
            error_message = f"Scenario metadata file not found: {metadata_filepath}"
            raise FileNotFoundError(error_message)

        with metadata_filepath.open("rb") as f:
            scenario_metadata: list[dict[str, Any]] = pickle.load(f)
        _LOGGER.info("Loaded %d scenarios for split '%s' from %s", len(scenario_metadata), split, metadata_filepath)

        output_split_path = output_data_path / split
        num_not_found = 0
        for scenario in scenario_metadata:
            scenario_id = scenario["scenario_id"]
            input_filepath = scenario_filepaths.get(scenario_id)
            if input_filepath is None:
                num_not_found += 1
                continue
            tasks.append((scenario_id, input_filepath, output_split_path / f"{scenario_id}.pkl"))

        if num_not_found:
            _LOGGER.warning("Split '%s': %d scenarios not found in input directory", split, num_not_found)

    _LOGGER.info("Starting parallel copy of %d scenarios with %d workers", len(tasks), config.num_workers)
    with multiprocessing.Pool(config.num_workers) as pool:
        list(
            tqdm(
                pool.starmap(partial(copy_scenario, unlink_source=config.unlink_source), tasks),
                total=len(tasks),
                desc="Copying scenarios",
            )
        )

    _verify_splits(output_data_path, scores_path, config.prefix)
    _LOGGER.info("SafeShift benchmark creation complete")
