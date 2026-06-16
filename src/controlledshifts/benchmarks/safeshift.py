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

import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from omegaconf import DictConfig

from controlledshifts import utils
from controlledshifts.benchmarks.common import (
    Benchmark,
    BenchmarkSplit,
    collect_scenario_filepaths,
    split_mapping_to_lists,
)


_LOGGER = utils.get_pylogger(__name__)


def _plot_score_density(scores_path: Path, prefix: str) -> None:
    """Saves a per-split score-density (KDE) plot to ``scores_pdf.png`` from the SafeShift score metadata.

    Args:
        scores_path: Directory containing the ``*_extra_processed_scenarios_<split>_infos.pkl`` metadata files.
        prefix: Filename prefix used to locate the metadata files.
    """
    colors = {"training": "green", "testing": "red", "validation": "blue"}
    for split in ("training", "validation", "testing"):
        split_infos = "test" if split == "testing" else "val" if split == "validation" else "training"
        metadata_filepath = scores_path / f"{prefix}extra_processed_scenarios_{split_infos}_infos.pkl"

        with metadata_filepath.open("rb") as f:
            scenario_metadata = pickle.load(f)  # nosec B301

        scores_ac = np.asarray([scenario["traj_scores_asym_combined"].max() for scenario in scenario_metadata])
        scores_fe = np.asarray([scenario["traj_scores_fe"].max() for scenario in scenario_metadata])

        sns.kdeplot(scores_ac, fill=True, color=colors[split], label=f"{split}_ac")
        sns.kdeplot(scores_fe, fill=True, color=colors[split], label=f"{split}_fe", alpha=0.7)

    plt.title("Score density plot")
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig("scores_pdf.png")
    _LOGGER.info("Saved score density plot to scores_pdf.png")


def create_safeshift_benchmark(config: DictConfig) -> BenchmarkSplit:
    """Creates the SafeShift benchmark split.

    Reads the predetermined per-split assignment from the SafeShift metadata pickle files and builds the train/val/test
    scenario lists. Scenarios listed in the metadata but absent from the input directory are recorded as invalid. When
    ``config.plot_scores`` is true, a per-split score-density plot is saved to ``scores_pdf.png``.

    Args:
        config: Hydra config with keys: input_data_path, scores_path, prefix, plot_scores.

    Returns:
        The train/val/test benchmark split.
    """
    input_data_path = Path(config.input_data_path)
    scores_path = Path(config.scores_path)

    _LOGGER.info("Creating %s benchmark", Benchmark.SAFESHIFT.value)

    available_ids = {fp.stem for fp in collect_scenario_filepaths(input_data_path)}
    _LOGGER.info("Found %d total scenario files in %s", len(available_ids), input_data_path)

    split_by_id: dict[str, str] = {}
    invalid: list[str] = []
    for split in ("training", "validation", "testing"):
        split_infos = "test" if split == "testing" else "val" if split == "validation" else "training"
        metadata_filepath = scores_path / f"{config.prefix}processed_scenarios_{split_infos}_infos.pkl"

        if not metadata_filepath.exists():
            error_message = f"Scenario metadata file not found: {metadata_filepath}"
            raise FileNotFoundError(error_message)

        with metadata_filepath.open("rb") as f:
            scenario_metadata: list[dict[str, Any]] = pickle.load(f)  # nosec B301
        _LOGGER.info("Loaded %d scenarios for split '%s' from %s", len(scenario_metadata), split, metadata_filepath)

        num_not_found = 0
        for scenario in scenario_metadata:
            scenario_id = scenario["scenario_id"]
            if scenario_id not in available_ids:
                num_not_found += 1
                invalid.append(scenario_id)
                continue
            split_by_id[scenario_id] = split

        if num_not_found:
            _LOGGER.warning("Split '%s': %d scenarios not found in input directory", split, num_not_found)

    split = split_mapping_to_lists(split_by_id, invalid=invalid)

    if config.get("plot_scores", False):
        _plot_score_density(scores_path, config.prefix)

    return split
