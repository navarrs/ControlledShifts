r"""Benchmark creation for the Non-Causal Agents benchmark.

A harder variant of the Causal Agents benchmark that focuses on a single perturbation (removing non-causal agents) and
re-organizes scenarios into train/validation/testing splits by difficulty. Difficulty is the number of non-causal agents
in a scenario: scenarios are sorted ascending by that count and the hardest ones (at or above ``cutoff_percentile``)
form the test set, mirroring how ego_safeshift/safeshift move the hardest scenarios to test.

Each scenario is materialized twice under the same split: an unperturbed ``original`` copy and a ``perturbed`` copy with
non-causal agents removed. This keeps a 1-1 correspondence so the original and perturbed versions of the same held-out
scenarios can be compared. Existing ``remove_noncausal`` perturbed files are reused when found; otherwise they are
generated on the fly.

Output layout under output_data_path::

    non_causal_agents/
    ├── original/   {training, validation, testing}
    └── perturbed/  {training, validation, testing}

Example usage:

    uv run -m controlledshifts.create_benchmark benchmark=non_causal_agents \\
        input_data_path=/data/driving/waymo/processed/mini_causal \\
        output_data_path=/data/driving/waymo/processed/non_causal_agents \\
        causal_labels_path=/data/driving/waymo/causal_agents/processed_labels \\
        perturbed_data_path=/data/driving/waymo/processed/remove_noncausal

See configs/benchmark/non_causal_agents.yaml for all available options.
"""

import json
import multiprocessing
import pickle  # nosec B403
from functools import partial
from pathlib import Path

import numpy as np
from numpy.random import Generator, default_rng
from omegaconf import DictConfig
from tqdm import tqdm

from controlledshifts import utils
from controlledshifts.benchmarks.causal_agents import remove_noncausal
from controlledshifts.benchmarks.common import (
    collect_scenario_filepaths,
    copy_scenario,
    create_split_dirs,
    get_noncausal_mask,
    verify_splits,
)


_LOGGER = utils.get_pylogger(__name__)


def _count_noncausal(input_filepath: Path, causal_labels_path: Path) -> tuple[str, int] | None:
    """Counts the non-causal agents in a scenario.

    Args:
        input_filepath: Path to the input scenario pkl.
        causal_labels_path: Directory with per-scenario JSON causal labels.

    Returns:
        Tuple of (scenario_id, non-causal count), or None if the scenario or its causal labels are missing.
    """
    if not input_filepath.exists():
        return None

    with input_filepath.open("rb") as f:
        scenario = pickle.load(f)  # nosec B301

    scenario_id = scenario["scenario_id"]
    causal_labels_filepath = causal_labels_path / f"{scenario_id}.json"
    if not causal_labels_filepath.exists():
        return None

    with causal_labels_filepath.open("r") as f:
        causal_labels = json.load(f)

    noncausal_mask = get_noncausal_mask(scenario, causal_labels)
    return scenario_id, int(noncausal_mask.sum())


def _materialize_scenario(
    task: tuple[str, Path, Path | None, str], causal_labels_path: Path, output_data_path: Path
) -> None:
    """Writes the original and perturbed copies of a scenario into its assigned split.

    The original copy is taken verbatim from the input file. The perturbed copy reuses an existing ``remove_noncausal``
    file when available, and is otherwise generated on the fly from the input scenario and its causal labels.

    Args:
        task: Tuple of (scenario_id, original_filepath, perturbed_filepath_or_None, split).
        causal_labels_path: Directory with per-scenario JSON causal labels (used when generating the perturbed copy).
        output_data_path: Root output directory containing the ``original`` and ``perturbed`` subtrees.
    """
    scenario_id, original_filepath, perturbed_filepath, split = task

    original_output = output_data_path / "original" / split / f"{scenario_id}.pkl"
    copy_scenario(scenario_id, original_filepath, original_output)

    perturbed_output = output_data_path / "perturbed" / split / f"{scenario_id}.pkl"
    if perturbed_filepath is not None and perturbed_filepath.exists():
        copy_scenario(scenario_id, perturbed_filepath, perturbed_output)
        return

    causal_labels_filepath = causal_labels_path / f"{scenario_id}.json"
    if not original_filepath.exists() or not causal_labels_filepath.exists():
        return

    with original_filepath.open("rb") as f:
        scenario = pickle.load(f)  # nosec B301
    with causal_labels_filepath.open("r") as f:
        causal_labels = json.load(f)

    remove_noncausal(scenario, causal_labels, perturbed_output)


def _assign_splits(
    counts: list[tuple[str, int]], cutoff_percentile: float, validation_percentage: float, random_generator: Generator
) -> dict[str, str]:
    """Assigns scenarios to train/validation/testing splits by non-causal count.

    Scenarios at or above the ``cutoff_percentile`` of non-causal counts form the test set; the remainder is the
    train/val pool, shuffled with ``validation_percentage`` held out for validation.

    Args:
        counts: List of (scenario_id, non-causal count) tuples.
        cutoff_percentile: Percentile of non-causal counts at/above which scenarios go to the test set.
        validation_percentage: Percentage of the train/val pool held out for validation.
        random_generator: Random number generator used to shuffle the train/val pool.

    Returns:
        Mapping from scenario_id to split name ("training", "validation", or "testing").
    """
    scenario_ids = np.array([scenario_id for scenario_id, _ in counts])
    count_values = np.array([count for _, count in counts])
    cutoff = np.percentile(count_values, cutoff_percentile)

    testing_scenarios = scenario_ids[count_values >= cutoff].tolist()

    train_val_scenarios = scenario_ids[count_values < cutoff].tolist()
    random_generator.shuffle(train_val_scenarios)
    num_validation_scenarios = int(len(train_val_scenarios) * (validation_percentage / 100.0))
    validation_scenarios = train_val_scenarios[:num_validation_scenarios]
    training_scenarios = train_val_scenarios[num_validation_scenarios:]

    _LOGGER.info(
        "Non-causal count cutoff (p%.1f): %.2f. Splits -> training: %d, validation: %d, testing: %d",
        cutoff_percentile,
        cutoff,
        len(training_scenarios),
        len(validation_scenarios),
        len(testing_scenarios),
    )

    split_by_id = dict.fromkeys(training_scenarios, "training")
    split_by_id.update(dict.fromkeys(validation_scenarios, "validation"))
    split_by_id.update(dict.fromkeys(testing_scenarios, "testing"))
    return split_by_id


def create_non_causal_agents_benchmark(config: DictConfig) -> None:
    """Creates benchmark splits for the Non-Causal Agents benchmark.

    Computes the non-causal agent count for each scenario, re-splits scenarios into train/validation/testing by that
    count, then materializes an ``original`` and a ``perturbed`` (remove_noncausal) copy of each scenario under its
    assigned split. Perturbed copies are reused from ``perturbed_data_path`` when present and generated otherwise.

    Args:
        config: Hydra config.
            Expected keys: input_data_path, output_data_path, causal_labels_path, perturbed_data_path,
            cutoff_percentile, validation_percentage, num_workers, seed.
    """
    output_data_path = Path(config.output_data_path)
    causal_labels_path = Path(config.causal_labels_path)
    random_generator: Generator = default_rng(config.seed)

    filepaths = collect_scenario_filepaths(Path(config.input_data_path))
    original_mapping = {fp.stem: fp for fp in filepaths}

    perturbed_data_path = Path(config.perturbed_data_path)
    perturbed_mapping: dict[str, Path] = {}
    if perturbed_data_path.exists():
        perturbed_mapping = {fp.stem: fp for fp in collect_scenario_filepaths(perturbed_data_path)}
    _LOGGER.info("Found %d original scenarios, %d reusable perturbed scenarios", len(filepaths), len(perturbed_mapping))

    _LOGGER.info("Creating Non-Causal Agents benchmark")
    create_split_dirs(output_data_path / "original")
    create_split_dirs(output_data_path / "perturbed")

    with multiprocessing.Pool(config.num_workers) as pool:
        count_results = list(
            tqdm(
                pool.starmap(
                    partial(_count_noncausal, causal_labels_path=causal_labels_path),
                    [(fp,) for fp in filepaths],
                ),
                total=len(filepaths),
                desc="Counting non-causal agents",
            )
        )
    counts = [result for result in count_results if result is not None]

    split_by_id = _assign_splits(counts, config.cutoff_percentile, config.validation_percentage, random_generator)

    tasks = [
        (scenario_id, original_mapping[scenario_id], perturbed_mapping.get(scenario_id), split)
        for scenario_id, split in split_by_id.items()
        if scenario_id in original_mapping
    ]

    with multiprocessing.Pool(config.num_workers) as pool:
        list(
            tqdm(
                pool.starmap(
                    partial(
                        _materialize_scenario,
                        causal_labels_path=causal_labels_path,
                        output_data_path=output_data_path,
                    ),
                    [(task,) for task in tasks],
                ),
                total=len(tasks),
                desc="Materializing scenarios",
            )
        )

    verify_splits(output_data_path / "original")
    verify_splits(output_data_path / "perturbed")
    _LOGGER.info("Non-Causal Agents benchmark creation complete")
