r"""Benchmark creation for the Background Agents Hard benchmark.

A harder variant of the Background Agents benchmark that focuses on a single perturbation (removing background
agents) and re-organizes scenarios into train/validation/testing splits by difficulty. Difficulty is the number of
background agents in a scenario: the scenarios with the most background agents form the test set (following
``split_ratios``), mirroring how ego_safeshift/safeshift move the hardest scenarios to test.

The benchmark produces a single split JSON (``splits/causal_agents_hard.json``). The unperturbed scenes are served from
the ``base`` variant and the perturbed scenes from the ``remove_noncausal`` variant; both are selected by the same split
so the original and perturbed versions of the same held-out scenarios can be compared. As a preparation step, the
``remove_noncausal`` perturbations are written flat to ``perturbed_data_path`` (a variant store), reusing existing files
when found.

Example usage:

    uv run -m controlledshifts.create_benchmark benchmark=background_agents_hard \\
        input_data_path=/data/driving/waymo/variants/base \\
        causal_labels_path=/data/driving/waymo/meta/causal_agents/processed_labels \\
        perturbed_data_path=/data/driving/waymo/variants/remove_noncausal

See configs/benchmark/background_agents_hard.yaml for all available options.
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

from controlledshifts.benchmarks.background_agents import remove_background
from controlledshifts.benchmarks.common import (
    BenchmarkSplit,
    collect_scenario_filepaths,
    get_background_mask,
    split_ids_by_score,
    split_mapping_to_lists,
)
from controlledshifts.utils.pylogger import get_pylogger


_LOGGER = get_pylogger(__name__)


def _count_background(input_filepath: Path, causal_labels_path: Path) -> tuple[str, int] | None:
    """Counts the background agents in a scenario.

    Args:
        input_filepath: Path to the input scenario pkl.
        causal_labels_path: Directory with per-scenario JSON causal labels.

    Returns:
        Tuple of (scenario_id, background count), or None if the scenario or its causal labels are missing.
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

    background_mask = get_background_mask(scenario, causal_labels)
    return scenario_id, int(background_mask.sum())


def _prepare_perturbed_scenario(
    input_filepath: Path, perturbed_path: Path, causal_labels_path: Path, *, overwrite: bool = False
) -> None:
    """Generates the ``remove_noncausal`` perturbation of a scenario flat to ``perturbed_path/<id>.pkl`` if missing.

    Args:
        input_filepath: Path to the input scenario pkl.
        perturbed_path: Flat output directory for the perturbed dataset (no split subdirs).
        causal_labels_path: Directory with per-scenario JSON causal labels.
        overwrite: If False, skip scenarios already present in ``perturbed_path``. Defaults to False.
    """
    if not input_filepath.exists():
        return

    output_filepath = perturbed_path / f"{input_filepath.stem}.pkl"
    if output_filepath.exists() and not overwrite:
        return

    causal_labels_filepath = causal_labels_path / f"{input_filepath.stem}.json"
    if not causal_labels_filepath.exists():
        return

    with input_filepath.open("rb") as f:
        scenario = pickle.load(f)  # nosec B301
    with causal_labels_filepath.open("r") as f:
        causal_labels = json.load(f)

    remove_background(scenario, causal_labels, output_filepath)


def create_background_agents_hard_benchmark(config: DictConfig) -> BenchmarkSplit:
    """Creates the Background Agents Hard benchmark split and prepares the perturbed dataset.

    Computes the background agent count for each scenario and splits scenarios into train/validation/testing by that
    count (the scenarios with the most background agents form the test set, following split_ratios). The
    ``remove_noncausal`` perturbed dataset is generated flat under ``perturbed_data_path`` (a variant store, reusing
    existing files), so both the ``base`` and ``remove_noncausal`` variants share this split. Only the split JSON is
    produced; training/eval select IDs from it and read agent-centric records from the per-variant cache.

    Args:
        config: Hydra config.
            Expected keys: input_data_path, causal_labels_path, perturbed_data_path, split_ratios, num_workers, seed,
            overwrite.

    Returns:
        The shared BenchmarkSplit (used for both the original and perturbed datasets).
    """
    input_data_path = Path(config.input_data_path)
    causal_labels_path = Path(config.causal_labels_path)
    perturbed_data_path = Path(config.perturbed_data_path)
    random_generator: Generator = default_rng(config.seed)

    filepaths = collect_scenario_filepaths(input_data_path)
    chunksize = max(1, len(filepaths) // (config.num_workers * 8))

    with multiprocessing.Pool(config.num_workers) as pool:
        count_results = list(
            tqdm(
                pool.imap_unordered(
                    partial(_count_background, causal_labels_path=causal_labels_path),
                    filepaths,
                    chunksize=chunksize,
                ),
                total=len(filepaths),
                desc="Counting background agents",
            )
        )
    counts = [result for result in count_results if result is not None]
    counted_ids = {scenario_id for scenario_id, _ in counts}
    invalid = [fp.stem for fp in filepaths if fp.stem not in counted_ids]

    split_by_id = split_ids_by_score(
        [scenario_id for scenario_id, _ in counts],
        np.array([count for _, count in counts]),
        tuple(config.split_ratios),
        random_generator,
        hardest_highest=True,
    )
    split = split_mapping_to_lists(split_by_id, invalid=invalid)

    perturbed_data_path.mkdir(parents=True, exist_ok=True)
    _LOGGER.info("Preparing remove_noncausal perturbations for %d scenarios at %s", len(filepaths), perturbed_data_path)
    with multiprocessing.Pool(config.num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    partial(
                        _prepare_perturbed_scenario,
                        perturbed_path=perturbed_data_path,
                        causal_labels_path=causal_labels_path,
                        overwrite=config.overwrite,
                    ),
                    filepaths,
                    chunksize=chunksize,
                ),
                total=len(filepaths),
                desc="Preparing perturbations",
            )
        )

    return split
