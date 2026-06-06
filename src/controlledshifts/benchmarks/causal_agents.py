r"""Benchmark creation for the Causal Agents benchmark.

Reuses the split of a reference benchmark (``reference_benchmark``, by default ``uniform``) rather than computing its
own, so the perturbed scenes land in the same train/validation/testing bucket as their unperturbed counterparts. As a
preparation step, it generates the perturbed dataset for every masking strategy (causal, non-causal, non-causal-equal,
static), written flat under ``output_data_path/<strategy>/`` with no split subdirectories. The shared split is returned
(and saved as JSON by the entry point); the optional copy step organizes each perturbed dataset into
``output_data_path/causal_agents/<strategy>/<split>/``. The unperturbed "original" data is not re-copied; it is served
directly from the reference benchmark's split directories (e.g. ``processed/uniform/<split>/``).

The reference split must exist before running this benchmark. Create it first with, e.g.:

    uv run -m controlledshifts.create_benchmark benchmark=uniform copy_splits=true

Example usage:

    uv run -m controlledshifts.create_benchmark benchmark=causal_agents \\
        input_data_path=/datasets/waymo/processed/mini_causal \\
        causal_labels_path=/datasets/waymo/causal_agents/processed_labels

See configs/benchmark/causal_agents.yaml for all available options.
"""

import json
import multiprocessing
import pickle  # nosec B403
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator, default_rng
from omegaconf import DictConfig
from tqdm import tqdm

from controlledshifts import utils
from controlledshifts.benchmarks.common import (
    CAUSAL_STRATEGIES,
    BenchmarkSplit,
    collect_scenario_filepaths,
    get_noncausal_mask,
    load_benchmark_split,
)
from controlledshifts.utils.constants import MIN_VALID_POINTS


_LOGGER = utils.get_pylogger(__name__)


def _remove_causal(scenario: dict[str, Any], causal_labels: dict[str, Any], output_filepath: Path) -> None:
    """Removes causal objects from a scenario by setting the last column of the trajectories to 0 for causal objects.

    Args:
        scenario: Scenario dictionary.
        causal_labels: Causal labels dictionary.
        output_filepath: Path to the output file.
    """
    causal_ids = np.array(causal_labels["causal_ids"], dtype=np.int64)
    object_ids = np.array(scenario["track_infos"]["object_id"])

    causal_mask = np.isin(object_ids, causal_ids)

    track_infos = scenario["track_infos"]
    track_infos["causal_ids"] = causal_labels["causal_ids"]

    trajectories = track_infos["trajs"].copy()
    trajectories[..., -1][causal_mask] = 0
    track_infos["trajs"] = trajectories
    scenario["track_infos"] = track_infos

    agent_idxs = np.arange(len(object_ids))
    causal_idxs = agent_idxs[causal_mask]

    tracks_to_predict = scenario["tracks_to_predict"]
    track_index = np.array(tracks_to_predict["track_index"])
    track_difficulty = np.array(tracks_to_predict["difficulty"])
    object_type = np.array(tracks_to_predict["object_type"])

    causal_track_index_mask = ~np.isin(track_index, causal_idxs)
    scenario["tracks_to_predict"] = {
        "track_index": track_index[causal_track_index_mask].tolist(),
        "track_difficulty": track_difficulty[causal_track_index_mask].tolist(),
        "object_type": object_type[causal_track_index_mask].tolist(),
    }

    with output_filepath.open("wb") as f:
        pickle.dump(scenario, f)


def remove_noncausal(scenario: dict[str, Any], causal_labels: dict[str, Any], output_filepath: Path) -> None:
    """Removes non-causal objects from a scenario by setting the last column of the trajectories to 0 for non-causal
    objects.

    Args:
        scenario: Scenario dictionary.
        causal_labels: Causal labels dictionary.
        output_filepath: Path to the output file.
    """
    object_ids = np.array(scenario["track_infos"]["object_id"])
    noncausal_mask = get_noncausal_mask(scenario, causal_labels)

    track_infos = scenario["track_infos"]
    track_infos["causal_ids"] = causal_labels["causal_ids"]

    trajectories = track_infos["trajs"].copy()
    trajectories[..., -1][noncausal_mask] = 0
    track_infos["trajs"] = trajectories
    scenario["track_infos"] = track_infos

    agent_idxs = np.arange(len(object_ids))
    noncausal_idxs = agent_idxs[noncausal_mask]

    tracks_to_predict = scenario["tracks_to_predict"]
    track_index = np.array(tracks_to_predict["track_index"])
    track_difficulty = np.array(tracks_to_predict["difficulty"])
    object_type = np.array(tracks_to_predict["object_type"])

    noncausal_track_index_mask = ~np.isin(track_index, noncausal_idxs)
    scenario["tracks_to_predict"] = {
        "track_index": track_index[noncausal_track_index_mask].tolist(),
        "track_difficulty": track_difficulty[noncausal_track_index_mask].tolist(),
        "object_type": object_type[noncausal_track_index_mask].tolist(),
    }

    with output_filepath.open("wb") as f:
        pickle.dump(scenario, f)


def _remove_noncausalequal(
    scenario: dict[str, Any], causal_labels: dict[str, Any], output_filepath: Path, random_generator: Generator
) -> None:
    """Removes a random subset of non-causal objects equal in count to the causal objects.

    Args:
        scenario: Scenario dictionary.
        causal_labels: Causal labels dictionary.
        output_filepath: Path to the output file.
        random_generator: Random number generator.
    """
    object_ids = np.array(scenario["track_infos"]["object_id"])
    noncausal_mask = get_noncausal_mask(scenario, causal_labels)

    num_to_remove = len(causal_labels["causal_ids"])
    agent_idxs = np.arange(len(object_ids))
    noncausal_idxs = agent_idxs[noncausal_mask]
    noncausal_idxs_to_remove = random_generator.choice(
        noncausal_idxs, size=min(num_to_remove, len(noncausal_idxs)), replace=False
    )

    track_infos = scenario["track_infos"]
    track_infos["causal_ids"] = causal_labels["causal_ids"]

    trajectories = track_infos["trajs"].copy()
    trajectories[..., -1][noncausal_idxs_to_remove] = 0
    track_infos["trajs"] = trajectories
    scenario["track_infos"] = track_infos

    tracks_to_predict = scenario["tracks_to_predict"]
    track_index = np.array(tracks_to_predict["track_index"])
    track_difficulty = np.array(tracks_to_predict["difficulty"])
    object_type = np.array(tracks_to_predict["object_type"])

    noncausal_track_index_mask = ~np.isin(track_index, noncausal_idxs_to_remove)
    scenario["tracks_to_predict"] = {
        "track_index": track_index[noncausal_track_index_mask].tolist(),
        "track_difficulty": track_difficulty[noncausal_track_index_mask].tolist(),
        "object_type": object_type[noncausal_track_index_mask].tolist(),
    }

    with output_filepath.open("wb") as f:
        pickle.dump(scenario, f)


def _remove_static(scenario: dict[str, Any], output_filepath: Path, threshold_distance: float = 0.1) -> None:
    """Removes static objects from a scenario by masking trajectories with displacement below threshold_distance.

    Args:
        scenario: Scenario dictionary.
        output_filepath: Path to the output file.
        threshold_distance: Minimum displacement to consider an object dynamic. Defaults to 0.1.
    """
    track_infos = scenario["track_infos"]
    track_infos["static_threshold_distance"] = threshold_distance

    trajectories = track_infos["trajs"].copy()
    static_mask = np.zeros(trajectories.shape[0], dtype=bool)

    for n, traj in enumerate(trajectories):
        valid_mask = traj[..., -1].astype(bool)
        if valid_mask.sum() < MIN_VALID_POINTS:
            continue
        pos = traj[..., :2][valid_mask]
        static_mask[n] = np.linalg.norm(pos[-1] - pos[0], axis=-1) < threshold_distance

    static_mask[scenario["sdc_track_index"]] = False

    trajectories[..., -1][static_mask] = 0
    track_infos["trajs"] = trajectories
    scenario["track_infos"] = track_infos

    tracks_to_predict = scenario["tracks_to_predict"]
    track_index = np.array(tracks_to_predict["track_index"])
    track_difficulty = np.array(tracks_to_predict["difficulty"])
    object_type = np.array(tracks_to_predict["object_type"])

    object_ids = np.array(scenario["track_infos"]["object_id"])
    agent_idxs = np.arange(len(object_ids))
    static_idxs = agent_idxs[static_mask]
    static_track_index_mask = ~np.isin(track_index, static_idxs)

    filtered_track_index = track_index[static_track_index_mask].tolist()
    if not filtered_track_index:
        return

    scenario["tracks_to_predict"] = {
        "track_index": filtered_track_index,
        "track_difficulty": track_difficulty[static_track_index_mask].tolist(),
        "object_type": object_type[static_track_index_mask].tolist(),
    }

    with output_filepath.open("wb") as f:
        pickle.dump(scenario, f)


def _perturb_scenario(  # noqa: PLR0913
    input_filepath: Path,
    perturbed_path: Path,
    causal_labels_path: Path,
    strategy: str,
    random_generator: Generator,
    *,
    overwrite: bool = False,
) -> None:
    """Applies a masking strategy to a single scenario and writes it flat to ``perturbed_path/<id>.pkl``.

    Args:
        input_filepath: Path to the input scenario pkl.
        perturbed_path: Flat output directory for the perturbed dataset (no split subdirs).
        causal_labels_path: Directory with per-scenario JSON causal labels (unused by ``remove_static``).
        strategy: Masking strategy name.
        random_generator: Random number generator (used by ``remove_noncausalequal``).
        overwrite: If False, skip scenarios already present in ``perturbed_path``. Defaults to False.
    """
    if not input_filepath.exists():
        return

    output_filepath = perturbed_path / f"{input_filepath.stem}.pkl"
    if output_filepath.exists() and not overwrite:
        return

    with input_filepath.open("rb") as f:
        scenario = pickle.load(f)  # nosec B301

    if strategy == "remove_static":
        _remove_static(scenario, output_filepath)
        return

    causal_labels_filepath = causal_labels_path / f"{input_filepath.stem}.json"
    if not causal_labels_filepath.exists():
        return
    with causal_labels_filepath.open("r") as f:
        causal_labels = json.load(f)

    match strategy:
        case "remove_causal":
            _remove_causal(scenario, causal_labels, output_filepath)
        case "remove_noncausal":
            remove_noncausal(scenario, causal_labels, output_filepath)
        case "remove_noncausalequal":
            _remove_noncausalequal(scenario, causal_labels, output_filepath, random_generator)
        case _:
            error_message = f"Strategy '{strategy}' is not supported. "
            error_message += "Choose from: remove_causal, remove_noncausal, remove_noncausalequal, remove_static."
            raise ValueError(error_message)


def _prepare_perturbations(  # noqa: PLR0913
    filepaths: list[Path],
    output_data_path: Path,
    causal_labels_path: Path,
    random_generator: Generator,
    num_workers: int,
    *,
    overwrite: bool = False,
) -> None:
    """Generates the flat perturbed dataset for every masking strategy under ``output_data_path/<strategy>/``.

    Args:
        filepaths: Input scenario filepaths to perturb.
        output_data_path: Root under which each ``<strategy>`` flat dataset directory is created.
        causal_labels_path: Directory with per-scenario JSON causal labels.
        random_generator: Random number generator (used by ``remove_noncausalequal``).
        num_workers: Number of parallel worker processes.
        overwrite: If False, scenarios already present in a strategy's directory are skipped. Defaults to False.
    """
    for strategy in CAUSAL_STRATEGIES:
        perturbed_path = output_data_path / strategy
        perturbed_path.mkdir(parents=True, exist_ok=True)
        _LOGGER.info("Generating '%s' perturbations for %d scenarios at %s", strategy, len(filepaths), perturbed_path)
        chunksize = max(1, len(filepaths) // (num_workers * 8))
        with multiprocessing.Pool(num_workers) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        partial(
                            _perturb_scenario,
                            perturbed_path=perturbed_path,
                            causal_labels_path=causal_labels_path,
                            strategy=strategy,
                            random_generator=random_generator,
                            overwrite=overwrite,
                        ),
                        filepaths,
                        chunksize=chunksize,
                    ),
                    total=len(filepaths),
                    desc=f"Perturbing ({strategy})",
                )
            )


def create_causal_agents_benchmark(config: DictConfig) -> BenchmarkSplit:
    """Reuses the reference benchmark's split and (optionally) prepares the perturbed datasets.

    Loads the split saved by ``config.reference_benchmark`` (by default ``uniform``) from
    ``config.splits_path/<reference_benchmark>.json`` instead of computing its own, so the perturbed scenes are
    organized into the same train/validation/testing buckets as their unperturbed counterparts. When
    config.prepare_perturbations is true, the perturbed dataset for every masking strategy is generated up front and
    written flat under output_data_path/<strategy>/, mirroring the input. The copy targets (see
    ``common.plan_copy_targets``) later organize each perturbed dataset into
    output_data_path/causal_agents/<strategy>/<split>/; the unperturbed data is served directly from the reference
    benchmark's split directories and is not re-copied.

    Args:
        config: Hydra config.
            Expected keys: input_data_path, output_data_path, causal_labels_path, prepare_perturbations,
            reference_benchmark, splits_path, num_workers, seed, overwrite.

    Returns:
        The reference BenchmarkSplit (shared across all strategies).

    Raises:
        FileNotFoundError: If the reference benchmark's split JSON does not exist yet.
    """
    input_data_path = Path(config.input_data_path)
    output_data_path = Path(config.output_data_path)

    reference_split_filepath = Path(config.splits_path) / f"{config.reference_benchmark}.json"
    if not reference_split_filepath.exists():
        error_message = (
            f"Reference split '{reference_split_filepath}' not found. Create the '{config.reference_benchmark}' "
            f"benchmark first, e.g.: uv run -m controlledshifts.create_benchmark benchmark={config.reference_benchmark}"
        )
        raise FileNotFoundError(error_message)
    split = load_benchmark_split(reference_split_filepath)

    filepaths = collect_scenario_filepaths(input_data_path)
    random_generator: Generator = default_rng(config.seed)

    if config.prepare_perturbations:
        _prepare_perturbations(
            filepaths,
            output_data_path,
            Path(config.causal_labels_path),
            random_generator,
            config.num_workers,
            overwrite=config.overwrite,
        )

    return split
