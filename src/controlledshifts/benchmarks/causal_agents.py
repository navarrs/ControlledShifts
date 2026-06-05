r"""Benchmark creation for the Causal Agents benchmark.

Randomly re-splits the input scenarios into training/validation/testing (by ``split_ratios``) and materializes each
scenario twice under its assigned split: an unperturbed copy under ``output_data_path/original/`` and a copy with a
specific object category (causal, non-causal, static) masked out under ``output_data_path/<strategy>/``. Keeping a
1-1 ``original``/perturbed correspondence per split lets the original and perturbed versions of the same held-out
scenarios be compared.

Example usage:

    uv run -m controlledshifts.create_benchmark benchmark=causal_agents \\
        input_data_path=/datasets/waymo/processed/mini_causal \\
        output_data_path=/datasets/waymo/processed/causal_agents \\
        causal_labels_path=/datasets/waymo/causal_agents/processed_labels \\
        strategy=remove_causal

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

from controlledshifts.benchmarks.common import (
    collect_scenario_filepaths,
    copy_scenario,
    create_split_dirs,
    get_noncausal_mask,
    split_ids_by_ratio,
)
from controlledshifts.utils.constants import MIN_VALID_POINTS


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


def _create_scenario(  # noqa: PLR0913
    input_filepath: Path,
    original_path: Path,
    perturbed_path: Path,
    causal_labels_path: Path,
    scenario_mapping: dict[str, str],
    strategy: str,
    random_generator: Generator,
) -> None:
    """Materializes the unperturbed and perturbed copies of a scenario into its assigned split.

    Args:
        input_filepath: Path to the input file.
        original_path: Root directory for the re-split unperturbed copies.
        perturbed_path: Root directory for the re-split perturbed copies.
        causal_labels_path: Path to the causal labels.
        scenario_mapping: Maps scenario_id to its split name (e.g. 'training').
        strategy: Benchmark strategy name.
        random_generator: Random number generator.
    """
    if not input_filepath.exists():
        return

    with input_filepath.open("rb") as f:
        scenario = pickle.load(f)  # nosec B301

    scenario_id = scenario["scenario_id"]

    causal_labels_filepath = causal_labels_path / f"{scenario_id}.json"
    if not causal_labels_filepath.exists():
        return

    with causal_labels_filepath.open("r") as f:
        causal_labels = json.load(f)

    split = scenario_mapping[scenario_id]

    # Materialize the unperturbed copy under the re-split original tree (taken from the file, so it is unaffected by the
    # in-memory mutation the masking strategies apply below).
    copy_scenario(scenario_id, input_filepath, original_path / split / f"{scenario_id}.pkl")

    output_filepath = perturbed_path / split / f"{scenario_id}.pkl"

    match strategy:
        case "remove_causal":
            _remove_causal(scenario, causal_labels, output_filepath)
        case "remove_noncausal":
            remove_noncausal(scenario, causal_labels, output_filepath)
        case "remove_noncausalequal":
            _remove_noncausalequal(scenario, causal_labels, output_filepath, random_generator)
        case "remove_static":
            _remove_static(scenario, output_filepath)
        case _:
            error_message = f"Strategy '{strategy}' is not supported. "
            error_message += "Choose from: remove_causal, remove_noncausal, remove_noncausalequal, remove_static."
            raise ValueError(error_message)


def create_causal_agents_benchmark(config: DictConfig) -> None:
    """Creates benchmark scenarios for Waymo dataset following the CausalAgents strategy.

    Reads scenario pkl files from config.input_data_path (which may be flat / unorganized), randomly re-splits the
    scenarios into training/validation/testing by config.split_ratios, and materializes each scenario twice under its
    assigned split: an unperturbed copy under output_data_path/original/ and a perturbed copy under
    output_data_path/<strategy>/. The split is deterministic for a fixed seed, so repeated runs with different
    strategies stay aligned.

    Args:
        config: Hydra config.
            Expected keys: input_data_path, output_data_path, causal_labels_path, strategy, split_ratios, num_workers,
            seed.
    """
    filepaths = collect_scenario_filepaths(Path(config.input_data_path))

    random_generator: Generator = default_rng(config.seed)
    scenario_mapping = split_ids_by_ratio([fp.stem for fp in filepaths], tuple(config.split_ratios), random_generator)

    output_data_path = Path(config.output_data_path)
    original_path = output_data_path / "original"
    perturbed_path = output_data_path / config.strategy
    print(f"Processing Causal Agents benchmark: {config.strategy}")
    create_split_dirs(original_path)
    create_split_dirs(perturbed_path)

    with multiprocessing.Pool(config.num_workers) as pool:
        pool.starmap(
            partial(
                _create_scenario,
                original_path=original_path,
                perturbed_path=perturbed_path,
                causal_labels_path=Path(config.causal_labels_path),
                scenario_mapping=scenario_mapping,
                strategy=config.strategy,
                random_generator=random_generator,
            ),
            [(file,) for file in tqdm(filepaths, total=len(filepaths))],
        )
