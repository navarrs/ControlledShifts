"""Shared helpers used across benchmark creation modules."""

import itertools
import shutil
from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator

from controlledshifts import utils


_LOGGER = utils.get_pylogger(__name__)


class Benchmark(Enum):
    CAUSAL_AGENTS = "causal_agents"
    NON_CAUSAL_AGENTS = "non_causal_agents"
    EGO_SAFESHIFT = "ego_safeshift"
    SAFESHIFT = "safeshift"
    ENVIRONMENTS = "environments"


_DEFAULT_SPLITS: tuple[str, ...] = ("training", "validation", "testing")


def verify_splits(output_path: Path, splits: Iterable[str] = _DEFAULT_SPLITS) -> None:
    """Reads scenario files from train/val/test subdirectories of output_path and prints any filename overlap."""
    split_data: dict[str, set[str]] = {
        split: {p.name for p in (output_path / split).rglob("*.pkl") if p.is_file()} for split in splits
    }
    for set1, set2 in itertools.combinations(splits, 2):
        intersection = split_data[set1] & split_data[set2]
        if intersection:
            _LOGGER.warning("Overlap between %s and %s: %d scenarios", set1, set2, len(intersection))
        else:
            _LOGGER.info("No overlap between %s and %s", set1, set2)


def create_split_dirs(output_path: Path, splits: Iterable[str] = _DEFAULT_SPLITS) -> None:
    """Creates split subdirectories under output_path.

    Args:
        output_path: Root directory under which split subdirs are created.
        splits: Split names. Defaults to ("training", "validation", "testing").
    """
    for split in splits:
        (output_path / split).mkdir(parents=True, exist_ok=True)
        _LOGGER.info("Creating benchmark subdir: %s", output_path / split)


def get_noncausal_mask(scenario: dict[str, Any], causal_labels: dict[str, Any]) -> np.ndarray:
    """Returns a boolean mask over a scenario's agents that is True for non-causal agents.

    Non-causal agents are those whose object_id is neither in the causal labels nor the ego agent.

    Args:
        scenario: Scenario dictionary.
        causal_labels: Causal labels dictionary with a "causal_ids" key.

    Returns:
        Boolean array of shape (num_agents,), True where the agent is non-causal.
    """
    object_ids = np.array(scenario["track_infos"]["object_id"])
    ego_id = object_ids[scenario["sdc_track_index"]]
    causal_ids = np.array(causal_labels["causal_ids"] + [ego_id], dtype=np.int64)
    return ~np.isin(object_ids, causal_ids)


def _build_split_mapping(training: list[str], validation: list[str], testing: list[str]) -> dict[str, str]:
    """Builds a scenario_id -> split-name mapping and logs the resulting split sizes."""
    _LOGGER.info(
        "Split sizes -> training: %d, validation: %d, testing: %d", len(training), len(validation), len(testing)
    )
    split_mapping = dict.fromkeys(training, "training")
    split_mapping.update(dict.fromkeys(validation, "validation"))
    split_mapping.update(dict.fromkeys(testing, "testing"))
    return split_mapping


def split_ids_by_ratio(
    scenario_ids: Sequence[str], split_ratios: tuple[float, float, float], random_generator: Generator
) -> dict[str, str]:
    """Randomly assigns scenario IDs to training/validation/testing by (train, val, test) fractions of the total.

    The ids are sorted before shuffling so the assignment is reproducible regardless of input ordering.

    Args:
        scenario_ids: Scenario IDs to split.
        split_ratios: (train, val, test) fractions of the full dataset. Should sum to 1.0.
        random_generator: Random number generator used to shuffle the ids.

    Returns:
        Mapping from scenario_id to split name ("training", "validation", or "testing").
    """
    ids = sorted(scenario_ids)
    random_generator.shuffle(ids)
    total = len(ids)
    num_test = int(total * split_ratios[2])
    num_val = int(total * split_ratios[1])
    testing = ids[:num_test]
    validation = ids[num_test : num_test + num_val]
    training = ids[num_test + num_val :]
    return _build_split_mapping(training, validation, testing)


def split_ids_by_score(
    scenario_ids: Sequence[str],
    scores: np.ndarray,
    split_ratios: tuple[float, float, float],
    random_generator: Generator,
    *,
    hardest_highest: bool = True,
) -> dict[str, str]:
    """Assigns scenarios to splits by a difficulty score, sending the hardest scenarios to the test set.

    The hardest ``int(total * split_ratios[2])`` scenarios by ``scores`` form the test set; the remainder is shuffled
    and the first ``int(total * split_ratios[1])`` become validation, the rest training. Ids are sorted before ranking
    so the assignment is reproducible regardless of input ordering.

    Args:
        scenario_ids: Scenario IDs to split.
        scores: Per-scenario difficulty scores aligned with ``scenario_ids``.
        split_ratios: (train, val, test) fractions of the full dataset. Should sum to 1.0.
        random_generator: Random number generator used to shuffle the train/val remainder.
        hardest_highest: If True, the highest scores are hardest (go to testing); if False, the lowest are hardest.

    Returns:
        Mapping from scenario_id to split name ("training", "validation", or "testing").
    """
    order = np.argsort(scenario_ids, kind="stable")
    ids = np.asarray(scenario_ids)[order]
    ranked = np.asarray(scores)[order]

    rank_order = np.argsort(ranked, kind="stable")
    if hardest_highest:
        rank_order = rank_order[::-1]
    ids_by_hardness = ids[rank_order].tolist()

    total = len(ids_by_hardness)
    num_test = int(total * split_ratios[2])
    num_val = int(total * split_ratios[1])

    testing = ids_by_hardness[:num_test]
    remainder = ids_by_hardness[num_test:]
    random_generator.shuffle(remainder)
    validation = remainder[:num_val]
    training = remainder[num_val:]
    return _build_split_mapping(training, validation, testing)


def collect_scenario_filepaths(data_path: Path) -> list[Path]:
    """Returns all .pkl scenario filepaths under data_path, excluding info files.

    Args:
        data_path: Root directory to search recursively.

    Returns:
        List of .pkl filepaths whose stem does not contain 'infos'.
    """
    return [fp for fp in data_path.rglob("*.pkl") if "infos" not in fp.stem]


def get_scenario_mapping(
    scenario_ids: list[str],
    output_data_path: Path,
    split: str,
) -> dict[str, Path]:
    """Creates a mapping from scenario IDs to output file paths.

    Args:
        scenario_ids: List of scenario IDs.
        output_data_path: Path to the output data.
        split: Data split (e.g., 'training', 'validation', 'testing').

    Returns:
        Mapping from scenario IDs to output file paths (each ending in .pkl).
    """
    return {scenario_id: output_data_path / split / f"{scenario_id}.pkl" for scenario_id in scenario_ids}


def copy_scenario(
    scenario_id: str,
    input_filepath: Path,
    output_filepath: Path,
    *,
    unlink_source: bool = False,
) -> None:
    """Copies a scenario file from input_filepath to output_filepath.

    Args:
        scenario_id: Scenario ID, used only for warning messages.
        input_filepath: Source file path.
        output_filepath: Destination file path.
        unlink_source: If True, delete the source file after a successful copy. Defaults to False.
    """
    if not input_filepath.exists():
        _LOGGER.warning("Scenario %s not found at %s.", scenario_id, input_filepath)
        return
    shutil.copy2(input_filepath, output_filepath)
    if unlink_source:
        input_filepath.unlink()
