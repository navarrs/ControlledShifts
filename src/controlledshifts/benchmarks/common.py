"""Shared helpers used across benchmark creation modules."""

import itertools
import json
from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.random import Generator

from controlledshifts import utils


_LOGGER = utils.get_pylogger(__name__)

# Masking strategies generated up front by the causal_agents benchmark; each becomes a flat perturbed dataset.
CAUSAL_STRATEGIES: tuple[str, ...] = ("remove_causal", "remove_noncausal", "remove_noncausalequal", "remove_static")


class Benchmark(Enum):
    CAUSAL_AGENTS = "causal_agents"
    CAUSAL_AGENTS_HARD = "causal_agents_hard"
    EGO_SAFESHIFT = "ego_safeshift"
    SAFESHIFT = "safeshift"
    ENVIRONMENTS = "environments"
    UNIFORM = "uniform"


class BenchmarkSplit(NamedTuple):
    """Scenario IDs assigned to each split of a benchmark.

    The train/validation/testing lists are mutually exclusive. ``invalid`` holds scenarios that were considered but
    could not be placed (e.g. missing from the input directory, missing causal labels, or left empty after masking).
    ``benchmark_name`` records which benchmark produced the split; it is empty for freshly computed splits and
    populated by ``load_benchmark_split`` when reading a saved JSON.
    """

    training: list[str]
    validation: list[str]
    testing: list[str]
    invalid: list[str] = []  # noqa: RUF012
    benchmark_name: str = ""


_DEFAULT_SPLITS: tuple[str, ...] = ("training", "validation", "testing")


def get_noncausal_mask(scenario: dict[str, Any], causal_labels: dict[str, Any]) -> np.ndarray:
    """Returns a boolean mask over a scenario's agents that is True for non-causal agents.

    Non-causal agents are those whose object_id is neither in the causal labels nor the ego agent.

    Args:
        scenario: Decoded raw scenario dictionary.
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


def split_mapping_to_lists(mapping: dict[str, str], invalid: Iterable[str] | None = None) -> BenchmarkSplit:
    """Converts a scenario_id -> split-name mapping into a BenchmarkSplit.

    Args:
        mapping: Mapping from scenario_id to split name ("training", "validation", or "testing").
        invalid: Scenario IDs that were considered but could not be placed. Defaults to None (empty).

    Returns:
        BenchmarkSplit with the scenario IDs grouped by split, sorted for reproducibility.
    """
    grouped: dict[str, list[str]] = {split: [] for split in _DEFAULT_SPLITS}
    for scenario_id, split in mapping.items():
        grouped[split].append(scenario_id)
    return BenchmarkSplit(
        training=sorted(grouped["training"]),
        validation=sorted(grouped["validation"]),
        testing=sorted(grouped["testing"]),
        invalid=sorted(invalid) if invalid is not None else [],
    )


def load_benchmark_split(filepath: Path) -> BenchmarkSplit:
    """Reads a BenchmarkSplit from a JSON file written by ``save_benchmark_split``.

    Args:
        filepath: Path to the split JSON file.

    Returns:
        The deserialized BenchmarkSplit.
    """
    with filepath.open("r") as f:
        payload = json.load(f)
    return BenchmarkSplit(
        training=payload["training"],
        validation=payload["validation"],
        testing=payload["testing"],
        invalid=payload.get("invalid", []),
        benchmark_name=payload.get("benchmark_name", ""),
    )


def load_split_if_exists(splits_path: Path, benchmark_name: str, *, overwrite: bool) -> BenchmarkSplit | None:
    """Returns the existing split for a benchmark, or None if it should be (re)generated.

    Args:
        splits_path: Directory holding the split JSON files.
        benchmark_name: Benchmark name (the JSON filename stem).
        overwrite: If True, always return None so the caller regenerates the split.

    Returns:
        The loaded BenchmarkSplit when the JSON exists and ``overwrite`` is False; otherwise None.
    """
    splits_file = splits_path / f"{benchmark_name}.json"
    if splits_file.exists() and not overwrite:
        _LOGGER.info("Loading existing split from %s (set overwrite=true to regenerate)", splits_file)
        return load_benchmark_split(splits_file)
    return None


def save_benchmark_split(
    split: BenchmarkSplit, benchmark_name: str, splits_path: Path, *, overwrite: bool = False
) -> Path:
    """Writes a BenchmarkSplit to ``${splits_path}/${benchmark_name}.json`` and returns the path.

    Args:
        split: Split to serialize.
        benchmark_name: Benchmark name, used as the JSON filename and stored inside the file.
        splits_path: Directory under which the JSON file is written (created if missing).
        overwrite: If False, skip writing when the JSON already exists. Defaults to False.

    Returns:
        Path to the written (or already-existing) JSON file.
    """
    splits_path.mkdir(parents=True, exist_ok=True)
    output_filepath = splits_path / f"{benchmark_name}.json"
    if output_filepath.exists() and not overwrite:
        _LOGGER.info("Split %s already exists; not overwriting (set overwrite=true to regenerate)", output_filepath)
        return output_filepath
    payload = {
        "benchmark_name": benchmark_name,
        "training": split.training,
        "validation": split.validation,
        "testing": split.testing,
        "invalid": split.invalid,
    }
    with output_filepath.open("w") as f:
        json.dump(payload, f, indent=2)
    _LOGGER.info(
        "Saved %s split to %s (training: %d, validation: %d, testing: %d, invalid: %d)",
        benchmark_name,
        output_filepath,
        len(split.training),
        len(split.validation),
        len(split.testing),
        len(split.invalid),
    )
    return output_filepath


def check_overlap(split: BenchmarkSplit) -> None:
    """Raises a ValueError if the training/validation/testing ID sets of a split intersect.

    Args:
        split: Split to check for overlapping scenario IDs.

    Raises:
        ValueError: If any pair of splits shares one or more scenario IDs.
    """
    sets = {
        "training": set(split.training),
        "validation": set(split.validation),
        "testing": set(split.testing),
    }
    for name1, name2 in itertools.combinations(sets, 2):
        intersection = sets[name1] & sets[name2]
        if intersection:
            error_message = f"Overlap between {name1} and {name2}: {len(intersection)} scenarios"
            raise ValueError(error_message)
    _LOGGER.info("No overlap between splits (training: %d, validation: %d, testing: %d)", *map(len, sets.values()))
