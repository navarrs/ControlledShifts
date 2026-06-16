r"""Benchmark creation for the Ego-SafeShift benchmark.

Ranks scenarios by safety score and copies them into training/validation/testing splits following split_ratios. The
hardest (highest-scoring) scenarios form the test set; the remainder is split into train/val.

Example usage:

    uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift \\
        input_data_path=/datasets/waymo/processed/mini_causal \\
        output_data_path=/datasets/waymo/processed/causal_ego_safeshift \\
        scenario_score_mapping_filepath=meta/ego-safeshift/scores_8/scenario_to_scores_mapping.csv

See configs/benchmark/ego_safeshift.yaml for all available options.
"""

from pathlib import Path

import pandas as pd
from numpy.random import Generator, default_rng
from omegaconf import DictConfig

from controlledshifts.benchmarks.common import (
    BenchmarkSplit,
    collect_scenario_filepaths,
    split_ids_by_score,
    split_mapping_to_lists,
)


def create_ego_safeshift_benchmark(config: DictConfig) -> BenchmarkSplit:
    """Creates the Ego-SafeShift benchmark split.

    Splits the dataset by safety score: the hardest (highest-scoring) scenarios form the test set and the remainder is
    split into training/validation, following split_ratios. Scenarios listed in the score CSV but absent from the
    input directory are recorded as invalid.

    Args:
        config: Hydra config with keys: input_data_path, scenario_score_mapping_filepath, score_type,
            split_ratios, seed.

    Returns:
        The train/val/test benchmark split.
    """
    input_data_path = Path(config.input_data_path)
    random_generator: Generator = default_rng(config.seed)
    available_ids = {fp.stem for fp in collect_scenario_filepaths(input_data_path)}

    scenario_scores_df = pd.read_csv(Path(config.scenario_score_mapping_filepath))
    scenario_ids = [Path(scenario_id).stem for scenario_id in scenario_scores_df["scenario_ids"].tolist()]
    split_by_id = split_ids_by_score(
        scenario_ids,
        scenario_scores_df[config.score_type].to_numpy(),
        tuple(config.split_ratios),
        random_generator,
        hardest_highest=True,
    )

    invalid = [scenario_id for scenario_id in split_by_id if scenario_id not in available_ids]
    placed = {scenario_id: split for scenario_id, split in split_by_id.items() if scenario_id in available_ids}
    return split_mapping_to_lists(placed, invalid=invalid)
