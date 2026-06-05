r"""Benchmark creation for the Uniform benchmark.

Splits the input dataset uniformly at random into training/validation/testing following split_ratios. This is the
plain IID baseline (no distribution shift) used as a control to compare against the shift-inducing benchmarks.

Example usage:

    uv run -m controlledshifts.create_benchmark benchmark=uniform \\
        input_data_path=/datasets/waymo/processed/mini \\
        output_data_path=/datasets/waymo/processed/uniform

See configs/benchmark/uniform.yaml for all available options.
"""

from pathlib import Path

from numpy.random import Generator, default_rng
from omegaconf import DictConfig

from controlledshifts.benchmarks.common import (
    BenchmarkSplit,
    collect_scenario_filepaths,
    split_ids_by_ratio,
    split_mapping_to_lists,
)


def create_uniform_benchmark(config: DictConfig) -> BenchmarkSplit:
    """Creates the Uniform benchmark split.

    Randomly assigns the input scenarios to training/validation/testing following split_ratios, with no distribution
    shift between splits.

    Args:
        config: Hydra config.
            Expected keys: input_data_path, split_ratios, seed.

    Returns:
        The BenchmarkSplit.
    """
    input_data_path = Path(config.input_data_path)
    random_generator: Generator = default_rng(config.seed)
    filepaths = collect_scenario_filepaths(input_data_path)
    scenario_mapping = split_ids_by_ratio([fp.stem for fp in filepaths], tuple(config.split_ratios), random_generator)
    return split_mapping_to_lists(scenario_mapping)
