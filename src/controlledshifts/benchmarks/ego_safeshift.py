r"""Benchmark creation for the Ego-SafeShift benchmark.

Ranks scenarios by safety score and copies them into training/validation/testing splits following split_ratios. The
hardest (highest-scoring) scenarios form the test set; the remainder is split into train/val.

Example usage:

    uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift \\
        input_data_path=/datasets/waymo/processed/mini_causal \\
        output_data_path=/datasets/waymo/processed/causal_ego_safeshift \\
        scenario_score_mapping_filepath=meta/scenario_to_scores_mapping.csv

See configs/benchmark/ego_safeshift.yaml for all available options.
"""

import multiprocessing
from functools import partial
from pathlib import Path

import pandas as pd
from numpy.random import Generator, default_rng
from omegaconf import DictConfig
from tqdm import tqdm

from controlledshifts.benchmarks.common import (
    collect_scenario_filepaths,
    copy_scenario,
    create_split_dirs,
    get_scenario_mapping,
    split_ids_by_score,
)


def create_ego_safeshift_benchmark(config: DictConfig) -> None:
    """Creates benchmark scenarios for the Ego-SafeShift benchmark.

    Splits the dataset by safety score: the hardest (highest-scoring) scenarios form the test set and the remainder is
    split into training/validation, following split_ratios. Files are copied from the input directory into the
    appropriate split subdirectory.

    Args:
        config: Hydra config.
            Expected keys: input_data_path, output_data_path, scenario_score_mapping_filepath, score_type,
            split_ratios, num_workers, seed.
    """
    output_data_path = Path(config.output_data_path)
    random_generator: Generator = default_rng(config.seed)

    filepaths = collect_scenario_filepaths(Path(config.input_data_path))
    input_scenario_mapping = {fp.stem: fp for fp in filepaths}

    print("Processing Ego-SafeShift benchmark")
    create_split_dirs(output_data_path)

    scenario_scores_df = pd.read_csv(Path(config.scenario_score_mapping_filepath))
    split_by_id = split_ids_by_score(
        scenario_scores_df["scenario_ids"].tolist(),
        scenario_scores_df[config.score_type].to_numpy(),
        tuple(config.split_ratios),
        random_generator,
        hardest_highest=True,
    )

    output_scenario_mapping: dict[str, Path] = {}
    for split in ("training", "validation", "testing"):
        split_scenarios = [
            scenario_id for scenario_id, scenario_split in split_by_id.items() if scenario_split == split
        ]
        output_scenario_mapping.update(get_scenario_mapping(split_scenarios, output_data_path, split))

    tasks = [
        (scenario_id, input_scenario_mapping[scenario_id], output_scenario_mapping[scenario_id])
        for scenario_id in output_scenario_mapping
        if scenario_id in input_scenario_mapping
    ]

    with multiprocessing.Pool(config.num_workers) as pool:
        list(
            tqdm(
                pool.starmap(partial(copy_scenario, unlink_source=config.unlink_source), tasks),
                total=len(tasks),
                desc="Copying scenarios",
            )
        )
