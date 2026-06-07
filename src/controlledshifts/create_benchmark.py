r"""Script used for creating benchmark dataset splits.

Each benchmark computes its train/validation/testing split and saves it as JSON under ``splits_path``. Pass
``copy_splits=true`` to also organize the data into split subdirectories after the splits are saved.

Example usage:

    # Uniform benchmark (plain IID random split, no distribution shift)
    uv run -m controlledshifts.create_benchmark benchmark=uniform

    # Causal Agents benchmark (generates all perturbation strategies up front)
    uv run -m controlledshifts.create_benchmark benchmark=causal_agents

    # Causal Agents Hard benchmark
    uv run -m controlledshifts.create_benchmark benchmark=causal_agents_hard

    # Ego-SafeShift benchmark
    uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift \\
        scenario_score_mapping_filepath=meta/ego-safeshift/scores_8/scenario_to_scores_mapping.csv

    # Environments benchmark
    uv run -m controlledshifts.create_benchmark benchmark=environments

    # Save the splits and copy the data into training/validation/testing subdirs
    uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift copy_splits=true \\
        scenario_score_mapping_filepath=meta/ego-safeshift/scores_8/scenario_to_scores_mapping.csv

See `configs/create_benchmark.yaml` and the per-benchmark configs under `configs/benchmark/` for all options.
"""

from pathlib import Path

import hydra
import pyrootutils
from omegaconf import DictConfig

from controlledshifts import benchmarks, utils
from controlledshifts.benchmarks import Benchmark, BenchmarkSplit


_LOGGER = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _compute_split(benchmark: Benchmark, cfg: DictConfig) -> BenchmarkSplit:
    """Dispatches to the benchmark that computes the split (and runs its data preparation)."""
    match benchmark:
        case Benchmark.UNIFORM:
            return benchmarks.create_uniform_benchmark(cfg)
        case Benchmark.CAUSAL_AGENTS:
            return benchmarks.create_causal_agents_benchmark(cfg)
        case Benchmark.CAUSAL_AGENTS_HARD:
            return benchmarks.create_causal_agents_hard_benchmark(cfg)
        case Benchmark.EGO_SAFESHIFT:
            return benchmarks.create_ego_safeshift_benchmark(cfg)
        case Benchmark.SAFESHIFT:
            return benchmarks.create_safeshift_benchmark(cfg)
        case Benchmark.ENVIRONMENTS:
            return benchmarks.create_environments_benchmark(cfg)


@hydra.main(version_base="1.3", config_path="configs", config_name="create_benchmark.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for creating benchmark dataset splits.

    Reuses the split saved at ``${splits_path}/${benchmark_name}.json`` when it exists (unless ``overwrite`` is set);
    otherwise the benchmark computes the split, runs any data preparation (such as causal_agents' perturbations), the
    split is checked for overlap (aborting on overlap) and saved as JSON. When ``copy_splits`` is enabled, the
    benchmark's copy targets are organized into training/validation/testing subdirectories.

    Raises:
        ValueError: If the computed training/validation/testing splits overlap.
    """
    _LOGGER.info("Printing config tree")
    utils.print_config_tree(cfg, resolve=True, save_to_file=False)

    benchmark = Benchmark(cfg.benchmark_name)
    splits_path = Path(cfg.splits_path)

    split = benchmarks.load_split_if_exists(splits_path, cfg.benchmark_name, overwrite=cfg.overwrite)
    if split is None:
        split = _compute_split(benchmark, cfg)
        benchmarks.check_overlap(split)
        benchmarks.save_benchmark_split(split, cfg.benchmark_name, splits_path, overwrite=cfg.overwrite)

    if cfg.copy_splits:
        for target in benchmarks.plan_copy_targets(benchmark, cfg):
            benchmarks.copy_split_dataset(
                split,
                target.source,
                target.output,
                num_workers=cfg.num_workers,
                unlink_source=cfg.unlink_source,
                overwrite=cfg.overwrite,
            )


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
