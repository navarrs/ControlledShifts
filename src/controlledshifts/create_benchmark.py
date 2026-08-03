r"""Script used for creating benchmark dataset splits.

Each benchmark computes its train/validation/testing split of scenario IDs and saves it as JSON under ``splits_path``.
Scenarios are never copied into per-split directories: training/eval select IDs from the split JSON and read
agent-centric records from the canonical per-variant cache (see ``controlledshifts.build_ac_cache``). The
non_background_agents benchmark additionally writes its perturbed variant stores under ``variants_path`` as a
preparation step.

Example usage:

    # Uniform benchmark (plain IID random split, no distribution shift)
    uv run -m controlledshifts.create_benchmark benchmark=uniform

    # Non-Background Agents benchmark (generates all perturbation variant stores up front)
    uv run -m controlledshifts.create_benchmark benchmark=non_background_agents

    # Non-Background Agents Hard benchmark
    uv run -m controlledshifts.create_benchmark benchmark=non_background_agents_hard

    # Ego-SafeShift benchmark (with a precomputed score CSV)
    uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift \\
        scenario_score_mapping_filepath=meta/ego-safeshift/scores_8/scenario_to_scores_mapping.csv

    # Ego-SafeShift benchmark (no CSV: scores are computed and cached; split saved as ego_safeshift_<tag>.json,
    # or pass split_name=... for a stable filename)
    uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift split_name=ego_safeshift_scores8

    # Environments benchmark
    uv run -m controlledshifts.create_benchmark benchmark=environments

See `configs/create_benchmark.yaml` and the per-benchmark configs under `configs/benchmark/` for all options.
"""

from pathlib import Path

import hydra
import pyrootutils
from omegaconf import DictConfig

from controlledshifts import benchmarks, utils
from controlledshifts.benchmarks import Benchmark, BenchmarkSplit
from controlledshifts.utils.plotting import configure_fonts


_LOGGER = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _compute_split(benchmark: Benchmark, cfg: DictConfig) -> BenchmarkSplit:
    """Dispatches to the benchmark that computes the split (and runs its data preparation)."""
    match benchmark:
        case Benchmark.UNIFORM:
            return benchmarks.create_uniform_benchmark(cfg)
        case Benchmark.NON_BACKGROUND_AGENTS:
            return benchmarks.create_non_background_agents_benchmark(cfg)
        case Benchmark.NON_BACKGROUND_AGENTS_HARD:
            return benchmarks.create_non_background_agents_hard_benchmark(cfg)
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
    otherwise the benchmark computes the split, runs any data preparation (such as non_background_agents' perturbations
    writing a perturbed variant store), checks the split for overlap (aborting on overlap) and saves it as JSON. The
    split JSON is all that is produced: scenarios are never copied into per-split directories. Training/eval select
    scenario IDs from the split JSON and read agent-centric records from the canonical per-variant cache (see
    ``controlledshifts.build_ac_cache``).

    Raises:
        ValueError: If the computed training/validation/testing splits overlap.
    """
    _LOGGER.info("Printing config tree")
    configure_fonts(log=_LOGGER)
    utils.print_config_tree(cfg, resolve=True, save_to_file=False)

    benchmark = Benchmark(cfg.benchmark_name)
    splits_path = Path(cfg.splits_path)

    # The split filename may be per-variant (e.g. ego_safeshift_<tag>) while the stored label stays the benchmark name.
    split_name = benchmarks.resolve_split_name(benchmark, cfg)
    split = benchmarks.load_split_if_exists(splits_path, split_name, overwrite=cfg.overwrite)
    if split is None:
        split = _compute_split(benchmark, cfg)
        benchmarks.check_overlap(split)
        benchmarks.save_benchmark_split(
            split, split_name, splits_path, overwrite=cfg.overwrite, benchmark_label=cfg.benchmark_name
        )


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
