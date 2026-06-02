r"""Script used for creating benchmark dataset splits.

Example usage:

    # Causal Agents benchmark
    uv run -m controlledshifts.create_benchmark benchmark=causal_agents \\
        input_data_path=/data/driving/waymo/processed/mini_causal \\
        output_data_path=/data/driving/waymo/processed/causal_agents \\
        causal_labels_path=/data/driving/waymo/causal_agents/processed_labels \\
        strategy=remove_causal

    # Ego-SafeShift benchmark
    uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift \\
        scenario_score_mapping_filepath=meta/scenario_to_scores_mapping.csv

    # Environments benchmark
    uv run -m controlledshifts.create_benchmark benchmark=environments

See `configs/create_benchmark.yaml` and the per-benchmark configs under `configs/benchmark/` for all options.
"""

import hydra
import pyrootutils
from omegaconf import DictConfig

from controlledshifts import benchmarks, utils
from controlledshifts.benchmarks import Benchmark


_LOGGER = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="create_benchmark.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for creating benchmark dataset splits."""
    _LOGGER.info("Printing config tree")
    utils.print_config_tree(cfg, resolve=True, save_to_file=False)

    benchmark = Benchmark(cfg.benchmark_name)
    match benchmark:
        case Benchmark.CAUSAL_AGENTS:
            benchmarks.create_causal_agents_benchmark(cfg)
        case Benchmark.EGO_SAFESHIFT:
            benchmarks.create_ego_safeshift_benchmark(cfg)
        case Benchmark.SAFESHIFT:
            benchmarks.create_safeshift_benchmark(cfg)
        case Benchmark.ENVIRONMENTS:
            benchmarks.create_environments_benchmark(cfg)


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
