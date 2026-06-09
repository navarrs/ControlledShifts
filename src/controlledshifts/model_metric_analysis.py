"""Models Metrics Analysis Script.

Example usage:

    uv run -m controlledshifts.model_metric_analysis group_name=[name]

See `docs/ANALYSIS.md` and 'configs/model_metric_analysis.yaml' for more argument details.
"""

import copy
import random
from pathlib import Path
from time import time

import hydra
import pyrootutils
from omegaconf import DictConfig

from controlledshifts import utils


log = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="analysis.yaml")
def main(config: DictConfig) -> float | None:
    """Hydra's entrypoint for running scenario analysis training."""
    random.seed(config.seed)

    start = time()
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if config.run_causal_agents_sample_selection_sweep_lineplot_analysis:
        causal_config = copy.deepcopy(config)
        causal_config.sample_selection_benchmark = config.causal_agents_sample_selection_benchmark
        causal_config.sample_selection_files = config.causal_agents_sample_selection_files
        causal_config.sample_selection_splits_to_compare = config.causal_agents_sample_selection_splits_to_compare
        utils.plot_sample_selection_sweep_lineplot(causal_config, log, output_path)

    if config.run_causal_agents_sample_selection_sweep_heatmap_analysis:
        causal_config = copy.deepcopy(config)
        causal_config.sample_selection_benchmark = config.causal_agents_sample_selection_benchmark
        causal_config.sample_selection_files = config.causal_agents_sample_selection_files
        causal_config.sample_selection_splits_to_compare = config.causal_agents_sample_selection_splits_to_compare
        summary = utils.plot_sample_selection_sweep_heatmap(causal_config, log, output_path)
        log.info("CausalAgents heatmap summary: %d metrics", len(summary))

    if config.run_ego_safeshift_sample_selection_sweep_lineplot_analysis:
        ego_config = copy.deepcopy(config)
        ego_config.sample_selection_benchmark = config.ego_safeshift_sample_selection_benchmark
        ego_config.sample_selection_files = config.ego_safeshift_sample_selection_files
        ego_config.sample_selection_splits_to_compare = config.ego_safeshift_sample_selection_splits_to_compare
        utils.plot_sample_selection_sweep_lineplot(ego_config, log, output_path)

    if config.run_ego_safeshift_sample_selection_sweep_heatmap_analysis:
        ego_config = copy.deepcopy(config)
        ego_config.sample_selection_benchmark = config.ego_safeshift_sample_selection_benchmark
        ego_config.sample_selection_files = config.ego_safeshift_sample_selection_files
        ego_config.sample_selection_splits_to_compare = config.ego_safeshift_sample_selection_splits_to_compare
        summary = utils.plot_sample_selection_sweep_heatmap(ego_config, log, output_path)
        log.info("EgoSafeShift heatmap summary: %d metrics", len(summary))

    if config.run_environments_sample_selection_sweep_lineplot_analysis:
        env_config = copy.deepcopy(config)
        env_config.sample_selection_benchmark = config.environments_sample_selection_benchmark
        env_config.sample_selection_files = config.environments_sample_selection_files
        env_config.sample_selection_splits_to_compare = config.environments_sample_selection_splits_to_compare
        utils.plot_sample_selection_sweep_lineplot(env_config, log, output_path)

    if config.run_environments_sample_selection_sweep_heatmap_analysis:
        env_config = copy.deepcopy(config)
        env_config.sample_selection_benchmark = config.environments_sample_selection_benchmark
        env_config.sample_selection_files = config.environments_sample_selection_files
        env_config.sample_selection_splits_to_compare = config.environments_sample_selection_splits_to_compare
        summary = utils.plot_sample_selection_sweep_heatmap(env_config, log, output_path)
        log.info("Environments heatmap summary: %d metrics", len(summary))

    log.info("Total time: %s second", time() - start)
    log.info("Process completed!")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
