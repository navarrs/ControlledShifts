"""Scenario analysis entry point.

Example:
    uv run -m controlledshifts.model_analysis analysis=tokenization num_batches=<x> num_scenarios=<y>

See `docs/ANALYSIS.md` and `configs/analysis.yaml` for more argument details.
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


def _run_experiment_analysis(config: DictConfig, output_path: Path) -> None:
    """Run selected analyses over tokenized scenarios for one experiment.

    Args:
        config (DictConfig): Configuration with analysis flags and runtime parameters.
        output_path (Path): Directory where analysis results are saved.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Load batches only if at least one analysis is enabled.
    if not any(
        [
            config.run_scenario_class_distribution,
            config.run_token_consistency_analysis,
            config.run_group_uniqueness_analysis,
            config.run_intergroup_analysis,
            config.run_dim_reduction_analysis,
            config.run_score_analysis,
        ]
    ):
        log.info("No analyses selected to run. Skipping batch loading and analysis.")
        return

    # Load tokenized scenario information.
    log.info("Loading batches from %s", config.paths.batch_cache_path)
    batches = utils.load_batches(
        config.paths.batch_cache_path, config.num_batches, config.num_scenarios, config.seed, config.split
    )

    # Run selected analyses.

    # Plot a histogram over tokenized scenarios to show token utilization.
    if config.run_scenario_class_distribution:
        log.info("Running tokenization distribution analysis...")
        utils.plot_scenario_class_distribution(config, batches, output_path)

    # Compute token consistency across tokenized groups.
    if config.run_token_consistency_analysis:
        log.info("Running tokenization consistency distribution analysis...")
        utils.compute_token_consistency_matrix(config, batches, output_path)

    # Compute group-uniqueness distribution over tokenized groups.
    if config.run_group_uniqueness_analysis:
        log.info("Running group uniqueness distribution analysis...")
        utils.compute_group_uniqueness(config, batches, output_path)

    # Compute intergroup-uniqueness distribution across tokenized groups.
    if config.run_intergroup_analysis:
        log.info("Running intergroup distribution analysis...")
        utils.compute_intergroup_uniqueness(config, batches, output_path)

    # Visualize scenario embeddings with dimensionality-reduction methods.
    if config.run_dim_reduction_analysis:
        log.info("Running dimensionality reduction analysis...")
        dim_reduction_result = utils.compute_dimensionality_reduction(config, batches, output_path)
        utils.plot_manifold_by_tokens(config, dim_reduction_result, batches, output_path)

    # Compute per-class score analysis over scenarios.
    if config.run_score_analysis:
        log.info("Running score analysis...")
        score_analysis = utils.compute_score_analysis(config, batches, output_path)
        if config.viz_scored_scenarios:
            utils.plot_tokenized_scenarios_by_score_percentile(config, batches, score_analysis, output_path)


@hydra.main(version_base="1.3", config_path="configs", config_name="analysis.yaml")
def main(config: DictConfig) -> float | None:
    """Hydra entry point for running scenario analyses."""
    analysis_paths = {}

    # Run selected analyses for each configured experiment and save outputs.
    for experiment_dir in config.experiment_dirs:
        random.seed(config.seed)
        log.info("Running analysis for experiment: %s", experiment_dir)
        start = time()

        # Print configuration to validate the experiment_dir used for this run.
        experiment_config = copy.deepcopy(config)
        experiment_config.paths.experiment_dir = experiment_dir
        utils.print_config_tree(
            experiment_config,
            resolve=True,
            save_to_file=False,
            print_order=["analysis", "paths"],
            print_field_not_in_print_order=False,
        )

        # Run the tokenization analysis for the given experiment.
        output_path = Path(experiment_config.paths.analysis_path) / f"{experiment_config.split}_model-analysis"
        _run_experiment_analysis(experiment_config, output_path)
        analysis_paths[experiment_dir] = output_path

        log.info("Total time: %s seconds", time() - start)
        log.info("Process completed!")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
