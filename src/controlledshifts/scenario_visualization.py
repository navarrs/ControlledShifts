"""Scenario Visualization Script

Example usage:

    uv run -m controlledshifts.scenario_visualization \
        experiment_name=[experiment_name] \
        analysis=[visualizer_type] \
        num_batches=[num_batches] \
        num_scenarios=[num_scenarios]

See `docs/ANALYSIS.md` and `configs/analysis.yaml` for more argument details.
"""

import pickle
from pathlib import Path
from time import time

import hydra
import pyrootutils
from omegaconf import DictConfig

from controlledshifts import utils
from controlledshifts.schemas import AgentCentricScenario


log = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="analysis.yaml")
def main(config: DictConfig) -> float | None:
    """Hydra's entrypoint for running scenario analysis training."""
    log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
    utils.print_config_tree(config, resolve=True, save_to_file=False)

    start = time()
    output_path = Path(config.paths.viz_path) / f"{config.split}_scenarios" / f"{config.visualization.tag}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Loads model outputs
    batches = utils.load_batches(
        config.paths.batch_cache_path, config.num_batches, config.num_scenarios, config.seed, config.split
    )

    # Creates a dictionary of scenarios following the format {scenario_id: path/to/scenario_id.pkl}
    scenario_ids = batches.keys()
    scenario_files = {str(f).split("/")[-1].split(".")[0]: f for f in Path(config.scenarios_path).glob("*/*")}
    scenario_files = {k: v for k, v in scenario_files.items() if k in scenario_ids}

    # Plot scenarios
    visualizer = hydra.utils.instantiate(config.visualization.visualizer)
    dataset = hydra.utils.instantiate(config.dataset)
    for scenario_id, scenario_output in batches.items():
        if scenario_id not in scenario_files:
            log.warning("Scenario ID %s not found in scenario files, skipping...", scenario_id)
            continue
        scenario_file = scenario_files[scenario_id]
        with scenario_file.open("rb") as f:
            scenario = pickle.load(f)

        scenario = dataset.repack_scenario(scenario)
        scenario_features = dataset.scenario_features_processor.compute(scenario)
        scenario_scores = dataset.scenario_scores_processor.compute(scenario, scenario_features)

        if visualizer.is_ego_centric:
            scenario = dataset.process_agent_centric_scenario(scenario)[0]
            scenario = AgentCentricScenario(**scenario)

        scenario_output_path = output_path

        visualizer.visualize_scenario(
            scenario,
            scores=scenario_scores,
            model_output=scenario_output,
            output_dir=scenario_output_path,
        )
    log.info("Total time: %s second", time() - start)
    log.info("Process completed!")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
