import matplotlib.pyplot as plt
from characterization.schemas import Scenario, ScenarioScores
from characterization.utils.io_utils import get_logger
from omegaconf import DictConfig

from controlledshifts.schemas import AgentCentricScenario, ModelOutput
from controlledshifts.utils.scenario_visualizers.base_visualizer import BaseVisualizer


logger = get_logger(__name__)


class ScenarioAnimatedVisualizer(BaseVisualizer):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

    def visualize_scenario(
        self,
        scenario: Scenario | AgentCentricScenario,
        scores: ScenarioScores | None = None,
        model_output: ModelOutput | None = None,  # noqa: ARG002
        output_dir: str = "temp",
    ) -> None:
        """Visualizes a single scenario and saves the output to a file.

        ScenarioAnimatedVisualizer visualizes the scenario as an per-timestep animation.

        Args:
            scenario (Scenario | AgentCentricScenario): encapsulates the scenario to visualize.
            scores (ScenarioScores | None): encapsulates the scenario and agent scores.
            model_output (ModelOutput | None): encapsulates model outputs.
            output_dir: (str): the directory where to save the scenario visualization.
        """
        if not isinstance(scenario, Scenario):
            error_message = "Scenario visualization only supported in global frame."
            raise TypeError(error_message)

        scenario_id = scenario.metadata.scenario_id
        scene_score = BaseVisualizer.get_scenario_score(scores)
        suffix = "" if scene_score is None else f"_{scene_score}"
        output_filepath = f"{output_dir}/{scenario_id}{suffix}.gif"
        logger.info("Visualizing scenario to %s", output_filepath)

        total_timesteps = scenario.metadata.track_length
        for timestep in range(2, total_timesteps):
            _, ax = plt.subplots(1, 1, figsize=(5, 5))

            # Plot static and dynamic map information in the scenario
            self.plot_map_data(ax, scenario)

            self.plot_sequences(ax, scenario, scores, show_relevant=True, end_timestep=timestep)

            # Prepare and save plot
            self.set_axes(ax, scenario)
            ax.set_title(f"Scenario: {scenario_id}")
            plt.subplots_adjust(wspace=0.05)

            plt.savefig(f"{output_dir}/temp_{timestep}.png", dpi=300, bbox_inches="tight")
            plt.close()
        BaseVisualizer.to_gif(output_dir, output_filepath)
