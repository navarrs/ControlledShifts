import matplotlib.pyplot as plt
import numpy as np
from characterization.schemas import Scenario, ScenarioScores
from characterization.utils.io_utils import get_logger
from omegaconf import DictConfig

from controlledshifts.schemas import AgentCentricScenario, ModelOutput
from controlledshifts.utils.scenario_visualizers.base_visualizer import PANE_TITLES, BaseVisualizer


logger = get_logger(__name__)


class ScenarioVisualizer(BaseVisualizer):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

    def visualize_scenario(
        self,
        scenario: Scenario | AgentCentricScenario,
        scores: ScenarioScores | None = None,
        model_output: ModelOutput | None = None,
        output_dir: str = "temp",
    ) -> None:
        """Visualizes a single scenario and saves the output to a file.

        ScenarioVisualizer renders one window per pane in ``panes_to_plot``, dispatching each pane through
        ``plot_pane``.

        Args:
            scenario (Scenario): encapsulates the scenario to visualize.
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
        output_filepath = f"{output_dir}/{scenario_id}{suffix}.png"
        logger.info("Visualizing scenario to %s", output_filepath)

        num_windows = len(self.panes_to_plot)
        _, axs = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))

        # Plot static and dynamic map information in the scenario
        self.plot_map_data(axs, scenario, num_windows)

        # Plot each requested pane on its own window
        axs_list = np.atleast_1d(axs)
        for ax, pane in zip(axs_list, self.panes_to_plot, strict=True):
            self.plot_pane(ax, pane, scenario, scores, model_output)
            ax.set_title(PANE_TITLES[pane])

        # Prepare and save plot
        self.set_axes(axs, scenario, num_windows)
        plt.suptitle(f"Scenario: {scenario_id}")
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
        plt.close()
