import matplotlib.pyplot as plt
from characterization.schemas import Scenario, ScenarioScores
from characterization.utils.io_utils import get_logger
from omegaconf import DictConfig

from controlledshifts.schemas import AgentCentricScenario, ModelOutput
from controlledshifts.utils.constants import CausalOutputType
from controlledshifts.utils.scenario_visualizers.base_visualizer import BaseVisualizer
from controlledshifts.utils.scenario_visualizers.scenario_causal import ScenarioCausalVisualizer


logger = get_logger(__name__)


class ScenarioCausalAnimatedVisualizer(ScenarioCausalVisualizer, BaseVisualizer):
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

        ScenarioCausalAnimatedVisualizer visualizes the scenario on three windows:
            window 1: displays the full scene zoomed out
            window 2: displays the scene with GT causal agents marked in a different color.
            window 3: displays the scene with predicted causal agents marked in a different color and with a
                probability-based alpha value.

        Args:
            scenario (Scenario | AgentCentricScenario): encapsulates the scenario to visualize.
            scores (ScenarioScores | None): encapsulates the scenario and agent scores.
            model_output (ModelOutput | None): encapsulates model outputs.
            output_dir: (str): the directory where to save the scenario visualization.
        """
        if not isinstance(scenario, Scenario):
            error_message = "Scenario visualization only supported in global frame."
            raise TypeError(error_message)

        if model_output is None:
            error_message = "Model output is required for causal scenario visualization."
            raise ValueError(error_message)

        scenario_id = scenario.metadata.scenario_id
        scene_score = BaseVisualizer.get_scenario_score(scores)
        suffix = "" if scene_score is None else f"_{scene_score}"
        output_filepath = f"{output_dir}/{scenario_id}_causal{suffix}.gif"
        logger.info("Visualizing scenario to %s", output_filepath)

        num_windows = 3
        _, axs = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))

        total_timesteps = scenario.metadata.track_length
        for timestep in range(2, total_timesteps):
            _, axs = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))

            # Plot static and dynamic map information in the scenario
            self.plot_map_data(axs, scenario, num_windows)

            axs[0].set_title("Full Scene")
            self.plot_sequences(axs[0], scenario, scores, show_relevant=True, end_timestep=timestep)

            # Plot causal scene
            axs[1].set_title("GT Causal")
            self.plot_causal(
                axs[1],
                scenario,
                model_output=model_output,
                show_causal=CausalOutputType.GROUND_TRUTH,
                end_timestep=timestep,
            )

            # Plot causal scene
            axs[2].set_title("Pred Causal")
            self.plot_causal(
                axs[2],
                scenario,
                model_output=model_output,
                show_causal=CausalOutputType.PREDICTION,
                end_timestep=timestep,
            )

            # Prepare and save plot
            self.set_axes(axs, scenario, num_windows)
            plt.suptitle(f"Scenario: {scenario_id}")
            plt.subplots_adjust(wspace=0.05)
            plt.savefig(f"{output_dir}/temp_{timestep}.png", dpi=300, bbox_inches="tight")
            for ax in axs.reshape(-1):
                ax.cla()
            plt.close()
        BaseVisualizer.to_gif(output_dir, output_filepath)
