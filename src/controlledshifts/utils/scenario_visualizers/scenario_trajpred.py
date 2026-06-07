import matplotlib.pyplot as plt
import numpy as np
from characterization.schemas import Scenario, ScenarioScores
from characterization.utils.io_utils import get_logger
from matplotlib import cm
from omegaconf import DictConfig

from controlledshifts.schemas import AgentCentricScenario, ModelOutput
from controlledshifts.utils.scenario_visualizers.base_visualizer import BaseVisualizer


logger = get_logger(__name__)


class ScenarioTrajpredVisualizer(BaseVisualizer):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

    @staticmethod
    def _decode_map(map_polylines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        map_xy = map_polylines[..., :2]
        map_type = map_polylines[..., 0, 9:29]
        map_type = np.argmax(map_type, axis=-1)
        return map_xy, map_type

    @staticmethod
    def _interpolate_color_ego(t: int, total_t: int) -> tuple[float, float, float]:
        # Start is red, end is blue
        return (1 - t / total_t, 0, t / total_t)

    @staticmethod
    def _interpolate_color(t: int, total_t: int) -> tuple[float, float, float]:
        # Start is green, end is blue
        return (0, 1 - t / total_t, t / total_t)

    @staticmethod
    def _draw_line_with_mask(
        point1: np.ndarray,
        point2: np.ndarray,
        color: str | tuple[float, ...] | np.ndarray,
        line_width: float = 4,
    ) -> None:
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)

    @staticmethod
    def _draw_trajectory(trajectory: np.ndarray, line_width: float, ego: bool = False) -> None:
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = ScenarioTrajpredVisualizer._interpolate_color_ego(t, total_t)
            else:
                color = ScenarioTrajpredVisualizer._interpolate_color(t, total_t)
            if trajectory[t, 0] and trajectory[t + 1, 0]:
                ScenarioTrajpredVisualizer._draw_line_with_mask(
                    trajectory[t], trajectory[t + 1], color=color, line_width=line_width
                )

    def visualize_scenario(
        self,
        scenario: Scenario | AgentCentricScenario,
        scores: ScenarioScores | None = None,
        model_output: ModelOutput | None = None,
        output_dir: str = "temp",
    ) -> None:
        """Visualizes a single scenario and saves the output to a file.

        ScenarioTrajpredVisualizer visualizes the scenario on two windows:
            window 1: displays the full scene zoomed out
            window 2: displays the scene with relevant agents in different colors.

        Args:
            scenario (Scenario): encapsulates the scenario to visualize.
            scores (ScenarioScores | None): encapsulates the scenario and agent scores.
            model_output (ModelOutput | None): encapsulates model outputs.
            output_dir: (str): the directory where to save the scenario visualization.
        """
        if not isinstance(scenario, AgentCentricScenario):
            error_message = "Scenario needs to be of AgentCentricScenario"
            raise TypeError(error_message)

        if model_output is None or model_output.trajectory_decoder_output is None:
            error_message = "Trajectory decoder output is required for TrajPred scenario visualization."
            raise ValueError(error_message)

        scenario_id = scenario.scenario_id
        scene_score = BaseVisualizer.get_scenario_score(scores)
        suffix = "" if scene_score is None else f"_{scene_score}"
        output_filepath = f"{output_dir}/{scenario_id}{suffix}.png"
        logger.info("Visualizing scenario to %s", output_filepath)

        _, ax = plt.subplots(1, 1, figsize=(5, 5))

        # draw map
        map_xy, map_type = self._decode_map(scenario.map_polylines)
        map_mask = scenario.map_polylines_mask

        # Plot the map with mask check
        for idx, lane in enumerate(map_xy):
            lane_type = map_type[idx]
            # convert onehot to index
            lane_type = np.argmax(lane_type)
            if lane_type in [1, 2, 3]:
                continue
            for i in range(len(lane) - 1):
                if map_mask[idx, i] and map_mask[idx, i + 1]:
                    self._draw_line_with_mask(lane[i], lane[i + 1], color="grey", line_width=1.5)

        # draw past trajectory
        for traj in scenario.obj_trajs:
            self._draw_trajectory(traj, line_width=2)

        # draw future trajectory
        for traj in scenario.obj_trajs_future_state:
            self._draw_trajectory(traj, line_width=2)

        # predicted future trajectory is (n,future_len,2) with n possible future trajectories, visualize all of the
        pred_future_traj = model_output.trajectory_decoder_output.decoded_trajectories.value.detach().cpu().numpy()
        pred_future_prob = model_output.trajectory_decoder_output.mode_probabilities.value.detach().cpu().numpy()
        for idx, traj in enumerate(pred_future_traj):
            # calculate color based on probability
            color = cm.hot(pred_future_prob[idx])
            for i in range(len(traj) - 1):
                self._draw_line_with_mask(traj[i], traj[i + 1], color=color, line_width=2)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.grid(visible=True)
        plt.suptitle(f"Scenario: {scenario_id}")
        plt.subplots_adjust(wspace=0.05)
        plt.tight_layout()
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
        ax.cla()
        plt.close()
