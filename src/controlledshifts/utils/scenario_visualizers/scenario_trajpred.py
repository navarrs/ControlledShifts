import matplotlib.pyplot as plt
import numpy as np
from characterization.schemas import Scenario, ScenarioScores
from characterization.utils.io_utils import get_logger
from matplotlib import cm
from matplotlib.axes import Axes
from numpy.typing import NDArray
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
        ax: Axes,
        point1: np.ndarray,
        point2: np.ndarray,
        color: str | tuple[float, ...] | np.ndarray,
        line_width: float = 4,
        alpha: float = 1.0,
    ) -> None:
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color, alpha=alpha)

    @staticmethod
    def _draw_trajectory(
        ax: Axes, trajectory: np.ndarray, line_width: float, ego: bool = False, alpha: float = 1.0
    ) -> None:
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = ScenarioTrajpredVisualizer._interpolate_color_ego(t, total_t)
            else:
                color = ScenarioTrajpredVisualizer._interpolate_color(t, total_t)
            if trajectory[t, 0] and trajectory[t + 1, 0]:
                ScenarioTrajpredVisualizer._draw_line_with_mask(
                    ax, trajectory[t], trajectory[t + 1], color=color, line_width=line_width, alpha=alpha
                )

    def _draw_scene_context(
        self, ax: Axes, scenario: AgentCentricScenario, *, draw_future_gt: bool = True, gt_alpha: float = 0.3
    ) -> None:
        """Draws the shared scene context on a pane: map lanes, agent history, and optionally the GT future.

        Args:
            ax: Axes to plot on.
            scenario: encapsulates the agent-centric scenario to visualize.
            draw_future_gt: if True, overlays the ground-truth future trajectories for reference.
            gt_alpha: transparency for the ground-truth future trajectories, dimmed against the model prediction.
        """
        map_xy, map_type = self._decode_map(scenario.map_polylines)
        map_mask = scenario.map_polylines_mask

        for idx, lane in enumerate(map_xy):
            lane_type = map_type[idx]
            lane_type = np.argmax(lane_type)
            if lane_type in [1, 2, 3]:
                continue
            for i in range(len(lane) - 1):
                if map_mask[idx, i] and map_mask[idx, i + 1]:
                    self._draw_line_with_mask(ax, lane[i], lane[i + 1], color="grey", line_width=1.5)

        for traj in scenario.obj_trajs:
            self._draw_trajectory(ax, traj, line_width=2)

        if draw_future_gt:
            for traj in scenario.obj_trajs_future_state:
                self._draw_trajectory(ax, traj, line_width=2, alpha=gt_alpha)

    def _draw_predictions(self, ax: Axes, model_output: ModelOutput) -> None:
        """Draws a model's predicted future trajectories, colored by mode probability.

        Args:
            ax: Axes to plot on.
            model_output: encapsulates the model outputs; must carry a trajectory decoder output.
        """
        # predicted future trajectory is (n, future_len, 2): n possible futures, visualize all of them
        pred_future_traj = model_output.trajectory_decoder_output.decoded_trajectories.value.detach().cpu().numpy()
        pred_future_prob = model_output.trajectory_decoder_output.mode_probabilities.value.detach().cpu().numpy()
        for idx, traj in enumerate(pred_future_traj):
            color = cm.hot(pred_future_prob[idx])
            for i in range(len(traj) - 1):
                self._draw_line_with_mask(ax, traj[i], traj[i + 1], color=color, line_width=2)

    def visualize_scenario(  # noqa: PLR0913
        self,
        scenario: Scenario | AgentCentricScenario,
        scores: ScenarioScores | None = None,
        model_output: ModelOutput | None = None,
        output_dir: str = "temp",
        causal_gt_ids: NDArray[np.int_] | None = None,
        model_outputs: dict[str, ModelOutput] | None = None,
    ) -> None:
        """Visualizes a single scenario as one comparison pane per model and saves the output to a file.

        ScenarioTrajpredVisualizer renders N side-by-side panes, one per model in ``model_outputs``. Each pane draws
        the shared scene context (map, agent history, and the dimmed ground-truth future) plus that model's predicted
        trajectories, titled with the model's name. A single ``model_output`` is accepted as a one-model fallback.

        Args:
            scenario: encapsulates the scenario to visualize.
            scores: encapsulates the scenario and agent scores.
            model_output: a single model's outputs, used only when ``model_outputs`` is not provided.
            output_dir: the directory where to save the scenario visualization.
            causal_gt_ids: unused; trajpred does not render causal panes.
            model_outputs: per-model outputs keyed by model name; each becomes one pane.
        """
        del causal_gt_ids
        if not isinstance(scenario, AgentCentricScenario):
            error_message = "Scenario needs to be of AgentCentricScenario"
            raise TypeError(error_message)

        models = model_outputs or ({"model": model_output} if model_output is not None else None)
        if not models:
            error_message = "At least one model output is required for TrajPred scenario visualization."
            raise ValueError(error_message)

        scenario_id = scenario.scenario_id
        scene_score = BaseVisualizer.get_scenario_score(scores)
        suffix = "" if scene_score is None else f"_{scene_score}"
        output_filepath = f"{output_dir}/{scenario_id}{suffix}.png"
        logger.info("Visualizing scenario to %s", output_filepath)

        num_panes = len(models)
        gt_alpha = self.config.get("gt_future_alpha", 0.3)
        _, axs = plt.subplots(1, num_panes, figsize=(5 * num_panes, 5), sharex=True, sharey=True)
        axs = np.atleast_1d(axs)

        for ax, (name, output) in zip(axs, models.items(), strict=True):
            self._draw_scene_context(ax, scenario, draw_future_gt=True, gt_alpha=gt_alpha)
            if output is None or output.trajectory_decoder_output is None:
                logger.warning(
                    "No trajectory decoder output for model '%s' on scenario %s; drawing scene context only.",
                    name,
                    scenario_id,
                )
            else:
                self._draw_predictions(ax, output)
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.axis("off")

        plt.suptitle(f"Scenario: {scenario_id}")
        plt.subplots_adjust(wspace=0.05)
        plt.tight_layout()
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
        for ax in axs:
            ax.cla()
        plt.close()
