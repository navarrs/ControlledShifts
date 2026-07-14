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
            # Skip lane centerlines (1/2/3 = freeway/surface/bike); draw only road lines, edges, and crosswalks.
            if map_type[idx] in [1, 2, 3]:
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
        model_grid: dict[str, dict[str, ModelOutput]] | None = None,
        row_scenarios: dict[str, AgentCentricScenario] | None = None,
    ) -> None:
        """Visualizes a single scenario as a grid of comparison panes and saves the output to a file.

        Each pane draws a scene context (map, agent history, and the dimmed ground-truth future) plus one model's
        predicted trajectories. ``model_grid`` lays the panes out as rows x columns (e.g. training benchmarks x models),
        labelling rows on the left and columns on top. ``model_outputs`` renders a single unlabelled row, one pane per
        model; a lone ``model_output`` is accepted as a one-pane fallback.

        Args:
            scenario: encapsulates the scenario to visualize; the scene every row draws unless ``row_scenarios``
                overrides it.
            scores: encapsulates the scenario and agent scores.
            model_output: a single model's outputs, used only when neither grid nor ``model_outputs`` is provided.
            output_dir: the directory where to save the scenario visualization.
            causal_gt_ids: unused; trajpred does not render causal panes.
            model_outputs: per-model outputs keyed by model name; each becomes one pane of a single row.
            model_grid: per-model outputs keyed by row label then column label. Takes precedence over ``model_outputs``.
            row_scenarios: the scene each row draws, keyed by row label. A benchmark may evaluate a split on a perturbed
                variant (causal-agents-hard tests on ``remove_noncausal``), so its row must draw the scene its models
                were actually given rather than the unperturbed one.
        """
        del causal_gt_ids
        if not isinstance(scenario, AgentCentricScenario):
            error_message = "Scenario needs to be of AgentCentricScenario"
            raise TypeError(error_message)

        models = model_outputs or ({"model": model_output} if model_output is not None else None)
        grid = model_grid or ({"": models} if models else None)
        if not grid:
            error_message = "At least one model output is required for TrajPred scenario visualization."
            raise ValueError(error_message)
        rows, columns = self._grid_layout(grid)

        scenario_id = scenario.scenario_id
        scene_score = BaseVisualizer.get_scenario_score(scores)
        suffix = "" if scene_score is None else f"_{scene_score}"
        output_filepath = f"{output_dir}/{scenario_id}{suffix}.png"
        logger.info("Visualizing scenario to %s", output_filepath)

        gt_alpha = self.config.get("gt_future_alpha", 0.3)
        pane_size = self.config.get("pane_size", 5)
        # `constrained` rather than tight_layout: it reserves room for the suptitle, which otherwise overlaps the column
        # titles once the figure has more than one row.
        _, axs = plt.subplots(
            len(rows),
            len(columns),
            figsize=(pane_size * len(columns), pane_size * len(rows)),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )

        for row_index, row in enumerate(rows):
            # The scene this row's models were evaluated on, which is not always the unperturbed one.
            row_scenario = (row_scenarios or {}).get(row, scenario)
            for column_index, column in enumerate(columns):
                ax = axs[row_index][column_index]
                self._draw_scene_context(ax, row_scenario, draw_future_gt=True, gt_alpha=gt_alpha)
                output = grid[row].get(column)
                if output is None or output.trajectory_decoder_output is None:
                    logger.warning(
                        "No trajectory decoder output for pane '%s' on scenario %s; drawing scene context only.",
                        f"{row}/{column}" if row else column,
                        scenario_id,
                    )
                else:
                    self._draw_predictions(ax, output)
                if row_index == 0:
                    ax.set_title(column)
                if column_index == 0 and row:
                    ax.set_ylabel(row)
                # Not `ax.axis("off")`: that would also hide the row label drawn as the y-axis label.
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")
                for spine in ax.spines.values():
                    spine.set_visible(False)

        plt.suptitle(f"Scenario: {scenario_id}")
        plt.savefig(output_filepath, dpi=self.config.get("dpi", 300), bbox_inches="tight")
        for ax in axs.flat:
            ax.cla()
        plt.close()

    @staticmethod
    def _grid_layout(model_grid: dict[str, dict[str, ModelOutput]]) -> tuple[list[str], list[str]]:
        """Resolves the pane grid into its row and column labels, both in first-seen order."""
        rows = list(model_grid)
        columns = list(dict.fromkeys(column for row in model_grid.values() for column in row))
        return rows, columns
