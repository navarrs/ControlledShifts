import matplotlib.pyplot as plt
import numpy as np
from characterization.schemas import Scenario, ScenarioScores
from characterization.utils.io_utils import get_logger
from matplotlib.axes import Axes
from numpy.typing import NDArray
from omegaconf import DictConfig

from controlledshifts.schemas import AgentCentricScenario, ModelOutput
from controlledshifts.utils.analysis.common import MODEL_NAME_MAP
from controlledshifts.utils.constants import MIN_VALID_POINTS
from controlledshifts.utils.scenario_visualizers.base_visualizer import BaseVisualizer


logger = get_logger(__name__)

# Feature offsets within the agent-centric `obj_trajs` last dimension (see AgentCentricProcessor.get_centered_agent_data).
_SIZE_SLICE = slice(3, 6)  # length, width, height
_TYPE_SLICE = slice(6, 9)  # one-hot: vehicle, pedestrian, cyclist
_EGO_CHANNEL = 10  # is-SDC flag
_HEADING_SLICE = slice(23, 25)  # sin(theta), cos(theta)

# obj_trajs type-onehot index -> agent_colors key.
_AGENT_TYPE_KEYS = ("TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST")

# Draw order, low to high: other agents, ego track, predictions, then the ego box on top so predictions do not overdraw
# the agent making them.
_AGENT_ZORDER = 100
_EGO_ZORDER = 1000
_PRED_ZORDER = 1500
_PRED_BEST_ZORDER = 2000
_EGO_BOX_ZORDER = 2500

# Waymo polyline_type integer (the argmax of map_polylines[..., 9:29]) -> map_colors key. Index 0 is padding/masked and
# is intentionally absent so it is skipped; 1/2/3 are lane centerlines, now drawn rather than skipped.
_MAP_TYPE_TO_KEY: dict[int, str] = {
    1: "lane",
    2: "lane",
    3: "lane",
    6: "road_line",
    7: "road_line",
    8: "road_line",
    9: "road_line",
    10: "road_line",
    11: "road_line",
    12: "road_line",
    13: "road_line",
    15: "road_edge",
    16: "road_edge",
    17: "stop_sign",
    18: "crosswalk",
    19: "speed_bump",
}


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
    def _decode_agent(agent_history: np.ndarray, valid_step: int) -> tuple[str, float, float, float, bool]:
        """Decodes one agent's ``agent_colors`` key, box heading/length/width and ego flag from ``obj_trajs``.

        Args:
            agent_history: one agent's ``obj_trajs`` row, shape ``(timesteps, features)``.
            valid_step: the timestep to read the pose and size from (the last valid history step).

        Returns:
            The agent-type color key, heading (radians), length, width, and whether the agent is the ego.
        """
        step = agent_history[valid_step]
        is_ego = bool(step[_EGO_CHANNEL])
        type_key = "TYPE_SDC" if is_ego else _AGENT_TYPE_KEYS[int(np.argmax(step[_TYPE_SLICE]))]
        heading = float(np.arctan2(step[_HEADING_SLICE][0], step[_HEADING_SLICE][1]))
        length, width, _ = step[_SIZE_SLICE]
        return type_key, heading, float(length), float(width), is_ego

    def _draw_scene_context(
        self, ax: Axes, scenario: AgentCentricScenario, *, draw_future_gt: bool = True, gt_alpha: float = 0.3
    ) -> None:
        """Draws the scene context on a pane: the map, agent history boxes/tracks, and optionally the GT future.

        Styling is inherited from the shared visualization config (``map_colors``/``map_alphas``/``agent_colors``), so
        the pane matches the regular/causal visualizers.

        Args:
            ax: Axes to plot on.
            scenario: encapsulates the agent-centric scenario to visualize.
            draw_future_gt: if True, overlays the ground-truth future trajectories for reference.
            gt_alpha: transparency for the ground-truth future trajectories, dimmed against the model prediction.
        """
        self._draw_map(ax, scenario)
        self._draw_agents(ax, scenario, draw_future_gt=draw_future_gt, gt_alpha=gt_alpha)

    def _draw_map(self, ax: Axes, scenario: AgentCentricScenario) -> None:
        """Draws each map polyline in its semantic color, keyed by Waymo polyline type."""
        map_xy, map_type = self._decode_map(scenario.map_polylines)
        map_mask = np.asarray(scenario.map_polylines_mask)
        for idx, polyline in enumerate(map_xy):
            key = _MAP_TYPE_TO_KEY.get(int(map_type[idx]))
            if key is None:  # padding/masked polyline
                continue
            # One plot per polyline (as in the regular plot_polylines); masked points become NaN so matplotlib breaks
            # the line there instead of us drawing segment-by-segment.
            xy = np.where(map_mask[idx, :, None], polyline, np.nan)
            ax.plot(xy[:, 0], xy[:, 1], color=self.map_colors[key], alpha=self.map_alphas[key], linewidth=0.5)

    def _draw_agents(
        self, ax: Axes, scenario: AgentCentricScenario, *, draw_future_gt: bool, gt_alpha: float
    ) -> None:
        """Draws each agent's history track and a rotated bounding box, colored by agent type (ego highlighted)."""
        futures = scenario.obj_trajs_future_state
        future_mask = scenario.obj_trajs_future_mask
        for idx, (history, mask) in enumerate(zip(scenario.obj_trajs, scenario.obj_trajs_mask, strict=True)):
            valid = np.flatnonzero(mask)
            if len(valid) < MIN_VALID_POINTS:
                continue
            type_key, heading, length, width, is_ego = self._decode_agent(history, valid[-1])
            color = self.agent_colors[type_key]
            zorder = _EGO_ZORDER if is_ego else _AGENT_ZORDER
            # The ego box sits above the predictions so they never overdraw the agent making them; other agents stay low.
            box_zorder = _EGO_BOX_ZORDER if is_ego else _AGENT_ZORDER

            track = history[valid, :2]
            ax.plot(track[:, 0], track[:, 1], color=color, linewidth=2, zorder=zorder)
            if draw_future_gt:
                future_valid = np.flatnonzero(future_mask[idx])
                if len(future_valid) >= MIN_VALID_POINTS:
                    gt = futures[idx][future_valid, :2]
                    ax.plot(gt[:, 0], gt[:, 1], color=color, linewidth=2, alpha=gt_alpha, zorder=zorder)
            self.plot_agent(
                ax, track[-1, 0], track[-1, 1], heading, length, width, 1.0, color, plot_rectangle=True, zorder=box_zorder
            )

    def _draw_predictions(self, ax: Axes, model_output: ModelOutput) -> None:
        """Draws a model's predicted future trajectories in coral, with the most likely mode emphasized.

        Args:
            ax: Axes to plot on.
            model_output: encapsulates the model outputs; must carry a trajectory decoder output.
        """
        pred_future_traj = model_output.trajectory_decoder_output.decoded_trajectories.value.detach().cpu().numpy()
        pred_future_prob = model_output.trajectory_decoder_output.mode_probabilities.value.detach().cpu().numpy()
        color = self.agent_colors["TYPE_RELEVANT"]
        best_mode = int(np.argmax(pred_future_prob))
        # Scale opacity by mode probability, keeping unlikely modes faintly visible rather than invisible.
        peak = max(float(pred_future_prob.max()), 1e-6)
        for idx, traj in enumerate(pred_future_traj):
            is_best = idx == best_mode
            alpha = 1.0 if is_best else max(0.2, float(pred_future_prob[idx]) / peak)
            # The most likely mode is drawn thickest and above the other modes so it reads first.
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=color,
                linewidth=3 if is_best else 2,
                alpha=alpha,
                zorder=_PRED_BEST_ZORDER if is_best else _PRED_ZORDER,
            )

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
                    # Show the canonical display name (e.g. "wayformer" -> "Wayformer"); custom labels pass through.
                    ax.set_title(MODEL_NAME_MAP.get(column, column))
                if column_index == 0 and row:
                    ax.set_ylabel(row)
                # Ego-centred square window (the ego sits at the origin in the agent-centric frame), matching the regular
                # visualizers' framing. Ticks off but the black spines kept, so panes read as framed like the others.
                distance = self.distance_to_ego_zoom_in + self.buffer_distance
                ax.set_xlim(-distance, distance)
                ax.set_ylim(-distance, distance)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")

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
