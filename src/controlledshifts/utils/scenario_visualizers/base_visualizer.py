import os
from abc import ABC, abstractmethod
from glob import glob

import numpy as np
from characterization.schemas import DynamicMapData, Scenario, ScenarioScores, StaticMapData
from characterization.utils.common import SUPPORTED_SCENARIO_TYPES, AgentTrajectoryMasker
from characterization.utils.io_utils import get_logger
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from natsort import natsorted
from numpy.typing import NDArray
from omegaconf import DictConfig
from PIL import Image

from controlledshifts.schemas import AgentCentricScenario, ModelOutput
from controlledshifts.utils.constants import INVALID_AGENT_ID, MIN_VALID_POINTS, NonBackgroundSource, SupportedPanes


logger = get_logger(__name__)

PANE_TITLES: dict[SupportedPanes, str] = {
    SupportedPanes.ALL_AGENTS: "All Agents Trajectories",
    SupportedPanes.HIGHLIGHT_RELEVANT: "Highlighted Relevant and SDC Agent Trajectories",
    SupportedPanes.NON_BACKGROUND_AGENTS_GT: "GT Non-Background",
    SupportedPanes.NON_BACKGROUND_AGENTS_PRED: "Pred Non-Background",
    SupportedPanes.TRAJECTORY_PREDICTION: "Trajectory Prediction",
}


class BaseVisualizer(ABC):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseVisualizer with visualization configuration and validates required keys.

        This base class provides a flexible interface for scenario visualizers, supporting custom map and agent color
        schemes, transparency, and scenario type validation. Subclasses should implement scenario-specific visualization
        logic.

        Args:
            config: Configuration for the visualizer, including scenario type, map/agent keys, colors, and
                alpha values.

        Raises:
            AssertionError: If the scenario type or any required configuration key is missing or unsupported.
        """
        self.config = config
        self.scenario_type = config.scenario_type
        if self.scenario_type not in SUPPORTED_SCENARIO_TYPES:
            error_message = f"Scenario type {self.scenario_type} not in supported types: {SUPPORTED_SCENARIO_TYPES}"
            raise AssertionError(error_message)

        self.static_map_keys = config.get("static_map_keys", None)
        if self.static_map_keys is None:
            error_message = "static_map_keys must be provided in the configuration."
            raise AssertionError(error_message)

        self.dynamic_map_keys = config.get("dynamic_map_keys", None)
        if self.dynamic_map_keys is None:
            error_message = "dynamic_map_keys must be provided in the configuration."
            raise AssertionError(error_message)

        self.map_colors = config.get("map_colors", None)
        if self.map_colors is None:
            error_message = "map_colors must be provided in the configuration."
            raise AssertionError(error_message)

        self.map_alphas = config.get("map_alphas", None)
        if self.map_alphas is None:
            error_message = "map_alphas must be provided in the configuration."
            raise AssertionError(error_message)

        self.agent_colors = config.get("agent_colors", None)
        if self.agent_colors is None:
            error_message = "agent_colors must be provided in the configuration."
            raise AssertionError(error_message)

        panes_to_plot = config.get("panes_to_plot", None)
        if panes_to_plot is None:
            error_message = "panes_to_plot must be provided in the configuration."
            raise AssertionError(error_message)
        self.panes_to_plot = [SupportedPanes[pane] for pane in panes_to_plot]

        self.buffer_distance = config.get("distance_to_ego_zoom_in", 5.0)  # in meters
        self.distance_to_ego_zoom_in = config.get("distance_to_ego_zoom_in", 50.0)  # in meters

        self.background_alpha = config.get("background_alpha", 0.6)

    @property
    def is_ego_centric(self) -> bool:
        return self.config.get("is_ego_centric", False)

    @property
    def is_animated(self) -> bool:
        # Animated subclasses override this to return True.
        return False

    @staticmethod
    def get_scenario_score(scores: ScenarioScores | None) -> float | None:
        """Gets the scene score from the ScenarioScores.

        Args:
            scores: encapsulates the scenario and agent scores.

        Returns:
            The scene score if available, otherwise None.
        """
        return (
            None
            if scores is None or scores.safeshift_scores is None or scores.safeshift_scores.scene_score is None
            else round(scores.safeshift_scores.scene_score, 2)
        )

    def plot_map_data(self, ax: Axes, scenario: Scenario, num_windows: int = 1) -> None:
        """Plots the map data.

        Args:
            ax: Axes to plot on.
            scenario: encapsulates the scenario to visualize.
            num_windows: Number of subplot windows. Defaults to 1.
        """
        if scenario.static_map_data is None:
            logger.warning("Scenario does not contain map_polylines, skipping static map visualization.")
        else:
            self.plot_static_map_data(ax, static_map_data=scenario.static_map_data, num_windows=num_windows)

        if scenario.dynamic_map_data is None:
            logger.warning("Scenario does not contain dynamic_map_info, skipping dynamic map visualization.")
        else:
            self.plot_dynamic_map_data(ax, dynamic_map_data=scenario.dynamic_map_data, num_windows=num_windows)

    def plot_sequences(  # noqa: PLR0913
        self,
        ax: Axes,
        scenario: Scenario,
        scores: ScenarioScores | None = None,
        *,
        show_relevant: bool = False,
        start_timestep: int = 0,
        end_timestep: int = -1,
    ) -> None:
        """Plots agent trajectories for a scenario, with optional highlighting and score-based transparency.

        Args:
            ax: Axes to plot on.
            scenario: encapsulates the scenario to visualize.
            scores: encapsulates the scenario and agent scores.
            show_relevant: if True, highlights relevant and SDC agents. Defaults to False.
            start_timestep: starting timestep to plot the sequences.
            end_timestep: ending timestep to plot the sequences.
        """
        agent_data = scenario.agent_data
        agent_relevance = agent_data.agent_relevance
        agent_types = np.asarray([atype.name for atype in agent_data.agent_types])
        ego_index = scenario.metadata.ego_vehicle_index

        agent_scores = np.ones(agent_data.num_agents, float) if scores is None else scores.safeshift_scores.agent_scores
        agent_scores = BaseVisualizer.get_normalized_agent_scores(agent_scores, ego_index)

        if show_relevant and agent_relevance is not None:
            relevant_indeces = np.where(agent_relevance > 0.0)[0]
            agent_types[relevant_indeces] = "TYPE_RELEVANT"
        agent_types[ego_index] = "TYPE_SDC"

        agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
        zipped = zip(
            agent_trajectories.agent_xy_pos,
            agent_trajectories.agent_lengths,
            agent_trajectories.agent_widths,
            agent_trajectories.agent_headings,
            agent_trajectories.agent_valid.squeeze(-1).astype(bool),
            agent_types,
            agent_scores,
            strict=False,
        )
        for apos, alen, awid, ahead, amask, atype, score in zipped:
            mask = amask[start_timestep:end_timestep]
            if not mask.any() or mask.sum() < MIN_VALID_POINTS:
                continue
            pos = apos[start_timestep:end_timestep][mask]
            # plot_agent expects scalars; the per-agent/per-timestep values carry a trailing channel dim, so squeeze it.
            heading = ahead[end_timestep].item()
            length = alen[end_timestep].item()
            width = awid[end_timestep].item()
            color = self.agent_colors[atype]
            ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2, alpha=score)
            self.plot_agent(ax, pos[-1, 0], pos[-1, 1], heading, length, width, score, color, plot_rectangle=True)

    def plot_pane(  # noqa: PLR0913
        self,
        ax: Axes,
        pane: SupportedPanes,
        scenario: Scenario,
        scores: ScenarioScores | None = None,
        model_output: ModelOutput | None = None,
        *,
        non_background_gt_ids: NDArray[np.int_] | None = None,
        start_timestep: int = 0,
        end_timestep: int = -1,
    ) -> None:
        """Plots a single pane of a scenario visualization based on the requested pane type.

        Args:
            ax: Axes to plot on.
            pane: the pane to plot.
            scenario: encapsulates the scenario to visualize.
            scores: encapsulates the scenario and agent scores.
            model_output: encapsulates model outputs. Required for the predicted non-background pane and
                for the GT non-background pane when ``non_background_gt_ids`` is not provided.
            non_background_gt_ids: ground-truth non-background agent ids loaded from the non-background label files.
                When provided, the GT non-background pane is rendered from these instead of from model outputs.
            start_timestep: starting timestep to plot the sequences.
            end_timestep: ending timestep to plot the sequences.

        Raises:
            ValueError: if a non-background pane is requested without a usable source, or if the pane is not supported.
        """
        match pane:
            case SupportedPanes.ALL_AGENTS:
                self.plot_sequences(ax, scenario, scores, start_timestep=start_timestep, end_timestep=end_timestep)
            case SupportedPanes.HIGHLIGHT_RELEVANT:
                self.plot_sequences(
                    ax, scenario, scores, show_relevant=True, start_timestep=start_timestep, end_timestep=end_timestep
                )
            case SupportedPanes.NON_BACKGROUND_AGENTS_GT:
                if non_background_gt_ids is not None:
                    self.plot_non_background_gt(
                        ax, scenario, non_background_gt_ids, start_timestep=start_timestep, end_timestep=end_timestep
                    )
                elif model_output is not None:
                    self.plot_non_background(
                        ax,
                        scenario,
                        model_output,
                        NonBackgroundSource.GROUND_TRUTH,
                        start_timestep=start_timestep,
                        end_timestep=end_timestep,
                    )
                else:
                    error_message = (
                        "Non-background ground-truth ids or model output are required for the GT non-background pane."
                    )
                    raise ValueError(error_message)
            case SupportedPanes.NON_BACKGROUND_AGENTS_PRED:
                if model_output is None:
                    error_message = "Model output is required for non-background scenario visualization."
                    raise ValueError(error_message)
                self.plot_non_background(
                    ax,
                    scenario,
                    model_output,
                    NonBackgroundSource.PREDICTION,
                    start_timestep=start_timestep,
                    end_timestep=end_timestep,
                )
            case _:
                error_message = f"Pane {pane} is not supported by this visualizer."
                raise ValueError(error_message)

    def plot_non_background(  # noqa: PLR0913
        self,
        ax: Axes,
        scenario: Scenario,
        model_output: ModelOutput,
        source: NonBackgroundSource,
        start_timestep: int = 0,
        end_timestep: int = -1,
    ) -> None:
        """Plots agent trajectories, colouring background agents orange and the ego agent blue.

        Background agents are drawn in the ``TYPE_BACKGROUND`` colour at ``background_alpha``; the ego agent uses the
        ``TYPE_SDC`` colour, and every other (non-background) agent keeps its regular agent-type colour.

        Args:
            ax: Axes to plot on.
            scenario: Scenario data with agent positions, types, and relevance.
            model_output: encapsulates model outputs.
            source: Source in (GROUND_TRUTH, PREDICTION) providing the per-agent non-background classification.
            start_timestep: starting timestep to plot the sequences.
            end_timestep: ending timestep to plot the sequences.

        Raises:
            ValueError: if the model output does not contain a non-background output.
        """
        # `causal_output` is the on-disk schema field name; it holds the non-background classification.
        non_background_output = model_output.causal_output
        if non_background_output is None:
            error_message = "Non-background output is required for non-background scenario visualization."
            raise ValueError(error_message)

        agent_data = scenario.agent_data
        agent_ids = agent_data.agent_ids
        ego_index = scenario.metadata.ego_vehicle_index

        # Agents keep their regular type colour unless they are background; object dtype so the wider "TYPE_BACKGROUND"
        # marker is not truncated by the fixed-width dtype numpy infers from the type names present.
        agent_types = np.asarray([atype.name for atype in agent_data.agent_types], dtype=object)
        agent_scores = np.full(agent_data.num_agents, self.background_alpha, float)
        is_background = np.ones(agent_data.num_agents, bool)

        modeled_agent_ids = model_output.agent_ids.value.detach().cpu().numpy()
        mask = modeled_agent_ids != INVALID_AGENT_ID
        modeled_agent_ids = modeled_agent_ids[mask]
        match source:
            case NonBackgroundSource.GROUND_TRUTH:
                non_background = non_background_output.causal_gt.value.detach().cpu().numpy()[mask]
                non_background_indices = np.where(non_background > 0.0)[0]
                non_background_agent_ids = modeled_agent_ids[non_background_indices]
                idxs = np.isin(agent_ids, non_background_agent_ids)
                is_background[idxs] = False
                agent_scores[idxs] = 1.0
            case NonBackgroundSource.PREDICTION:
                non_background = non_background_output.causal_pred.value.detach().cpu().numpy()[mask]
                non_background_probs = non_background_output.causal_pred_probs.value.detach().cpu().numpy()[mask]
                for n, (pred, prob) in enumerate(zip(non_background.astype(int), non_background_probs, strict=False)):
                    agent_id = modeled_agent_ids[n]
                    idx = np.isin(agent_ids, agent_id)
                    if pred == 1:
                        is_background[idx] = False
                    agent_scores[idx] = prob[pred]
        agent_types[is_background] = "TYPE_BACKGROUND"
        agent_types[ego_index] = "TYPE_SDC"
        agent_scores[ego_index] = 1.0

        self._draw_agents(
            ax, scenario, agent_types, agent_scores, start_timestep=start_timestep, end_timestep=end_timestep
        )

    def plot_non_background_gt(
        self,
        ax: Axes,
        scenario: Scenario,
        non_background_agent_ids: NDArray[np.int_],
        start_timestep: int = 0,
        end_timestep: int = -1,
    ) -> None:
        """Plots ground-truth non-background agents loaded from the label files (no model output required).

        Agents whose ids are in ``non_background_agent_ids`` keep their regular agent-type colour at full opacity; every
        other agent is drawn in the ``TYPE_BACKGROUND`` colour (orange) at ``background_alpha``, and the ego agent uses
        the ``TYPE_SDC`` colour (blue). This mirrors the GROUND_TRUTH branch of ``plot_non_background`` but sources the
        labels from disk.

        Args:
            ax: Axes to plot on.
            scenario: Scenario data with agent positions, types, and ids.
            non_background_agent_ids: the ids of the ground-truth non-background agents (including the ego agent).
            start_timestep: starting timestep to plot the sequences.
            end_timestep: ending timestep to plot the sequences.
        """
        agent_data = scenario.agent_data
        agent_ids = agent_data.agent_ids
        ego_index = scenario.metadata.ego_vehicle_index

        # Agents keep their regular type colour unless they are background; object dtype so the wider "TYPE_BACKGROUND"
        # marker is not truncated by the fixed-width dtype numpy infers from the type names present.
        agent_types = np.asarray([atype.name for atype in agent_data.agent_types], dtype=object)
        agent_scores = np.full(agent_data.num_agents, self.background_alpha, float)
        non_background_idxs = np.isin(agent_ids, non_background_agent_ids)
        agent_types[~non_background_idxs] = "TYPE_BACKGROUND"
        agent_scores[non_background_idxs] = 1.0
        agent_types[ego_index] = "TYPE_SDC"
        agent_scores[ego_index] = 1.0

        self._draw_agents(
            ax, scenario, agent_types, agent_scores, start_timestep=start_timestep, end_timestep=end_timestep
        )

    def _draw_agents(  # noqa: PLR0913
        self,
        ax: Axes,
        scenario: Scenario,
        agent_types: np.ndarray,
        agent_scores: np.ndarray,
        *,
        start_timestep: int = 0,
        end_timestep: int = -1,
    ) -> None:
        """Draws agent trajectories using precomputed per-agent types and score-based alphas.

        Background agents typically carry ``background_alpha``; non-background agents and the ego get a higher score.
        Shared by the model-output (``plot_non_background``) and ground-truth (``plot_non_background_gt``) panes.

        Args:
            ax: Axes to plot on.
            scenario: encapsulates the scenario to visualize.
            agent_types: per-agent type names used to look up colors (e.g. "TYPE_VEHICLE", "TYPE_BACKGROUND",
                "TYPE_SDC").
            agent_scores: per-agent alpha values; 0.0 hides an agent, positive values highlight it.
            start_timestep: starting timestep to plot the sequences.
            end_timestep: ending timestep to plot the sequences.
        """
        agent_trajectories = AgentTrajectoryMasker(scenario.agent_data.agent_trajectories)
        zipped = zip(
            agent_trajectories.agent_xy_pos,
            agent_trajectories.agent_lengths,
            agent_trajectories.agent_widths,
            agent_trajectories.agent_headings,
            agent_trajectories.agent_valid.squeeze(-1).astype(bool),
            agent_types,
            agent_scores,
            strict=False,
        )
        for apos, alen, awid, ahead, amask, atype, score in zipped:
            mask = amask[start_timestep:end_timestep]
            if not mask.any() or mask.sum() < MIN_VALID_POINTS:
                continue

            pos = apos[start_timestep:end_timestep][mask]
            # plot_agent expects scalars; the per-agent/per-timestep values carry a trailing channel dim, so squeeze it.
            heading = ahead[end_timestep].item()
            length = alen[end_timestep].item()
            width = awid[end_timestep].item()
            color = self.agent_colors[atype]
            zorder = 1000 if atype == "TYPE_SDC" else 100
            ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2, alpha=score, zorder=zorder)
            self.plot_agent(
                ax, pos[-1, 0], pos[-1, 1], heading, length, width, score, color, plot_rectangle=True, zorder=zorder
            )

    def plot_agent(  # noqa: PLR0913
        self,
        ax: Axes,
        x: float,
        y: float,
        heading: float,
        width: float,
        height: float,
        alpha: float,
        color: str = "magenta",
        *,
        plot_rectangle: bool = False,
        linewidth: float = 0.5,
        edgecolor: str = "black",
        zorder: int = 100,
        marker: str = "o",
        marker_size: int = 8,
    ) -> None:
        """Plots a single agent as a point (optionally as a rectangle) on the axes.

        Args:
            ax: axes to plot on.
            x: x position of the agent.
            y: y position of the agent.
            heading: heading angle of the agent.
            width: width of the agent.
            height: height of the agent.
            alpha: transparency for the agent marker.
            color: color of the agent marker.
            plot_rectangle: if True it will plot the agent as rectangle, otherwise it will plot it as 'marker'.
            edgecolor: color of the agent's edge if 'plot_rectangle' is True.
            linewidth: width of the agent's border if 'plot_rectangle' is True.
            zorder: z order of agent to plot.
            marker: marker type to plot the agent as, if 'plot_rectangle' is False.
            marker_size: size of the marker if to plot the agent.
        """
        if plot_rectangle:
            angle_deg = np.rad2deg(heading)
            cx, cy = -width / 2.0, -height / 2.0
            x_offset = cx * np.cos(heading) - cy * np.sin(heading)
            y_offset = cx * np.sin(heading) + cy * np.cos(heading)
            rect = Rectangle(
                (x + x_offset, y + y_offset),
                width,
                height,
                angle=angle_deg,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor=color,
                alpha=alpha,
                zorder=zorder,
            )
            ax.add_patch(rect)
        else:
            ax.scatter(x, y, s=marker_size, zorder=zorder, c=color, marker=marker, alpha=alpha)

    def plot_static_map_data(
        self, ax: Axes, static_map_data: StaticMapData, num_windows: int = 1, dim: int = 2
    ) -> None:
        """Plots static map features (lanes, road lines, crosswalks, etc.) for a scenario.

        Args:
            ax: Axes to plot on.
            static_map_data: static map information.
            num_windows: Number of subplot windows. Defaults to 1.
            dim: Number of dimensions to plot. Defaults to 2.
        """
        road_graph = static_map_data.map_polylines
        if road_graph is None:
            return

        road_graph = road_graph[:, :dim]
        if static_map_data.lane_polyline_idxs is not None:
            color, alpha = self.map_colors["lane"], self.map_alphas["lane"]
            BaseVisualizer.plot_polylines(
                ax, road_graph, static_map_data.lane_polyline_idxs, num_windows, color=color, alpha=alpha
            )
        if static_map_data.road_line_polyline_idxs is not None:
            color, alpha = self.map_colors["road_line"], self.map_alphas["road_line"]
            BaseVisualizer.plot_polylines(
                ax, road_graph, static_map_data.road_line_polyline_idxs, num_windows, color=color, alpha=alpha
            )
        if static_map_data.road_edge_polyline_idxs is not None:
            color, alpha = self.map_colors["road_edge"], self.map_alphas["road_edge"]
            BaseVisualizer.plot_polylines(
                ax, road_graph, static_map_data.road_edge_polyline_idxs, num_windows, color=color, alpha=alpha
            )
        if static_map_data.crosswalk_polyline_idxs is not None:
            color, alpha = self.map_colors["crosswalk"], self.map_alphas["crosswalk"]
            BaseVisualizer.plot_polylines(
                ax, road_graph, static_map_data.crosswalk_polyline_idxs, num_windows, color, alpha
            )
        if static_map_data.speed_bump_polyline_idxs is not None:
            color, alpha = self.map_colors["speed_bump"], self.map_alphas["speed_bump"]
            BaseVisualizer.plot_polylines(
                ax, road_graph, static_map_data.speed_bump_polyline_idxs, num_windows, color=color, alpha=alpha
            )
        if static_map_data.stop_sign_polyline_idxs is not None:
            color, alpha = self.map_colors["stop_sign"], self.map_alphas["stop_sign"]
            BaseVisualizer.plot_stop_signs(
                ax, road_graph, static_map_data.stop_sign_polyline_idxs, num_windows, color=color
            )

    def plot_dynamic_map_data(self, ax: Axes, dynamic_map_data: DynamicMapData, num_windows: int = 0) -> None:
        """Plots dynamic map features (e.g., stop points) for a scenario.

        Args:
            ax: Axes to plot on.
            dynamic_map_data: Dynamic map information.
            num_windows: Number of subplot windows. Defaults to 0.
        """
        stop_points = dynamic_map_data.stop_points
        if stop_points is None:
            return
        x_pos = stop_points[0][0][:, 0]
        y_pos = stop_points[0][0][:, 1]
        color = self.map_colors["stop_point"]
        alpha = self.map_alphas["stop_point"]
        if num_windows == 1:
            ax.scatter(x_pos, y_pos, s=6, c=color, marker="s", alpha=alpha)
        else:
            for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
                a.scatter(x_pos, y_pos, s=6, c=color, marker="s", alpha=alpha)

    @staticmethod
    def plot_stop_signs(  # noqa: PLR0913
        ax: Axes,
        road_graph: np.ndarray,
        polyline_idxs: np.ndarray,
        num_windows: int = 0,
        color: str = "red",
        dim: int = 2,
    ) -> None:
        """Plots stop signs on the axes for a scenario using polyline indices.

        Args:
            ax: Axes to plot on.
            road_graph: Road graph points.
            polyline_idxs: Indices for stop sign polylines.
            num_windows: Number of subplot windows. Defaults to 0.
            color: Color for stop signs. Defaults to "red".
            dim: Number of dimensions to plot. Defaults to 2.
        """
        for polyline in polyline_idxs:
            start_idx, end_idx = polyline
            pos = road_graph[start_idx:end_idx, :dim]
            if num_windows == 1:
                ax.scatter(pos[:, 0], pos[:, 1], s=16, c=color, marker="H", alpha=1.0)
            else:
                for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
                    a.scatter(pos[:, 0], pos[:, 1], s=16, c=color, marker="H", alpha=1.0)

    @staticmethod
    def plot_polylines(  # noqa: PLR0913
        ax: Axes,
        road_graph: np.ndarray,
        polyline_idxs: np.ndarray,
        num_windows: int = 0,
        color: str = "k",
        alpha: float = 1.0,
        linewidth: float = 0.5,
    ) -> None:
        """Plots polylines (e.g., lanes, crosswalks) on the axes for a scenario.

        Args:
            ax: Axes to plot on.
            road_graph: Road graph points.
            polyline_idxs: Indices for polylines to plot.
            num_windows: Number of subplot windows. Defaults to 0.
            color: Color for polylines. Defaults to "k".
            alpha: Alpha transparency. Defaults to 1.0.
            linewidth: Line width. Defaults to 0.5.
        """
        for polyline in polyline_idxs:
            start_idx, end_idx = polyline
            pos = road_graph[start_idx:end_idx]
            if num_windows == 1:
                ax.plot(pos[:, 0], pos[:, 1], color, alpha=alpha, linewidth=linewidth)
            else:
                for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
                    a.plot(pos[:, 0], pos[:, 1], color, alpha=alpha, linewidth=linewidth)

    @staticmethod
    def to_gif(
        output_dir: str,
        output_filepath: str,
        *,
        duration: int = 100,
        disposal: int = 2,
        loop: int = 0,
    ) -> None:
        """Saves scenario as a GIF.

        Args:
            output_dir: directory where temporary scenario files have been saved.
            output_filepath: output filepath to save the GIF.
            duration: duration of each frame.
            disposal: specifies how the previous frame should be treated before displaying the next frame.
                (Default value is 2 (restores background color, clear the previous frame))
            loop: number of times the GIF should loop.
        """
        files = glob(f"{output_dir}/temp_*.png")  # noqa: PTH207
        imgs = [Image.open(f) for f in natsorted(files)]

        imgs[0].save(
            output_filepath,
            format="GIF",
            append_images=imgs[1:],
            save_all=True,  # Ensures all frames are saved. Needed for preserving animation.
            duration=duration,
            disposal=disposal,
            loop=loop,
        )

        for f in files:
            os.remove(f)  # noqa: PTH107

    @staticmethod
    def get_normalized_agent_scores(
        agent_scores: np.ndarray, ego_index: int, amin: float = 0.05, amax: float = 1.0
    ) -> np.ndarray:
        """Gets the agent scores and returns a normalized score.

        Args:
            agent_scores: array containing the agent scores.
            ego_index: index of the ego agent.
            amin: minimum value to clip the array.
            amax: maximum value to clip the array.
        """
        min_score = np.nanmin(agent_scores)
        max_score = np.nanmax(agent_scores)
        if max_score > min_score:
            agent_scores = np.clip((agent_scores - min_score) / (max_score - min_score), a_min=amin, a_max=amax)
        else:
            agent_scores = 1.0 - 2 * np.ones_like(agent_scores) / agent_scores.shape[0]
        agent_scores[ego_index] = amax
        return agent_scores

    def set_axes(self, ax: Axes, scenario: Scenario, num_windows: int = 1) -> None:
        """Sets axis limits to zoom in around the ego agent, hiding ticks.

        Args:
            ax: Axes to plot on.
            scenario: encapsulates the scenario to visualize.
            num_windows: Number of subplot windows. Defaults to 1.
        """
        ego_index = scenario.metadata.ego_vehicle_index
        agent_positions = AgentTrajectoryMasker(scenario.agent_data.agent_trajectories).agent_xy_pos
        ego_position = agent_positions[ego_index, 0]
        last_ego_position = agent_positions[ego_index, -1]
        ego_displacement = np.linalg.norm(ego_position - last_ego_position, axis=-1)
        distance = max(self.distance_to_ego_zoom_in, ego_displacement) + self.buffer_distance

        if num_windows == 1:
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlim(ego_position[0] - distance, ego_position[0] + distance)
            ax.set_ylim(ego_position[1] - distance, ego_position[1] + distance)

        else:
            for n, a in enumerate(ax.reshape(-1)):
                a.set_xticks([])
                a.set_yticks([])
                if n == 0:
                    continue

                a.set_xlim(ego_position[0] - distance, ego_position[0] + distance)
                a.set_ylim(ego_position[1] - distance, ego_position[1] + distance)

    @abstractmethod
    def visualize_scenario(  # noqa: PLR0913
        self,
        scenario: Scenario | AgentCentricScenario,
        scores: ScenarioScores | None = None,
        model_output: ModelOutput | None = None,
        output_dir: str = "temp",
        non_background_gt_ids: NDArray[np.int_] | None = None,
        model_outputs: dict[str, ModelOutput] | None = None,
    ) -> None:
        """Visualizes a single scenario and saves the output to a file.

        This method should be implemented by subclasses to provide scenario-specific visualization, supporting flexible
        titles and output paths. It is designed to handle both static and dynamic map features, as well as agent
        trajectories and attributes.

        Args:
            scenario: encapsulates the scenario to visualize.
            scores: encapsulates the scenario and agent scores.
            model_output: encapsulates model outputs.
            output_dir: the directory where to save the scenario visualization.
            non_background_gt_ids: ground-truth non-background agent ids for the GT non-background pane, loaded from
                the non-background label files; used when no model output is available.
            model_outputs: per-model outputs keyed by model name, used by the trajpred visualizer to render one
                comparison pane per model. Ignored by visualizers that render a single model output.
        """
