# pyright: reportOptionalMemberAccess=false
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from characterization.schemas import Scenario, ScenarioScores
from characterization.utils.common import AgentTrajectoryMasker
from characterization.utils.io_utils import get_logger
from matplotlib.axes import Axes
from omegaconf import DictConfig

from controlledshifts.schemas import AgentCentricScenario, ModelOutput
from controlledshifts.utils.constants import MIN_VALID_POINTS, CausalOutputType
from controlledshifts.utils.scenario_visualizers.base_visualizer import BaseVisualizer


logger = get_logger(__name__)


class ScenarioCausalVisualizer(BaseVisualizer):
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

        ScenarioCausalVisualizer visualizes the scenario on three or four windows:
            window 1: displays the full scene zoomed out
            window 2: displays the scene with GT causal agents marked in a different color.
            window 3: displays the scene with predicted causal agents marked in a different color and with a
                probability-based alpha value.
            window 4: displays the scene with each agent in a different color, based on it's learned tokenization.

        Args:
            scenario (Scenario | AgentCentricScenario): encapsulates the scenario to visualize.
            scores (ScenarioScores | None): encapsulates the scenario and agent scores.
            model_output (ModelOutput | None): encapsulates model outputs.
            output_dir: (str): the directory where to save the scenario visualization.
        """
        if not isinstance(scenario, Scenario):
            error_message = "Scenario visualization only supported in global frame."
            raise TypeError(error_message)

        if model_output is None or model_output.causal_output is None:
            error_message = "Model output is required for causal scenario visualization."
            raise ValueError(error_message)

        scenario_id = scenario.metadata.scenario_id
        scene_score = BaseVisualizer.get_scenario_score(scores)
        suffix = "" if scene_score is None else f"_{scene_score}"
        output_filepath = f"{output_dir}/{scenario_id}_causal{suffix}.png"
        logger.info("Visualizing scenario to %s", output_filepath)

        causal_tokenization_output = model_output.causal_tokenization_output
        num_windows = 3 if causal_tokenization_output is None else 4
        _, axs = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))

        # Plot static and dynamic map information in the scenario
        self.plot_map_data(axs, scenario, num_windows)

        # Window 1: full scene
        axs[0].set_title("Full Scene")
        self.plot_sequences(axs[0], scenario, scores, show_relevant=True)

        # Window 2: Ground truth causal scene
        axs[1].set_title("GT Causal")
        self.plot_causal(axs[1], scenario, model_output=model_output, show_causal=CausalOutputType.GROUND_TRUTH)

        # Window 2: Predicted causal scene
        axs[2].set_title("Pred Causal")
        self.plot_causal(axs[2], scenario, model_output=model_output, show_causal=CausalOutputType.PREDICTION)

        # Plot remove-noncausalequal scene
        if causal_tokenization_output is not None:
            axs[3].set_title("Causal Token")
            self.plot_tokenized(axs[3], scenario, model_output=model_output)

        # Prepare and save plot
        self.set_axes(axs, scenario, num_windows)
        plt.suptitle(f"Scenario: {scenario_id}")
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_causal(  # noqa: PLR0913
        self,
        ax: Axes,
        scenario: Scenario,
        model_output: ModelOutput,
        show_causal: CausalOutputType,
        start_timestep: int = 0,
        end_timestep: int = -1,
    ) -> None:
        """Plots agent trajectories for a scenario, with optional highlighting and score-based transparency.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on.
            scenario (Scenario): Scenario data with agent positions, types, and relevance.
            model_output (ModelOutput | None): encapsulates model outputs.
            show_causal (CausalOutputType): Source in (GROUND_TRUTH, PREDICTED) to show agent causality.
            start_timestep (int): starting timestep to plot the sequences.
            end_timestep (int): ending timestep to plot the sequences.
        """
        if model_output.causal_output is None:
            error_message = "Causal output is required for causal scenario visualization."
            raise ValueError(error_message)

        agent_data = scenario.agent_data
        agent_ids = agent_data.agent_ids
        agent_types = np.asarray([atype.name for atype in agent_data.agent_types])
        ego_index = scenario.metadata.ego_vehicle_index

        ego_index = scenario.metadata.ego_vehicle_index

        # Get the agents normalized scores
        # agent_scores = np.ones(agent_data.num_agents, float)
        # agent_scores = BaseVisualizer.get_normalized_agent_scores(agent_scores, ego_index)
        agent_scores = np.zeros(agent_data.num_agents, float)

        # Mark causal agents as 'TYPE_RELEVANT'
        modeled_agent_ids = model_output.agent_ids.value.detach().cpu().numpy()
        mask = modeled_agent_ids != -1
        modeled_agent_ids = modeled_agent_ids[mask]
        match show_causal:
            case CausalOutputType.GROUND_TRUTH:
                causal = model_output.causal_output.causal_gt.value.detach().cpu().numpy()[mask]
                relevant_indeces = np.where(causal > 0.0)[0]
                relevant_agent_ids = modeled_agent_ids[relevant_indeces]
                idxs = np.isin(agent_ids, relevant_agent_ids)
                agent_types[idxs] = "TYPE_RELEVANT"
                agent_scores[idxs] = 1.0
            case CausalOutputType.PREDICTION:
                # TODO: make this parallized
                causal = model_output.causal_output.causal_pred.value.detach().cpu().numpy()[mask]
                causal_probs = model_output.causal_output.causal_pred_probs.value.detach().cpu().numpy()[mask]
                for n, (pred, prob) in enumerate(zip(causal.astype(int), causal_probs, strict=False)):
                    agent_id = modeled_agent_ids[n]
                    idx = np.isin(agent_ids, agent_id)
                    if pred == 1:
                        agent_types[idx] = "TYPE_RELEVANT"
                    agent_scores[idx] = prob[pred]
        agent_types[ego_index] = "TYPE_SDC"  # Mark ego agent for visualization

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
            heading = ahead[end_timestep]
            length = alen[end_timestep]
            width = awid[end_timestep]
            color = self.agent_colors[atype]
            zorder = 1000 if atype == "TYPE_SDC" else 100
            ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2, alpha=score, zorder=zorder)
            # Plot the agent
            self.plot_agent(
                ax, pos[-1, 0], pos[-1, 1], heading, length, width, score, color, plot_rectangle=True, zorder=zorder
            )

    def plot_tokenized(
        self,
        ax: Axes,
        scenario: Scenario,
        model_output: ModelOutput,
        start_timestep: int = 0,
        end_timestep: int = -1,
    ) -> None:
        """Plots agent trajectories for a scenario, with optional highlighting and score-based transparency.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on.
            scenario (Scenario): Scenario data with agent positions, types, and relevance.
            model_output (ModelOutput | None): encapsulates model outputs.
            start_timestep (int): starting timestep to plot the sequences.
            end_timestep (int): ending timestep to plot the sequences.
        """
        agent_data = scenario.agent_data
        agent_ids = np.asarray(agent_data.agent_ids)
        agent_types = np.asarray([atype.name for atype in agent_data.agent_types])
        ego_index = scenario.metadata.ego_vehicle_index

        # Get the agents normalized scores
        agent_scores = np.ones(agent_data.num_agents, float)
        agent_scores = BaseVisualizer.get_normalized_agent_scores(agent_scores, ego_index)

        causal_tokenization = model_output.causal_tokenization_output
        num_tokens = causal_tokenization.quantized_embedding.value.shape[-1]
        colors = sns.color_palette(palette="tab20", n_colors=num_tokens)
        unique_classes = np.arange(num_tokens).tolist()
        color_map = dict(zip(unique_classes, colors, strict=False))

        # TODO: Make this more efficient
        tokens = causal_tokenization.token_indices.value.detach().cpu().numpy()
        modeled_agent_ids = model_output.agent_ids.value.detach().cpu().numpy()
        mask = modeled_agent_ids != -1
        modeled_agent_ids = modeled_agent_ids[mask]

        agent_types[ego_index] = "TYPE_SDC"  # Mark ego agent for visualization
        color_list = [self.agent_colors[atype] for atype in agent_types]
        for n, agent_id in enumerate(modeled_agent_ids):
            idx = np.where(agent_ids == agent_id)[0]
            if idx.shape[0] == 0:
                continue
            idx = idx[0].item()
            if idx == ego_index:
                continue
            token_idx = tokens[n].item()
            color_list[idx] = color_map[token_idx]

        agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
        zipped = zip(
            agent_ids,
            agent_trajectories.agent_xy_pos,
            agent_trajectories.agent_lengths,
            agent_trajectories.agent_widths,
            agent_trajectories.agent_headings,
            agent_trajectories.agent_valid.squeeze(-1).astype(bool),
            color_list,
            agent_scores,
            strict=False,
        )
        for aid, apos, alen, awid, ahead, amask, color, score in zipped:
            if aid not in modeled_agent_ids:
                score = 0.1  # noqa: PLW2901
            # Skip if there are less than 2 valid points.
            mask = amask[start_timestep:end_timestep]
            if not mask.any() or mask.sum() < MIN_VALID_POINTS:
                continue

            pos = apos[start_timestep:end_timestep][mask]
            heading = ahead[end_timestep]
            length = alen[end_timestep]
            width = awid[end_timestep]
            ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2, alpha=score)
            # Plot the agent
            self.plot_agent(ax, pos[-1, 0], pos[-1, 1], heading, length, width, score, plot_rectangle=True, color=color)
