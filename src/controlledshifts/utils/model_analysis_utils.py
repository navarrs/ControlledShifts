"""Utility functions for model analysis.

See `docs/ANALYSIS.md` for usage details.
"""

import json
import pickle
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from scipy import stats
from sklearn.manifold import TSNE

from controlledshifts import utils
from controlledshifts.schemas import output_schemas as output
from controlledshifts.utils import metrics as metrics_utils


def get_scenario_dec_embeddings(
    model_outputs: dict[str, output.ModelOutput],
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.float64]]:
    """Extracts and flattens scenario_dec embeddings from model outputs.

    Args:
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.

    Returns:
        scenario_ids (npt.NDArray[np.str_]): array of scenario IDs in insertion order.
        embeddings (npt.NDArray[np.float64]): array of shape (num_scenarios, embedding_dim).
    """
    scenario_ids = []
    embeddings = []
    for scenario_id, model_output in model_outputs.items():
        emb = model_output.scenario_embedding.scenario_dec.value.detach().cpu().numpy().flatten()
        scenario_ids.append(scenario_id)
        embeddings.append(emb)
    return np.asarray(scenario_ids), np.stack(embeddings).astype(np.float64)


def get_scenario_classes_best_mode(
    model_outputs: dict[str, output.ModelOutput],
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.int64], npt.NDArray[np.float64], int]:
    """Iterates over the model outputs and extracts the scenario class corresponding to the best predicted mode.

    Args:
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.

    Returns:
        scenario_ids (npt.NDArray[np.str_]): list of scenario IDs in the model outputs list.
        scenario_classes (npt.NDArray[np.int64]): list of classes.
        scenario_scores (npt.NDArray[np.float64]): list of scenario scores.
        num_classes (int): number of scenario classes.
    """
    scenario_ids, scenario_classes, scenario_scores, num_classes = [], [], [], 0
    # For each scenario, get the class corresponding to the best mode
    for scenario_id, model_output in model_outputs.items():
        tokenization_output = model_output.tokenization_output
        num_classes = tokenization_output.num_tokens if tokenization_output.num_tokens != 0 else 100
        indices = tokenization_output.token_indices.value.numpy()

        trajectory_decoder_output = model_output.trajectory_decoder_output
        selected_mode = trajectory_decoder_output.mode_probabilities.value.argmax(dim=-1).item()

        scenario_ids.append(scenario_id)
        scenario_classes.append(indices[selected_mode])

        if model_output.scenario_scores is not None:
            # TODO: fix this by adding SafeShift Scores to the ScenarioScores schema
            indvidual_score = model_output.scenario_scores.individual_scenario_score.value.item()
            interaction_score = model_output.scenario_scores.interaction_scenario_score.value.item()
            safeshift_score = indvidual_score + interaction_score
            scenario_scores.append(safeshift_score)

    return np.asarray(scenario_ids), np.array(scenario_classes).reshape(-1, 1), np.array(scenario_scores), num_classes


def get_scenario_classes_per_mode(
    model_outputs: dict[str, output.ModelOutput], *, rank_by_probability: bool = True
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.float64], int]:
    """Iterates over the model outputs and extracts the scenario classes.

    Args:
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.
        rank_by_probability (bool): if True, organizes the tokens by the probability of the modes.

    Returns:
        scenario_ids (npt.NDArray[np.str_]): numpy array of shape (num_scenarios,) containing the scenario IDs.
        scenario_classes (npt.NDArray[np.float64]): numpy array of shape (num_scenarios, num_modes) containing the
            scenario tokens per mode.
        num_classes (int): number of scenario classes.
    """
    scenario_ids, scenario_classes, num_classes = [], [], 0
    # For each scenario, get the class corresponding to the best mode
    for scenario_id, model_output in model_outputs.items():
        tokenization_output = model_output.tokenization_output
        num_classes = tokenization_output.num_tokens if tokenization_output.num_tokens != 0 else 100
        indices = tokenization_output.token_indices.value.numpy()

        if rank_by_probability:
            trajectory_decoder_output = model_output.trajectory_decoder_output
            modes_order = np.argsort(trajectory_decoder_output.mode_probabilities.value.numpy())[::-1]
            indices = indices[modes_order]

        scenario_ids.append(scenario_id)
        scenario_classes.append(indices)

    return np.asarray(scenario_ids), np.stack(scenario_classes), num_classes


def plot_scenario_class_distribution(  # noqa: PLR0915
    config: DictConfig, model_outputs: dict[str, output.ModelOutput], output_path: Path
) -> None:
    """Creates a histogram over the scenario classes for the given batches.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.
        output_path (Path): output path where visualization will be saved to.
    """
    if config.best_mode_only:
        # Scenario classess shape: (num_scenarios, 1)
        scenario_ids, scenario_classes, _, num_classes = get_scenario_classes_best_mode(model_outputs)
    else:
        # Scenario classess shape: (num_scenarios, num_modes)
        scenario_ids, scenario_classes, num_classes = get_scenario_classes_per_mode(
            model_outputs, rank_by_probability=config.rank_by_probability
        )

    distribution_stats = {}

    num_total_tokens = config.model.config.num_classes if config.model.config.num_classes is not None else 100
    underutilization_threshold = 100.0 / num_total_tokens
    num_scenarios, num_modes = scenario_classes.shape
    unique_classes = np.unique(scenario_classes).tolist()

    # Calculate how many dead tokens there are (tokens that are not assigned to any scenario in any mode)
    dead_tokens = num_total_tokens - len(unique_classes)
    distribution_stats["num_total_tokens"] = num_total_tokens
    distribution_stats["dead_tokens"] = dead_tokens

    _, axs = plt.subplots(nrows=num_modes, figsize=(len(unique_classes), 20 * num_modes))

    colors = sns.color_palette(palette=config.palette, n_colors=num_classes)
    color_map = dict(zip(unique_classes, colors, strict=False))

    heatmap = np.zeros((num_modes, num_classes))
    arange = np.arange(0, num_classes + 1)
    sns.set_style("whitegrid")
    for num_mode in range(num_modes):
        scenario_classes_mode = scenario_classes[:, num_mode]

        # Create a histogram for numerical stats
        mode_stats = {}
        counts, _ = np.histogram(scenario_classes_mode, bins=unique_classes)
        normalized_counts = counts / num_scenarios
        mode_stats["mean"] = counts.mean().item()
        mode_stats["norm_mean"] = normalized_counts.mean().item()
        mode_stats["median"] = np.median(counts).item()
        mode_stats["norm_median"] = np.median(normalized_counts).item()
        mode_stats["num_underutilized_tokens"] = (normalized_counts < underutilization_threshold).sum().item()
        distribution_stats[f"mode_{num_mode}_stats"] = mode_stats

        # Create a histogram for visualization
        classes_dict = {"scenario_id": scenario_ids, "scenario_class": scenario_classes_mode}
        classes_df = pd.DataFrame(classes_dict)

        # Plot histograms
        ax = axs[num_mode] if num_modes > 1 else axs  # pyright: ignore[reportIndexIssue]
        num_scenarios = classes_df.shape
        sns.countplot(
            ax=ax,
            data=classes_df,
            x="scenario_class",
            hue="scenario_class",
            palette=color_map,
            edgecolor="white",
            linewidth=1.5,
            legend=False,
        )
        for container in ax.containers:
            ax.bar_label(container, fontsize=11, fontweight="bold")

        ax.set_xlabel("Scenario Class", fontsize=12, fontweight="bold", color="#2E2E2E")
        ax.set_ylabel("Scenario Count", fontsize=12, fontweight="bold", color="#2E2E2E")
        ax.set_title(
            f"Scenario Class Histogram (Total {num_scenarios})", fontsize=20, fontweight="bold", color="#1F1F1F"
        )
        ax.tick_params(axis="x", labelsize=12, rotation=45 if len(classes_df["scenario_class"].unique()) > 5 else 0)  # noqa: PLR2004
        ax.tick_params(axis="y", labelsize=12)
        ax.set_axisbelow(True)

        counts, _ = np.histogram(scenario_classes_mode, bins=arange)
        heatmap[num_mode] = counts

    suffix = "_best_mode" if config.best_mode_only else "_ranked_modes"
    output_filepath = output_path / f"class_histogram{suffix}.png"
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=200)
    plt.close()

    # Plot heatmap
    output_filepath = output_path / f"class_heatmap{suffix}.png"
    plt.subplots(figsize=(2 * len(unique_classes), 5 * num_modes))
    sns.heatmap(heatmap, xticklabels=arange, linewidth=1.0)
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=200)

    output_filepath = output_path / f"class_log-heatmap{suffix}.png"
    log_heatmap = np.log10(heatmap + 1)
    plt.subplots(figsize=(2 * len(unique_classes), 5 * num_modes))
    sns.heatmap(log_heatmap, xticklabels=arange, linewidth=1.0)
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=200)

    distribution_stats_output_filepath = output_path / f"class_distribution_stats{suffix}.json"
    with distribution_stats_output_filepath.open("w") as f:
        json.dump(distribution_stats, f, indent=4)
    print("\tDone")


def compute_dimensionality_reduction(
    config: DictConfig, model_outputs: dict[str, output.ModelOutput], output_path: Path
) -> npt.NDArray[np.float64]:
    """Uses a manifold learning algorithm (TSNE, UMAP) to reduce the dimensionality of the scenario embeddings.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.
        output_path (Path): output path where visualization will be saved to.

    Returns:
        model_results (npt.NDArray[np.float64]): a numpy array of shape (num_scenarios, config.num_components)
            encapsulating the dimensionality reduction results.
    """
    algorithm = config.dim_reduction_algorithm
    match algorithm:
        case "tsne":
            model = TSNE(n_components=config.num_components, random_state=config.seed)
        case "umap":
            from umap import UMAP  # noqa: PLC0415

            model = UMAP(n_components=config.num_components, transform_seed=config.seed)
        case _:
            error_message = f"Algorithm: {config.dim_reduction_algorithm} not supported."
            raise ValueError(error_message)

    print(f"Computing {algorithm} analysis...")
    print(f"\tNumber of components {config.num_components}")
    embeddings = []
    for model_output in model_outputs.values():
        match config.reduce:
            case "input_embedding":
                embedding = model_output.tokenization_output.input_embedding.value.detach().cpu().numpy()
            case "quantized_embedding":
                embedding = model_output.tokenization_output.quantized_embedding.value.detach().cpu().numpy()
            case _:
                error_message = f"Reduce: {config.reduce} not supported"
                raise ValueError(error_message)

        if config.best_mode_only:
            trajectory_decoder_output = model_output.trajectory_decoder_output
            selected_mode = trajectory_decoder_output.mode_probabilities.value.argmax(dim=-1).detach().cpu().item()
            embeddings.append(embedding[selected_mode].reshape(1, -1))
        else:
            embeddings.append(embedding)
    embeddings = np.concatenate(embeddings, axis=0)
    result = model.fit_transform(embeddings)

    if config.save_result:
        output_filepath = Path(f"{output_path}/{config.dim_reduction_algorithm}.pkl")
        with output_filepath.open("wb") as f:
            pickle.dump(result, f)

    print("\tDone")
    return result  # pyright: ignore[reportReturnType]


def plot_manifold_by_tokens(
    config: DictConfig, manifold: np.ndarray, model_outputs: dict[str, output.ModelOutput], output_path: Path
) -> None:
    """Visualizes the manifold results colored by token IDs.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        manifold (np.ndarray): learned manifold from 'compute_dimensionality_reduction'.
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.
        output_path (Path): output path where visualization will be saved to.

    Returns:
        model_results (np.ndarray): a numpy array of shape (num_scenarios, config.num_components) encapsulating the
            dimensionality reduction results.
    """
    print(f"Visualizing {config.dim_reduction_algorithm} tokens...")

    if config.best_mode_only:
        # Scenario classess shape: (num_scenarios, 1)
        _, scenario_classes, _, num_classes = get_scenario_classes_best_mode(model_outputs)
    else:
        # Scenario classess shape: (num_scenarios, num_modes)
        _, scenario_classes, num_classes = get_scenario_classes_per_mode(
            model_outputs, rank_by_probability=config.rank_by_probability
        )
    scenario_classes = scenario_classes.reshape(-1)

    # Get color map
    unique_classes, classes_counts = np.unique(scenario_classes, return_counts=True)
    unique_classes = unique_classes.tolist()
    colors = sns.color_palette(palette=config.palette, n_colors=num_classes)
    color_map = dict(zip(unique_classes, colors, strict=False))

    match config.alpha_type:
        case "class_counts":
            class_counts = utils.minmax_scaler(classes_counts)
            alphas = np.zeros(scenario_classes.shape, dtype=np.float32)
            for i, scenario_class in enumerate(unique_classes):
                idx = scenario_classes == scenario_class
                alphas[idx] = class_counts[i]
        case "scores":
            scores = np.array([output.scene_score.value.detach().cpu().item() for output in model_outputs.values()])
            alphas = utils.minmax_scaler(scores)
        case _:
            alphas = np.ones(scenario_classes.shape, dtype=np.float32)

    # scale and move the coordinates so they fit [0; 1] range
    tx, ty = manifold[:, 0], manifold[:, 1]
    if config.normalize:
        tx = utils.minmax_scaler(tx)
        ty = utils.minmax_scaler(ty)

    fig = plt.figure(dpi=1000)  # , facecolor="#444141")
    ax = fig.add_subplot(111)
    for label, color in color_map.items():
        idx = scenario_classes == label
        ax.scatter(tx[idx], ty[idx], marker=".", s=3, color=color, alpha=alphas[idx], edgecolors=color)

    # for n, idx in enumerate(idx_to_highlight):
    #     ax.scatter(tx[idx], ty[idx], marker='.', s=10, color='black', edgecolors='black')
    #     ax.text(tx[idx], ty[idx], s=idx_tags[n], fontsize=8)

    ax.axis("off")
    plt.tight_layout()
    suffix = f"_{config.reduce}" + ("_normalized" if config.normalize else "") + f"_{config.alpha_type}"
    output_filepath = f"{output_path}/{config.dim_reduction_algorithm}{suffix}.png"
    plt.savefig(output_filepath)
    print("\tDone")


def compute_score_analysis(
    config: DictConfig, model_outputs: dict[str, output.ModelOutput], output_path: Path
) -> pd.DataFrame:
    """Uses a manifold learning algorithm (TSNE, UMAP) to reduce the dimensionality of the scenario embeddings.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.
        output_path (Path): output path where visualization will be saved to.

    Returns:
        score_analysis (pd.DataFrane): a dataframe containing scenario score stats per token.
    """
    # Get the predicted classes for the best mode
    _, scenario_classes, scenario_scores, _ = get_scenario_classes_best_mode(model_outputs)
    scenario_classes = scenario_classes.squeeze(-1)

    unique_classes = np.unique(scenario_classes)

    score_analysis = {
        "scenario_class": [],
        "num_scenarios": [],
        "max_score": [],
        "min_score": [],
        "mean_score": [],
        "std_score": [],
        "min_max_diff_score": [],
    }
    for percentile in config.percentiles_of_interest:
        key = f"score_percentile_{percentile}"
        score_analysis[key] = []

    for scenario_class in unique_classes:
        idxs = scenario_classes == scenario_class
        num_scenarios = idxs.sum()
        class_scores = scenario_scores[idxs]

        # Add stats
        max_score, min_score = class_scores.max(), class_scores.min()
        score_analysis["scenario_class"].append(scenario_class)
        score_analysis["num_scenarios"].append(num_scenarios)
        score_analysis["max_score"].append(max_score)
        score_analysis["min_score"].append(min_score)
        score_analysis["mean_score"].append(class_scores.mean())
        score_analysis["std_score"].append(class_scores.std())
        score_analysis["min_max_diff_score"].append(max_score - min_score)

        # Add percentiles
        for percentile in config.percentiles_of_interest:
            key = f"score_percentile_{percentile}"
            score_analysis[key].append(np.percentile(class_scores, percentile))

    score_analysis = pd.DataFrame(score_analysis)
    output_file = Path(output_path / "scenario_score_analysis.csv")
    score_analysis.to_csv(output_file)

    return score_analysis


def read_score_analysis(config: DictConfig) -> pd.DataFrame:
    """Reads score analysis file.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.

    Returns:
        score_analysis (pd.DataFrane): a dataframe containing scenario score stats per token.

    Raise:
        FileNotFoundError: if file does not exist.
    """
    score_analysis_path = Path(config.output_path / "scenario_score_analysis.csv")
    if not score_analysis_path.exists():
        error_message = f"File: {score_analysis_path} not found. Run with 'run_score_analysis=True'."
        raise FileNotFoundError(error_message)
    return pd.read_csv(score_analysis_path)


def get_scenario_percentiles(
    scenario_scores: npt.NDArray[np.float32], score_info: pd.DataFrame
) -> npt.NDArray[np.float32]:
    """Given a list of scenario scores and score percentile information, get the percentile to which each scenario
    belongs to.

    Args:
        scenario_scores (npt.NDArray[np.float32]): array of N scenario scores to be assigned to a percentile.
        score_info (pd.DataFrame): Dataframe containing score percentile information.

    Returns:
        percentiles (npt.NDArray[np.float32]): an array of N percentile assignments.
    """
    percentile_columns = [column for column in score_info.columns if "percentile" in column]
    percentile_scores = score_info[percentile_columns].to_numpy().squeeze()
    percentiles = []
    a_max = len(percentile_columns) - 1
    for scenario_score in scenario_scores:
        percentile_index = np.clip(
            np.searchsorted(percentile_scores, scenario_score, side="left"), a_min=0, a_max=a_max
        )
        percentiles.append(percentile_columns[percentile_index])
    return np.asarray(percentiles)


def plot_scenarios(
    config: DictConfig,
    scenario_files: dict[str, Path],
    scenario_ids: npt.NDArray[np.str_],
    scenario_percentiles: npt.NDArray[np.str_],
    output_path: Path,
) -> None:
    """Plots scenarios by percentile.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        scenario_files (dict[str, str]): dictionary mapping scenario_ids to their corresponding filepath.
        scenario_ids (npt.NDArray[np.str_]): scenario IDs to plot.
        scenario_percentiles (npt.NDArray[np.str_]): array of scores corresponding to each scenario ID to plot.
        output_path (Path): path to save the scenarios to.
    """
    visualizer = hydra.utils.instantiate(config.visualization.visualizer)
    waymo_tf = hydra.utils.instantiate(config.dataset)
    for scenario_id, scenario_percentile in zip(scenario_ids, scenario_percentiles, strict=False):
        scenario_file = scenario_files[scenario_id]
        with scenario_file.open("rb") as f:
            scenario = pickle.load(f)
        scenario = waymo_tf.repack_scenario(scenario)
        features = waymo_tf.scenario_features_processor.compute(scenario)
        score = waymo_tf.scenario_scores_processor.compute(scenario, features)

        scenario_output_path = output_path / scenario_percentile
        scenario_output_path.mkdir(parents=True, exist_ok=True)
        visualizer.visualize_scenario(scenario, score, output_dir=scenario_output_path)


def plot_tokenized_scenarios_by_score_percentile(
    config: DictConfig, model_outputs: dict[str, output.ModelOutput], score_analysis: pd.DataFrame, output_path: Path
) -> None:
    """Plots tokenized scenarios by percentile.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.
        score_analysis (pd.DataFrame): a dataframe containing scenario score stats per token.
        output_path (Path): output path where visualization will be saved to.
    """
    # Get the paths to the GT scenarios
    scenario_files = {str(f).split("/")[-1].split(".")[0]: f for f in Path(config.scenarios_path).glob("*/*")}
    scenario_files = {k: scenario_files[k] for k in model_outputs}

    scenario_ids, scenario_classes, scenario_scores, _ = get_scenario_classes_best_mode(model_outputs)
    scenario_classes = scenario_classes.squeeze()
    unique_classes = np.unique(scenario_classes)
    for scenario_class in unique_classes:
        print(f"Running scenario token: {scenario_class}")
        scenario_output_path = output_path / f"scenario_viz_by_percentile/{scenario_class}"
        scenario_output_path.mkdir(parents=True, exist_ok=True)

        idx = scenario_classes == scenario_class
        class_scores = scenario_scores[idx]
        class_scenarios_ids = scenario_ids[idx]

        class_score_info = score_analysis[score_analysis.scenario_class == scenario_class]
        class_scenario_percentiles = get_scenario_percentiles(class_scores, class_score_info)
        # TODO: modify to use class scores
        plot_scenarios(config, scenario_files, class_scenarios_ids, class_scenario_percentiles, scenario_output_path)


def get_tokenization_groups(
    config: DictConfig,
    model_outputs: dict[str, output.ModelOutput],
) -> tuple[dict[int, npt.NDArray[np.float64] | None], dict[int, npt.NDArray[np.str_] | None]]:
    """Organizes the predicted tokens based on their corresponding mode probabilities.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.

    Returns:
        tokenization_groups (dict[int, npt.NDArray | None]): dictionary containing scenario tokenizations organized by
            the highest-likelihood token. If the token does not have any assignments it gets set to None.
        group_scenario_ids (dict[int, np.NDArray[str] | None]): dictionary containing scenario ids organized by
            the highest-likelihood token. If the token does not have any assignments it gets set to None.
    """
    num_tokens = config.model.config.num_classes
    # Initialize the tokenization groups dictionary
    tokenization_groups = {i: [] for i in range(num_tokens)}
    groups_scenario_ids = {i: [] for i in range(num_tokens)}

    # Organize the model outputs based on the highest-probability mode
    for model_output in model_outputs.values():
        tokenization_output = model_output.tokenization_output
        indices = tokenization_output.token_indices.value.detach().cpu().numpy()
        trajectory_decoder_output = model_output.trajectory_decoder_output
        # Get probability-based mode order
        order = np.argsort(trajectory_decoder_output.mode_probabilities.value.detach().cpu().numpy())[::-1]
        indices = indices[order]
        # Add the tokenization to dictionary
        tokenization_groups[indices[0]].append(indices)
        groups_scenario_ids[indices[0]].append(model_output.scenario_id)

    stacked_tokenization_groups: dict[int, npt.NDArray[np.int32] | None] = {}
    stacked_groups_scenario_ids: dict[int, npt.NDArray[np.str_] | None] = {}
    for best_token, other_tokens in tokenization_groups.items():
        if len(other_tokens) == 0:
            stacked_tokenization_groups[best_token] = None
            stacked_groups_scenario_ids[best_token] = None
            continue
        stacked_tokenization_groups[best_token] = np.stack(other_tokens)
        stacked_groups_scenario_ids[best_token] = np.stack(groups_scenario_ids[best_token])

    return stacked_tokenization_groups, stacked_groups_scenario_ids


def get_group_modes(tokenization_groups: dict[int, npt.NDArray[np.float64] | None]) -> dict[int, npt.NDArray[np.int32]]:
    """Computes the modes per tokenized group.

    Args:
        tokenization_groups (dict[int, npt.NDArray[np.float64] | None]): dictionary containing the tokenized groups.

    Returns:
        modes (dict[int, npt.NDArray[np.int32]]): a dictionary containing the statistical mode for each token group.
    """
    modes = dict.fromkeys(tokenization_groups.keys())
    for base_token, token_group in tokenization_groups.items():
        if token_group is None:
            continue
        modes[base_token] = stats.mode(token_group, axis=0).mode
    return modes


def get_group_unique(
    tokenization_groups: dict[int, npt.NDArray[np.float64] | None],
) -> tuple[dict[int, npt.NDArray[np.int32]], dict[int, npt.NDArray[np.int32]]]:
    """Computes the unique tokens and respective counts for each of the tokenization groups.

    Args:
        tokenization_groups (dict[int, npt.NDArray[np.float64] | None]): dictionary containing the tokenized groups.

    Returns:
        unique (dict[int, npt.NDArray[np.int32]]: a dictionary containing per-group unique tokens.
        counts (dict[int, npt.NDArray[np.int32]]: a dictionary containing per-group token counts.
    """
    group_unique = dict.fromkeys(tokenization_groups.keys())
    group_counts = dict.fromkeys(tokenization_groups.keys())
    for base_token, token_group in tokenization_groups.items():
        if token_group is None:
            continue
        values, counts = np.unique(token_group, return_counts=True)
        group_unique[base_token] = values
        group_counts[base_token] = counts
    return group_unique, group_counts


def plot_heatmap(  # noqa: PLR0913
    heatmap: npt.NDArray[np.float64],
    title: str,
    x_label: str,
    y_label: str,
    cbar_label: str,
    output_filepath: Path,
    colormap: str = "viridis",
) -> None:
    """Visualizes a heatmap matrix.

    Args:
        heatmap (npt.NDArray[np.float64]): a heatmap matrix to plot.
        title (str): the title of the heatmap.
        x_label (str): the label of the x-axis.
        y_label (str): the label of the y-axis.
        cbar_label (str): the label of the heatmap's colorbar.
        colormap (str): the colormap to use for the heatmap.
        output_filepath (Path): filepath to save the visualization.
    """
    plt.figure(figsize=(35, 30))

    plt.imshow(heatmap, cmap=colormap, aspect="auto")
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=40)
    cbar.set_label(cbar_label, size=40)

    plt.title(title, fontsize=50)
    plt.xlabel(x_label, fontsize=40)
    plt.ylabel(y_label, fontsize=40)
    plt.xticks(range(heatmap.shape[0]))
    plt.yticks(range(heatmap.shape[1]))
    plt.grid(visible=False)

    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()
    print(f"Heatmap saved to {output_filepath}")


def compute_alignment_scores(
    target: list[int], samples: npt.NDArray[np.int32], alignment_measure: str
) -> npt.NDArray[np.float64]:
    """Computes an alignment score between a target and samples.

    Args:
        target (list[int]): a list of values representing the target.
        samples (npt.NDArray[np.int32]): an array of samples.
        alignment_measure (str): the measure to compute the alignment between target and samples.

    Returns:
        scores (npt.NDArray[np.float64]): an array representing the alignement.
    """
    match alignment_measure:
        case "jaccard":
            scores = [metrics_utils.compute_jaccard_index(set(target), set(sample.tolist())) for sample in samples]
        case "hamming":
            scores = [
                metrics_utils.compute_hamming_distance(target, sample.tolist(), return_inverse=True)
                for sample in samples
            ]
        case _:
            error_message = f"{alignment_measure} is not supported"
            raise ValueError(error_message)
    return np.array(scores, dtype=np.float64)


def compute_token_consistency_matrix(
    config: DictConfig,
    model_outputs: dict[str, output.ModelOutput],
    output_path: Path,
) -> npt.NDArray[np.float64]:
    """Creates a histogram over the scenario classes for the given batches.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.
        output_path (Path): output path where visualization will be saved to.

    Returns:
        consistency_matrix (npt.NDArray[np.float64]): a matrix representing the consistency of each group.
    """
    num_tokens = config.model.config.num_classes
    tokenization_groups, _ = get_tokenization_groups(config, model_outputs)
    group_modes = get_group_modes(tokenization_groups)

    consistency_matrix = np.zeros(shape=(num_tokens, num_tokens))
    # Get a weight factor based on group sizes
    # group_sizes = np.array(
    #     [tokenization_groups[i].shape[0] if tokenization_groups[i] is not None else 0 for i in range(num_tokens)]
    # )
    # weight_factors = (group_sizes - group_sizes.min()) / (group_sizes.max() - group_sizes.min() + 1e-8)
    #
    for (i, j), _ in np.ndenumerate(consistency_matrix):
        if tokenization_groups[i] is None or group_modes.get(j, None) is None:
            continue
        assigned_samples = tokenization_groups[i]
        scores = compute_alignment_scores(group_modes[j].tolist(), assigned_samples, config.consistency_measure)
        consistency_matrix[i, j] = scores.mean()

    # Visualize the consistency matrix as a heatmap
    plot_heatmap(
        consistency_matrix,
        title="Token Asignment Consistency",
        x_label="Group Mode",
        y_label="Token Group",
        cbar_label="Average Consistency Index",
        output_filepath=output_path / f"token_consistency_matrix_{config.consistency_measure}.png",
        colormap=config.token_consistency_colormap,
    )

    # Compute stats
    diagonal = np.diag(consistency_matrix)
    tokenization_stats = {
        "diagonal": {
            "min": diagonal.min().item(),
            "max": diagonal.max().item(),
            "mean": diagonal.mean().item(),
            "std": diagonal.std().item(),
            "median": np.median(diagonal).item(),
        }
    }
    tokenization_stats_output_filepath = output_path / f"token_consistency_stats_{config.consistency_measure}.json"
    with tokenization_stats_output_filepath.open("w") as f:
        json.dump(tokenization_stats, f, indent=4)

    return consistency_matrix


def plot_uniqueness_index(
    config: DictConfig, uniqueness_index: npt.NDArray[np.float64], output_path: Path, bar_width: float = 0.8
) -> None:
    """Plots the uniqueness index for each tokenization group.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        uniqueness_index (npt.NDArray[np.float64]): an array representing the uniqueness of each group.
        output_path (Path): filepath to save the visualization.
        bar_width (float): width of each bar in the plot.
    """
    num_tokens = config.model.config.num_classes
    x_indices = np.arange(num_tokens)

    # Create the bar plot
    _, ax = plt.subplots(figsize=(30, 4))

    # Plot the bars
    plt.bar(x_indices, uniqueness_index, color="skyblue", width=bar_width)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlim(-bar_width / 2, num_tokens - 1 + bar_width / 2)

    # Set the y-axis limit from 0 to 1 as requested
    plt.ylim(0, 1)

    # Add labels and title
    plt.xlabel("Token ID")
    plt.ylabel("Uniqueness Index")
    plt.title("Token Group Uniqueness Index")
    plt.xticks(x_indices)  # Ensure all indices are shown on the x-axis

    plt.tight_layout()

    filepath = output_path / "group_uniqueness_index_jaccard.png"
    plt.savefig(filepath)
    plt.close()
    print(f"Jaccard uniqueness saved to {filepath}")


def compute_group_uniqueness(
    config: DictConfig,
    model_outputs: dict[str, output.ModelOutput],
    output_path: Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Computes the uniquenes index and counts for each tokenization group w.r.t the scenario vocabulary.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a list of model outputs per scenario.
        output_path (Path): filepath to save the visualization.

    Returns:
        group_uniqueness (npt.NDArray[np.float64]): a matrix representing the uniqueness index of each group.
        group_counts (npt.NDArray[np.float64]): a matrix representing the counts of each group.
    """
    num_tokens = config.model.config.num_classes
    tokenization_groups, _ = get_tokenization_groups(config, model_outputs)
    group_unique, group_counts = get_group_unique(tokenization_groups)

    scenario_vocabulary = set(range(num_tokens))
    group_uniqueness = np.zeros(shape=(num_tokens))
    group_vocab_counts = np.zeros(shape=(num_tokens, num_tokens))
    for n in range(num_tokens):
        unique, counts = group_unique.get(n, None), group_counts[n]
        if unique is None:
            continue
        counts = counts.astype(np.float64)
        group_uniqueness[n] = metrics_utils.compute_jaccard_index(scenario_vocabulary, set(unique.tolist()))
        match config.normalize_counts:
            case "group_scenarios":
                norm_value = tokenization_groups[n].shape[0]
            case "all_scenarios":
                norm_value = len(model_outputs)
            case _:
                norm_value = 1.0
        counts /= norm_value
        group_vocab_counts[n, unique] = counts

    # Visualize results
    plot_uniqueness_index(config, group_uniqueness, output_path)
    plot_heatmap(
        group_vocab_counts,
        title="Group Uniqueness Heatmap",
        x_label="Group",
        y_label="Vocabulary",
        cbar_label="Uniqueness Index",
        output_filepath=output_path / f"group_uniqueness_counts_{config.normalize_counts}.png",
        colormap=config.group_uniqueness_colormap,
    )

    # Save stats
    group_uniqueness_stats = {
        "uniqueness": {
            "min": group_uniqueness.min().item(),
            "max": group_uniqueness.max().item(),
            "mean": group_uniqueness.mean().item(),
            "std": group_uniqueness.std().item(),
            "median": np.median(group_uniqueness).item(),
        }
    }
    group_uniqueness_stats_output_filepath = output_path / f"group_uniqueness_stats_{config.normalize_counts}.json"
    with group_uniqueness_stats_output_filepath.open("w") as f:
        json.dump(group_uniqueness_stats, f, indent=4)

    return group_uniqueness, group_vocab_counts


def compute_intergroup_uniqueness(
    config: DictConfig,
    model_outputs: dict[str, output.ModelOutput],
    output_path: Path,
) -> npt.NDArray[np.float64]:
    """Computes the uniquenes index and counts for each tokenization group w.r.t all other tokenization groups.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a list of model outputs per scenario.
        output_path (Path): filepath to save the visualization.

    Returns:
        intergroup_uniqueness (npt.NDArray[np.float64]): a matrix representing the consistency of each group.
    """
    num_tokens = config.model.config.num_classes
    tokenization_groups, _ = get_tokenization_groups(config, model_outputs)
    group_unique, group_counts = get_group_unique(tokenization_groups)

    intergroup_uniqueness = np.zeros(shape=(num_tokens, num_tokens))
    categorical_vector = np.arange(num_tokens)
    wd_list = []
    for (i, j), _ in np.ndenumerate(intergroup_uniqueness):
        if i < j:
            continue
        group_i = group_unique.get(i, None)
        counts_i = group_counts.get(i, None)

        group_j = group_unique.get(j, None)
        counts_j = group_counts.get(j, None)
        if group_i is None or group_j is None or counts_i is None or counts_j is None:
            continue

        # If group information is valid, compute the uniqueness index and counts
        cat_i = np.zeros(shape=(num_tokens))
        cat_i[group_i] = categorical_vector[group_i]
        norm_counts_i = counts_i / counts_i.sum()
        norm_i = np.zeros(shape=(num_tokens))
        norm_i[group_i] = norm_counts_i.astype(np.float64)

        cat_j = np.zeros(shape=(num_tokens))
        cat_j[group_j] = categorical_vector[group_j]
        norm_counts_j = counts_j / counts_j.sum()
        norm_j = np.zeros(shape=(num_tokens))
        norm_j[group_j] = norm_counts_j.astype(np.float64)

        wd = stats.wasserstein_distance(cat_i, cat_j, norm_i, norm_j)
        if i != j:
            wd_list.append(wd)
        wd_inv = 1 - wd / (categorical_vector.max() - categorical_vector.min())
        # jaccard_index_ij = compute_jaccard_index(set(group_i.tolist()), set(group_j.tolist()))
        # intergroup_uniqueness[i, j] = jaccard_index_ij * (wd_inv)
        intergroup_uniqueness[i, j] = wd_inv

    # Visualize the uniqueness matrix as a heatmap
    plot_heatmap(
        intergroup_uniqueness,
        title="Intergroup Uniqueness Heatmap",
        x_label="Group",
        y_label="Group",
        cbar_label="Uniqueness Index",
        output_filepath=output_path / "intergroup_uniqueness.png",
        colormap=config.intergroup_uniqueness_colormap,
    )

    # Compute stats
    wd = np.array(wd_list)
    intergroup_stats = {
        "intergroup_uniqueness": {
            "min": intergroup_uniqueness.min().item(),
            "max": intergroup_uniqueness.max().item(),
            "mean": intergroup_uniqueness.mean().item(),
            "std": intergroup_uniqueness.std().item(),
            "median": np.median(intergroup_uniqueness).item(),
        },
        "wasserstein_distance": {
            "min": wd.min().item(),
            "max": wd.max().item(),
            "mean": wd.mean().item(),
            "std": wd.std().item(),
            "median": np.median(wd).item(),
        },
    }
    intergroup_uniqueness_stats_output_filepath = output_path / "intergroup_uniqueness_stats.json"
    with intergroup_uniqueness_stats_output_filepath.open("w") as f:
        json.dump(intergroup_stats, f, indent=4)

    return intergroup_uniqueness
