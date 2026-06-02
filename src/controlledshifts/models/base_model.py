"""BaseModel class for scenario modeling."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from characterization.utils.common import AgentType
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from controlledshifts.schemas.output_schemas import (
    CausalOutput,
    ModelOutput,
    SafetyOutput,
    ScenarioScores,
    TrajectoryDecoderOutput,
)
from controlledshifts.utils import metrics as metric_utils
from controlledshifts.utils import save_cache
from controlledshifts.utils.constants import KALMAN_DIFFICULTY, MILLION, ModelStatus, TrajectoryType


class BaseModel(LightningModule, ABC):
    """Scenario model wrapper based on the PyTorch Lightning `LightningModule` API.

    Reference:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html

    `BaseModel` defines shared structure for scenario modeling architectures, including the forward pass,
    optimizer/scheduler configuration, train/validation/test steps, and metric logging utilities.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the base model.

        Args:
            config (DictConfig): Configuration for the model.
        """
        super().__init__()
        self.config = config
        self.future_len = config.future_len
        self.past_len = config.past_len
        self.num_modes = config.num_modes
        self.max_num_agents = config.max_num_agents
        self.num_bivdist_params = 5  # (µ_x, µ_y, sig_x, sig_y, p)

        self.max_points_per_lane = config.max_points_per_lane
        self.max_num_roads = config.max_num_roads

        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.monitor = config.monitor
        self.criterion = config.criterion

        self.cache_batch = config.cache_batch
        self.cache_every_batch_idx = config.cache_every_batch_idx
        self.batch_cache_path = Path(config.batch_cache_path)
        self.sample_selection = config.get("sample_selection", False)
        self.batch_cache_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def forward(self, batch: dict) -> ModelOutput:
        """Run the model forward pass.

        Args:
            batch (dict): dictionary containing the input data batch:
                batch_size (int)
                input_dict (dict): dictionary containing the scenario information

        Returns:
            model_output (ModelOutput): model schema encapsulating all model outputs.
        """

    def configure_optimizers(self) -> dict[str, Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Define the optimizer and learning-rate scheduler.

        Returns:
            optimizer_config (dict): dictionary containing the optimizer and learning-rate scheduler from the model
            configuration.
        """
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.monitor,
            },
        }

    def print_and_get_num_params(self) -> int:
        """Return the number of model parameters.

        Returns:
            total_parameters (int): total number of parameters in the model.
        """
        trainable_parameters = 0
        nontrainable_parameters = 0
        for parameter in self.parameters():
            num_params = parameter.numel()
            if parameter.requires_grad:
                trainable_parameters += num_params
            else:
                nontrainable_parameters += num_params

        total_parameters = trainable_parameters + nontrainable_parameters
        print("Total number of parameters %.2fM" % (total_parameters / MILLION))
        print("\tTrainable parameters: %.2fM" % (trainable_parameters / MILLION))
        print("\tNon-Trainable parameters: %.2fM" % (nontrainable_parameters / MILLION))
        return total_parameters

    def model_step(self, batch: dict, batch_idx: int, status: ModelStatus) -> torch.Tensor:
        """Take one model step, compute loss, and log model outputs.

        Args:
            batch (dict): dictionary containing the batch information from the collate function.
            batch_idx (int): index of current batch.
            status (ModelStatus): status of the model, either of ModelStatus.TRAIN, ModelStatus.VALIDATION, or
                ModelStatus.TEST.

        Returns:
            loss (torch.Tensor): a tensor containing the model's loss.
        """
        model_output = self.forward(batch)
        loss = self.criterion(model_output)
        self.log_info(batch["input_dict"], model_output, loss, status=status)
        if self.sample_selection:
            cache_filepath = Path(self.batch_cache_path, f"train_batch_{batch_idx}.pkl")
            save_cache(model_output, cache_filepath)
        elif status != ModelStatus.TRAIN and self.cache_batch and batch_idx % self.cache_every_batch_idx == 0:
            cache_filepath = Path(self.batch_cache_path, f"{status.value}_batch_{batch_idx}.pkl")
            save_cache(model_output, cache_filepath)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Performs a model step on a training batch.

        Args:
            batch (dict): dictionary containing the batch information from the collate function.
            batch_idx (int): index of current batch.

        Returns:
            loss (torch.Tensor): model's loss value.
        """
        return self.model_step(batch, batch_idx, status=ModelStatus.TRAIN)

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Performs a model step on a validation batch.

        Args:
            batch (dict): dictionary containing the batch parameters.
            batch_idx (int): index of current batch.

        Returns:
            loss (torch.Tensor): model's loss value.
        """
        return self.model_step(batch, batch_idx, status=ModelStatus.VALIDATION)

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Performs a model step on a testing batch.

        Args:
            batch (dict): dictionary containing the batch parameters.
            batch_idx (int): index of current batch.

        Returns:
            loss (torch.Tensor): model's loss value.
        """
        return self.model_step(batch, batch_idx, status=ModelStatus.TEST)

    @staticmethod
    def gather_input(inputs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather scenario tensors and masks for ego, other agents, and map data.

        Notation:
            B: batch size
            N: number of agents in a scene
            H: history length
            P: number of map polylines
            L: number of points per polyline
            Da: number of agent features
            Dr: number of road features

        Args:
            inputs (dict): dictionary containing scenario data according to
                `collate_fn()` in `controlledshifts/datasets/base_dataset.py`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - ego_in: tensor with shape (B, H, Da + 1).
                - agents_in: tensor with shape (B, H, N, Da + 1).
                - roads: tensor with shape (B, P, L, Dr + 1).
        """
        # Agent trajectories shape: (B, N, H, Da)
        agents_in = inputs["obj_trajs"]
        # Index to predict shape: (B, 1, H, Da)
        index_to_predict = inputs["track_index_to_predict"].view(-1, 1, 1, 1).repeat(1, 1, *agents_in.shape[-2:])
        # Ego-agent input shape: (B, H, Da)
        ego_in = torch.gather(agents_in, 1, index_to_predict).squeeze(1)

        # Agent masks shape: (B, N, H)
        agents_mask = inputs["obj_trajs_mask"]
        # Index to predict shape: (B, 1, H)
        index_to_predict = inputs["track_index_to_predict"].view(-1, 1, 1).repeat(1, 1, agents_mask.shape[-1])
        # Ego mask shape: (B, H)
        ego_mask = torch.gather(agents_mask, 1, index_to_predict).squeeze(1)

        # Combined agent trajectories shape: (B, N, H, Da + 1) -> (B, H, N, Da + 1)
        agents_in = torch.cat([agents_in, agents_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)
        # Combined ego trajectory shape: (B, H, Da + 1)
        ego_in = torch.cat([ego_in, ego_mask.unsqueeze(-1)], dim=-1)

        # Road information shape: (B, P, L, D_r)
        roads = inputs["map_polylines"]
        # Road mask shape: (B, P, L)
        roads_mask = inputs["map_polylines_mask"].unsqueeze(-1)
        # Combined road information shape: (B, P, L, D_r + 1)
        roads = torch.cat([roads, roads_mask], dim=-1)
        return ego_in, agents_in, roads

    @staticmethod
    def gather_scores(inputs: dict) -> ScenarioScores | None:
        """Gather scenario scores for individual and interaction safety.

        Args:
            inputs (dict): dictionary containing scenario data according to `collate_fn()` in
                `controlledshifts/datasets/base_dataset.py`.

        Returns:
            scenario_scores (ScenarioScores | None): Pydantic model containing scenario score information.
        """
        scenario_scores = None
        if "individual_agent_scores" in inputs and "interaction_agent_scores" in inputs:
            individual_safety_scores = inputs["individual_agent_scores"].squeeze(-1)
            interaction_safety_scores = inputs["interaction_agent_scores"].squeeze(-1)
            scenario_scores = ScenarioScores(
                individual_agent_scores=individual_safety_scores,
                individual_scenario_score=inputs["individual_scene_scores"],
                interaction_agent_scores=interaction_safety_scores,
                interaction_scenario_score=inputs["interaction_scene_scores"],
            )
        return scenario_scores

    @staticmethod
    def split_by_trajectory_type(trajectory_types: np.ndarray, metric_dict: dict, metric_keys: list[str]) -> dict:
        """Splits a metric dictionary by trajectory type.

        Args:
            trajectory_types (np.ndarray(B)): array containing the trajectory types for each ego-agent in the batch.
            metric_dict (dict): dictionary containing the computed metrics.
            metric_keys (list[str]): list of metrics to split by trajectory type.

        Returns:
            split_dict (dict): a dictionary of metrics split by trajectory type.
        """
        split_dict = {}
        for traj_type in list(TrajectoryType):
            traj_type_idx = np.where(trajectory_types == traj_type.value)[0]
            if len(traj_type_idx) > 0:
                for key in metric_keys:
                    split_dict["traj_type/" + traj_type.name.lower() + "_" + key] = metric_dict[key][traj_type_idx]
        return split_dict

    @staticmethod
    def split_by_kalman_difficulty(kalman_difficulties: np.ndarray, metric_dict: dict, metric_keys: list[str]) -> dict:
        """Splits a metric dictionary by Kalman difficulty type.

        Args:
            kalman_difficulties (np.ndarray(B)): array containing Kalman difficulty values for each ego in the batch.
            metric_dict (dict): dictionary containing the computed metrics.
            metric_keys (list[str]): list of metrics to split by Kalman difficulty.

        Returns:
            split_dict (dict): dictionary of metrics split by Kalman difficulty.
        """
        split_dict = {}
        for kalman_bucket, (low, high) in KALMAN_DIFFICULTY.items():
            in_range = np.logical_and(low <= kalman_difficulties, kalman_difficulties < high)
            kalman_diff_idx = np.where(in_range)[0]
            if len(kalman_diff_idx) > 0:
                for key in metric_keys:
                    split_dict["kalman/" + kalman_bucket + "_" + key] = metric_dict[key][kalman_diff_idx]
        return split_dict

    @staticmethod
    def split_by_agent_type(agent_types: np.ndarray, metric_dict: dict, metric_keys: list[str]) -> dict:
        """Splits a metric dictionary by agent type.

        Args:
            agent_types (np.ndarray(B)): array containing the agent types for each ego-agent in the batch.
            metric_dict (dict): dictionary containing the computed metrics.
            metric_keys (list[str]): list of metrics to split by agent type.

        Returns:
            split_dict (dict): a dictionary of metrics split by agent type.
        """
        split_dict = {}
        for agent_type in list(AgentType):
            agent_type_idx = np.where(agent_types == agent_type.value)[0]
            if len(agent_type_idx) > 0:
                for key in metric_keys:
                    split_dict["agent_types/" + agent_type.name.lower() + "_" + key] = metric_dict[key][agent_type_idx]
        return split_dict

    @staticmethod
    def split_by_dataset_name(dataset_names: np.ndarray, metric_dict: dict, metric_keys: list[str]) -> dict:
        """Splits a metric dictionary by dataset names.

        Args:
            dataset_names (np.ndarray(B)): array containing the dataset names for each ego-agent in the batch.
            metric_dict (dict): dictionary containing the computed metrics.
            metric_keys (list[str]): list of metrics to split by dataset names.

        Returns:
            split_dict (dict): a dictionary of metrics split by dataset names.
        """
        split_dict = {}
        unique_dataset_names = np.unique(dataset_names)
        for dataset_name in unique_dataset_names:
            dataset_idx = np.argwhere([n == str(dataset_name) for n in dataset_names])[:, 0]
            for key in metric_keys:
                split_dict[dataset_name + "/" + key] = metric_dict[key][dataset_idx]
        return split_dict

    @staticmethod
    def _compute_trajectory_metrics(
        inputs: dict[str, Any], trajectory_output: TrajectoryDecoderOutput, status: ModelStatus
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Computes trajectory metrics on model outputs.

        Notation:
            B: batch size
            N: number of agents in the scene
            M: number of predicted modes
            F: future trajectory length
            Dp: number of predicted dimensions (mu_x, mu_y, sigma_x, sigma_y, p)
            Dg: number of state dimensions in ground-truth trajectories

        Args:
            inputs (dict): dictionary containing input scenario information
            trajectory_output: trajectory decoder output from the model's forward pass.
            status (ModelStatus): status of the model, either of ModelStatus.TRAIN, ModelStatus.VALIDATION, or
                ModelStatus.TEST.

        Returns:
            dict[str, npt.NDArray[np.float64]]: dictionary containing computed trajectory metrics
        """
        # Get ground truth trajectory and mask
        gt_traj = inputs["center_gt_trajs"].unsqueeze(1)  # shape (B, 1, F, Dg)
        gt_traj_mask = inputs["center_gt_trajs_mask"].unsqueeze(1)  # shape (B, 1, F)
        center_gt_final_valid_idx = inputs["center_gt_final_valid_idx"]  # shape (B)
        index_to_predict = inputs["track_index_to_predict"].squeeze(-1)  # shape (B)

        # Gather other agents' ground truth trajectories and masks
        other_trajs = inputs["obj_trajs_future_state"]  # shape (B, N, F, Dg)
        other_trajs_mask = inputs["obj_trajs_future_mask"]  # shape (B, N, F)

        # Get predicted trajectories and probabilities
        predicted_traj = trajectory_output.decoded_trajectories.value  # shape (B, M, F, Dp)
        predicted_prob = trajectory_output.mode_probabilities.value  # shape (B, M)

        batch_size, num_modes = predicted_prob.shape

        # Compute trajectory metrics.
        center_gt_final_valid_idx = center_gt_final_valid_idx.view(-1, 1, 1).repeat(1, num_modes, 1).to(torch.int64)

        # Compute average and final displacement errors.
        ade, fde = metric_utils.compute_displacement_error(
            predicted_traj[:, :, :, :2],
            gt_traj[:, :, :, :2],
            gt_traj_mask,
            center_gt_final_valid_idx,
        )
        min_ade, _ = ade.min(dim=-1)  # shape (B)
        min_fde, best_fde_idx = fde.min(dim=-1)  # both are shape (B)

        # Miss rate measures whether final displacement exceeds a threshold.
        miss_rate_all_modes = metric_utils.compute_miss_rate(fde)
        miss_rate_best_mode = metric_utils.compute_miss_rate(min_fde.unsqueeze(-1))

        best_fde_predicted_prob = predicted_prob[torch.arange(batch_size), best_fde_idx]
        brier_fde = min_fde + torch.square(1 - best_fde_predicted_prob)

        metrics = {
            "minADE6": min_ade.cpu().detach().numpy(),
            "minFDE6": min_fde.cpu().detach().numpy(),
            "missRate6": miss_rate_all_modes.cpu().detach().numpy(),
            "missRate": miss_rate_best_mode.cpu().detach().numpy(),
            "brierFDE": brier_fde.cpu().detach().numpy(),
        }

        # NOTE: These metrics slow down training, so compute them only during evaluation.
        if status in [ModelStatus.VALIDATION, ModelStatus.TEST]:
            collision_rate = metric_utils.compute_collision_rate(
                predicted_traj[:, :, :, :2],
                predicted_prob,
                index_to_predict,
                other_trajs[:, :, :, :2],
                other_trajs_mask.bool(),
            )
            collision_rate_best_mode = metric_utils.compute_collision_rate(
                predicted_traj[:, :, :, :2],
                predicted_prob,
                index_to_predict,
                other_trajs[:, :, :, :2],
                other_trajs_mask.bool(),
                best_mode_only=True,
            )
            metrics.update(
                {f"collisionRate{threshold}": rate.cpu().detach().numpy() for threshold, rate in collision_rate.items()}
            )
            metrics.update(
                {
                    f"collisionRateBestMode{threshold}": rate.cpu().detach().numpy()
                    for threshold, rate in collision_rate_best_mode.items()
                }
            )
        return metrics

    @staticmethod
    def _compute_causal_metrics(causal_output: CausalOutput) -> dict[str, npt.NDArray[np.float64]]:
        """Computes causal metrics on model outputs.

        Args:
            causal_output: causal output from the model's forward pass.

        Returns:
            dict[str, npt.NDArray[np.float64]]: dictionary containing computed causal metrics
        """
        labels = causal_output.causal_gt.value
        predictions = causal_output.causal_pred.value
        tp, tn, fp, fn = metric_utils.compute_binary_confusion_matrix(labels, predictions)
        precision, recall, f1_score = metric_utils.compute_accuracy(labels, predictions)
        return {
            "causalTP": tp.cpu().detach().numpy(),
            "causalTN": tn.cpu().detach().numpy(),
            "causalFP": fp.cpu().detach().numpy(),
            "causalFN": fn.cpu().detach().numpy(),
            "precision": precision.cpu().detach().numpy(),
            "recall": recall.cpu().detach().numpy(),
            "f1Score": f1_score.cpu().detach().numpy(),
        }

    @staticmethod
    def _compute_safety_metrics(safety_output: SafetyOutput) -> dict[str, npt.NDArray[np.float64]]:
        """Computes safety metrics on model outputs.

        Args:
            safety_output: safety output from the model's forward pass.

        Returns:
            dict[str, npt.NDArray[np.float64]]: dictionary containing computed safety metrics
        """
        individual_labels = safety_output.individual_safety_gt.value.squeeze(-1)
        individual_predictions = safety_output.individual_safety_pred.value
        num_classes = safety_output.individual_safety_pred_probs.value.shape[-1]
        ind_precision, ind_recall, ind_f1_score = metric_utils.compute_multiclass_accuracy(
            individual_labels, individual_predictions, num_classes
        )

        interaction_labels = safety_output.interaction_safety_gt.value.squeeze(-1)
        interaction_predictions = safety_output.interaction_safety_pred.value
        num_classes = safety_output.interaction_safety_pred_probs.value.shape[-1]
        int_precision, int_recall, int_f1_score = metric_utils.compute_multiclass_accuracy(
            interaction_labels, interaction_predictions, num_classes
        )

        return {
            "individualPrecision": ind_precision.cpu().detach().numpy(),
            "individualRecall": ind_recall.cpu().detach().numpy(),
            "individualF1Score": ind_f1_score.cpu().detach().numpy(),
            "interactionPrecision": int_precision.cpu().detach().numpy(),
            "interactionRecall": int_recall.cpu().detach().numpy(),
            "interactionF1Score": int_f1_score.cpu().detach().numpy(),
        }

    @staticmethod
    def compute_metrics(
        inputs: dict[str, Any], outputs: ModelOutput, status: ModelStatus
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Computes task metrics on model outputs.

        Args:
            inputs (dict): dictionary containing input scenario information
            outputs (ModelOutput): Pydantic model with model output information.
            status (ModelStatus): status of the model, either of ModelStatus.TRAIN, ModelStatus.VALIDATION, or
                ModelStatus.TEST.

        Returns:
            dict[str, npt.NDArray[np.float64]]: dictionary containing computed metrics.
        """
        # Report trajectory metrics if trajectory outputs are available
        metric_dict = {}
        trajectory_output = outputs.trajectory_decoder_output
        if trajectory_output is not None:
            trajectory_metrics = BaseModel._compute_trajectory_metrics(inputs, trajectory_output, status)
            metric_dict.update(trajectory_metrics)

        # Report causal metrics if causal outputs are available
        causal_output = outputs.causal_output
        if causal_output is not None:
            causal_metrics = BaseModel._compute_causal_metrics(causal_output)
            metric_dict.update(causal_metrics)

        safety_output = outputs.safety_output
        if safety_output is not None:
            safety_metrics = BaseModel._compute_safety_metrics(safety_output)
            metric_dict.update(safety_metrics)

        return metric_dict

    def log_info(
        self, inputs: dict, outputs: ModelOutput, loss: torch.Tensor, status: ModelStatus = ModelStatus.TRAIN
    ) -> None:
        """Log metric values after training, validation, and test steps.

        Args:
            inputs (dict): dictionary containing input scenario information
            outputs (ModelOutput): Pydantic model with model output information.
            loss (torch.Tensor): model's loss value.
            status (ModelStatus): status of the model, either of ModelStatus.TRAIN, ModelStatus.VALIDATION, or
                ModelStatus.TEST.
        """
        # Split metrics by dataset.
        metric_dict = BaseModel.compute_metrics(inputs, outputs, status)
        metric_list = list(metric_dict.keys())

        # Split metric dictionary by sub-dataset name.
        dataset_names = inputs["dataset_name"]
        new_dict = BaseModel.split_by_dataset_name(dataset_names, metric_dict, metric_list)
        metric_dict.update(new_dict)

        if status == ModelStatus.VALIDATION and self.config.get("eval", False):
            # Split metric dictionary by trajectory type.
            trajectory_types = inputs["trajectory_type"].cpu().numpy()
            new_dict = BaseModel.split_by_trajectory_type(trajectory_types, metric_dict, metric_list)
            metric_dict.update(new_dict)

            # Split metric dictionary by Kalman difficulty.
            kalman_difficulties = inputs["kalman_difficulty"][:, -1].cpu().numpy()
            new_dict = BaseModel.split_by_kalman_difficulty(kalman_difficulties, metric_dict, metric_list)
            metric_dict.update(new_dict)

            agent_types = inputs["center_objects_type"]
            new_dict = BaseModel.split_by_agent_type(agent_types, metric_dict, metric_list)
            metric_dict.update(new_dict)

        # Take mean for each key but store original length first (useful for aggregation).
        size_dict = {key: len(value) for key, value in metric_dict.items()}
        metric_dict = {key: np.mean(value) for key, value in metric_dict.items()}

        # Log information
        total_loss = loss.cpu().detach().item()
        self.log(f"losses/{status.value}", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in metric_dict.items():
            batch_size = size_dict[k]
            self.log(status.value + "/" + k, v, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        # TODO: Add support for visualization of scenarios.
        # if self.local_rank == 0 and status == 'val' and batch_idx == 0:
        #     img = visualization.visualize_prediction(batch, prediction)
        #     wandb.log({"prediction": [wandb.Image(img)]})
