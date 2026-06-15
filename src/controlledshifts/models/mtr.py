"""Motion Transformer (MTR) trajectory forecasting model.

Reference:
    Shi et al., "Motion Transformer with Global Intention Localization and Local Movement Refinement", NeurIPS 2022.
    https://arxiv.org/abs/2209.13508

This re-implementation follows the same architecture as UniTraj's MTR
(https://github.com/vita-epfl/UniTraj/tree/main/unitraj/models/mtr) while adapting the input/output interface to match
this repository's conventions.

Key differences from the original:
    - Extends :class:`controlledshifts.models.base_model.BaseModel` (PyTorch Lightning).
    - Returns :class:`controlledshifts.schemas.output_schemas.ModelOutput`.
    - The encoder's global self-attention is used by default (no CUDA KNN).
    - The decoder's map cross-attention uses a pure-PyTorch local attention
      (see ``components/mtr/transformer/multi_head_attention_local.py``).
    - Intention points are per-agent-type k-means cluster centers of training trajectory endpoints. During training,
      they are auto-computed via
      :func:`controlledshifts.utils.intention_points_utils.compute_and_cache_intention_points`
      if the configured ``motion_decoder.intention_points_file`` does not exist, falling back to random initialisation
      with a warning.
"""

import torch
from omegaconf import DictConfig

from controlledshifts.models.base_model import BaseModel
from controlledshifts.models.components.mtr.mtr_decoder import MTRDecoder
from controlledshifts.models.components.mtr.mtr_encoder import MTREncoder
from controlledshifts.schemas.output_schemas import ModelOutput, ScenarioEmbedding, TrajectoryDecoderOutput
from controlledshifts.utils import save_cache
from controlledshifts.utils.constants import ModelStatus


class MTR(BaseModel):
    """Motion Transformer for multi-modal trajectory prediction.

    The model consists of:
    1. :class:`MTREncoder`: PointNet polyline encoders + global transformer self-attention over agent and map tokens.
    2. :class:`MTRDecoder`: Intention-point motion queries refined through iterative cross-attention with object and map
        features, producing M trajectory distributions per sample.

    Loss is computed internally by the decoder across all decoder layers:
    - GMM regression loss (nearest mode)
    - Velocity regression loss (L1)
    - Classification loss (cross-entropy)
    - Dense future prediction auxiliary loss
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.context_encoder = MTREncoder(config.context_encoder)
        self.motion_decoder = MTRDecoder(self.context_encoder.num_out_channels, config.motion_decoder)

        self.print_and_get_num_params()

    def forward(self, batch: dict) -> ModelOutput:
        """Run the MTR forward pass.

        Notation:
            B: batch size
            N: max number of agents in the scene (including ego)
            H: history length (timesteps)
            F: future length (timesteps)
            M: number of predicted modes (after NMS at eval time)
            P: number of map polylines
            L: number of points per polyline
            Da: agent feature dimension
            Dm: encoder/decoder hidden dimension

        Args:
            batch (dict):
                batch_size (int)
                input_dict (dict): scenario tensors from the dataset's collate_fn

        Returns:
            ModelOutput: model outputs containing scenario embeddings, trajectory decoder outputs, ground-truth tensors,
                metadata, and optional scenario scores.
        """
        input_dict = batch["input_dict"]

        # Encode
        batch = self.context_encoder(batch)

        # Decode
        batch = self.motion_decoder(batch)

        # Build output
        pred_list = batch["pred_list"]

        if self.training:
            # During training return the raw (un-NMS'd) last-layer predictions
            pred_scores, pred_trajs = pred_list[-1]
            pred_scores = torch.softmax(pred_scores, dim=-1)
        else:
            pred_scores = batch["pred_scores"]  # (B, M)
            pred_trajs = batch["pred_trajs"]  # (B, M, F, 7)

        # The first 5 channels of pred_trajs are [mu_x, mu_y, log_sigma_x, log_sigma_y, rho] which is the
        # parameterisation used by the GMM loss and is compatible with the framework's TrajectoryDecoderOutput
        # (decoded_trajectories: B, M, F, 5).
        decoded_trajectories = pred_trajs[:, :, :, :5]  # (B, M, F, 5)

        # Ground truth tensors. MTR uses the ego-agent (track_index_to_predict) history only.
        # history_ground_truth shape: (B, H, Da + 1); future_ground_truth shape: (B, F, 3) = (x, y, mask).
        history_ground_truth, future_ground_truth = BaseModel.gather_ground_truth(input_dict, ego_only_history=True)

        # Scenario embedding: expose the center-object feature
        center_feat = batch["center_objects_feature"]  # (B, Dm)
        scenario_embedding = ScenarioEmbedding(scenario_dec=center_feat.unsqueeze(1))  # (B, 1, Dm)

        return ModelOutput(
            scenario_embedding=scenario_embedding,
            trajectory_decoder_output=TrajectoryDecoderOutput(
                decoded_trajectories=decoded_trajectories,
                mode_probabilities=pred_scores,  # pyright: ignore[reportArgumentType]
                mode_logits=None,
            ),
            history_ground_truth=history_ground_truth,  # pyright: ignore[reportArgumentType]
            future_ground_truth=future_ground_truth,  # pyright: ignore[reportArgumentType]
            dataset_name=input_dict["dataset_name"],
            scenario_id=input_dict["scenario_id"],
            agent_ids=input_dict["obj_ids"].squeeze(-1).squeeze(-1),
            scenario_scores=BaseModel.gather_scores(input_dict),
        )

    def configure_optimizers(self) -> dict:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Define the optimizer and LambdaLR learning-rate scheduler.

        Overrides :meth:`BaseModel.configure_optimizers` to use the step-based LambdaLR schedule from the
        original UniTraj/MTR implementation instead of the default OneCycleLR.  The learning rate is
        multiplied by ``lr_decay`` at each epoch listed in ``learning_rate_sched``, then clipped at
        ``lr_clip``.

        Returns:
            optimizer_config (dict): dictionary containing the optimizer and learning-rate scheduler.
        """
        decay_steps = list(self.config.learning_rate_sched)
        base_lr = self.config.lr

        def lr_lbmd(cur_epoch: int) -> float:
            cur_decay = 1.0
            for decay_step in decay_steps:
                if cur_epoch >= decay_step:
                    cur_decay *= self.config.lr_decay
            return max(cur_decay, self.config.lr_clip / base_lr)

        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer=optimizer, lr_lambda=lr_lbmd)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.monitor,
            },
        }

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

        # NOTE: MTR computes its own loss internally across all decoder layers, so we call a method on the decoder to
        # get it, and thus override the BaseModel's model_step (which would normally compute loss using self.criterion)
        # to use MTR's internal loss instead.
        loss, _, _ = self.motion_decoder.get_loss()

        self.log_info(batch["input_dict"], model_output, loss, status=status)
        if self.sample_selection:
            save_cache(model_output, self.batch_cache_path, "train")
        elif status != ModelStatus.TRAIN and self.cache_batch and batch_idx % self.cache_every_batch_idx == 0:
            save_cache(model_output, self.batch_cache_path, f"{status}")
        return loss
