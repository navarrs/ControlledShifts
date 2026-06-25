"""Trajectory decoder for multi-modal future prediction.

The decoder maps scenario context to trajectory distribution parameters and mode probabilities.
"""

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from controlledshifts.models.components import common
from controlledshifts.schemas import TrajectoryDecoderOutput


class TrajectoryDecoder(nn.Module):
    """Decode scenario context into multi-modal trajectory predictions."""

    def __init__(
        self,
        num_modes: int,
        decoding_length: int,
        hidden_size: int,
    ) -> None:
        """Initialize the trajectory decoder.

        Args:
            num_modes (int): Number of trajectory modes to predict.
            decoding_length (int): Number of future steps per mode.
            hidden_size (int): Channel size of each context query.
        """
        super().__init__()
        self.num_modes = num_modes
        self.num_bivdist_params = 5  # (µ_x, µ_y, sig_x, sig_y, p)
        self.decoding_length = decoding_length
        self.decoder_input_size = hidden_size
        self.decoder_output = nn.Sequential(
            nn.Linear(self.decoder_input_size, self.num_bivdist_params * self.decoding_length)
        )
        self.decoder_probs = nn.Sequential(nn.Linear(self.decoder_input_size, 1))
        self.apply(common.initialize_weights_with_xavier)

    def forward(self, context: torch.Tensor) -> TrajectoryDecoderOutput:
        """Decode scenario context into trajectory modes.

        Notation:
            B: Batch size.
            Q: Number of context queries.
            H: Hidden size (feature channels).
            M: Number of trajectory modes.
            F: Number of decoded future steps.

        Args:
            context (torch.Tensor): Context tensor with shape `(B, Q, H)`.

        Returns:
            TrajectoryDecoderOutput: A container with:
                decoded_trajectories (torch.Tensor): Bivariate Gaussian parameters with shape `(B, M, F, 5)`.
                mode_probabilities (torch.Tensor): Softmax-normalized mode probabilities with shape `(B, M)`.
                mode_logits (torch.Tensor): Pre-softmax mode scores with shape `(B, M)`.
        """
        batch_size, _, _ = context.shape

        # Decode M query slots into M trajectory modes across the F-step horizon.
        # Shape: (B, M, F * 5) -> (B, M, F, 5).
        decoded_trajectories = self.decoder_output(context[:, : self.num_modes])
        decoded_trajectories = decoded_trajectories.reshape(batch_size, self.num_modes, self.decoding_length, -1)

        # Predict one score per mode and normalize with softmax.
        # Shape: (B, M, 1) -> (B, M).
        mode_probabilities = self.decoder_probs(context[:, : self.num_modes])
        mode_probabilities = mode_probabilities.reshape(batch_size, self.num_modes)

        if len(np.argwhere(np.isnan(decoded_trajectories.detach().cpu().numpy()))) > 1:
            error_message = "Found NaNs during decoding step."
            raise ValueError(error_message)

        return TrajectoryDecoderOutput(
            decoded_trajectories=decoded_trajectories,
            mode_probabilities=F.softmax(mode_probabilities, dim=-1),
            mode_logits=mode_probabilities,
        )
