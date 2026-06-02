"""Base interface for loss criteria."""

from abc import ABC, abstractmethod

from omegaconf import DictConfig
from torch import Tensor, nn

from controlledshifts.schemas.output_schemas import ModelOutput


class Criterion(nn.Module, ABC):
    """Abstract base class for all loss criteria."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, model_output: ModelOutput) -> Tensor:
        """Compute the loss for a batch of model outputs."""
