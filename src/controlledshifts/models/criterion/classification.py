"""Classification criteria for safety and causal prediction tasks."""

import torch
from omegaconf import DictConfig, ListConfig

from controlledshifts.models.criterion.base_criterion import Criterion
from controlledshifts.schemas.output_schemas import ModelOutput


class SafetyClassification(Criterion):
    """Multi-class safety classification loss."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.classification_weight = config.get("classification_weight", 1.0)
        self.safety_type = config.get("safety_type", "individual")  # "individual" or "interaction"

        # CrossEntropyLoss supports multi-class classification
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Compute multi-class cross-entropy loss for safety labels.

        Notation:
            B: batch size
            N: number of elements per batch
            C: number of classes

        Args:
            model_output (ModelOutput): Structured model outputs.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        safety_output = model_output.safety_output
        if self.safety_type == "individual":
            gt = safety_output.individual_safety_gt.value
            logits = safety_output.individual_safety_logits.value
            mask = safety_output.individual_safety_mask.value
        elif self.safety_type == "interaction":
            gt = safety_output.interaction_safety_gt.value
            logits = safety_output.interaction_safety_logits.value
            mask = safety_output.interaction_safety_mask.value
        else:
            error_msg = f"Unknown safety_type: {self.safety_type}. Supported types are 'individual' and 'interaction'."
            raise ValueError(error_msg)

        # Ground truth labels: (B, N) → (B*N,)
        gt = gt.view(-1).long()

        # Logits: (B, N, C) → (B*N, C)
        logits = logits.view(-1, logits.shape[-1])

        # Filter out invalid and padded entries using the validity mask if provided.
        valid = mask.view(-1).bool()
        gt = gt[valid]
        logits = logits[valid]

        if gt.numel() == 0:
            return gt.new_tensor(0.0)

        loss = self.loss_function(logits, gt)
        return self.classification_weight * loss.mean()


class CausalClassification(Criterion):
    """Multi-class causal classification loss."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.classification_weight = config.get("classification_weight", 1.0)

        # CrossEntropyLoss supports multi-class classification
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Compute multi-class cross-entropy loss for causal labels.

        Notation:
            B: batch size
            N: number of elements per batch
            C: number of classes

        Args:
            model_output (ModelOutput): Structured model outputs.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        causal_output = model_output.causal_output

        # Ground truth labels: (B, N) → (B*N,)
        gt = causal_output.causal_gt.value.view(-1).long()

        # Logits: (B, N, C) → (B*N, C)
        logits = causal_output.causal_logits.value.view(
            -1,
            causal_output.causal_logits.value.shape[-1],
        )

        # Filter out invalid and padded entries using the validity mask if provided.
        if causal_output.causal_mask is not None:
            valid = causal_output.causal_mask.value.view(-1).bool()
            gt = gt[valid]
            logits = logits[valid]

        if gt.numel() == 0:
            return gt.new_tensor(0.0)

        loss = self.loss_function(logits, gt)
        return self.classification_weight * loss.mean()


class FocalSafetyClassification(Criterion):
    """Focal Multi-class safety classification loss."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.classification_weight = config.get("classification_weight", 1.0)
        self.safety_type = config.get("safety_type", "individual")  # "individual" or "interaction"
        self.num_classes = config.get("num_classes", 2)

        self.gamma = config.get("gamma", 2.0)
        match self.safety_type:
            case "individual":
                alpha = config.get("individual_alpha", None)
            case "interaction":
                alpha = config.get("interaction_alpha", None)
            case _:
                error_message = (
                    f"Unknown safety_type: {self.safety_type}. Supported types are 'individual' and 'interaction'."
                )
                raise ValueError(error_message)

        if alpha is None:
            # Default will be set to 1.0 for all classes, which means no class weighting.
            self.alpha = torch.ones(self.num_classes, dtype=torch.float32)
        elif isinstance(alpha, (list, tuple, ListConfig)):
            assert len(alpha) == self.num_classes, (
                f"Need a per-class alpha value. Num classes {self.num_classes}, num alphas: {len(alpha)}"
            )
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            # If a single value is provided, broadcast it across classes.
            self.alpha = torch.tensor([alpha] * self.num_classes, dtype=torch.float32)

        # CrossEntropyLoss supports multi-class classification
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Compute focal multi-class cross-entropy loss for safety labels.

        Notation:
            B: batch size
            N: number of elements per batch
            C: number of classes

        Args:
            model_output (ModelOutput): Structured model outputs.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        safety_output = model_output.safety_output
        if self.safety_type == "individual":
            gt = safety_output.individual_safety_gt.value
            logits = safety_output.individual_safety_logits.value
            mask = safety_output.individual_safety_mask.value
        elif self.safety_type == "interaction":
            gt = safety_output.interaction_safety_gt.value
            logits = safety_output.interaction_safety_logits.value
            mask = safety_output.interaction_safety_mask.value
        else:
            error_msg = f"Unknown safety_type: {self.safety_type}. Supported types are 'individual' and 'interaction'."
            raise ValueError(error_msg)

        # Ground truth labels: (B, N) → (B*N,)
        gt = gt.view(-1).long()

        # Logits: (B, N, C) → (B*N, C)
        logits = logits.view(-1, logits.shape[-1])

        # Filter out invalid and padded entries using the validity mask if provided.
        valid = mask.view(-1).bool()
        gt = gt[valid]
        logits = logits[valid]

        if gt.numel() == 0:
            return gt.new_tensor(0.0)

        ce_loss = self.loss_function(logits, gt)
        pt = torch.exp(-ce_loss)

        # Get alpha value per sample
        alpha = self.alpha.to(logits.device)
        alpha_targets = alpha[gt]

        focal_loss = alpha_targets * (1 - pt) ** self.gamma * ce_loss
        return self.classification_weight * focal_loss.mean()


class FocalCausalClassification(Criterion):
    """Focal loss for class-imbalanced causal classification."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.classification_weight = config.get("classification_weight", 1.0)
        self.gamma = config.get("gamma", 2.0)

        self.num_classes = config.get("num_classes", 2)

        self.alpha = config.get("alpha", [0.25, 1.0])
        assert self.alpha is not None, "Alpha value(s) must be provided as a single or per-class value."
        if isinstance(self.alpha, (list, tuple, ListConfig)):
            assert len(self.alpha) == self.num_classes, (
                f"Need a per-class alpha value. Num classes {self.num_classes}, num alphas: {len(self.alpha)}"
            )
            self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        else:
            # If a single value is provided, broadcast it across classes.
            self.alpha = torch.tensor([self.alpha] * self.num_classes, dtype=torch.float32)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Compute focal cross-entropy loss for class-imbalanced causal labels.

        Reference: https://arxiv.org/pdf/1708.02002

        Notation:
            B: batch size
            N: number of elements per batch
            C: number of classes

        Args:
            model_output (ModelOutput): Structured model outputs.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        causal_output = model_output.causal_output

        # Target has shape (B, N)
        gt = causal_output.causal_gt.value.view(-1).long()

        # Logits has shape (B, N, C)
        logits = causal_output.causal_logits.value.view(-1, 2)

        # Filter out invalid and padded entries using the validity mask if provided.
        if causal_output.causal_mask is not None:
            valid = causal_output.causal_mask.value.view(-1).bool()
            gt = gt[valid]
            logits = logits[valid]

        if gt.numel() == 0:
            return gt.new_tensor(0.0)

        # Apply the cross entropy loss
        ce_loss = self.loss_function(logits, gt)
        pt = torch.exp(-ce_loss)

        # Apply alpha weighting.
        self.alpha = self.alpha.to(logits.device)

        # Get alpha value per sample
        alpha = self.alpha[gt]
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        return self.classification_weight * focal_loss.mean()
