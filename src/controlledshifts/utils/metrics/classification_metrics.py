import torch

from controlledshifts.utils.constants import EPSILON


def compute_binary_confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Calculates the confusion matrix between predictions and labels.

    Args:
        labels (torch.Tensor(B, N)): Tensor of target values.
        predictions (torch.Tensor(B, N)): Tensor of predicted values.

    Returns:
        true_positives (torch.Tensor(B)): True postive counts per sample.
        true_negatives (torch.Tensor(B)): True negative counts per sample.
        false_positives (torch.Tensor(B)): False positive counts per sample.
        false_negatives (torch.Tensor(B)): False negative counts per sample.
    """
    assert predictions.shape == labels.shape, "Shapes of predictions and labels must be the same."

    # Calculating True Positives
    true_positives = ((predictions == 1) & (labels == 1)).sum(dim=-1).float()

    # Calculating True Negatives
    true_negatives = ((predictions == 0) & (labels == 0)).sum(dim=-1).float()

    # Calculating False Positives
    false_positives = ((predictions == 1) & (labels == 0)).sum(dim=-1).float()

    # Calculating False Negatives
    false_negatives = ((predictions == 0) & (labels == 1)).sum(dim=-1).float()

    return true_positives, true_negatives, false_positives, false_negatives


def compute_multiclass_accuracy(
    labels: torch.Tensor, predictions: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, ...]:
    """Computes the precision, recall and F1 scores for multiclass classification.

    Args:
        labels (torch.Tensor(B, N)): Tensor of target values.
        predictions (torch.Tensor(B, N)): Tensor of predicted values.
        num_classes (int): number of classes.

    Returns:
        precision (torch.Tensor(B)): Accuracy of positive predictions.
        recall (torch.Tensor(B)): Sensitivity of possitive predictions.
        f1_score (torch.Tensor(B)): Balance between precision and recall.
    """
    assert predictions.shape == labels.shape, "Shapes of predictions and labels must be the same."

    batch_size = labels.shape[0]
    confusion_matrix = torch.zeros((batch_size, num_classes, num_classes), dtype=torch.float32, device=labels.device)
    for i in range(batch_size):
        for target, prediction in zip(labels[i].view(-1), predictions[i].view(-1), strict=False):
            confusion_matrix[i, target.long(), prediction.long()] += 1

    true_positives = confusion_matrix.diagonal(dim1=1, dim2=2)
    false_positives = confusion_matrix.sum(dim=1) - true_positives
    false_negatives = confusion_matrix.sum(dim=2) - true_positives

    precision = true_positives / (true_positives + false_positives + EPSILON)
    recall = true_positives / (true_positives + false_negatives + EPSILON)
    f1_score = 2 * (precision * recall) / (precision + recall + EPSILON)

    return precision.mean(dim=1), recall.mean(dim=1), f1_score.mean(dim=1)


def compute_accuracy(labels: torch.Tensor, predictions: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Computes the precision, recall and F1 scores.

    Args:
        labels (torch.Tensor(B, N)): Tensor of target values.
        predictions (torch.Tensor(B, N)): Tensor of predicted values.

    Returns:
        precision (torch.Tensor(B)): Accuracy of positive predictions.
        recall (torch.Tensor(B)): Sensitivity of possitive predictions.
        f1_score (torch.Tensor(B)): Balance between precision and recall.
    """
    true_positives, _, false_positives, false_negatives = compute_binary_confusion_matrix(labels, predictions)

    # Precision
    precision = true_positives / (true_positives + false_positives + EPSILON)

    # Recall
    recall = true_positives / (true_positives + false_negatives + EPSILON)

    # F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + EPSILON)
    return precision, recall, f1_score
