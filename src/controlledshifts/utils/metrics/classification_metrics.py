import torch

from controlledshifts.utils.constants import EPSILON


def compute_binary_confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Calculates the confusion matrix between predictions and labels.

    Args:
        labels: target values, shape (B, N).
        predictions: predicted values, shape (B, N).

    Returns:
        true_positives: true-positive counts per sample, shape (B).
        true_negatives: true-negative counts per sample, shape (B).
        false_positives: false-positive counts per sample, shape (B).
        false_negatives: false-negative counts per sample, shape (B).
    """
    assert predictions.shape == labels.shape, "Shapes of predictions and labels must be the same."

    true_positives = ((predictions == 1) & (labels == 1)).sum(dim=-1).float()
    true_negatives = ((predictions == 0) & (labels == 0)).sum(dim=-1).float()
    false_positives = ((predictions == 1) & (labels == 0)).sum(dim=-1).float()
    false_negatives = ((predictions == 0) & (labels == 1)).sum(dim=-1).float()

    return true_positives, true_negatives, false_positives, false_negatives


def compute_multiclass_accuracy(
    labels: torch.Tensor, predictions: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, ...]:
    """Computes the precision, recall and F1 scores for multiclass classification.

    Args:
        labels: target values, shape (B, N).
        predictions: predicted values, shape (B, N).
        num_classes: number of classes.

    Returns:
        precision: accuracy of positive predictions, shape (B).
        recall: sensitivity of positive predictions, shape (B).
        f1_score: balance between precision and recall, shape (B).
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
        labels: target values, shape (B, N).
        predictions: predicted values, shape (B, N).

    Returns:
        precision: accuracy of positive predictions, shape (B).
        recall: sensitivity of positive predictions, shape (B).
        f1_score: balance between precision and recall, shape (B).
    """
    true_positives, _, false_positives, false_negatives = compute_binary_confusion_matrix(labels, predictions)

    precision = true_positives / (true_positives + false_positives + EPSILON)
    recall = true_positives / (true_positives + false_negatives + EPSILON)
    f1_score = 2 * (precision * recall) / (precision + recall + EPSILON)
    return precision, recall, f1_score
