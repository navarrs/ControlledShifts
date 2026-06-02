import numpy as np
from numpy.typing import NDArray

from controlledshifts.utils.constants import EPSILON, LARGE_FLOAT


def compute_cosine_similarity(samples: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes cosine similarity between each embedding and a centroid, normalized to [0, 1].

    Args:
        samples (NDArray[np.float64]): array of shape (num_samples, embedding_dim).
        target (NDArray[np.float64]): array of shape (embedding_dim,).

    Returns:
        similarities (NDArray[np.float64]): per-sample cosine similarities in [0, 1].
    """
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    norms = np.where(norms < EPSILON, LARGE_FLOAT, norms)
    target_norm = np.linalg.norm(target)
    target_norm = target_norm if target_norm >= EPSILON else LARGE_FLOAT
    cosine_sim = np.clip((samples / norms) @ (target / target_norm), -1.0, 1.0)
    # Map from [-1, 1] to [0, 1]
    return ((cosine_sim + 1.0) / 2.0).astype(np.float64)


def compute_pairwise_cosine_similarity(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes the pairwise cosine similarity matrix for a set of embeddings, normalized to [0, 1].

    Args:
        samples: array of shape (num_samples, embedding_dim).

    Returns:
        sim_matrix: array of shape (num_samples, num_samples) with values in [0, 1].
    """
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    norms = np.where(norms < EPSILON, LARGE_FLOAT, norms)
    normalized = samples / norms
    cosine_sim = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return ((cosine_sim + 1.0) / 2.0).astype(np.float64)


def compute_pairwise_hamming_similarity(sequences: NDArray[np.int32]) -> NDArray[np.float64]:
    """Computes the pairwise Hamming similarity matrix for a set of integer sequences.

    Hamming similarity between two sequences is the fraction of positions where their values match.

    Args:
        sequences: array of shape (N, L) with integer values.

    Returns:
        sim_matrix: array of shape (N, N) with values in [0, 1].
    """
    length = sequences.shape[1]
    matches = (sequences[:, None, :] == sequences[None, :, :]).sum(axis=-1)
    return (matches / length).astype(np.float64)


def compute_pairwise_jaccard_similarity(sequences: NDArray[np.int32]) -> NDArray[np.float64]:
    """Computes the pairwise Jaccard similarity matrix for a set of integer sequences.

    Jaccard similarity between two sequences is the size of their value-set intersection divided by the size of
    their union, treating each sequence as a set (positional order is ignored).

    Args:
        sequences: array of shape (N, L) with integer values.

    Returns:
        sim_matrix: array of shape (N, N) with values in [0, 1].
    """
    num_labels = int(sequences.max()) + 1
    # indicator[i, k] = 1 if value k appears in sequences[i]
    indicator = np.zeros((len(sequences), num_labels), dtype=np.float64)
    for i, row in enumerate(sequences):
        indicator[i, row] = 1.0
    intersections = indicator @ indicator.T  # (N, N)
    set_sizes = indicator.sum(axis=1)  # (N,)
    unions = set_sizes[:, None] + set_sizes[None, :] - intersections  # (N, N)
    return np.where(unions > 0, intersections / unions, 0.0).astype(np.float64)


def compute_jaccard_index(a: set[int], b: set[int]) -> float:
    """Computes the Jaccard Index (Intersection over Union) between two sets.

    Args:
        a (set[int]): set of integer values.
        b (set[int]): set of integer values.

    Returns:
        jaccard_index (float): the intersection over union value [0, 1].
    """
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0


def compute_hamming_distance(a: list[int], b: list[int], *, return_inverse: bool = False) -> float:
    """Computes the Hamming distance between two categorical vectors.  It sums '1' for every position where elements are
    not equal and '0' otherwise.

    Args:
        a (list or tuple): The first categorical vector.
        b (list or tuple): The second categorical vector.
        return_inverse (bool): If 'True', returns the inverse of the Hamming distance.

    Returns:
        int: The number of mismatched positions, or -1 if lengths are unequal.
    """
    if len(a) != len(b):
        error_message = "Error: Vectors must have the same length."
        raise ValueError(error_message)
    hamming_distance = sum(1 for ai, bi in zip(a, b, strict=False) if ai != bi) / len(a)
    if return_inverse:
        return 1.0 - hamming_distance
    return hamming_distance
