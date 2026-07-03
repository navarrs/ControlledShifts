"""Sample selection strategies and their shared helpers.

Each strategy returns a dict with the scenario IDs to ``keep`` and ``drop`` (the ``drop`` list is the blacklist consumed
by the dataset at train time). Model-independent strategies (e.g. ``random_selection``) operate on the training scenario
IDs; embedding-based strategies operate on the cached per-scenario model outputs. The helpers in :mod:`common` are the
shared primitives the group- and embedding-based strategies build on.
"""

from .common import (
    SampleSelection,
    aggregate_selected_samples,
    compute_proportional_number_to_drop,
    greedy_select_from_sim_matrix,
    make_group_result,
    sort_ids_by_score,
    weighted_sorting,
    weighted_sorting_gumbel,
)
from .random_drop import random_selection


__all__ = [
    "SampleSelection",
    "aggregate_selected_samples",
    "compute_proportional_number_to_drop",
    "greedy_select_from_sim_matrix",
    "make_group_result",
    "random_selection",
    "sort_ids_by_score",
    "weighted_sorting",
    "weighted_sorting_gumbel",
]
