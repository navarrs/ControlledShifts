"""Utilities for computing MTR intention-point cluster centers.

MTR uses per-agent-type k-means cluster centers of trajectory endpoints as
initial intention points for its motion queries.  Call
:func:`compute_and_save_mtr_intention_points` after creating the training
DataLoader to generate (or reuse) the ``.pkl`` file referenced by
``model.config.motion_decoder.intention_points_file``.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch.multiprocessing
from characterization.utils.common import AgentType
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

from controlledshifts.utils.pylogger import get_pylogger


_LOGGER = get_pylogger(__name__)

# Agent type integer codes (must match characterization.utils.common.AgentType)
_TYPE_NAMES: dict[int, str] = {
    AgentType.TYPE_VEHICLE.value: "VEHICLE",
    AgentType.TYPE_PEDESTRIAN.value: "PEDESTRIAN",
    AgentType.TYPE_CYCLIST.value: "CYCLIST",
}


def _collect_endpoints(
    dataloader: DataLoader, object_types: list[str], max_samples: int | None = None
) -> dict[str, NDArray[np.float32]]:
    """Iterate the dataloader and collect trajectory endpoints per agent type.

    Args:
        dataloader: DataLoader yielding batches whose ``input_dict`` contains ``center_gt_trajs``,
            ``center_gt_trajs_mask``, and ``center_objects_type``.
        object_types: agent type names to collect.
        max_samples: stop after this many total samples. ``None`` = full dataset.

    Returns:
        Dict mapping type name → (N, 2) float32 array of (x, y) endpoints.
    """
    endpoints: dict[str, list[NDArray[np.float32]]] = {t: [] for t in object_types}
    type_name_by_code = {v: k for k, v in _TYPE_NAMES.items()}
    type_codes = {type_name_by_code[t] for t in object_types if t in type_name_by_code}

    total = 0
    for batch in tqdm(dataloader, desc="Collecting endpoints"):
        input_dict = batch["input_dict"]

        center_gt_trajs = input_dict["center_gt_trajs"].numpy()  # (B, F, 4)
        center_gt_mask = input_dict["center_gt_trajs_mask"].numpy()  # (B, F)
        center_types = input_dict["center_objects_type"]

        center_types = center_types.numpy() if hasattr(center_types, "numpy") else np.asarray(center_types)

        n_batch = center_gt_trajs.shape[0]
        for i in range(n_batch):
            type_code = int(center_types[i])
            if type_code not in type_codes:
                continue
            type_name = _TYPE_NAMES[type_code]

            valid_steps = np.where(center_gt_mask[i])[0]
            if len(valid_steps) == 0:
                continue
            endpoint = center_gt_trajs[i, valid_steps[-1], :2]  # (x, y)
            endpoints[type_name].append(endpoint)

        total += n_batch
        if max_samples is not None and total >= max_samples:
            break

    return {
        t: np.stack(pts, axis=0).astype(np.float32) if pts else np.zeros((0, 2), dtype=np.float32)
        for t, pts in endpoints.items()
    }


def _kmeans_cluster(points: NDArray[np.floating[Any]], k: int, seed: int = 0) -> NDArray[np.float32]:
    """Return k-means cluster centers for the given points.

    Uses ``sklearn.cluster.KMeans`` when available, otherwise falls back to a simple Lloyd's-algorithm implementation.

    Args:
        points: (N, 2) trajectory endpoints.
        k: number of clusters.
        seed: random seed.

    Returns:
        centers: (k, 2) cluster centers.
    """
    if len(points) < k:
        _LOGGER.warning("Only %d samples available for %d clusters. Padding centers with zeros.", len(points), k)
        centers = np.zeros((k, 2), dtype=np.float32)
        centers[: len(points)] = points
        return centers

    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    km.fit(points)
    return km.cluster_centers_.astype(np.float32)


def compute_and_cache_intention_points(train_loader: DataLoader, cfg: DictConfig) -> str:
    """Compute MTR intention-point clusters and save them to disk.

    Reads ``cfg.mtr_intention_points_output``, ``cfg.mtr_intention_points_num_clusters``, and
    ``cfg.mtr_intention_points_max_samples`` for configuration.

    Args:
        train_loader: training DataLoader (used to collect trajectory endpoints).
        cfg: Hydra config (global ``_global_`` package).

    Returns:
        Path to the saved ``.pkl`` file (absolute string).
    """
    object_type_config = OmegaConf.select(cfg, "model.config.motion_decoder.object_type")
    if object_type_config is not None:
        container = OmegaConf.to_container(object_type_config, resolve=True)
        object_types: list[str] = (
            list(container) if isinstance(container, list) else ["VEHICLE", "PEDESTRIAN", "CYCLIST"]
        )
    else:
        object_types: list[str] = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]

    num_clusters: int = OmegaConf.select(cfg, "model.config.intention_points_num_clusters")
    max_samples: int | None = OmegaConf.select(cfg, "model.config.intention_points_max_samples")
    out_path = Path(OmegaConf.select(cfg, "model.config.intention_points_file", default="./meta/intention_points.pkl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Switch to file_system sharing to avoid "too many open files" when iterating
    # large DataLoaders with multiple workers (default file_descriptor strategy exhausts ulimit).
    torch.multiprocessing.set_sharing_strategy("file_system")

    _LOGGER.info("Collecting trajectory endpoints for MTR intention points (types: %s) ...", object_types)
    endpoints = _collect_endpoints(train_loader, object_types, max_samples=max_samples)

    for t, pts in endpoints.items():
        _LOGGER.info("  %s: %d samples collected", t, len(pts))

    _LOGGER.info("Running k-means with k=%d per agent type ...", num_clusters)
    intention_points = {}
    for t, pts in endpoints.items():
        _LOGGER.info("  Clustering %s ...", t)
        centers = _kmeans_cluster(pts, k=num_clusters)
        intention_points[t] = centers
        _LOGGER.info(
            "    Center range: x=[%.2f, %.2f]  y=[%.2f, %.2f]",
            centers[:, 0].min(),
            centers[:, 0].max(),
            centers[:, 1].min(),
            centers[:, 1].max(),
        )

    with out_path.open("wb") as f:
        pickle.dump(intention_points, f)

    _LOGGER.info("MTR intention points saved to: %s", out_path)
    return str(out_path.resolve())
