"""Builder for the canonical agent-centric (AC) cache (Stage C of the data pipeline).

Processes one variant's canonical raw scenario store (``variants/<variant>/<scenario_id>.pkl``, decoded Waymo dicts)
into a per-scenario agent-centric cache (``ac_cache/<variant>/<profile_hash>/scenarios/<scenario_id>.pkl``), keyed by
the dataset's processing profile. Each raw dict is repacked to a ``Scenario`` (``waymo.repacker.load_scenario``) before
the transform. The cache is split-agnostic and built once per (variant, processing profile); benchmark split JSONs
later select which scenario IDs to load. Two bookkeeping files are written alongside the per-scenario files:

- ``_index.pkl``: maps each scenario_id to ``{num_records, kalman_difficulty, rel_path}`` so the loader can assemble its
  flat sample list without opening every file.
- ``_profile.json``: the resolved processing-profile key/values the hash was computed from; the loader hard-errors if
  it does not match the current config, guarding against silently reading a mismatched cache.

The actual ``Scenario`` -> records transform lives in ``agent_centric_processor.AgentCentricProcessor``; this module
only fans the work across processes and writes the index/profile.
"""

import json
import pickle  # nosec B403
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from controlledshifts.datasets.agent_centric_processor import AgentCentricProcessor
from controlledshifts.datasets.waymo.repacker import load_scenario
from controlledshifts.utils import pylogger


_LOGGER = pylogger.get_pylogger(__name__)

# Per-worker state, initialized once per process by ``_init_worker`` (avoids pickling the processor and its optional
# SafeShift sub-processors across the pool).
_WORKER: dict[str, Any] = {}


def scenario_filepaths(variant_dir: Path) -> list[Path]:
    """Returns the variant's scenario ``.pkl`` files, excluding bookkeeping (``_*``) and ``*infos*`` files."""
    return sorted(fp for fp in variant_dir.glob("*.pkl") if not fp.name.startswith("_") and "infos" not in fp.stem)


def _init_worker(config: DictConfig, scenarios_dir: Path, overwrite: bool) -> None:  # noqa: FBT001
    """Pool initializer: build a processor and stash the per-scenario output dir / overwrite flag for this worker."""
    _WORKER["processor"] = AgentCentricProcessor(config)
    _WORKER["scenarios_dir"] = scenarios_dir
    _WORKER["overwrite"] = overwrite


def _process_chunk(filepaths: list[Path]) -> dict[str, dict[str, Any]]:
    """Processes a chunk of variant ``Scenario`` files into per-scenario record pickles; returns a partial index.

    Scenarios that produce no records (e.g. all tracks masked out by a perturbation) are recorded with
    ``num_records == 0`` and no file is written.
    """
    processor: AgentCentricProcessor = _WORKER["processor"]
    scenarios_dir: Path = _WORKER["scenarios_dir"]
    overwrite: bool = _WORKER["overwrite"]

    index: dict[str, dict[str, Any]] = {}
    for filepath in filepaths:
        scenario_id = filepath.stem
        output_path = scenarios_dir / f"{scenario_id}.pkl"
        rel_path = f"scenarios/{scenario_id}.pkl"

        if output_path.exists() and not overwrite:
            with output_path.open("rb") as f:
                records = pickle.load(f)  # nosec B301
        else:
            scenario = load_scenario(filepath)
            records = processor.process(scenario)
            if records:
                with output_path.open("wb") as f:
                    pickle.dump(records, f)

        if not records:
            index[scenario_id] = {
                "num_records": 0,
                "kalman_difficulty": np.zeros((0, 3), dtype=np.float32),
                "rel_path": rel_path,
            }
            continue

        kalman_difficulty = np.stack([np.asarray(record["kalman_difficulty"]) for record in records])
        index[scenario_id] = {
            "num_records": len(records),
            "kalman_difficulty": kalman_difficulty,
            "rel_path": rel_path,
        }
    return index


def build_variant_cache(  # noqa: PLR0913
    processor: AgentCentricProcessor,
    variant: str,
    variant_dir: Path,
    cache_dir: Path,
    *,
    num_workers: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Builds the agent-centric cache for a single variant and writes its index and profile files.

    Args:
        processor (AgentCentricProcessor): processor providing the processing profile and the agent-centric transform.
        variant (str): variant name (e.g. ``base``, ``remove_noncausal``).
        variant_dir (Path): directory of canonical raw scenario ``.pkl`` files (decoded Waymo dicts) for the variant.
        cache_dir (Path): output cache directory (``ac_cache/<variant>/<profile_hash>``).
        num_workers (int): number of parallel worker processes.
        overwrite (bool): if False, reuse already-cached per-scenario files. Defaults to False.

    Returns:
        dict[str, Any]: the written index.

    Raises:
        FileNotFoundError: if ``variant_dir`` does not exist.
    """
    if not variant_dir.exists():
        error_message = f"Variant directory not found: {variant_dir}"
        raise FileNotFoundError(error_message)

    scenarios_dir = cache_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    filepaths = scenario_filepaths(variant_dir)
    _LOGGER.info(
        "Building agent-centric cache for variant '%s': %d scenarios -> %s", variant, len(filepaths), cache_dir
    )

    scenarios: dict[str, dict[str, Any]] = {}
    if filepaths:
        workers = max(1, min(num_workers, len(filepaths)))
        chunks = [list(chunk) for chunk in np.array_split(filepaths, workers) if len(chunk)]
        with Pool(
            processes=workers, initializer=_init_worker, initargs=(processor.config, scenarios_dir, overwrite)
        ) as pool:
            for partial_index in pool.map(_process_chunk, chunks):
                scenarios.update(partial_index)

    empty_scenarios = sorted(sid for sid, info in scenarios.items() if info["num_records"] == 0)
    index = {
        "variant": variant,
        "profile_alias": processor.config.get("profile_alias", "unknown"),
        "profile_hash": processor.processing_profile_hash(),
        "format": "pkl",
        "scenarios": scenarios,
        "empty_scenarios": empty_scenarios,
    }
    with (cache_dir / "_index.pkl").open("wb") as f:
        pickle.dump(index, f)
    with (cache_dir / "_profile.json").open("w") as f:
        json.dump(processor.processing_profile(), f, indent=2, sort_keys=True, default=str)

    _LOGGER.info(
        "Built variant '%s': %d scenarios (%d empty) at %s", variant, len(scenarios), len(empty_scenarios), cache_dir
    )
    return index
