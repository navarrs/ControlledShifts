r"""Benchmark creation for the Ego-SafeShift benchmark.

Ranks scenarios by safety score and assigns them to training/validation/testing splits following split_ratios. The
hardest (highest-scoring) scenarios form the test set; the remainder is split into train/val.

The per-scenario scores come from a CSV (``scenario_score_mapping_filepath``) with columns ``scenario_ids`` and the
score column named by ``score_type``. When that CSV is not provided or does not exist, the scores are computed here from
the scenarios under ``input_data_path`` using the SafeShift characterization API (see
``controlledshifts.datasets.scenario_scorer.ScenarioScorer`` and the shared ``scoring`` config group), and written to
``${splits_path}/ego_safeshift/scenario_to_scores_mapping_<hash>.csv`` (keyed by a hash of the scoring config so
different scoring configurations do not clobber each other).

Each scoring variant gets its own split JSON: the split filename is ``split_name`` when set, otherwise an auto-derived
``ego_safeshift_<tag>`` keyed by the score source, ``score_type``, ``split_ratios`` and ``seed`` (see
``ego_safeshift_split_name``), so multiple score CSVs each produce a distinct split file.

Example usage:

    # With a precomputed score CSV:
    uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift \\
        input_data_path=/datasets/waymo/processed/mini_causal \\
        scenario_score_mapping_filepath=meta/ego-safeshift/scores_8/scenario_to_scores_mapping.csv

    # Without a CSV (scores are computed and cached, split saved as ego_safeshift_<tag>.json):
    uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift \\
        input_data_path=/datasets/waymo/processed/mini_causal split_name=ego_safeshift_scores8

See configs/benchmark/ego_safeshift.yaml for all available options.
"""

import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from characterization.schemas import Score
from numpy.random import Generator, default_rng
from omegaconf import DictConfig, OmegaConf

from controlledshifts.benchmarks.common import (
    BenchmarkSplit,
    collect_scenario_filepaths,
    split_ids_by_score,
    split_mapping_to_lists,
)
from controlledshifts.datasets.scenario_scorer import ScenarioScorer
from controlledshifts.datasets.waymo.repacker import load_scenario
from controlledshifts.utils.pylogger import get_pylogger


_LOGGER = get_pylogger(__name__)

# Columns written to a computed score CSV; the safeshift column is the default `score_type` for ranking.
SCORE_COLUMNS: dict[str, str] = {
    "safeshift": "gt_critical_continuous_safeshift",
    "individual": "gt_critical_continuous_individual",
    "interaction": "gt_critical_continuous_interaction",
}

# Set by ``_init_score_worker`` to avoid pickling the scorer (and its SafeShift sub-processors) across the pool.
_WORKER: dict[str, Any] = {}


def _scene_score(score: Score | None) -> float | None:
    """Returns a Score's scalar scene score, or None when unavailable."""
    return None if score is None or score.scene_score is None else float(score.scene_score)


def _init_score_worker(scoring_config: DictConfig) -> None:
    """Pool initializer: build one ScenarioScorer per worker."""
    _WORKER["scorer"] = ScenarioScorer(scoring_config)


def _score_chunk(filepaths: list[Path]) -> list[dict[str, Any]]:
    """Scores a chunk of scenario ``.pkl`` files into score rows; scenarios that fail to score are skipped."""
    scorer: ScenarioScorer = _WORKER["scorer"]
    rows: list[dict[str, Any]] = []
    for filepath in filepaths:
        try:
            scores = scorer.compute(load_scenario(filepath))
        except Exception:
            _LOGGER.exception("error scoring scenario: %s", filepath.stem)
            continue
        rows.append(
            {
                "scenario_ids": filepath.stem,
                SCORE_COLUMNS["safeshift"]: _scene_score(scores.safeshift_scores),
                SCORE_COLUMNS["individual"]: _scene_score(scores.individual_scores),
                SCORE_COLUMNS["interaction"]: _scene_score(scores.interaction_scores),
            }
        )
    return rows


def compute_scores_dataframe(scoring_config: DictConfig, input_data_path: Path, num_workers: int) -> pd.DataFrame:
    """Scores every scenario under ``input_data_path`` in parallel and returns a scenario-to-scores DataFrame.

    Args:
        scoring_config: Focused scoring config (the ``scenario_characterization``/``conflict_points``/``closest_lanes``
            blocks) used to build a :class:`ScenarioScorer`.
        input_data_path: Root directory of scenario ``.pkl`` files to score.
        num_workers: Number of parallel worker processes.

    Returns:
        DataFrame with a ``scenario_ids`` column and the three :data:`SCORE_COLUMNS` score columns.

    Raises:
        ValueError: If no scenarios are found under ``input_data_path``.
    """
    filepaths = collect_scenario_filepaths(input_data_path)
    if not filepaths:
        error_message = f"No scenarios found under {input_data_path}; cannot compute Ego-SafeShift scores."
        raise ValueError(error_message)

    workers = max(1, min(num_workers, len(filepaths)))
    chunks = [list(chunk) for chunk in np.array_split(np.asarray(filepaths, dtype=object), workers) if len(chunk)]
    _LOGGER.info("Scoring %d scenarios with %d workers", len(filepaths), workers)
    with multiprocessing.Pool(workers, initializer=_init_score_worker, initargs=(scoring_config,)) as pool:
        rows = [row for chunk_rows in pool.map(_score_chunk, chunks) for row in chunk_rows]
    return pd.DataFrame(rows)


def _validate_score_type(scenario_scores_df: pd.DataFrame, score_type: str) -> None:
    """Raises a ValueError if a computed score DataFrame lacks the requested ``score_type`` column."""
    if score_type not in scenario_scores_df.columns:
        error_message = (
            f"score_type '{score_type}' is not among the computed score columns {list(SCORE_COLUMNS.values())}. "
            f"Use one of those, or provide a scenario_score_mapping_filepath that contains '{score_type}'."
        )
        raise ValueError(error_message)


def _scoring_hash(config: DictConfig) -> str:
    """Returns a short stable hash of the resolved scoring config (keys the computed score CSV filename)."""
    payload = json.dumps(OmegaConf.to_container(config.scoring, resolve=True), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _resolve_scores_csv_path(config: DictConfig) -> Path:
    """Returns the default path for a computed score CSV, keyed by the scoring-config hash."""
    return Path(config.splits_path) / "ego_safeshift" / f"scenario_to_scores_mapping_{_scoring_hash(config)}.csv"


def _score_source_id(config: DictConfig) -> str:
    """Identifies the score source: the provided CSV's content hash, or the scoring-config hash for the fallback."""
    csv_path = config.get("scenario_score_mapping_filepath", None)
    if csv_path:
        csv_path = Path(csv_path)
        if csv_path.exists():
            return hashlib.sha256(csv_path.read_bytes()).hexdigest()[:12]
    return _scoring_hash(config)


def ego_safeshift_split_name(config: DictConfig) -> str:
    """Returns the split filename stem for an Ego-SafeShift run.

    Uses ``config.split_name`` verbatim when set; otherwise auto-derives ``ego_safeshift_<tag>`` where ``tag`` hashes
    everything that determines the split (score source, score_type, split_ratios, seed), so distinct score CSVs /
    scoring configs each get their own split file.
    """
    explicit = config.get("split_name", None)
    if explicit:
        return str(explicit)
    payload = json.dumps(
        {
            "source": _score_source_id(config),
            "score_type": config.score_type,
            "split_ratios": list(config.split_ratios),
            "seed": config.seed,
        },
        sort_keys=True,
        default=str,
    )
    tag = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"ego_safeshift_{tag}"


def _load_or_compute_scores(config: DictConfig, input_data_path: Path) -> tuple[pd.DataFrame, bool]:
    """Loads the score CSV if available, else computes scores and writes a hashed CSV.

    Returns:
        The scenario-to-scores DataFrame and a flag that is True when scores were computed here (so on-disk scenarios
        that failed to score should be reported as invalid).
    """
    csv_path = config.get("scenario_score_mapping_filepath", None)
    csv_path = Path(csv_path) if csv_path else None
    if csv_path is not None and csv_path.exists():
        return pd.read_csv(csv_path), False

    output_csv = _resolve_scores_csv_path(config)
    if output_csv.exists() and not config.get("overwrite", False):
        _LOGGER.info("Reusing computed scores at %s (set overwrite=true to recompute)", output_csv)
        return pd.read_csv(output_csv), True

    scenario_scores_df = compute_scores_dataframe(config.scoring, input_data_path, config.num_workers)
    _validate_score_type(scenario_scores_df, config.score_type)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scenario_scores_df.to_csv(output_csv, index=False)
    _LOGGER.info("Computed and wrote scenario scores to %s", output_csv)
    return scenario_scores_df, True


def create_ego_safeshift_benchmark(config: DictConfig) -> BenchmarkSplit:
    """Creates the Ego-SafeShift benchmark split.

    Splits the dataset by safety score: the hardest (highest-scoring) scenarios form the test set and the remainder is
    split into training/validation, following split_ratios. Scores are read from ``scenario_score_mapping_filepath``
    or, when that is null/missing, computed from the scenarios under ``input_data_path`` (see module docstring).
    Scenarios in the score mapping but absent from the input directory -- and, when scores are computed here, on-disk
    scenarios that failed to score -- are recorded as invalid.

    Args:
        config: Hydra config with keys: input_data_path, scenario_score_mapping_filepath, score_type, split_ratios,
            seed, and (for the fallback) scoring, splits_path, num_workers, overwrite.

    Returns:
        The train/val/test benchmark split.
    """
    input_data_path = Path(config.input_data_path)
    random_generator: Generator = default_rng(config.seed)
    available_ids = {fp.stem for fp in collect_scenario_filepaths(input_data_path)}

    scenario_scores_df, computed = _load_or_compute_scores(config, input_data_path)
    scenario_ids = [Path(scenario_id).stem for scenario_id in scenario_scores_df["scenario_ids"].tolist()]
    split_by_id = split_ids_by_score(
        scenario_ids,
        scenario_scores_df[config.score_type].to_numpy(),
        tuple(config.split_ratios),
        random_generator,
        hardest_highest=True,
    )

    missing_from_input = {scenario_id for scenario_id in split_by_id if scenario_id not in available_ids}
    # When scores were computed here, every on-disk scenario was scored, so any unscored one is a scoring failure.
    failed_to_score = available_ids - set(scenario_ids) if computed else set()
    invalid = missing_from_input | failed_to_score
    placed = {scenario_id: split for scenario_id, split in split_by_id.items() if scenario_id in available_ids}
    return split_mapping_to_lists(placed, invalid=invalid)
