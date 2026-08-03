"""Locating trained runs and their cached model outputs on disk.

A run lives at ``<cache_root>/<dataset>/<model>/<timestamp>`` (i.e. ``paths.experiment_dir``) and its cached outputs at
``<run_dir>/batch_cache/<split>/<source>/<scenario_id>.pkl``. The runs themselves are enumerated from a W&B results
export, whose dataset/model/timestamp columns locate each run directory.

Written by ``run_model_cache_sweep``; read by anything that needs a trained run's cached outputs.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from controlledshifts.utils.constants import ModelStatus


# The splits `save_cache` writes during eval, named after the ModelStatus values.
CACHE_SPLITS = (ModelStatus.VALIDATION, ModelStatus.TEST)


@dataclass(frozen=True)
class Run:
    """One row of the results CSV, resolved onto its on-disk training run."""

    dataset: str  # paths.tag, hyphenated (e.g. "background-agents")
    model: str  # model.config.model_name (e.g. "wayformer")
    timestamp: str  # trailing path segment (e.g. "2026-06-13_20-43-20")
    run_dir: Path

    @property
    def paths_group(self) -> str:
        """Hydra `paths` group name, which is underscored where `paths.tag` is hyphenated."""
        return self.dataset.replace("-", "_")

    @property
    def experiment_dir(self) -> str:
        """Value to pin `paths.experiment_dir` to, so eval reads and writes inside this training run."""
        return f"{self.dataset}/{self.model}/{self.timestamp}"

    @property
    def name(self) -> str:
        """Short label for logs."""
        return f"{self.dataset}/{self.model}"


def read_runs(csv_filepath: Path, cache_root: Path) -> list[Run]:
    """Parses the results export into its finished runs, in CSV order.

    Args:
        csv_filepath: W&B results export listing the runs.
        cache_root: root holding the per-run directories (``paths.experiment_cache_path``).

    Returns:
        The finished runs, each resolved onto its on-disk run directory.
    """
    with csv_filepath.open(newline="") as f:
        rows = list(csv.DictReader(f))

    runs = []
    for row in rows:
        if row.get("State") != "finished":
            continue
        dataset, model, timestamp = row["_content.dataset"], row["_content.model"], row["_content.timestamp"]
        runs.append(
            Run(
                dataset=dataset,
                model=model,
                timestamp=timestamp,
                run_dir=cache_root / dataset / model / timestamp,
            )
        )
    return runs


def filter_runs(runs: list[Run], models: list[str] | None, benchmarks: list[str] | None) -> list[Run]:
    """Keeps only the runs matching the requested models and benchmarks (None keeps everything)."""
    if models:
        wanted_models = {model.strip() for model in models}
        runs = [run for run in runs if run.model in wanted_models]
    if benchmarks:
        # Accept either spelling: the paths group (`background_agents`) or the tag (`background-agents`).
        wanted_benchmarks = {benchmark.strip().replace("-", "_") for benchmark in benchmarks}
        runs = [run for run in runs if run.paths_group in wanted_benchmarks]
    return runs


def resolve_checkpoint(run: Run, ckpt: str) -> str:
    """Returns the checkpoint stem to pass to eval as ``ckpt_name``.

    Args:
        run: the training run to load a checkpoint from.
        ckpt: either ``best`` (the run's single ``epoch_XXX.ckpt``) or ``last``.

    Returns:
        The checkpoint file stem, e.g. ``epoch_057``.

    Raises:
        ValueError: if the run has no usable checkpoint, or if ``best`` is ambiguous.
    """
    ckpt_dir = run.run_dir / "ckpts"
    if ckpt == "last":
        if not (ckpt_dir / "last.ckpt").is_file():
            error_message = f"No last.ckpt in {ckpt_dir}"
            raise ValueError(error_message)
        return "last"

    # Training checkpoints with save_top_k=1 on val/brierFDE, so a run holds exactly one epoch_XXX.ckpt: its best
    # weights, and the ones the results CSV reported metrics for.
    candidates = sorted(ckpt_dir.glob("epoch_*.ckpt"))
    if not candidates:
        error_message = f"No epoch_*.ckpt in {ckpt_dir}"
        raise ValueError(error_message)
    if len(candidates) > 1:
        names = ", ".join(candidate.name for candidate in candidates)
        error_message = f"Ambiguous best checkpoint in {ckpt_dir}: {names}"
        raise ValueError(error_message)
    return candidates[0].stem


def has_cache(run: Run) -> bool:
    """Returns True when both split caches already hold at least one scenario pickle.

    Cached outputs are namespaced by source (``<split>/<dataset_name>/<scenario_id>.pkl``), hence the recursive glob.
    """
    return all(any((run.run_dir / "batch_cache" / split).rglob("*.pkl")) for split in CACHE_SPLITS)


def resolve_csv_filepath(csv_filepath: str, project_root: Path) -> Path:
    """Resolves the results CSV against the project root when given as a relative path.

    Raises:
        ValueError: if the CSV does not exist.
    """
    filepath = Path(csv_filepath)
    if not filepath.is_absolute():
        filepath = project_root / filepath
    if not filepath.is_file():
        error_message = f"Results CSV not found: {filepath}"
        raise ValueError(error_message)
    return filepath
