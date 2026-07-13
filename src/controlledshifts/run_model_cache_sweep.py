"""Script used for generating the validation/testing model-output caches of already-trained runs.

Models train with ``cache_batch: False``, so a finished run's ``batch_cache/`` is empty. This re-runs each run's
checkpoint through ``controlledshifts.eval`` with caching enabled and writes the outputs back into the *original* run
directory -- the one already holding the checkpoint -- rather than a fresh timestamped one. The runs to cache are read
from a W&B results export (one row per run; its dataset/model/timestamp columns locate the run directory).

The run directory is ``<cache_root>/<dataset>/<model>/<timestamp>``, which is exactly ``paths.experiment_dir``. That
config interpolates ``${now:...}``, so a plain eval invocation would mint a new dated directory containing no
checkpoint; pinning ``paths.experiment_dir`` to the training run's path makes ``ckpt_path``, ``batch_cache_path`` and
``log_path`` all resolve inside it. Outputs land in ``<run_dir>/batch_cache/{val,test}/<source>/<scenario_id>.pkl``.

Example usage:

    # Preview the eval command for every run in the CSV without running anything.
    uv run -m controlledshifts.run_model_cache_sweep dry_run=true

    # Cache the validation/testing outputs of every run in the CSV.
    uv run -m controlledshifts.run_model_cache_sweep

    # Resume an interrupted sweep, skipping runs that already have both caches.
    uv run -m controlledshifts.run_model_cache_sweep skip_existing=true

    # Restrict to some models and benchmarks, and pick the GPU.
    uv run -m controlledshifts.run_model_cache_sweep 'models=[wayformer,mtr]' 'benchmarks=[causal_agents]' devices=1

    # Use the final-epoch weights and subsample every 10th batch instead of caching every scenario.
    uv run -m controlledshifts.run_model_cache_sweep ckpt=last cache_every_batch_idx=10

A failing run does not abort the sweep: failures are collected and reported in a summary at the end.

See `docs/ANALYSIS.md` and `configs/model_cache_sweep.yaml` for more argument details.
"""

import csv
import subprocess  # nosec B404
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import hydra
import pyrootutils
from omegaconf import DictConfig

from controlledshifts import utils
from controlledshifts.utils.constants import ModelStatus


log = utils.get_pylogger(__name__)

# Eval is spawned from the project root: unlike `train.py`, `eval.py` does not call `_add_mtr_extras`, so MTR resolves
# its intention points through the relative `./meta/<tag>/intention_points.pkl`.
PROJECT_ROOT = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# The splits `save_cache` writes during eval, named after the ModelStatus values.
CACHE_SPLITS = (ModelStatus.VALIDATION, ModelStatus.TEST)


class RunOutcome(StrEnum):
    """What happened to one run of the sweep."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    PENDING = "pending"  # counted but not run, i.e. a dry run


@dataclass(frozen=True)
class Run:
    """One row of the results CSV, resolved onto its on-disk training run."""

    dataset: str  # paths.tag, hyphenated (e.g. "causal-agents")
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
        csv_filepath: W&B results export listing the runs to cache.
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
        # Accept either spelling: the paths group (`causal_agents`) or the tag (`causal-agents`).
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


def build_command(run: Run, ckpt_name: str, config: DictConfig) -> list[str]:
    """Builds the eval invocation that caches this run's outputs into its own run directory."""
    return [
        "uv",
        "run",
        "-m",
        "controlledshifts.eval",
        f"model={run.model}",
        f"paths={run.paths_group}",
        f"trainer.devices=[{config.devices}]",
        # Pins ckpt_path/batch_cache_path/log_path to the training run instead of a fresh ${now:...} directory.
        f"paths.experiment_dir={run.experiment_dir}",
        f"ckpt_name={ckpt_name}",
        "model.config.cache_batch=true",
        f"model.config.cache_every_batch_idx={config.cache_every_batch_idx}",
        "eval=true",
        "test=true",
        "logger=csv",
    ]


def cache_run(run: Run, config: DictConfig) -> tuple[RunOutcome, str]:
    """Caches one run's validation/testing outputs by spawning eval on its checkpoint.

    Args:
        run: the training run to cache.
        config: the sweep configuration.

    Returns:
        The run's outcome and a short reason describing it.
    """
    if not run.run_dir.is_dir():
        return RunOutcome.SKIPPED, f"run directory does not exist: {run.run_dir}"

    if config.skip_existing and has_cache(run):
        return RunOutcome.SKIPPED, "cache already present"

    try:
        ckpt_name = resolve_checkpoint(run, config.ckpt)
    except ValueError as exc:
        return RunOutcome.SKIPPED, str(exc)

    command = build_command(run, ckpt_name, config)
    log.info("Caching %s from %s.ckpt into %s", run.name, ckpt_name, run.run_dir / "batch_cache")
    if config.dry_run:
        log.info("[DRY RUN] %s", " ".join(command))
        return RunOutcome.PENDING, "dry run"

    # Keep going on failure so one bad run does not abort the remaining ones.
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)  # noqa: S603
    if result.returncode != 0:
        return RunOutcome.FAILED, f"exit code {result.returncode}"
    return RunOutcome.SUCCEEDED, "cached"


def resolve_csv_filepath(csv_filepath: str) -> Path:
    """Resolves the results CSV against the project root when given as a relative path.

    Raises:
        ValueError: if the CSV does not exist.
    """
    filepath = Path(csv_filepath)
    if not filepath.is_absolute():
        filepath = PROJECT_ROOT / filepath
    if not filepath.is_file():
        error_message = f"Results CSV not found: {filepath}"
        raise ValueError(error_message)
    return filepath


@hydra.main(version_base="1.3", config_path="configs", config_name="model_cache_sweep.yaml")
def main(config: DictConfig) -> None:
    """Hydra entry point for caching the validation/testing outputs of the runs listed in the results CSV.

    Raises:
        ValueError: if the CSV is missing, no run matches the filters, or any run's eval failed.
    """
    utils.print_config_tree(config, resolve=True, save_to_file=False)

    csv_filepath = resolve_csv_filepath(config.csv_filepath)
    runs = filter_runs(read_runs(csv_filepath, Path(config.cache_root)), config.models, config.benchmarks)
    if not runs:
        error_message = f"No runs in {csv_filepath} matched models={config.models}, benchmarks={config.benchmarks}"
        raise ValueError(error_message)

    outcomes: dict[RunOutcome, list[tuple[str, str]]] = {outcome: [] for outcome in RunOutcome}
    for index, run in enumerate(runs, start=1):
        log.info("[%d/%d] %s (%s)", index, len(runs), run.name, run.timestamp)
        outcome, reason = cache_run(run, config)
        if outcome == RunOutcome.FAILED:
            log.error("Failed %s: %s", run.name, reason)
        outcomes[outcome].append((run.name, reason))

    skipped, failed = outcomes[RunOutcome.SKIPPED], outcomes[RunOutcome.FAILED]
    if config.dry_run:
        log.info("Dry run: %d command(s) would run, %d skipped", len(outcomes[RunOutcome.PENDING]), len(skipped))
    else:
        log.info(
            "Done: %d succeeded, %d failed, %d skipped", len(outcomes[RunOutcome.SUCCEEDED]), len(failed), len(skipped)
        )
    for name, reason in skipped:
        log.info("Skipped %s: %s", name, reason)

    if failed:
        summary = "; ".join(f"{name} ({reason})" for name, reason in failed)
        error_message = f"{len(failed)} of {len(runs)} runs failed: {summary}"
        raise ValueError(error_message)


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
