"""Locating trained runs and their cached model outputs on disk.

A run lives at ``<cache_root>/<dataset>/<model>/<timestamp>`` (i.e. ``paths.experiment_dir``) and its cached outputs at
``<run_dir>/batch_cache/<split>/<source>/<scenario_id>.pkl``. The runs themselves are enumerated from a W&B results
export, whose dataset/model/timestamp columns locate each run directory.

Written by ``run_model_cache_sweep``; read by anything that needs a trained run's cached outputs.
"""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from controlledshifts.utils.constants import ModelStatus


# The splits `save_cache` writes during eval, named after the ModelStatus values.
CACHE_SPLITS = (ModelStatus.VALIDATION, ModelStatus.TEST)

# Maps a benchmark split JSON key to the ModelStatus tag used as the cache subdirectory and in the viz output path.
SPLIT_TAGS: dict[str, ModelStatus] = {
    "training": ModelStatus.TRAIN,
    "validation": ModelStatus.VALIDATION,
    "testing": ModelStatus.TEST,
}

# Each benchmark declares, per split, the sources it is evaluated under. Read the `paths` group config directly rather
# than composing it: the scenario visualization deliberately does not compose `paths` (it interpolates ${exp_name}), and
# these keys need no interpolation.
_PATHS_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "paths"
_SOURCE_KEYS: dict[str, str] = {"validation": "val_sources", "testing": "test_sources"}

# Cached outputs are namespaced by the record's `dataset_name`, which `BaseDataset.__getitem__` stamps as `waymo-<tag>`.
_SOURCE_PREFIX = "waymo"


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
    def batch_cache_path(self) -> Path:
        """Root of this run's cached model outputs."""
        return self.run_dir / "batch_cache"

    @property
    def name(self) -> str:
        """Short label for logs."""
        return f"{self.dataset}/{self.model}"


@dataclass(frozen=True)
class ModelCacheSpec:
    """One pane of a trajpred figure: which cached run feeds it, which scene it saw, and how it is labelled."""

    name: str  # column label, i.e. the model (e.g. "wayformer")
    cache_path: Path  # <run_dir>/batch_cache
    source: str | None = None  # source directory under the split tag (e.g. "waymo-uniform-validation")
    group: str = ""  # row label, i.e. the training benchmark; "" is the flat, single-row layout
    variant: str = "base"  # scene variant the model was evaluated on, i.e. the one the pane must draw


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


def resolve_split_source(
    paths_group: str, split_key: str, paths_config_dir: Path = _PATHS_CONFIG_DIR
) -> tuple[str, str]:
    """Returns the ``(cache source, scene variant)`` a benchmark evaluates one split under.

    Reads ``configs/paths/<paths_group>.yaml`` and picks the entry in ``val_sources``/``test_sources`` whose ``split``
    is ``split_key``. Both halves matter:

    * the source must be pinned, because a run's ``test/`` cache holds several of them -- the seen (ID) split is
      re-evaluated under the ``test`` tag alongside the unseen (OOD) one -- so an unpinned load silently mixes them;
    * the variant says which scene the model was actually given (causal-agents-hard tests on ``remove_noncausal``), so
      the pane can draw that scene rather than the unperturbed one.

    Args:
        paths_group: Hydra `paths` group, i.e. the underscored benchmark name (e.g. ``causal_agents_hard``).
        split_key: benchmark split JSON key, either ``validation`` or ``testing``.
        paths_config_dir: directory holding the `paths` group configs.

    Returns:
        The source directory name (e.g. ``waymo-uniform-testing``) and the scene variant (e.g. ``base``).

    Raises:
        ValueError: if the split is not cached, the config is missing, or it declares no single such source.
    """
    if split_key not in _SOURCE_KEYS:
        error_message = f"Split '{split_key}' has no cached model outputs; expected one of {sorted(_SOURCE_KEYS)}."
        raise ValueError(error_message)

    config_filepath = paths_config_dir / f"{paths_group}.yaml"
    if not config_filepath.is_file():
        error_message = f"No `paths` config for benchmark '{paths_group}': {config_filepath}"
        raise ValueError(error_message)

    paths_config = OmegaConf.load(config_filepath)
    sources = [source for source in paths_config[_SOURCE_KEYS[split_key]] if source.split == split_key]
    if len(sources) != 1:
        error_message = (
            f"Expected exactly one '{split_key}' source in {config_filepath}, found {len(sources)}. The cache is "
            f"namespaced by source, so the split must resolve to a single one."
        )
        raise ValueError(error_message)
    return f"{_SOURCE_PREFIX}-{sources[0].tag}", sources[0].variant


def build_grid_specs(  # noqa: PLR0913
    benchmarks: Sequence[str],
    models: Sequence[str],
    split_key: str,
    runs: Sequence[Run],
    paths_groups: Mapping[str, str],
    paths_config_dir: Path = _PATHS_CONFIG_DIR,
) -> list[ModelCacheSpec]:
    """Builds one spec per (benchmark row, model column) of a trajpred grid.

    Args:
        benchmarks: display names of the benchmarks to render as rows, in row order.
        models: model names to render as columns, in column order.
        split_key: the benchmark split being rendered (``validation`` or ``testing``).
        runs: the finished runs to resolve the cells against, from the results CSV.
        paths_groups: maps each benchmark display name to its Hydra `paths` group.
        paths_config_dir: directory holding the `paths` group configs.

    Returns:
        The specs, row-major.

    Raises:
        ValueError: if a benchmark is unmapped, or a cell does not resolve to exactly one existing run.
    """
    unmapped = [benchmark for benchmark in benchmarks if benchmark not in paths_groups]
    if unmapped:
        error_message = (
            f"No `paths` group mapped for benchmark(s) {', '.join(unmapped)}. Add them to "
            f"`overlap_grid.benchmark_paths_groups` (known: {', '.join(sorted(paths_groups))})."
        )
        raise ValueError(error_message)

    specs: list[ModelCacheSpec] = []
    problems: list[str] = []
    for benchmark in benchmarks:
        paths_group = paths_groups[benchmark]
        source, variant = resolve_split_source(paths_group, split_key, paths_config_dir)
        for model in models:
            candidates = [run for run in runs if run.paths_group == paths_group and run.model == model]
            if len(candidates) != 1:
                problems.append(f"{benchmark}/{model}: {len(candidates)} finished runs in the CSV, expected 1")
                continue
            if not candidates[0].run_dir.is_dir():
                problems.append(f"{benchmark}/{model}: run directory does not exist: {candidates[0].run_dir}")
                continue
            specs.append(
                ModelCacheSpec(
                    name=model,
                    cache_path=candidates[0].batch_cache_path,
                    source=source,
                    group=benchmark,
                    variant=variant,
                )
            )

    if problems:
        error_message = "Could not resolve {} of {} grid cells:\n  {}".format(
            len(problems), len(benchmarks) * len(models), "\n  ".join(problems)
        )
        raise ValueError(error_message)
    return specs
