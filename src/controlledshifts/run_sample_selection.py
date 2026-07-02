"""Script used for generating sample selection blacklists for training experiments.

A sample selection strategy consumes the cached per-scenario model outputs for the training split and produces a JSON
file listing the scenarios to ``keep`` and ``drop``. Training then removes the ``drop`` scenarios (see
``controlledshifts.datasets.base_dataset``) so a model can be retrained on a smaller data regime.

Example usage:

    # Cache the training-set outputs with a pretrained model, then generate a random blacklist.
    uv run -m controlledshifts.run_sample_selection \
        paths=causal_agents model=wayformer ckpt_name=epoch_093 create_training_batch_cache=true

    # Reuse an existing train batch cache and only run the selection.
    uv run -m controlledshifts.run_sample_selection \
        paths=causal_agents model=wayformer selection_strategies=[random_drop] percentages_to_keep=[0.55]

See `docs/ANALYSIS.md` and `configs/sample_selection.yaml` for more argument details.
"""

import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from controlledshifts import utils
from controlledshifts.schemas import ModelOutput
from controlledshifts.utils import sample_selection
from controlledshifts.utils.constants import DataSplits, SampleSelection


if TYPE_CHECKING:
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.loggers.logger import Logger

log = utils.get_pylogger(__name__)

torch.set_float32_matmul_precision("medium")
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

utils.disable_mlflow_tls_verification()


def _load_training_scenario_ids(cfg: DictConfig) -> list[str]:
    """Collects the training scenario IDs from the benchmark split JSONs (the model-free selection universe).

    Mirrors how ``base_dataset`` resolves its training universe: for each training source, read
    ``<splits_root>/<split_json>.json`` and take the IDs under the source's split.

    Args:
        cfg: the sample selection configuration.

    Returns:
        The de-duplicated training scenario IDs across all training sources.
    """
    dataset_config = cfg.dataset.config
    splits_root = Path(dataset_config.splits_root)
    scenario_ids: list[str] = []
    seen: set[str] = set()
    for source in dataset_config.train_sources:
        split_path = splits_root / f"{source['split_json']}.json"
        with split_path.open() as f:
            payload = json.load(f)
        for scenario_id in payload[source["split"]]:
            if scenario_id not in seen:
                seen.add(scenario_id)
                scenario_ids.append(scenario_id)
    return scenario_ids


def _run_sample_selection(
    config: DictConfig,
    scenario_ids: list[str],
    model_outputs: dict[str, ModelOutput] | None,
    output_path: Path,
) -> None:
    """Runs the configured sample selection strategy and writes its keep/drop lists to disk.

    Args:
        config: the sample selection configuration, with ``selection_strategy`` and ``percentage_to_keep`` set.
        scenario_ids: the training scenario IDs (the selection universe for model-independent strategies).
        model_outputs: cached per-scenario model outputs for embedding-based strategies, or ``None`` when not loaded.
        output_path: directory the selection JSON is written to.
    """
    selection_strategy = SampleSelection(config.selection_strategy)
    match selection_strategy:
        case SampleSelection.RANDOM_DROP:
            selected_samples = sample_selection.random_selection(config, scenario_ids)
        case (
            SampleSelection.KMEANS_RANDOM_DROP
            | SampleSelection.SIMPLE_KMEANS_COSINE_DROP
            | SampleSelection.GUMBEL_KMEANS_COSINE_DROP
            | SampleSelection.DEN_TP
            | SampleSelection.VOCAB_CLUSTER_HAMMING_DROP
            | SampleSelection.VOCAB_CLUSTER_JACCARD_DROP
        ):
            if model_outputs is None:
                error_message = (
                    f"Strategy '{selection_strategy.value}' needs cached model outputs; "
                    "run with create_training_batch_cache=true and a valid ckpt_name."
                )
                raise ValueError(error_message)
            error_message = f"Selection strategy '{selection_strategy.value}' is not yet ported."
            raise NotImplementedError(error_message)
        case _:
            error_message = f"Unsupported selection strategy: {selection_strategy}"
            raise ValueError(error_message)

    output_filepath = output_path / f"sample_selection_{selection_strategy.value}_{config.percentage_to_keep}.json"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with output_filepath.open("w") as f:
        json.dump(selected_samples, f, indent=2)
    log.info(
        "Wrote %s (keep=%s, drop=%s)",
        output_filepath,
        selected_samples["num_to_keep"],
        selected_samples["num_to_drop"],
    )


@utils.task_wrapper
def _evaluate_and_cache_dataset(cfg: DictConfig) -> tuple[dict, dict]:
    """Evaluates and caches the training set so sample selection can read its per-scenario model outputs.

    Runs the pretrained model over the training split with ``model.config.sample_selection=true``, which dumps one
    ``ModelOutput`` pickle per scenario under ``paths.batch_cache_path/train`` (see ``models.base_model``).

    Args:
        cfg: configuration composed by Hydra.

    Returns:
        A tuple with the trainer metrics and the instantiated objects.
    """
    if not Path(cfg.ckpt_path).exists():
        error_message = f"Checkpoint path: {cfg.ckpt_path} does not exist!"
        raise ValueError(error_message)

    log.info("Instantiating model <%s>", cfg.model._target_)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: list[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {"cfg": cfg, "model": model, "logger": logger, "trainer": trainer}
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Instantiating training dataset <%s>", cfg.dataset._target_)
    cfg.dataset.config.split = DataSplits.TRAINING
    training_set: Dataset = hydra.utils.instantiate(cfg.dataset)
    train_loader = DataLoader(
        training_set,
        batch_size=cfg.model.config.eval_batch_size,
        num_workers=cfg.model.config.load_num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=training_set.collate_fn,
    )
    log.info("Starting test process to cache the training set.")
    trainer.test(model=model, dataloaders=train_loader, ckpt_path=cfg.ckpt_path)
    return trainer.callback_metrics, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="sample_selection.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint that caches the training set (optional) and sweeps the configured selection strategies."""
    utils.print_config_tree(cfg, resolve=True, save_to_file=False)

    # random_drop only needs the training scenario IDs; embedding-based strategies need cached model outputs. Only run
    # the (potentially expensive) caching + batch-loading pipeline when a model-dependent strategy is requested.
    needs_model_outputs = any(
        SampleSelection(strategy) != SampleSelection.RANDOM_DROP for strategy in cfg.selection_strategies
    )
    model_outputs: dict[str, ModelOutput] | None = None
    if needs_model_outputs:
        if cfg.create_training_batch_cache:
            _evaluate_and_cache_dataset(cfg)
        log.info("Loading batches from %s", cfg.paths.batch_cache_path)
        model_outputs = utils.load_batches(
            cfg.paths.batch_cache_path, cfg.num_batches, cfg.num_scenarios, cfg.seed, cfg.split
        )

    scenario_ids = _load_training_scenario_ids(cfg)
    log.info("Loaded %d training scenario IDs for the selection universe", len(scenario_ids))

    output_path = Path(cfg.sample_selection_path)
    output_path.mkdir(parents=True, exist_ok=True)
    log.info("Saving sample selection lists to %s", str(output_path))
    for selection_strategy, percentage_to_keep in product(cfg.selection_strategies, cfg.percentages_to_keep):
        ss_cfg = deepcopy(cfg)
        ss_cfg.selection_strategy = selection_strategy
        ss_cfg.percentage_to_keep = percentage_to_keep
        log.info(
            "Running sample selection with strategy: %s, percentage_to_keep: %s", selection_strategy, percentage_to_keep
        )
        _run_sample_selection(ss_cfg, scenario_ids, model_outputs, output_path)


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
