"""Script used for performing sample selection for training experiments.

Example usage:

    uv run -m controlledshifts.run_sample_selection \
        ckpt_path=/path/to/checkpoint.ckpt model=wayformer

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

from controlledshifts import sample_selection, utils
from controlledshifts.schemas import output_schemas as output
from controlledshifts.utils.constants import DataSplits, SampleSelection


if TYPE_CHECKING:
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.loggers.logger import Logger

log = utils.get_pylogger(__name__)

torch.set_float32_matmul_precision("medium")
root_path = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

utils.disable_mlflow_tls_verification()


def _run_sample_selection(config: DictConfig, model_outputs: dict[str, output.ModelOutput], output_path: Path) -> None:
    """Wrapper function which runs a specified sample selection strategy. A sample selection strategy produces a
    dictionary containing the a set of training scenarios to keep and to drop.

    Args:
        config: encapsulates model analysis configuration parameters.
        model_outputs: a dictionary containing model outputs per scenario.
        output_path: output path where visualization will be saved to.
    """
    selection_strategy = SampleSelection(config.selection_strategy)
    match selection_strategy:
        case SampleSelection.RANDOM_DROP:
            subdir = "random"
            selected_samples = sample_selection.random_selection(config, model_outputs)
        case SampleSelection.KMEANS_RANDOM_DROP:
            subdir = "kmeans"
            config.clustering_strategy = "kmeans"
            selected_samples = sample_selection.random_selection_per_cluster(config, model_outputs)
        case SampleSelection.SIMPLE_KMEANS_COSINE_DROP:
            subdir = "kmeans"
            config.sorting_strategy = "simple"
            selected_samples = sample_selection.cosine_selection_per_cluster(config, model_outputs)
        case SampleSelection.GUMBEL_KMEANS_COSINE_DROP:
            subdir = "kmeans"
            config.sorting_strategy = "gumbel"
            selected_samples = sample_selection.cosine_selection_per_cluster(config, model_outputs)
        case SampleSelection.DEN_TP:
            subdir = "dentp"
            selected_samples = sample_selection.dentp_selection(config, model_outputs)
        case SampleSelection.VOCAB_CLUSTER_HAMMING_DROP | SampleSelection.VOCAB_CLUSTER_JACCARD_DROP:
            subdir = "kmeans"
            config.alignment_strategy = (
                "hamming" if selection_strategy == SampleSelection.VOCAB_CLUSTER_HAMMING_DROP else "jaccard"
            )
            selected_samples = sample_selection.vocab_cluster_selection(config, model_outputs)
        case _:
            error_message = f"Unsupported selection strategy: {selection_strategy}"
            raise ValueError(error_message)

    output_filepath = (
        output_path / subdir / f"sample_selection_{selection_strategy.value}_{config.percentage_to_keep}.json"
    )
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with output_filepath.open("w") as f:
        json.dump(selected_samples, f, indent=2)


@utils.task_wrapper
def _evaluate_and_cache_dataset(cfg: DictConfig) -> tuple[dict, dict]:
    """Evaluates and caches the training set for sample selection purposes.

    Args:
        cfg (Dictcfg): configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
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
    test_loader = DataLoader(
        training_set,
        batch_size=cfg.model.config.eval_batch_size,
        num_workers=cfg.model.config.load_num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=training_set.collate_fn,
    )
    log.info("Starting test process to cache training set.")
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=cfg.ckpt_path)
    metric_dict = trainer.callback_metrics
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="sample_selection.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra's entrypoint for running the sample selection experiment."""
    log.info("Printing cfg tree with Rich! <cfg.extras.print_cfg=True>")
    utils.print_config_tree(cfg, resolve=True, save_to_file=False)

    # Run model evaluation to cache the training set embeddings
    if cfg.create_training_batch_cache:
        _evaluate_and_cache_dataset(cfg)

    log.info("Loading batches from %s", cfg.paths.batch_cache_path)
    batches = utils.load_batches(cfg.paths.batch_cache_path, cfg.num_batches, cfg.num_scenarios, cfg.seed, cfg.split)

    # Run sample selection
    output_path = Path(cfg.paths.meta_path)
    output_path.mkdir(parents=True, exist_ok=True)
    log.info("Saving sample selection lists to %s", str(output_path))
    for selection_strategy, percentage_to_keep in product(cfg.selection_strategies, cfg.percentages_to_keep):
        ss_cfg = deepcopy(cfg)
        ss_cfg.selection_strategy = selection_strategy
        ss_cfg.percentage_to_keep = percentage_to_keep
        log.info(
            "Running sample selection with strategy: %s, percentage_to_keep: %s", selection_strategy, percentage_to_keep
        )
        _run_sample_selection(ss_cfg, batches, output_path)


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
