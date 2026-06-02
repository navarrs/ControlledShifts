"""Evaluation script for ControlledShifts.

Example usage:

    uv run -m controlledshifts.eval model=[model_name] ckpt_path=/path/to/checkpoint.ckpt

See `docs/MODEL_TRAINING.md` and `configs/eval.yaml` for more argument details.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from controlledshifts import utils
from controlledshifts.utils.constants import DataSplits


if TYPE_CHECKING:
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.loggers.logger import Logger

log = utils.get_pylogger(__name__)

torch.set_float32_matmul_precision("medium")
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

utils.disable_mlflow_tls_verification()


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset. This method is wrapped in optional @task_wrapper decorator,
    that controls the behavior during failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

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

    object_dict = {
        "cfg": cfg,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    val_metrics = {}
    if cfg.eval:
        log.info("Instantiating testing dataset <%s>", cfg.dataset._target_)
        cfg.dataset.config.split = DataSplits.VALIDATION
        validation_dataset: Dataset = hydra.utils.instantiate(cfg.dataset)
        val_loader = DataLoader(
            validation_dataset,
            batch_size=cfg.model.config.eval_batch_size,
            num_workers=cfg.model.config.load_num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=validation_dataset.collate_fn,
        )
        log.info("Starting validation!")
        trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)
        val_metrics = trainer.callback_metrics

    test_metrics = {}
    if cfg.test:
        log.info("Instantiating testing dataset <%s>", cfg.dataset._target_)
        cfg.dataset.config.split = DataSplits.TESTING
        testing_dataset: Dataset = hydra.utils.instantiate(cfg.dataset)
        test_loader = DataLoader(
            testing_dataset,
            batch_size=cfg.model.config.eval_batch_size,
            num_workers=cfg.model.config.load_num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=testing_dataset.collate_fn,
        )
        log.info("Starting testing!")
        trainer.test(model=model, dataloaders=test_loader, ckpt_path=cfg.ckpt_path)
        test_metrics = trainer.callback_metrics

    metric_dict = {**val_metrics, **test_metrics}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra's entrypoint for running the model's evaluation."""
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
