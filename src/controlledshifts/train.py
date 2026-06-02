"""Training script for SocialTokens.

Example usage:

    uv run -m controlledshifts.train model=[model_name]

See `docs/MODEL_TRAINING.md` and `configs/train.yaml` for more argument details.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pyrootutils
import pytorch_lightning
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, Dataset

from controlledshifts import utils
from controlledshifts.utils.constants import DataSplits, SampleSelection


if TYPE_CHECKING:
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.loggers.logger import Logger

log = utils.get_pylogger(__name__)

torch.set_float32_matmul_precision("medium")
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

utils.disable_mlflow_tls_verification()


def _add_mtr_extras(cfg: DictConfig, train_loader: DataLoader) -> None:
    """Resolve the MTR intention points file and compute it if missing.

    Args:
        cfg (DictConfig): full Hydra config, used to check for the intention points file path
        train_loader (DataLoader): training dataloader, used to compute intention points if the file is missing.
    """
    if SampleSelection(cfg.sample_selection_strategy) != SampleSelection.ALL:
        sample_intention_file = cfg.get("sample_selection_intention_points_filepath")
        log.info("Sample selection active — overriding intention points file: %s", sample_intention_file)
        cfg.model.config.intention_points_file = sample_intention_file
        cfg.model.config.motion_decoder.intention_points_file = sample_intention_file

    intention_file = cfg.model.config.motion_decoder.get("intention_points_file")
    if intention_file is None or not Path(intention_file).exists():
        log.info("MTR intention points not found — computing from training data ...")
        saved_path = utils.compute_and_cache_intention_points(train_loader, cfg)
        cfg.model.config.motion_decoder.intention_points_file = saved_path


@utils.task_wrapper
def train(cfg: DictConfig) -> tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during training. This
    method is wrapped in optional @task_wrapper decorator, that controls the behavior during failure. Useful for
    multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    metric_dict, object_dict = {}, {}
    print("Check CUDA", torch.cuda.is_available())

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pytorch_lightning.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating loggers...")
    logger: list[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("Instantiating training dataset <%s>", cfg.dataset._target_)
    cfg.dataset.config.split = DataSplits.TRAINING
    training_dataset: Dataset = hydra.utils.instantiate(cfg.dataset)
    train_loader = DataLoader(
        training_dataset,
        batch_size=cfg.model.config.train_batch_size,
        num_workers=cfg.model.config.load_num_workers,
        drop_last=False,
        collate_fn=training_dataset.collate_fn,
    )

    if cfg.model._target_ == "controlledshifts.models.mtr.MTR":
        _add_mtr_extras(cfg, train_loader)

    log.info("Instantiating model <%s>", cfg.model._target_)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating validation dataset <%s>", cfg.dataset._target_)
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

    object_dict = {
        "cfg": cfg,
        "training_dataset": training_dataset,
        "validation_dataset": validation_dataset,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)  # pyright: ignore[reportAssignmentType]

    log.info("Starting training!")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.get("ckpt_path"),
    )
    train_metrics = trainer.callback_metrics
    log.info("Done training!")

    if cfg.get("test"):
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
        ckpt_path = trainer.checkpoint_callback.best_model_path
        log.info("Best ckpt path: %s", ckpt_path)
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, dataloaders=test_loader, ckpt_path=ckpt_path)
        log.info("Done testing!")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Hydra's entrypoint for running model training."""
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))
    log.info("Retrieved metrics: \n%s", metric_value)
    log.info("Process completed!")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
