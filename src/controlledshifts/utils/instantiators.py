import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger

from controlledshifts.utils import pylogger


log = pylogger.get_pylogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config.

    Args:
        callbacks_cfg: configuration for the callbacks to instantiate.

    Returns:
        List of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        error_message = "Callbacks config must be a DictConfig!"
        raise TypeError(error_message)

    for cb_conf in callbacks_cfg.values():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info("Instantiating callback <%s>", cb_conf._target_)
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiates loggers from config.

    Args:
        logger_cfg: configuration for the loggers to instantiate.

    Returns:
        List of instantiated loggers.
    """
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        error_message = "Logger config must be a DictConfig!"
        raise TypeError(error_message)

    for lg_conf in logger_cfg.values():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info("Instantiating logger <%s>", lg_conf._target_)
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
