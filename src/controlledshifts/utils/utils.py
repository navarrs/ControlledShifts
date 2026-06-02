import os
import warnings
from collections.abc import Callable
from importlib.util import find_spec

import urllib3
from omegaconf import DictConfig
from urllib3.exceptions import InsecureRequestWarning

from controlledshifts.utils import pylogger, rich_utils


log = pylogger.get_pylogger(__name__)


def disable_mlflow_tls_verification() -> None:
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    urllib3.disable_warnings(InsecureRequestWarning)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
           @utils.task_wrapper
           def train(cfg: DictConfig) -> tuple[dict, dict]:
               ...

               return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig) -> tuple[dict, dict]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise AssertionError(ex)  # noqa: B904

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info("Output dir: %s", cfg.paths.output_dir)

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb  # noqa: PLC0415

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float | None:
    """Safely retrieves value of the metric logged in LightningModule."""
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(  # noqa: TRY002, TRY003
            f"Metric value not found! <metric_name={metric_name}>\n"  # noqa: EM102
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!",
        )

    metric_value = metric_dict[metric_name].item()
    log.info("Retrieved metric value! <%s=%s>", metric_name, metric_value)

    return metric_value
