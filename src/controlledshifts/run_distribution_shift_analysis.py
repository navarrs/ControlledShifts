"""Distribution Shift Analysis Script.

Reads a single combined results file and, for each benchmark in the config, compares the specified seen (ID) and unseen
(OOD) metric columns. Produces per-benchmark comparison plots and one combined LaTeX table across all benchmarks.

Example usage:

    uv run -m controlledshifts.run_distribution_shift_analysis

See `docs/ANALYSIS.md` and `configs/analysis/distribution_shift.yaml` for more argument details.
"""

import random
from pathlib import Path
from time import time

import hydra
import pyrootutils
from omegaconf import DictConfig

from controlledshifts import utils


log = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="analysis/distribution_shift")
def main(config: DictConfig) -> None:
    """Hydra's entrypoint for running distribution-shift analysis over a single combined results file."""
    random.seed(config.seed)

    start = time()
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    utils.run_distribution_shift_analysis(config, log, output_path)

    log.info("Total time: %.2f seconds", time() - start)
    log.info("Process completed!")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
