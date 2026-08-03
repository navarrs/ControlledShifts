"""Unified Analysis Script.

Runs one of the analyses over a single combined results file, selected via the ``analysis`` config group:

    # In-distribution vs out-of-distribution benchmark comparison (per-benchmark plots + LaTeX table).
    uv run -m controlledshifts.run_analysis analysis=distribution_shift

    # One in-distribution reference vs many benchmarks with no artificial training shift (table + bars + gap heatmap).
    uv run -m controlledshifts.run_analysis analysis=unshifted_generalization

    # Per-model, per-metric distribution-shift robustness scores (radar plot + CSV + LaTeX table).
    uv run -m controlledshifts.run_analysis analysis=robustness

    # Non-background vs background agent count distributions across train/val/test for the background-agents
    # benchmarks.
    uv run -m controlledshifts.run_analysis analysis=background_distribution

    # Per-scenario criticality-score distributions across train/val/test for the ego-safeshift benchmark.
    uv run -m controlledshifts.run_analysis analysis=score_distribution

    # NetLSD-descriptor clustering for the environments benchmark (TSNE by cluster/split + silhouette plot).
    uv run -m controlledshifts.run_analysis analysis=environments_distribution

    # Pairwise scenario overlap between benchmarks per split (Jaccard heatmaps + CSV).
    uv run -m controlledshifts.run_analysis analysis=scenario_overlap

See `docs/ANALYSIS.md` and the per-analysis configs under `configs/analysis/` for more argument details.
"""

import random
from collections.abc import Callable
from logging import Logger
from pathlib import Path
from time import time

import hydra
import pyrootutils
from omegaconf import DictConfig

from controlledshifts import utils
from controlledshifts.utils import analysis
from controlledshifts.utils.plotting import configure_fonts


log = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# Maps the selected ``analysis`` config-group option (its ``analysis_name``) to its runner.
_ANALYSES: dict[str, Callable[[DictConfig, Logger, Path], None]] = {
    "distribution_shift": analysis.run_distribution_shift_analysis,
    "unshifted_generalization": analysis.run_unshifted_generalization_analysis,
    "robustness": analysis.run_robustness_scores_analysis,
    "background_distribution": analysis.run_background_distribution_analysis,
    "score_distribution": analysis.run_score_distribution_analysis,
    "environments_distribution": analysis.run_environments_distribution_analysis,
    "scenario_overlap": analysis.run_scenario_overlap_analysis,
}


@hydra.main(version_base="1.3", config_path="configs", config_name="analysis.yaml")
def main(config: DictConfig) -> None:
    """Hydra entry point dispatching to the analysis selected by ``analysis=<option>``.

    Raises:
        ValueError: If the selected analysis has no registered runner.
    """
    random.seed(config.seed)
    configure_fonts(log=log)

    runner = _ANALYSES.get(config.analysis_name)
    if runner is None:
        error_message = f"Unknown analysis '{config.analysis_name}'; options: {sorted(_ANALYSES)}"
        raise ValueError(error_message)

    start = time()
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    runner(config, log, output_path)

    log.info("Total time: %.2f seconds", time() - start)
    log.info("Process completed!")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
