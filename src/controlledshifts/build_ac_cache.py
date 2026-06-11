r"""Builds the canonical agent-centric (AC) cache for a single data variant.

The AC cache stores each scenario's agent-centric representation once per (variant, processing profile), keyed by a hash
of the tensor-affecting config. Training/eval then read per-scenario records straight from this cache (selected by the
benchmark split JSONs) with no reprocessing or copying. Build it once per variant per processing profile.

Example usage:

    # Base (unperturbed) variant for the AutoBot processing profile
    uv run -m controlledshifts.build_ac_cache variant=base model=autobot

    # A perturbed variant (must be generated first via create_benchmark benchmark=causal_agents)
    uv run -m controlledshifts.build_ac_cache variant=remove_noncausal model=autobot

    # MTR uses manually_split_lane, which yields a distinct processing profile / cache directory
    uv run -m controlledshifts.build_ac_cache variant=base model=mtr

See ``configs/build_ac_cache.yaml`` for all options.
"""

from pathlib import Path

import hydra
import pyrootutils
from omegaconf import DictConfig

from controlledshifts import utils
from controlledshifts.datasets import agent_centric_cacher
from controlledshifts.datasets.agent_centric_processor import AgentCentricProcessor


_LOGGER = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="build_ac_cache.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: builds the agent-centric cache for ``cfg.variant`` under the current processing profile."""
    # resolve=False: the composed experiment-centric paths interpolate ${exp_name}, which the builder never uses.
    utils.print_config_tree(cfg, resolve=False, save_to_file=False)

    processor = AgentCentricProcessor(cfg.dataset.config)
    variant = cfg.variant
    variant_dir = Path(cfg.paths.variants_path) / variant
    cache_dir = Path(cfg.paths.ac_cache_path) / variant / processor.processing_profile_hash()

    agent_centric_cacher.build_variant_cache(
        processor,
        variant,
        variant_dir,
        cache_dir,
        num_workers=cfg.num_workers,
        overwrite=cfg.overwrite,
    )


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
