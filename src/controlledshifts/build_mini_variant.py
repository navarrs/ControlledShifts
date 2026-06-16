r"""Builds a non-overlapping, same-size ``mini`` scenario variant from an external processed-scenario dump.

We have a large dump of processed Waymo training scenarios (``sample_<scenario_id>.pkl`` files) that share the exact
Stage-A raw scenario schema produced by our own ``waymo`` preprocessor. This script carves out a new variant that:

* has the same number of scenarios as a reference variant (``base`` by default),
* shares zero scenario IDs with that reference variant,
* is renamed from ``sample_<scenario_id>.pkl`` to our canonical ``<scenario_id>.pkl`` convention, and
* lands under ``${variants_path}/<variant_name>`` alongside a ``_variant_manifest.json`` and a matching split JSON.

Selection is seeded, so re-runs reproduce the same subset. Copies are skipped if already present, so the script is
safe to resume.

Example usage:

    # Default: 44,097 scenarios (matching base), copied into variants/mini, split mirroring base (30869/6614/6614)
    uv run -m controlledshifts.build_mini_variant

    # Hardlink instead of copy (same filesystem, ~0 extra space) and regenerate the split
    uv run -m controlledshifts.build_mini_variant --link --overwrite
"""

import argparse
import json
import multiprocessing
import os
import shutil
from pathlib import Path

import numpy as np
import pyrootutils
from tqdm import tqdm

from controlledshifts import utils
from controlledshifts.benchmarks.common import BenchmarkSplit, check_overlap, save_benchmark_split


_LOGGER = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

POOL_PREFIX = "sample_"


def _list_scenario_ids(directory: Path, prefix: str = "") -> set[str]:
    """Returns the scenario IDs of every ``*.pkl`` in ``directory``, stripping ``prefix`` from the filename stem."""
    ids: set[str] = set()
    for entry in directory.iterdir():
        if entry.suffix != ".pkl":
            continue
        stem = entry.stem
        if prefix and stem.startswith(prefix):
            stem = stem[len(prefix) :]
        ids.add(stem)
    return ids


def _materialize_scenario(task: tuple[Path, Path, bool]) -> None:
    """Copies (or hardlinks) one source pickle to its renamed destination, skipping if it already exists."""
    src, dst, use_link = task
    if dst.exists():
        return
    if use_link:
        os.link(src, dst)
    else:
        shutil.copyfile(src, dst)


def build_mini_variant(  # noqa: PLR0913
    *,
    pool_dir: Path,
    variants_path: Path,
    splits_path: Path,
    variant_name: str,
    reference_variant: str,
    count: int | None,
    seed: int,
    val_size: int,
    test_size: int,
    num_workers: int,
    use_link: bool,
    overwrite: bool,
) -> None:
    """Selects, copies, and splits a non-overlapping same-size subset into a new variant store."""
    reference_dir = variants_path / reference_variant
    output_dir = variants_path / variant_name

    base_ids = _list_scenario_ids(reference_dir)
    pool_ids = _list_scenario_ids(pool_dir, prefix=POOL_PREFIX)
    eligible = sorted(pool_ids - base_ids)
    target = count if count is not None else len(base_ids)
    _LOGGER.info(
        "Reference '%s': %d scenarios | pool: %d scenarios | eligible (pool - reference): %d | target: %d",
        reference_variant,
        len(base_ids),
        len(pool_ids),
        len(eligible),
        target,
    )
    if len(eligible) < target:
        msg = f"Not enough non-overlapping scenarios: need {target}, only {len(eligible)} available."
        raise ValueError(msg)

    train_size = target - val_size - test_size
    if train_size <= 0:
        msg = f"Invalid split sizes: train={train_size} (target={target}, val={val_size}, test={test_size})."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(np.array(eligible), size=target, replace=False).tolist())

    # Copy + rename: sample_<id>.pkl -> <id>.pkl.
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(pool_dir / f"{POOL_PREFIX}{sid}.pkl", output_dir / f"{sid}.pkl", use_link) for sid in selected]
    verb = "Hardlinking" if use_link else "Copying"
    _LOGGER.info("%s %d scenarios into %s", verb, len(tasks), output_dir)
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(_materialize_scenario, tasks, chunksize=64), total=len(tasks), desc=verb))

    # Provenance manifest: every selected scenario originates from the Waymo training dump.
    manifest_path = output_dir / "_variant_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(dict.fromkeys(selected, "training"), f)
    _LOGGER.info("Wrote variant manifest at %s (%d scenarios)", manifest_path, len(selected))

    # Split JSON mirroring the reference split sizes, using a seeded shuffle for the partition.
    shuffled = rng.permutation(np.array(selected)).tolist()
    split = BenchmarkSplit(
        training=sorted(shuffled[:train_size]),
        validation=sorted(shuffled[train_size : train_size + val_size]),
        testing=sorted(shuffled[train_size + val_size :]),
        invalid=[],
        benchmark_name=variant_name,
    )
    check_overlap(split)
    save_benchmark_split(split, variant_name, splits_path, overwrite=overwrite)


def main() -> None:
    """CLI entry point: parses arguments and builds the variant."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-dir", type=Path, default=Path("/data/waymo_mtr/new_processed_scenarios_training"))
    parser.add_argument("--variants-path", type=Path, default=Path("/data/driving/waymo/variants"))
    parser.add_argument("--splits-path", type=Path, default=Path("/data/driving/waymo/splits"))
    parser.add_argument("--variant-name", default="mini")
    parser.add_argument(
        "--reference-variant", default="base", help="Variant whose IDs are excluded and whose size is matched."
    )
    parser.add_argument("--count", type=int, default=None, help="Subset size; defaults to the reference variant size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=6614)
    parser.add_argument("--test-size", type=int, default=6614)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--link", action="store_true", help="Hardlink instead of copying (same filesystem, ~0 space).")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate the split JSON if it already exists.")
    args = parser.parse_args()

    build_mini_variant(
        pool_dir=args.pool_dir,
        variants_path=args.variants_path,
        splits_path=args.splits_path,
        variant_name=args.variant_name,
        reference_variant=args.reference_variant,
        count=args.count,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=args.num_workers,
        use_link=args.link,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
