# Benchmarks

## Splits and data copying

Benchmark creation is split into two stages. Every `create_benchmark` run first **computes** the
training/validation/testing assignment and saves it as a single JSON file at `${splits_path}/${benchmark_name}.json`:

```json
{
  "benchmark_name": "ego_safeshift",
  "training":   ["scenario_id", "..."],
  "validation": ["..."],
  "testing":    ["..."],
  "invalid":    ["..."]
}
```

The `training`/`validation`/`testing` lists are mutually exclusive; `invalid` holds scenarios that were considered but
could not be placed (missing from the input directory, missing causal labels, etc.). Pass `copy_splits=true` to also
**copy** the data into `training/validation/testing` subdirectories under each benchmark's `output_data_path` after the
splits are saved. Without `copy_splits`, only the JSON files are produced. Key options (see
`configs/create_benchmark.yaml`):
- `splits_path`: directory where the split JSON files are written. Default: `/data/driving/waymo/splits`.
- `copy_splits`: if true, organize the input (and any prepared perturbed datasets) into split subdirectories. Default: `false`.

## Waymo (default)

No benchmark creation step required. Use the default Waymo paths.

**Training and evaluation:**
```bash
uv run -m controlledshifts.train model=[model_name] paths=waymo
```

## Uniform

The plain IID baseline (no distribution shift). The input scenarios are split uniformly at random into training/validation/testing following `split_ratios`, with the same distribution across all three splits. Use it as a control to compare against the shift-inducing benchmarks below.

**Creating the benchmark:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=uniform \
    input_data_path=/data/driving/waymo/processed/mini
```

Key options (see `configs/benchmark/uniform.yaml`):
- `split_ratios`: `(train, val, test)` fractions of the full dataset. Default: `[0.70, 0.15, 0.15]`. The split is deterministic for a fixed `seed`.

## Causal Agents

Evaluates robustness to causal agent perturbations. Rather than computing its own split, this benchmark **reuses the split of a reference benchmark** (`reference_benchmark`, by default `uniform`), so the perturbed scenes land in the same train/validation/testing buckets as their unperturbed counterparts. Create the reference benchmark first — `uv run -m controlledshifts.create_benchmark benchmark=uniform copy_splits=true` — or the run aborts with a `FileNotFoundError`. As a preparation step, the perturbed dataset for **every** masking strategy (causal, non-causal, non-causal-equal, static) is generated up front and written flat under `output_data_path/<strategy>/`, mirroring the input dataset with no split subdirectories. With `copy_splits=true`, each perturbed dataset is organized into `output_data_path/causal_agents/<strategy>/{training,validation,testing}`. The unperturbed "original" data is **not** re-copied: it is served directly from the reference benchmark's split directories (e.g. `processed/uniform/{training,validation,testing}`), so the same held-out scenes can be compared with and without the perturbation.

**Creating the benchmark:**
```bash
# Create the reference (uniform) split first.
uv run -m controlledshifts.create_benchmark benchmark=uniform copy_splits=true

uv run -m controlledshifts.create_benchmark benchmark=causal_agents \
    input_data_path=/data/driving/waymo/processed/mini_causal \
    causal_labels_path=/data/driving/waymo/causal_agents/processed_labels \
    copy_splits=true
```

Key options (see `configs/benchmark/causal_agents.yaml`):
- `reference_benchmark`: benchmark whose saved split is reused and whose directories serve the unperturbed data. Default: `uniform`. It must be created first.
- `causal_labels_path`: directory containing per-scenario JSON causal labels.
- `prepare_perturbations`: if true, generate the flat perturbed dataset for every strategy up front. Default: `true`.

**Training and evaluation:**

`paths=causal_agents_all` evaluates against all perturbation strategies:
```bash
uv run -m controlledshifts.train model=[model_name] paths=causal_agents_all
```

`paths=causal_agents` evaluates against `remove_causal` and `remove_noncausal` only:
```bash
uv run -m controlledshifts.train model=[model_name] paths=causal_agents
```

Test subsets for `causal_agents_all`:
- *Original*: unperturbed scenes.
- *Remove causal*: removes agents labeled as causal to the ego.
- *Remove non-causal*: removes agents not labeled as causal to the ego.
- *Remove non-causal-equal*: removes N non-causal agents, where N equals the number of causal agents.
- *Remove static*: removes agents whose motion is below a threshold.

## Causal Agents Hard

A harder variant of Causal Agents that focuses on a single perturbation — removing non-causal agents — and re-organizes scenarios by difficulty instead of reusing the original mini-causal splits. Difficulty is the number of non-causal agents in a scenario: the scenarios with the most non-causal agents form the test set (following `split_ratios`). Each scenario is materialized twice under the same split: an unperturbed `original` copy and a `remove_noncausal` copy with non-causal agents removed (matching the folder naming of the Causal Agents benchmark), so the original and perturbed versions of the same held-out scenes can be compared. Existing `remove_noncausal` perturbed files are reused when found and generated on the fly otherwise.

**Creating the benchmark:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=causal_agents_hard \
    input_data_path=/data/driving/waymo/processed/mini_causal \
    output_data_path=/data/driving/waymo/processed/causal_agents_hard \
    causal_labels_path=/data/driving/waymo/causal_agents/processed_labels \
    perturbed_data_path=/data/driving/waymo/processed/remove_noncausal
```

Key options (see `configs/benchmark/causal_agents_hard.yaml`):
- `causal_labels_path`: directory containing per-scenario JSON causal labels.
- `perturbed_data_path`: existing `remove_noncausal` output to reuse; missing scenarios are generated on the fly.
- `split_ratios`: `(train, val, test)` fractions of the full dataset; scenarios with the most non-causal agents form the test set. Default: `[0.70, 0.15, 0.15]`.

**Training and evaluation:**

`paths=causal_agents_hard` trains on the reorganized original splits and evaluates on the original and perturbed versions of the hardest held-out scenes:
```bash
uv run -m controlledshifts.train model=[model_name] paths=causal_agents_hard
```

## SafeShift

Evaluates generalization to safety-critical scenarios. Splits are derived from the full SafeShift dataset using the asymmetric-combined scoring strategy. Train/val use the In-Distribution (ID) subset; test uses the Out-of-Distribution (OOD) subset.

**Creating the benchmark:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=safeshift \
    input_data_path=/datasets/scenarios/safeshift_all \
    output_data_path=/data/driving/waymo/processed/safeshift \
    scores_path=/datasets/waymo/mtr_process_splits \
    prefix=score_asym_combined_80_
```

Key options (see `configs/benchmark/safeshift.yaml`):
- `scores_path`: directory containing the SafeShift score metadata (`*_infos.pkl` files).
- `prefix`: filename prefix used to locate the metadata files. Default: `score_asym_combined_80_`.

**Training and evaluation:**

`paths=safeshift` trains on the SafeShift ID subset and evaluates on OOD:
```bash
uv run -m controlledshifts.train model=[model_name] paths=safeshift
```

`paths=safeshift_causal` combines the Causal Agents and SafeShift benchmarks, training on mini-causal data and evaluating on both:
```bash
uv run -m controlledshifts.train model=[model_name] paths=safeshift_causal
```

## Ego-SafeShift

Evaluates generalization to ego-safety-critical scenarios. Splits are derived from the causal dataset by ranking scenarios using ego safety scores from the ground-truth scoring strategy. The highest-scoring (hardest) scenarios form the OOD test set; the remainder forms the ID train/val pool, following `split_ratios`.

**Creating the benchmark:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift \
    input_data_path=/data/driving/waymo/processed/mini_causal \
    output_data_path=/data/driving/waymo/processed/causal_ego_safeshift \
    scenario_score_mapping_filepath=meta/scenario_to_scores_mapping.csv
```

Key options (see `configs/benchmark/ego_safeshift.yaml`):
- `scenario_score_mapping_filepath`: CSV file with columns `scenario_ids` and a score column.
- `score_type`: column to rank scenarios by (higher = harder = test). Default: `gt_critical_continuous_safeshift`.
- `split_ratios`: `(train, val, test)` fractions of the full dataset; the hardest scenarios form the test set. Default: `[0.70, 0.15, 0.15]`.

**NOTE:** The ego scores were computed using the [ScenarioCharacterization](https://github.com/navarrs/ScenarioCharacterization/) package. Please refer to the repository for instructions on how to obtain the scores. Here,
we faciliate `scenario_to_scores_mapping.csv` which maps each scenario to their ego-score.

**Training and evaluation:**

`paths=ego_safeshift_causal` trains on the Ego-SafeShift ID subset and evaluates on OOD:
```bash
uv run -m controlledshifts.train model=[model_name] paths=ego_safeshift_causal
```

## Environments

Evaluates generalization across road topology types. Scenarios are clustered by map structure using NetLSD graph descriptors and KMeans; splits are assigned based on cluster hardness (silhouette score).

**Creating the benchmark:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=environments \
    input_data_path=/datasets/waymo/processed/mini_causal \
    output_data_path=/datasets/waymo/processed/environment_benchmark
```

Key options (see `configs/benchmark/environments.yaml`):
- `n_clusters`: number of KMeans clusters. Default: `10`.
- `sample_percentage`: fraction of scenarios used to fit the model. Default: `0.20`.
- `reduction`: dimensionality reduction for the scatter plot — `pca` or `tsne`. Default: `pca`.

Training and evaluation paths are under development.
