# Benchmarks

## Waymo (default)

No benchmark creation step required. Use the default Waymo paths.

**Training and evaluation:**
```bash
uv run -m controlledshifts.train model=[model_name] paths=waymo
```

## Causal Agents

Evaluates robustness to causal agent perturbations. Scenarios are derived from the mini-causal dataset by masking specific agent categories (causal, non-causal, or static).

**Creating the benchmark:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=causal_agents \
    input_data_path=/data/driving/waymo/processed/mini_causal \
    output_data_path=/data/driving/waymo/processed/causal_agents \
    causal_labels_path=/data/driving/waymo/causal_agents/processed_labels \
    strategy=remove_causal
```

Key options (see `configs/benchmark/causal_agents.yaml`):
- `causal_labels_path`: directory containing per-scenario JSON causal labels.
- `strategy`: one of `remove_causal`, `remove_noncausal`, `remove_noncausalequal`, `remove_static`.

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

## Non-Causal Agents

A harder variant of Causal Agents that focuses on a single perturbation — removing non-causal agents — and re-organizes scenarios by difficulty instead of reusing the original mini-causal splits. Difficulty is the number of non-causal agents in a scenario: scenarios are sorted ascending by that count and the hardest ones (at or above `cutoff_percentile`) form the test set. Each scenario is materialized twice under the same split: an unperturbed `original` copy and a `perturbed` copy with non-causal agents removed, so the original and perturbed versions of the same held-out scenes can be compared. Existing `remove_noncausal` perturbed files are reused when found and generated on the fly otherwise.

**Creating the benchmark:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=non_causal_agents \
    input_data_path=/data/driving/waymo/processed/mini_causal \
    output_data_path=/data/driving/waymo/processed/non_causal_agents \
    causal_labels_path=/data/driving/waymo/causal_agents/processed_labels \
    perturbed_data_path=/data/driving/waymo/processed/remove_noncausal
```

Key options (see `configs/benchmark/non_causal_agents.yaml`):
- `causal_labels_path`: directory containing per-scenario JSON causal labels.
- `perturbed_data_path`: existing `remove_noncausal` output to reuse; missing scenarios are generated on the fly.
- `cutoff_percentile`: percentile of non-causal counts at/above which scenarios go to the test set. Default: `80.0`.
- `validation_percentage`: percentage of the train/val pool to hold out for validation. Default: `10.0`.

**Training and evaluation:**

`paths=non_causal_agents` trains on the reorganized original splits and evaluates on the original and perturbed versions of the hardest held-out scenes:
```bash
uv run -m controlledshifts.train model=[model_name] paths=non_causal_agents
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

Evaluates generalization to ego-safety-critical scenarios. Splits are derived from the causal dataset by filtering scenarios using ego safety scores from the ground-truth scoring strategy. Scenarios below the cutoff percentile form the ID train/val pool; scenarios above form the OOD test set.

**Creating the benchmark:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift \
    input_data_path=/data/driving/waymo/processed/mini_causal \
    output_data_path=/data/driving/waymo/processed/causal_ego_safeshift \
    scenario_score_mapping_filepath=meta/scenario_to_scores_mapping.csv
```

Key options (see `configs/benchmark/ego_safeshift.yaml`):
- `scenario_score_mapping_filepath`: CSV file with columns `scenario_ids` and a score column.
- `score_type`: column to use for filtering. Default: `gt_critical_continuous_safeshift`.
- `cutoff_percentile`: percentile threshold separating ID from OOD. Default: `80.0`.
- `validation_percentage`: percentage of the ID pool to hold out for validation. Default: `10.0`.

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
