# Benchmarks

A *benchmark* defines a train→test **distribution shift** over the single canonical scenario pool (`variants/base`, produced in [DATA_PREPARATION.md](DATA_PREPARATION.md)). It does this by deciding how the pool is re-split into train/validation/testing and — for the perturbation benchmarks — by generating perturbed copies of the scenes. See [DATA_PREPARATION.md](DATA_PREPARATION.md) for the full from-scratch pipeline (decode → benchmark → build agent-centric cache → train).

## How benchmarks are created and consumed

Benchmark creation only computes a **split** (lists of scenario IDs) and, for the causal benchmarks, writes **perturbed variant stores** — it never copies or duplicates the scenario data.

1. **Create** — `create_benchmark` writes a single split JSON at `${splits_path}/${benchmark_name}.json` with keys `benchmark_name`, `training`, `validation`, `testing` and `invalid`. The three split lists are mutually exclusive; `invalid` holds scenarios that were considered but could not be placed (missing from the input store, missing causal labels, etc.). The `causal_agents` and `causal_agents_hard` benchmarks additionally write perturbed variant stores under `variants/<strategy>/` (one flat `<scenario_id>.pkl` per scene; see [Causal Agents](#causal-agents)).

2. **Build the agent-centric cache** — before training, build the cache for each variant the benchmark uses (`base`, plus any perturbation variants): `uv run -m controlledshifts.build_ac_cache variant=<variant> model=<model>` (see [DATA_PREPARATION.md](DATA_PREPARATION.md) step 5).

3. **Train / evaluate** — `paths=<benchmark>` selects scenario IDs from the split JSON and reads the matching variant's cache. Unperturbed scenes come from the `base` variant; perturbed test sets come from the perturbation variants — both indexed by the **same** split, so a scene is paired with its perturbed counterpart in the same bucket.

Common `create_benchmark` options (see [`create_benchmark.yaml`](../src/controlledshifts/configs/create_benchmark.yaml)):

| Option | Default | Description |
|---|---|---|
| `input_data_path` | `/data/driving/waymo/variants/base` | The canonical scenario store to split. |
| `splits_path` | `/data/driving/waymo/splits` | Directory where the split JSON is written. |
| `seed` | `42` | Makes the split deterministic. |
| `overwrite` | `false` | When `false`, an existing split JSON and already-prepared perturbed variants are reused. Set `true` to regenerate. |

## Supported Benchmarks

| Benchmark | Description |
|---|---|
| [Uniform](#uniform) | IID control — random split, no shift. Also the reference split reused by Causal Agents. |
| [Causal Agents](#causal-agents) | Agent-removal robustness. Reuses the Uniform split and generates four perturbed variants. |
| [Causal Agents Hard](#causal-agents-hard) | Re-splits by non-causal agent count so the densest scenes form the test set; `remove_noncausal` only. |
| [SafeShift](#safeshift) | Safety-critical ID→OOD split derived from precomputed SafeShift score metadata. |
| [Ego-SafeShift](#ego-safeshift) | Ranks scenes by ego-centric safety score; the hardest form the OOD test set. |
| [Environments](#environments) | Clusters scenes by road topology (NetLSD descriptors); the hardest clusters form the OOD test set. |

Each benchmark is selected at creation with `benchmark=<name>` and at train/eval time with `paths=<name>`. Two benchmarks expose more than one `paths` option: `causal_agents` also has `causal_agents_all` (evaluates all four perturbations), and `safeshift` also has `safeshift_original` (pure ID→OOD, no perturbation). A separate `paths=mini` option evaluates a model trained on the unshifted `mini` variant against several benchmarks' OOD test sets at once.

## Uniform

The plain IID baseline (**no distribution shift**): the pool is split uniformly at random into training/validation/testing following `split_ratios`, with the same distribution across all three. Use it as a control, and as the **reference split** the perturbation benchmarks reuse.

**Create:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=uniform   # -> splits/uniform.json
```

Key options (see [`benchmark/uniform.yaml`](../src/controlledshifts/configs/benchmark/uniform.yaml)):

| Option | Default | Description |
|---|---|---|
| `split_ratios` | `[0.70, 0.15, 0.15]` | `(train, val, test)` fractions. Deterministic for a fixed `seed`. |

**Train / evaluate:**
```bash
uv run -m controlledshifts.train model=[model_name] paths=uniform
```

## Causal Agents

Evaluates robustness to **agent removal**: does the model rely on the *right* (causal) agents? Rather than computing its own split, it **reuses the `uniform` split** so every perturbed scene stays in the same train/validation/testing bucket as its unperturbed counterpart.

> [!WARNING]
> Create the `uniform` benchmark first, or the run aborts with a `FileNotFoundError`.

As preparation it generates a perturbed variant store for **every** masking strategy, written flat under `variants/<strategy>/` with the same scenario IDs as `base` (the selected agents have their trajectories invalidated and are dropped from the prediction targets):

- `variants/remove_causal/` — removes the agents labelled **causal** to the ego.
- `variants/remove_noncausal/` — removes the agents **not** causal to the ego (keeps only the causal agents + ego).
- `variants/remove_noncausalequal/` — removes a **random subset of non-causal** agents **equal in number** to the causal
  agents (a control for *how many* agents are removed).
- `variants/remove_static/` — removes **static** agents (start-to-end displacement below a threshold), keeping the ego.

The unperturbed "original" scenes are not re-stored — they are the `base` variant, paired with the perturbed variants by
the shared `uniform` split.

**Create:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=uniform          # reference split (run once)
uv run -m controlledshifts.create_benchmark benchmark=causal_agents    # -> variants/remove_{causal,noncausal,noncausalequal,static}/
```

Key options (see [`benchmark/causal_agents.yaml`](../src/controlledshifts/configs/benchmark/causal_agents.yaml)):

| Option | Default | Description |
|---|---|---|
| `reference_benchmark` | `uniform` | Benchmark whose split is reused. Must exist first. |
| `causal_labels_path` | `/data/driving/waymo/meta/causal_agents/processed_labels/` | Directory of per-scenario JSON causal labels. |
| `prepare_perturbations` | `true` | Generate the perturbed variant stores up front. |

**Train / evaluate** (build the `base` cache and the perturbation caches you will evaluate first):
```bash
# Original vs the remove_noncausal perturbation:
uv run -m controlledshifts.train model=[model_name] paths=causal_agents
# Original vs all four perturbations:
uv run -m controlledshifts.train model=[model_name] paths=causal_agents_all
```

## Causal Agents Hard

A harder variant of Causal Agents focused on a single perturbation (**remove non-causal**) that **re-splits by difficulty** instead of reusing `uniform`. Difficulty is the number of non-causal agents per scene: the scenes with the most non-causal agents form the test set (following `split_ratios`). It writes `splits/causal_agents_hard.json` and generates the `remove_noncausal` perturbed variant store (reusing existing perturbed files when present), so the `base` (original) and `remove_noncausal` versions of the same held-out scenes can be compared.

**Create:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=causal_agents_hard
# -> splits/causal_agents_hard.json (+ variants/remove_noncausal/ if not already present)
```

Key options (see [`benchmark/causal_agents_hard.yaml`](../src/controlledshifts/configs/benchmark/causal_agents_hard.yaml)):

| Option | Default | Description |
|---|---|---|
| `causal_labels_path` | `/data/driving/waymo/meta/causal_agents/processed_labels/` | Directory of per-scenario JSON causal labels. |
| `perturbed_data_path` | `/data/driving/waymo/variants/remove_noncausal/` | `remove_noncausal` variant store to reuse or populate. |
| `split_ratios` | `[0.70, 0.15, 0.15]` | `(train, val, test)` fractions; scenes with the most non-causal agents form the test set. |

**Train / evaluate** (trains on `base` under this split, evaluates `base` vs `remove_noncausal` on the hardest held-out scenes):
```bash
uv run -m controlledshifts.train model=[model_name] paths=causal_agents_hard
```

## SafeShift

Evaluates generalization to **safety-critical** scenes. The split is derived from the SafeShift score metadata (asymmetric-combined scoring): train/val are the In-Distribution (ID) subset, test is the Out-of-Distribution (OOD) subset. The scenario *data* is still the `base` variant — only the split assignment comes from the SafeShift metadata.

**Create:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=safeshift   # -> splits/safeshift.json
```

Key options (see [`benchmark/safeshift.yaml`](../src/controlledshifts/configs/benchmark/safeshift.yaml)):

| Option | Default | Description |
|---|---|---|
| `scores_path` | `/data/driving/waymo/meta/safeshift/mtr_process_splits` | Directory of SafeShift score metadata (`*_infos.pkl`). |
| `prefix` | `score_asym_combined_80_` | Filename prefix for the metadata files. |
| `plot_scores` | `false` | Write a score-density plot alongside the split. |

**Train / evaluate:**
```bash
# Pure SafeShift ID (train/val) -> OOD (test):
uv run -m controlledshifts.train model=[model_name] paths=safeshift_original
# SafeShift OOD combined with the remove_noncausal causal perturbation at test time:
uv run -m controlledshifts.train model=[model_name] paths=safeshift
```

## Ego-SafeShift

Like SafeShift, but scenes are ranked by **ego-centric** safety scores (ground-truth scoring): the highest-scoring (hardest) scenes form the OOD test set and the rest form the ID train/val pool, following `split_ratios`.

**Create (with a precomputed score CSV):**
```bash
uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift split_name=ego_safeshift \
    scenario_score_mapping_filepath=/data/driving/waymo/meta/ego-safeshift/scores_8/scenario_to_scores_mapping.csv
# -> splits/ego_safeshift.json
```

**Create (no CSV — scores are computed on the fly):**
```bash
uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift split_name=ego_safeshift_scores8
# -> computes scores via the SafeShift API, caches them under splits/ego_safeshift/, writes splits/ego_safeshift_scores8.json
```
When `scenario_score_mapping_filepath` is null or missing, the benchmark scores every scenario under `input_data_path` using the SafeShift characterization API (the shared `scoring` config group), writing the per-scenario scores to `${splits_path}/ego_safeshift/scenario_to_scores_mapping_<hash>.csv` (keyed by a hash of the scoring config, so changing the scoring config writes a new file; reused on re-runs unless `overwrite=true`).

Key options (see [`benchmark/ego_safeshift.yaml`](../src/controlledshifts/configs/benchmark/ego_safeshift.yaml)):

| Option | Default | Description |
|---|---|---|
| `scenario_score_mapping_filepath` | `null` | CSV with a `scenario_ids` column and a score column. When null or missing, scores are computed (see above). |
| `score_type` | `gt_critical_continuous_safeshift` | Column to rank by (higher = harder = test). Computed CSVs contain `gt_critical_continuous_{safeshift,individual,interaction}`. |
| `split_ratios` | `[0.70, 0.15, 0.15]` | `(train, val, test)` fractions; the hardest scenes form the test set. |
| `split_name` | `null` | Split JSON filename stem. When null, an `ego_safeshift_<tag>` name is auto-derived (keyed by the score source, `score_type`, `split_ratios` and `seed`) so multiple score CSVs each get their own file. Set it (e.g. `ego_safeshift_scores8`) to reference the split deterministically from [`configs/paths/`](../src/controlledshifts/configs/paths/). |

**Producing the score file (optional):** the score CSV can instead come from [ScenarioCharacterization](https://github.com/navarrs/ScenarioCharacterization/) — see its [scoring instructions](https://github.com/navarrs/ScenarioCharacterization/blob/main/docs/CHARACTERIZATION.md). A precomputed CSV is available [here](https://drive.google.com/file/d/1Ptv1JIM0qymo7180_a5svLZJXWn03yQx/view?usp=drive_link); place it under `./meta`.

**Train / evaluate:**
```bash
uv run -m controlledshifts.train model=[model_name] paths=ego_safeshift
```

## Environments

Evaluates generalization across **road-topology** types. Scenes are clustered by map structure (NetLSD graph descriptors) and clusters are ranked by hardness; the hardest clusters form the OOD test set. The clustering artifacts (model, scatter plots, per-cluster examples) are written under `cache_path`; the split assignment is written to `splits/environments.json`.

**Create:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=environments   # -> splits/environments.json
```

Key options (see [`benchmark/environments.yaml`](../src/controlledshifts/configs/benchmark/environments.yaml)):

| Option | Default | Description |
|---|---|---|
| `clustering_algorithm` | `ward` | One of `kmeans`, `hdbscan`, `agglomerative`, `ward`, `spectral`. |
| `n_clusters` | `10` | Number of clusters. |
| `hardness_metric` | `silhouette` | How cluster hardness is ranked — `silhouette` (lowest = hardest) or `dbi` (highest = hardest). |
| `sample_percentage` | `1.0` | Fraction of scenes used to fit the clustering model. |
| `num_scenarios` | `null` | Explicit scene count for fitting; overrides `sample_percentage` when set. |
| `cache_path` | `/data/driving/waymo/meta/environments` | Directory for clustering artifacts. |

**Train / evaluate:**
```bash
uv run -m controlledshifts.train model=[model_name] paths=environments
```
