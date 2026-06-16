# Benchmarks

A *benchmark* defines a train→test **distribution shift** over the single canonical scenario pool
(`variants/base`, produced in [DATA_PREPARATION.md](DATA_PREPARATION.md)). It does this by deciding how the pool is
re-split into train/validation/testing and — for the perturbation benchmarks — by generating perturbed copies of the
scenes. See [DATA_PREPARATION.md](DATA_PREPARATION.md) for the full from-scratch pipeline (decode → benchmark → build
agent-centric cache → train).

## How benchmarks are created and consumed

Benchmark creation only computes a **split** (lists of scenario IDs) and, for the causal benchmarks, writes **perturbed
variant stores** — it never copies or duplicates the scenario data.

1. **Create** — `create_benchmark` writes a single split JSON at `${splits_path}/${benchmark_name}.json`:
   ```json
   {
     "benchmark_name": "ego_safeshift",
     "training":   ["scenario_id", "..."],
     "validation": ["..."],
     "testing":    ["..."],
     "invalid":    ["..."]
   }
   ```
   The `training`/`validation`/`testing` lists are mutually exclusive; `invalid` holds scenarios that were considered
   but could not be placed (missing from the input store, missing causal labels, etc.). The `causal_agents` and
   `causal_agents_hard` benchmarks additionally write perturbed variant stores under `variants/<strategy>/` (one flat
   `<scenario_id>.pkl` per scene; see *Causal Agents*).

2. **Build the agent-centric cache** — before training, build the cache for each variant the benchmark uses
   (`base`, plus any perturbation variants): `uv run -m controlledshifts.build_ac_cache variant=<variant> model=<model>`
   (see [DATA_PREPARATION.md](DATA_PREPARATION.md) step 5).

3. **Train / evaluate** — `paths=<benchmark>` selects scenario IDs from the split JSON and reads the matching variant's
   cache. Unperturbed scenes come from the `base` variant; perturbed test sets come from the perturbation variants —
   both indexed by the **same** split, so a scene is paired with its perturbed counterpart in the same bucket.

Common `create_benchmark` options (see `configs/create_benchmark.yaml`):
- `input_data_path`: the canonical scenario store to split. Default: `/data/driving/waymo/variants/base`.
- `splits_path`: directory where the split JSON is written. Default: `/data/driving/waymo/splits`.
- `seed`: makes the split deterministic. Default: `42`.
- `overwrite`: if `false` (default), an existing split JSON and already-prepared perturbed variants are reused; set
  `true` to regenerate.

## Uniform

The plain IID baseline (**no distribution shift**): the pool is split uniformly at random into
training/validation/testing following `split_ratios`, with the same distribution across all three. Use it as a control,
and as the **reference split** the perturbation benchmarks reuse.

**Create:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=uniform   # -> splits/uniform.json
```

Key options (see `configs/benchmark/uniform.yaml`):
- `split_ratios`: `(train, val, test)` fractions. Default: `[0.70, 0.15, 0.15]`. Deterministic for a fixed `seed`.

**Train / evaluate:**
```bash
uv run -m controlledshifts.train model=[model_name] paths=uniform
```

## Causal Agents

Evaluates robustness to **agent removal**: does the model rely on the *right* (causal) agents? Rather than computing its
own split, it **reuses the `uniform` split** (`reference_benchmark`, default `uniform`) so every perturbed scene stays in
the same train/validation/testing bucket as its unperturbed counterpart — create `uniform` first or the run aborts with
a `FileNotFoundError`. As preparation it generates a perturbed variant store for **every** masking strategy, written flat
under `variants/<strategy>/` with the same scenario IDs as `base` (the selected agents have their trajectories
invalidated and are dropped from the prediction targets):

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

Key options (see `configs/benchmark/causal_agents.yaml`):
- `reference_benchmark`: benchmark whose split is reused. Default: `uniform` (must exist first).
- `causal_labels_path`: directory of per-scenario JSON causal labels. Default:
  `/data/driving/waymo/meta/causal_agents/processed_labels/`.
- `prepare_perturbations`: generate the perturbed variant stores up front. Default: `true`.

**Train / evaluate** (build the `base` cache and the perturbation caches you will evaluate first):
```bash
# Original vs the remove_noncausal perturbation:
uv run -m controlledshifts.train model=[model_name] paths=causal_agents
# Original vs all four perturbations:
uv run -m controlledshifts.train model=[model_name] paths=causal_agents_all
```

## Causal Agents Hard

A harder variant of Causal Agents focused on a single perturbation (**remove non-causal**) that **re-splits by
difficulty** instead of reusing `uniform`. Difficulty is the number of non-causal agents per scene: the scenes with the
most non-causal agents form the test set (following `split_ratios`). It writes `splits/causal_agents_hard.json` and
generates the `remove_noncausal` perturbed variant store (reusing existing perturbed files when present), so the
`base` (original) and `remove_noncausal` versions of the same held-out scenes can be compared.

**Create:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=causal_agents_hard
# -> splits/causal_agents_hard.json (+ variants/remove_noncausal/ if not already present)
```

Key options (see `configs/benchmark/causal_agents_hard.yaml`):
- `causal_labels_path`: directory of per-scenario JSON causal labels.
- `perturbed_data_path`: `remove_noncausal` variant store to reuse/populate. Default: `/data/driving/waymo/variants/remove_noncausal/`.
- `split_ratios`: `(train, val, test)` fractions; scenes with the most non-causal agents form the test set. Default: `[0.70, 0.15, 0.15]`.

**Train / evaluate** (trains on `base` under this split, evaluates `base` vs `remove_noncausal` on the hardest held-out scenes):
```bash
uv run -m controlledshifts.train model=[model_name] paths=causal_agents_hard
```

## SafeShift

Evaluates generalization to **safety-critical** scenes. The split is derived from the SafeShift score metadata
(asymmetric-combined scoring): train/val are the In-Distribution (ID) subset, test is the Out-of-Distribution (OOD)
subset. The scenario *data* is still the `base` variant — only the split assignment comes from the SafeShift metadata.

**Create:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=safeshift   # -> splits/safeshift.json
```

Key options (see `configs/benchmark/safeshift.yaml`):
- `scores_path`: directory of SafeShift score metadata (`*_infos.pkl`). Default: `/data/driving/waymo/meta/safeshift/mtr_process_splits`.
- `prefix`: filename prefix for the metadata files. Default: `score_asym_combined_80_`.

**Train / evaluate:**
```bash
# Pure SafeShift ID (train/val) -> OOD (test):
uv run -m controlledshifts.train model=[model_name] paths=safeshift_original
# SafeShift OOD combined with the remove_noncausal causal perturbation at test time:
uv run -m controlledshifts.train model=[model_name] paths=safeshift
```

## Ego-SafeShift

Like SafeShift, but scenes are ranked by **ego-centric** safety scores (ground-truth scoring): the highest-scoring
(hardest) scenes form the OOD test set and the rest form the ID train/val pool, following `split_ratios`.

**Create:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=ego_safeshift \
    scenario_score_mapping_filepath=/data/driving/waymo/meta/ego-safeshift/scores_8/scenario_to_scores_mapping.csv
# -> splits/ego_safeshift.json
```

Key options (see `configs/benchmark/ego_safeshift.yaml`):
- `scenario_score_mapping_filepath` (**required**): CSV with a `scenario_ids` column and a score column.
- `score_type`: column to rank by (higher = harder = test). Default: `gt_critical_continuous_safeshift`.
- `split_ratios`: `(train, val, test)` fractions; the hardest scenes form the test set. Default: `[0.70, 0.15, 0.15]`.

**Producing the score file:** the ego scores come from the [ScenarioCharacterization](https://github.com/navarrs/ScenarioCharacterization/)
package, which scores each scene from the perspective of the ego agent. Follow its scoring
[instructions](https://github.com/navarrs/ScenarioCharacterization/blob/main/docs/CHARACTERIZATION.md) to produce the
`scenario_to_scores_mapping.csv` required above — a CSV with a `scenario_ids` column plus the score column named by
`score_type` (default `gt_critical_continuous_safeshift`), e.g. at `meta/ego-safeshift/scores_8/scenario_to_scores_mapping.csv`.
A precomputed file is available [here](https://drive.google.com/file/d/1Ptv1JIM0qymo7180_a5svLZJXWn03yQx/view?usp=drive_link); place it in the `./meta` folder.

**Train / evaluate:**
```bash
uv run -m controlledshifts.train model=[model_name] paths=ego_safeshift
```

## Environments

Evaluates generalization across **road-topology** types. Scenes are clustered by map structure (NetLSD graph
descriptors) and clusters are ranked by hardness; the hardest clusters form the OOD test set. The clustering artifacts
(model, scatter plots, per-cluster examples) are written under `cache_path`; the split assignment is written to
`splits/environments.json`.

**Create:**
```bash
uv run -m controlledshifts.create_benchmark benchmark=environments   # -> splits/environments.json
```

Key options (see `configs/benchmark/environments.yaml`):
- `clustering_algorithm`: `kmeans` / `ward` / `spectral` / etc. Default: `ward`.
- `n_clusters`: number of clusters. Default: `10`.
- `hardness_metric`: how cluster hardness is ranked — `silhouette` (lowest = hardest) or `dbi` (highest = hardest). Default: `silhouette`.
- `sample_percentage` / `num_scenarios`: fraction (or count) of scenes used to fit the clustering model.
- `cache_path`: directory for clustering artifacts. Default: `/data/driving/waymo/meta/environments`.

**Train / evaluate:**
```bash
uv run -m controlledshifts.train model=[model_name] paths=environments
```
