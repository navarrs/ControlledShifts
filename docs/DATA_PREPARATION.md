# Dataset Preparation

## Run the full pipeline from scratch

The pipeline uses **two Python environments**. Only the raw Waymo proto decode needs Python 3.10; everything else runs
in the main Python 3.12 environment.

| Environment | Python | Install | Used for |
|---|---|---|---|
| **Main** | 3.12 | `uv sync` (or `uv run pip install -e .`) | `create_benchmark`, `build_ac_cache`, `train`, `eval`, visualization |
| **Decoder** | 3.10 | `uv venv .venv-waymo --python 3.10` then `uv pip install waymo-open-dataset-tf-2-12-0 numpy tqdm` | decoding raw Waymo protos (and the CausalAgents label scripts) |

The decoder must be a **separate, minimal** environment: `waymo-open-dataset-tf-2-12-0` pins **tensorflow 2.12**, which
conflicts with the project's core **tensorflow 2.19**, so the full project cannot be installed alongside it. The decoder
(`src/controlledshifts/datasets/waymo/preprocessor.py`) is self-contained (no `controlledshifts` imports), so run it
**by file path** with the decoder env's `python` — do **not** use `python -m controlledshifts...`, which would require
installing the conflicting full project into the 3.10 env.

End-to-end steps (the bracket marks which environment each step runs in):

1. **[3.12]** Install the main environment (see *Installation* in the README).
2. **[3.10]** Create the decoder env, download the raw Waymo data (`gsutil`, no Python needed), then decode each raw
   Waymo directory into the flat canonical `base` variant store. `--raw_split` selects which raw Waymo download
   directory to read (`training`/`validation`/`testing`) — it is **not** benchmark splitting (output is always flat;
   benchmark train/val/test lives in `splits/*.json`):
   ```bash
   # in the activated Python 3.10 decoder env, from the repo root
   python src/controlledshifts/datasets/waymo/preprocessor.py --raw_data_path /data/driving/waymo/raw/mini --proc_data_path /data/driving/waymo/variants/base --raw_split training
   python src/controlledshifts/datasets/waymo/preprocessor.py --raw_data_path /data/driving/waymo/raw/mini --proc_data_path /data/driving/waymo/variants/base --raw_split validation
   python src/controlledshifts/datasets/waymo/preprocessor.py --raw_data_path /data/driving/waymo/raw/mini --proc_data_path /data/driving/waymo/variants/base --raw_split testing
   ```
3. **[3.12] Create a benchmark.** A *benchmark* defines how the single `variants/base` pool is partitioned into
   train/validation/testing and — for distribution-shift benchmarks — what perturbed copies of the scenes to generate.
   The split itself is only lists of scenario IDs; no scenario data is moved or duplicated.

   This walkthrough creates two benchmarks as a concrete example — `uniform` (the baseline) and `causal_agents` (a shift
   benchmark, step 4) — but there are **six**; swap `benchmark=<name>` to create any of them. Each induces a different
   train→test distribution shift (or none, for `uniform`). See **[BENCHMARKS.md](BENCHMARKS.md)** for the full
   description and per-benchmark options:
   - `uniform` — IID baseline (no shift); also the reference split the others reuse.
   - `causal_agents` — remove causal / non-causal / static agents to test robustness to agent removal (step 4).
   - `causal_agents_hard` — the same perturbation, but re-split by difficulty (the scenes with the most non-causal agents form the test set).
   - `safeshift`, `ego_safeshift` — re-split by SafeShift safety scores (the most safety-critical scenes become the OOD test set).
   - `environments` — cluster scenes by road-network topology; the hardest clusters form the test set.

   **`uniform`** is the plain IID baseline: it randomly partitions the pool into train/validation/testing (default
   70/15/15, deterministic for a fixed `seed`) with no distribution shift — a control to compare the shift benchmarks
   against. It is also the *reference split* that other benchmarks reuse, so the original and perturbed versions of a
   scene always land in the same bucket.
   ```bash
   uv run -m controlledshifts.create_benchmark benchmark=uniform
   ```
   **Output:** `splits/uniform.json` = `{"training": [ids...], "validation": [ids...], "testing": [ids...]}`. No data is
   copied — training/eval later select these IDs out of `variants/base`.

4. **[3.12, optional — Causal Agents distribution-shift benchmark]** `causal_agents` tests whether a model relies on the
   *right* agents: it removes specific agents from each scene so you can compare predictions with and without them. It
   does **not** compute its own split — it **reuses `splits/uniform.json`** so each perturbed scene stays in the same
   train/val/test bucket as its original. It first needs the per-scenario causal labels, prepared by the **[3.10]** label
   scripts under `src/scripts/` (see *Prepare the Causal Agents (WOMD) Dataset* below).
   ```bash
   uv run -m controlledshifts.create_benchmark benchmark=causal_agents
   ```
   **Output:** four perturbed variant stores next to `base`. Each is a flat set of `<scenario_id>.pkl` with the **same
   scenario IDs** as `base`, but with the selected agents masked out (their trajectories invalidated and dropped from the
   prediction targets):
   - `variants/remove_causal/` — removes the agents labelled **causal** to the ego.
   - `variants/remove_noncausal/` — removes the agents that are **not** causal to the ego (keeps only the causal agents + ego).
   - `variants/remove_noncausalequal/` — removes a **random subset of non-causal** agents **equal in number** to the causal agents (a control for *how many* agents are removed).
   - `variants/remove_static/` — removes **static** agents (start-to-end displacement below a threshold), keeping the ego.

   No new split JSON is written. Training/eval pair these variants with `splits/uniform.json` via `paths=causal_agents`
   (evaluates the `remove_noncausal` perturbation) or `paths=causal_agents_all` (evaluates all four).

5. **[3.12] Build the agent-centric cache** — turn the raw scenarios of a variant into the model-ready tensors, cached
   once per `(variant, processing profile)` under `ac_cache/<variant>/<profile_hash>/`. Build it for each variant you
   will train or evaluate on.

   **`variant=` accepts any variant store under `variants/`:**
   - `base` — the unperturbed scenes (needed by every benchmark).
   - `remove_causal`, `remove_noncausal`, `remove_noncausalequal`, `remove_static` — the four causal_agents perturbations
     (only if you created the `causal_agents` benchmark in step 4, and only the ones you will evaluate).

   **`model=` only matters insofar as it changes the *processing profile*** (the tensor-affecting config — agent/road
   caps, `manually_split_lane`, past/future length, etc.), which is what the cache is keyed by. Models with the same
   profile share one cache, so you do **not** rebuild per model. With the shipped configs there are only **two** profiles:
   - **`autobot`, `cvm`, `mtr`, `naive`, `scenetransformer`, `wayformer`** all share **one** cache — build once with any of
     them (e.g. `model=autobot`).
   - **`mtr_mini`** is the **only** model that needs its **own** cache (it sets `manually_split_lane=true` and larger
     agent/road caps, a distinct profile).

   ```bash
   # Shared profile (covers autobot/cvm/mtr/naive/scenetransformer/wayformer) — build per variant you need:
   uv run -m controlledshifts.build_ac_cache variant=base model=autobot
   uv run -m controlledshifts.build_ac_cache variant=remove_noncausal model=autobot

   # mtr_mini only — its own cache (repeat for each variant you evaluate with mtr_mini):
   uv run -m controlledshifts.build_ac_cache variant=base model=mtr_mini
   ```

6. **[3.12]** Train / evaluate (reads records straight from `ac_cache/<variant>/<profile_hash>/`, selected by the split
   JSON; no reprocessing or copying):
   ```bash
   uv run -m controlledshifts.train model=autobot paths=uniform        # or paths=causal_agents, etc.
   ```

The dataset-specific sections below give the full download/label details for each benchmark.

## Prepare the Waymo Open Motion Dataset (WOMD)

1. Create output path
```bash
mkdir /datasets/waymo/raw/scenario
```

2. Install [gcloud CLI](https://cloud.google.com/sdk/docs/install#deb)

3. Download [Waymo Open Motion Dataset](https://scenarionet.readthedocs.io/en/latest/waymo.html#install-requirements)
```bash
cd /datasets/waymo/raw/scenario
gcloud init
gsutil -m cp -r "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training" .
gsutil -m cp -r "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/validation" .
gsutil -m cp -r "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/testing" .
```

4. **[Optional]** Sample a subset of the data

If skipping this step, simply add a symlink.

```shell
mkdir -p /datasets/waymo/raw
cd /datasets/waymo/raw
ln -s ../motion/scenario/ mini
```

To get a mini subset run the following.
```bash
cd ControlledShifts/src/scripts
uv run waymo_data_selection.py --parallel
```

By default it will randomly select 15% of the files within the training split and copy them to `/datasets/waymo/raw/mini`.

If disk-space is a constraint, delete the original split files:
```bash
cd /datasets/waymo/raw/scenario
rm -rf *
```

5. Prepare the data for training, using **custom processor**:

**NOTE:** the decoder (`src/controlledshifts/datasets/waymo/preprocessor.py`) **requires Python 3.10** and the separate
minimal decoder env from *Run the full pipeline from scratch* above (`waymo-open-dataset-tf-2-12-0` pins tensorflow 2.12,
which conflicts with the project's core tensorflow 2.19). It is self-contained, so run it **by file path** with the
decoder env's `python` — not `python -m ...`, which would need the conflicting full project installed. It writes the raw
decoded dicts; the thin repack to the `Scenario` schema happens later in the main Python 3.12 environment.

Scenarios from every raw split are written **flat** into the canonical `base` variant store (one decoded scenario dict
per `scenario_id`). The raw Waymo origin split of each scenario is recorded in `variants/base/_variant_manifest.json`
for provenance only; benchmark train/val/test splits are defined separately (see below).

```bash
python src/controlledshifts/datasets/waymo/preprocessor.py --raw_data_path /data/driving/waymo/raw/mini --proc_data_path /data/driving/waymo/variants/base --raw_split training
python src/controlledshifts/datasets/waymo/preprocessor.py --raw_data_path /data/driving/waymo/raw/mini --proc_data_path /data/driving/waymo/variants/base --raw_split validation
python src/controlledshifts/datasets/waymo/preprocessor.py --raw_data_path /data/driving/waymo/raw/mini --proc_data_path /data/driving/waymo/variants/base --raw_split testing
```

The training configuration will expect data in `/data/driving` by default (see [`default.yaml`](src/controlledshifts/configs/paths/default.yaml) for example).
It is convenient to create another symlink for it.

```shell
mkdir -p /data/driving
```

## Data layout: variants, splits, and the agent-centric cache

The pipeline keeps a single canonical copy of each *truly different* dataset and re-splits it logically, instead of
copying scenarios per benchmark:

- `variants/<variant>/<scenario_id>.pkl` — the canonical scenario store. `base` is the unperturbed data; each causal
  perturbation (`remove_causal`, `remove_noncausal`, `remove_noncausalequal`, `remove_static`) is its own variant.
- `splits/<benchmark>.json` — each benchmark's train/validation/testing lists of scenario IDs (no data is copied).
- `ac_cache/<variant>/<profile_hash>/` — the agent-centric tensors for a variant, built once per processing profile
  (a hash of the tensor-affecting dataset/model config). `scenarios/<scenario_id>.pkl` holds one scenario's records;
  `_index.pkl` lets the loader assemble its sample list without opening every file; `_profile.json` records the exact
  config and is checked at load time so a mismatched cache is never read silently.

After creating a benchmark split (see [BENCHMARKS.md](BENCHMARKS.md)), build the agent-centric cache once per variant
per processing profile. The `model` selects the profile — e.g. MTR's `manually_split_lane` yields a distinct cache:

```bash
# Base variant for the AutoBot processing profile
uv run -m controlledshifts.build_ac_cache variant=base model=autobot
```

Training/eval then read records straight from `ac_cache/<variant>/<profile_hash>/` — selected by the benchmark split
JSON — with no reprocessing or copying. A benchmark's `paths/*.yaml` declares its sources as `{variant, split_json,
split, tag}`.

## Prepare the Causal Agents (WOMD) Dataset

**NOTE:** Most of these steps require Python 3.10 to run because they depend on the `waymo_open_dataset` package.

1. Download [CausalAgents](https://github.com/google-research/causal-agents/tree/main?tab=readme-ov-file) labels:
```bash
mkdir /datasets/waymo/causal_agents
cd /datasets/waymo/causal_agents
gsutil cp -r "gs://waymo_open_dataset_causal_agents/cusal_labels.tfrecord" .
```

2. Install `protobuf`.

3. Get causal agents [proto](https://github.com/google-research/causal-agents/blob/main/protos/causal_labels.proto), and compile:
```bash
cd ControlledShifts/src/scripts
protoc --python_out=. causal_agents.proto
```

4. Process and verify causal agent dataset.

- Verify the labels have `scenario_id` information that can be traced back to the original validation data
and make the labels be in a format that does not depend on protobuf:
```bash
uv run process_causal_agents_labels.py
```
This script saves a summary file to `meta/summary.json` with the list and number of scenarios processed.

5. **[Optional Sanity Check]**: Cross-match scenario_ids with scenario_ids in original validation set with `find_causal_agent_scenarios.py`
```bash
uv run find_causal_agent_scenarios.py
```
This scripts saves a summary file to `meta/validation_records.json` containing the list of intersecting scenarios for each record file in the validation set, as well as unlabeled scenarios and scenarios not found.

6. Repeat the processing steps from the original Waymo subset.

- Create a uniform split with labeled (causal) data, i.e., using the validation data with:
```bash
uv run waymo_data_selection.py --parallel --input_data_path /datasets/waymo/raw/scenario/validation --output_dir mini_causal --percentage 1.0
```

- Prepare the data for training (write the labeled scenes into the `base` variant store, alongside the rest):
```bash
python src/controlledshifts/datasets/waymo/preprocessor.py --raw_data_path /data/driving/waymo/raw/mini_causal --proc_data_path /data/driving/waymo/variants/base --raw_split training
python src/controlledshifts/datasets/waymo/preprocessor.py --raw_data_path /data/driving/waymo/raw/mini_causal --proc_data_path /data/driving/waymo/variants/base --raw_split validation
python src/controlledshifts/datasets/waymo/preprocessor.py --raw_data_path /data/driving/waymo/raw/mini_causal --proc_data_path /data/driving/waymo/variants/base --raw_split testing
```

- Create the causal benchmark. This reuses the `uniform` split and writes a perturbed variant store under
  `variants/<strategy>/` for every masking strategy (`remove_causal`, `remove_noncausal`, `remove_noncausalequal`,
  `remove_static`). Create the `uniform` split first if you have not already:
```bash
uv run -m controlledshifts.create_benchmark benchmark=uniform
uv run -m controlledshifts.create_benchmark benchmark=causal_agents
```

- Build the agent-centric cache for the base variant and every perturbed variant you intend to evaluate (per
  processing profile; `model=autobot` shown):
```bash
uv run -m controlledshifts.build_ac_cache variant=base model=autobot
uv run -m controlledshifts.build_ac_cache variant=remove_noncausal model=autobot
```

7. **[Optional Sanity Check]**: Verify causal agent IDs exist in processed data:
```bash
uv run verify_agent_ids.py
```

## Prepare the SafeShift (WOMD) Dataset

1. Download the processed splits from [Box](https://cmu.app.box.com/s/ptl5vlsi5uwt6drejnrpcp8a9utfwuzo). This will download a file named `mtr_process_splits.zip` which contains all of the splits generated by SafeShift using different scoring strategies.

2. Unzip the folder downloaded file to `/datasets/`.

3. If not downloaded already, download [Waymo Open Motion Dataset](https://scenarionet.readthedocs.io/en/latest/waymo.html#install-requirements). For this benchmark only the **training** and **validation** sets are required:
```bash
cd /datasets/waymo/raw/scenario
gsutil -m cp -r "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training" .
gsutil -m cp -r "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/validation" .
```

4. Process the scenarios:
```bash
cd ControlledShifts/src/scripts
python src/controlledshifts/datasets/waymo/preprocessor.py --raw_data_path /datasets/waymo/raw/scenario/ --proc_data_path /datasets/safeshift_all --search_safeshift --safeshift_data_splits_path /datasets/mtr_process_splits --safeshift_prefix score_asym_combined_80_ --raw_split training
python src/controlledshifts/datasets/waymo/preprocessor.py --raw_data_path /datasets/waymo/raw/scenario/ --proc_data_path /datasets/safeshift_all --search_safeshift --safeshift_data_splits_path /datasets/mtr_process_splits --safeshift_prefix score_asym_combined_80_ --raw_split validation
```

2. Re-split the processed data:
```bash
cd ControlledShifts/src/scripts
uv run resplit_safeshift.py --scores_path /datasets/mtr_process_splits --scenarios_path /datasets/safeshift_all --output_path /datasets/processed/safeshift --prefix score_asym_combined_80_
```
Check the files inside `mtr_process_splits` for more `prefix` values allowed.

## Prepare the SafeShift-Causal (WOMD) Dataset

1. Make sure the SafeShift subset was prepared as instructed above.

2. **[Optional]** Make a copy of the subset:
```bash
cd /datasets/waymo/processed/
mkdir safeshift_causal
cp -r safeshift/testing safeshit_causal
cp -r safeshift/validation safeshit_causal
```

3. Verify there's no data leakage between the `train` set from Causal Agents and `test/val` sets from SafeShift:
```bash
cd ControlledShifts/src/scripts
uv run verify_safeshift_causal_splits.py
```

## Prepare the Ego-SafeShift-Causal (WOMD) Dataset

This benchmark is similar to **SafeShift-Causal**, but the main difference is that scenarios are scored only from the perspective of the ego-agent. We used the [ScenarioCharacterization](https://github.com/navarrs/ScenarioCharacterization) package to compute the scenario scores and produce the meta file `scenario_to_scores_mapping.csv` required by the benchmark creation script.

1. Follow steps 1 through 6 on preparing the Causal Agents dataset above.

2. Make sure you have [this](https://drive.google.com/file/d/1Ptv1JIM0qymo7180_a5svLZJXWn03yQx/view?usp=drive_link) file in the `./meta` folder. To generate this file follow the scoring [instructions](https://github.com/navarrs/ScenarioCharacterization/blob/main/docs/CHARACTERIZATION.md).

3. Create the benchmark:
```bash
cd ControlledShifts/src/scripts
uv run create_ego_safeshift_benchmark.py
```
