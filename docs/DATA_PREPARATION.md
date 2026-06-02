# Dataset Preparation

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
cd scene-tokens/src/scripts
uv run waymo_data_selection.py --parallel
```

By default it will randomly select 15% of the files within the training split and copy them to `/datasets/waymo/raw/mini`.

If disk-space is a constraint, delete the original split files:
```bash
cd /datasets/waymo/raw/scenario
rm -rf *
```

5. Prepare the data for training, using **custom processor**:

**NOTE:** this script was borrowed from [here](https://github.com/navarrs/ScenarioCharacterization/blob/main/src/characterization/utils/datasets/waymo_preprocess.py).
**It requires Python 3.10 to run**.
You may use `uv venv <name> --python 3.10` to create a secondary environment for running it.
You will need `uv pip install waymo-open-dataset-tf-2-12-0` after activation to install the dependency needed to deserialize the proto files.
You may need to use `python <script_path>` to run the script rather than `uv run` which invokes the primary (default) virtual environment intended for the main project.

```bash
uv run waymo_data_processing.py --raw_data_path /datasets/waymo/raw/mini --proc_data_path /datasets/waymo/processed/mini --split training
uv run waymo_data_processing.py --raw_data_path /datasets/waymo/raw/mini --proc_data_path /datasets/waymo/processed/mini --split validation
uv run waymo_data_processing.py --raw_data_path /datasets/waymo/raw/mini --proc_data_path /datasets/waymo/processed/mini --split testing
```

The training configuration will expect data in `/data/driving` by default (see [`default.yaml`](src/controlledshifts/configs/paths/default.yaml) for example).
It is convenient to create another symlink for it.

```shell
mkdir -p /data/driving
```

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
cd scene-tokens/src/scripts
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

- Prepare the data for training:
```bash
uv run waymo_data_processing.py --raw_data_path /datasets/waymo/raw/mini_causal --proc_data_path /datasets/waymo/processed/mini_causal --split training
uv run waymo_data_processing.py --raw_data_path /datasets/waymo/raw/mini_causal --proc_data_path /datasets/waymo/processed/mini_causal --split validation
uv run waymo_data_processing.py --raw_data_path /datasets/waymo/raw/mini_causal --proc_data_path /datasets/waymo/processed/mini_causal --split testing
```

- Prepare the causal benchmark:
```bash
uv run create_causal_benchmark.py --causal_data_path /datasets/waymo/processed/mini_causal --output_data_path /datasets/waymo/processed --causal_labels_path /datasets/waymo/causal_agents/processed_labels/ --benchmark remove_causal
uv run create_causal_benchmark.py --causal_data_path /datasets/waymo/processed/mini_causal --output_data_path /datasets/waymo/processed --causal_labels_path /datasets/waymo/causal_agents/processed_labels/ --benchmark remove_noncausal
uv run create_causal_benchmark.py --causal_data_path /datasets/waymo/processed/mini_causal --output_data_path /datasets/waymo/processed --causal_labels_path /datasets/waymo/causal_agents/processed_labels/ --benchmark remove_noncausalequal
uv run create_causal_benchmark.py --causal_data_path /datasets/waymo/processed/mini_causal --output_data_path /datasets/waymo/processed --causal_labels_path /datasets/waymo/causal_agents/processed_labels/ --benchmark remove_static
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
cd scene-tokens/src/scripts
uv run waymo_data_processing.py --raw_data_path /datasets/waymo/raw/scenario/ --proc_data_path /datasets/safeshift_all --search_safeshift --safeshift_data_splits_path /datasets/mtr_process_splits --safeshift_prefix score_asym_combined_80_ --split training
uv run waymo_data_processing.py --raw_data_path /datasets/waymo/raw/scenario/ --proc_data_path /datasets/safeshift_all --search_safeshift --safeshift_data_splits_path /datasets/mtr_process_splits --safeshift_prefix score_asym_combined_80_ --split validation
```

2. Re-split the processed data:
```bash
cd scene-tokens/src/scripts
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
cd scene-tokens/src/scripts
uv run verify_safeshift_causal_splits.py
```

## Prepare the Ego-SafeShift-Causal (WOMD) Dataset

This benchmark is similar to **SafeShift-Causal**, but the main difference is that scenarios are scored only from the perspective of the ego-agent. We used the [ScenarioCharacterization](https://github.com/navarrs/ScenarioCharacterization) package to compute the scenario scores and produce the meta file `scenario_to_scores_mapping.csv` required by the benchmark creation script.

1. Follow steps 1 through 6 on preparing the Causal Agents dataset above.

2. Make sure you have [this](https://drive.google.com/file/d/1Ptv1JIM0qymo7180_a5svLZJXWn03yQx/view?usp=drive_link) file in the `./meta` folder. To generate this file follow the scoring [instructions](https://github.com/navarrs/ScenarioCharacterization/blob/main/docs/CHARACTERIZATION.md).

3. Create the benchmark:
```bash
cd scene-tokens/src/scripts
uv run create_ego_safeshift_benchmark.py
```
