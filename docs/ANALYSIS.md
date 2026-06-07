# Experiment Analysis

## Scenario Visualization

The file `configs/scenario_visualization.yaml` specifies the configuration parameters to visualize scenarios in the data. Scenarios are read from a benchmark split JSON (produced by `create_benchmark.py`), and the visualization type is self-described by the chosen `visualization` config.

Run a scenario visualizer as:
```bash
uv run -m controlledshifts.run_scenario_visualization \
<<<<<<< HEAD
    visualization=[viz_config] \
    split_filepath=[path/to/benchmark_split.json] \
    scenarios_root=[dir/with/training,validation,testing/subdirs] \
    splits_to_visualize=[testing] \
    num_scenarios=[num_scenarios]
```
where:
* `visualization`: selects the visualizer and its `viz_type` (which drives what is loaded/computed). The available configs are:
  * `viz_static` (`regular`) / `viz_animated` (`regular`, animated): draw the scenarios as-is.
  * `viz_scored` (`scored`): compute scenario features → scores and render the scene score. **Requires** `dataset.config.autolabel_agents=true` so the feature/score processors are built.
  * `viz_causal` / `viz_causal_animated` (`model_output`): render ground-truth and predicted causal agents (needs cached model outputs).
  * `viz_trajpred` (`trajpred`): transform to agent-centric format and overlay model trajectory predictions (needs cached model outputs).

  Each visualization config declares a `panes_to_plot` list (values from `SupportedPanes`, e.g. `ALL_AGENTS`, `HIGHLIGHT_RELEVANT`, `CAUSAL_AGENTS_GT`, `CAUSAL_AGENTS_PRED`, `TRAJECTORY_PREDICTION`) that controls which panes are rendered, one window per pane.
* `split_filepath`: path to the benchmark split JSON. The visualized scenarios are taken from this file's `training`/`validation`/`testing` lists, and its `benchmark_name` becomes the `split_type` folder in the output path.
* `splits_to_visualize`: which of `training`/`validation`/`testing` to render (each becomes its own output subfolder).
* `scenarios_root`: directory holding the scenario pickles, organized into `training/`, `validation/`, `testing/` subdirectories of `<scenario_id>.pkl`.
* `num_batches` / `num_scenarios`: for the model-based types (`trajpred`, `model_output`), control how many cached batches/scenarios are loaded; scenarios are sampled if more are available than requested.
* `model_experiment`: for generic `model_output` visualizations, the tag used as the output `pane_type` folder.

Outputs are written under:
```
output_dir/<render>/<split_type>/<split>/<pane_type>
```
where `render` is `static`/`animated` (derived from the visualizer), `split_type` is the benchmark name, `split` is `train`/`val`/`test`, and `pane_type` is `scenario`/`scenario_scored`/`causal_scenario`/`trajectory_prediction` (or the `model_experiment` tag for generic model outputs).
=======
    experiment_name=[experiment_name] \
    analysis=[visualizer_type] \
    num_batches=[num_batches] \
    num_scenarios=[num_scenarios]
```
where:
* `experiment_name`: is the name of the experiment to analyze. Note that the experiment is assumed to be located at `${batch_cache_path}/${experiment_name}`.
* `analysis`: is either of `default` (the full scenario), `animated` (animated version of `default`), `causal` (scenario with causal labels and predictions), `causal_animated` (animated version of `causal`) or `trajpred` (scenario with trajectory predictions). Each visualization config declares a `panes_to_plot` list (values from `SupportedPanes`, e.g. `ALL_AGENTS`, `HIGHLIGHT_RELEVANT`, `CAUSAL_AGENTS_GT`, `CAUSAL_AGENTS_PRED`) that controls which panes are rendered, one window per pane.
* `num_batches`: is the number of cached batches the scenario visualizer script will load.
* `num_scenarios`: is the number of scenarios that will be visualized. The number of scenarios is sampled from the loaded batches if there are more scenarios than those specified.
>>>>>>> cd09afa (WIP: refactoring visualization)

**Example**: Result using the causal visualizer:

<img src="../assets/causal_scenario.png">

## Model Embedding Analysis

The file `configs/model_analysis.yaml` specifies the configuration parameters to run different analyses.

Run a scenario visualizer as:
```bash
uv run -m controlledshifts.model_analysis run_distribution_analysis=true run_dim_reduction_analysis=true run_score_analysis=true
```
the analyses can be run one at a time or all together.

**Example**: scenario codebook visualization.

<img src="../assets/codebook.png">

**Example**: scenario t-SNE visualization.

<img src="../assets/scenario_tsne.png">

## Model Metric Analysis


The file `configs/model_metric_analysis.yaml` specifies the configuration to compare the metrics from different models.

Run a scenario visualizer as:
```bash
uv run -m controlledshifts.model_metric_analysis group_name=[experiment_group]
```

An example of an expected input to this script is `assets/group.csv`, and an example of a corresponding result is shown below:

<img src="../assets/group_metric.png">


# Sample Selection

Cache training set embeddings:
```bash
uv run -m controlledshifts.run_sample_selection -m \
    paths=waymo_causal_labeled model=wayformer ckpt_name=epoch_118 +model.config.sample_selection=true cache=true
```

Run training analysis only:
```bash
uv run -m controlledshifts.run_sample_selection -m run_analysis=true
```
Run training experiment using blacklist created by sample selection experiment.

# Score Distribution Analysis

Visualize the agent score categorical distributions for the agents in a specified dataset:
```bash
uv run src/scripts/visualize_agent_score_distribution.py \
    --data_cache_path /path/to/data_cache
    --data_subsets name-of-training-set,name-of-validation-set,name-of-testing-set
    --output_path /path/to/save/the/plots
```

Output example shown [here](https://github.com/stackav-oss/social-twins/pull/16).
