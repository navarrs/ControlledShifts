# Experiment Analysis

## Scenario Visualization

The file `configs/scenario_visualization.yaml` specifies the configuration parameters to visualize scenarios in the data. Scenarios are read from a benchmark split JSON (produced by `create_benchmark.py`), and the visualization type is self-described by the chosen `visualization` config.

Run a scenario visualizer as:
```bash
uv run -m controlledshifts.run_scenario_visualization \
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
  * `viz_causal_gt` (`causal_gt`): render only the ground-truth causal agents, loaded from the causal-label JSON files (`dataset.config.causal_labels_path`). Needs no cached model outputs or predictions.
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


## Distribution Shift Analysis

The file `configs/analysis/distribution_shift.yaml` configures the in-distribution (ID) vs out-of-distribution (OOD)
benchmark analysis. Unlike the older per-benchmark CSVs, all results now live in a single combined file
(`benchmarks_filepath`, e.g. `meta/runs/distribution_shift_results.csv`), where each column is named
`<phase>/waymo-<split>/<metric>`. Each benchmark entry simply names the exact `seen` (ID) and `unseen` (OOD) column
prefixes to compare:

```yaml
benchmarks:
  - uniform:
      name: Uniform
      seen: "test/waymo-uniform-validation"
      unseen: "test/waymo-uniform-testing"
  # ...
```

For each model in `models_to_compare`, the seen and unseen metric values are looked up independently and joined by
model name, so a benchmark's seen and unseen splits may come from different training runs (or even different datasets).

Run the analysis as:
```bash
uv run -m controlledshifts.run_analysis analysis=distribution_shift
```

It writes per-benchmark comparison plots under `<output_path>/<benchmark>/` and one combined LaTeX table spanning all
benchmarks to `<output_path>/results.tex`.


## Robustness Score Analysis

The file `configs/analysis/sensitivity_score.yaml` reduces the same combined results file into comparable *robustness
scores* per model per metric, measured against a reference. Two reference modes are produced:

- `naive_relative` — each model vs the **Naive** baseline **within the same benchmark**.
- `uniform_relative` — each model vs **its own** performance in the **Uniform** benchmark (e.g. AutoBot on
  EgoSafeShift vs AutoBot on Uniform). The Uniform benchmark is excluded from this mode's aggregation.

All metrics are lower-is-better (errors). Working in natural-log space, a model's total out-of-distribution advantage
over the reference decomposes **exactly and additively** into two reference-relative robustness terms:

```
log(ref_unseen / model_unseen) = log(ref_seen / model_seen) + log((ref_unseen/ref_seen) / (model_unseen/model_seen))
   robustness_score (overall)  =     seen_robustness_score   +              shift_robustness_score
```

- `seen_robustness_score = log(ref_seen / model_seen)` — **ID-level robustness**: how much better the model already is
  on the seen split (its starting point).
- `shift_robustness_score = log((ref_unseen/ref_seen) / (model_unseen/model_seen))` — **shift robustness**: how much
  *less* the model degrades seen→unseen than the reference. Sign-preserving, so a model that improves under shift is
  rewarded.
- `robustness_score = seen_robustness_score + shift_robustness_score = log(ref_unseen / model_unseen)` — **overall OOD
  robustness** vs the reference.

All three share the same log units, are symmetric and unbounded both ways (a 2× improvement and a 2× degradation are
`±log 2`), and are `0` for the reference compared against itself (so the Naive row is a visible baseline in
`naive_relative`). In both reference modes the convention is the same: **higher == more robust than the reference**,
`0` == on par, negative == worse. There are no epsilon/clip knobs and no regression — the terms are exact log ratios.

Each term is aggregated across benchmarks (NaN-safe `mean`/`median`, set by `score.aggregate`); the `Combined` column
always holds the per-model mean across metrics.

Run the analysis as:
```bash
uv run -m controlledshifts.run_analysis analysis=sensitivity_score
```

For each reference mode it writes, under `<output_path>/<mode>/`, a radar plot, CSV table and LaTeX table for each term
(`seen_robustness_*`, `shift_robustness_*`, `robustness_*`) plus a `robustness_decomposition.png` scatter — one panel
per metric placing each model at `(seen_robustness_score, shift_robustness_score)`, with the reference at the origin and
anti-diagonals marking constant overall robustness. Upper-right points are both better in-distribution and more
shift-robust than the reference.


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
