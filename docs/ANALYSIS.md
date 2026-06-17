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
* `num_batches` / `num_scenarios`: for the model-based types (`trajpred`, `model_output`), control how many cached scenarios are loaded; scenarios are sampled if more are available than requested. Cached model outputs live as one pickle per scenario under `batch_cache_path/<split>/<scenario_id>.pkl` (split is `train`/`val`/`test`).
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
benchmarks to `<output_path>/results.tex`. Each benchmark block ends with a light-gray mean row holding the per-metric
mean seen value, mean unseen value, and mean OOD gap (the mean of the per-model gaps), and the table closes with a gray
overall-mean row giving the per-metric mean across the entire sweep. Because these rows are shaded with `\rowcolor`, the
consuming LaTeX document must load `\usepackage[table]{xcolor}`.


## Robustness Score Analysis

The file `configs/analysis/robustness.yaml` reduces the same combined results file into comparable *robustness
scores* per model per metric, measured against a reference. Two reference modes are produced:

- `naive_relative` — each model vs the **Naive** baseline **within the same benchmark**.
- `uniform_relative` — each model vs **its own** performance in the **Uniform** benchmark (e.g. AutoBot on
  EgoSafeShift vs AutoBot on Uniform). The Uniform benchmark is excluded from this mode's aggregation.

All metrics are lower-is-better (errors). Working in natural-log space, each model is characterized by two
reference-relative robustness axes:

- `seen_robustness_score = log(ref_seen / model_seen)` — **ID-level robustness**: how much better the model already is
  on the seen split (its starting point).
- `shift_robustness_score = log((ref_unseen/ref_seen) / (model_unseen/model_seen))` — **shift robustness**: how much
  *less* the model degrades seen→unseen than the reference. Sign-preserving, so a model that improves under shift is
  rewarded.

Both share the same log units, are symmetric and unbounded both ways (a 2× improvement and a 2× degradation are
`±log 2`), and are `0` for the reference compared against itself (so the Naive row is a visible baseline in
`naive_relative`): **higher == more robust than the reference**, `0` == on par, negative == worse. There are no
epsilon/clip knobs and no regression — the axes are exact log ratios. Each is aggregated across benchmarks (NaN-safe
`mean`/`median`, set by `score.aggregate`); the `Combined` column holds the per-model mean across metrics.

### Combined ranking score

To rank models by a single value, the two axes are reduced to a **combined** score. Their raw sum is deliberately *not*
used: it telescopes to `seen + shift = log(ref_unseen / model_unseen)`, which ranks models purely by OOD error (the
reference cancels to an additive constant) and adds nothing beyond the OOD numbers. Instead the combined score uses
**standardized equal-influence**: each axis is z-scored across the model cohort (per metric, NaN-aware), the two
z-scores are summed, and the per-model mean across metrics is the ranking value. This gives both axes — and every
metric — equal say regardless of their natural spread. The trade-off is that the combined score is **cohort-relative**:
`0` is the cohort average (not the reference), and scores recenter if the set of models changes. It is a ranking tool,
not an absolute metric.

Run the analysis as:
```bash
uv run -m controlledshifts.run_analysis analysis=robustness
```

For each reference mode it writes, under `<output_path>/<mode>/`:
- a radar plot, CSV and LaTeX table for each axis (`seen_robustness_*`, `shift_robustness_*`);
- a `robustness_decomposition.png` scatter — one panel per metric placing each model at
  `(seen_robustness_score, shift_robustness_score)`, with the reference at the origin; the upper-right quadrant is both
  better in-distribution and more shift-robust than the reference;
- the combined ranking as a sorted bar chart (`combined_robustness_ranking.png`, best first) plus its CSV and LaTeX
  table (`combined_robustness_scores.*`).


## Causal Agent Distribution Analysis

The file `configs/analysis/causal_distribution.yaml` configures a data-side comparison of how the two causal-agents
benchmarks distribute agents across their train/validation/testing splits. The standard `causal_agents` benchmark
reuses a random reference split, so its per-scenario agent counts should look the same across splits; `causal_agents_hard`
ranks scenarios by non-causal agent count and sends the hardest scenarios to the test set, so its test split should be
visibly shifted toward more non-causal agents.

Unlike the other analyses, this one reads raw data rather than the combined model-results CSV: per-scenario `base`
variant pkls (`variants_base_path`), per-scenario JSON causal labels (`causal_labels_path`), and the benchmark split
JSONs (`splits_path`). Each benchmark entry names the split JSON to read:

```yaml
benchmarks:
  - causal_agents:
      name: CausalAgents
      split_json: causal_agents
  - causal_agents_hard:
      name: CausalAgentsHard
      split_json: causal_agents_hard
```

For every scenario in the union of both splits, it counts causal agents (`causal_ids` + ego) and non-causal agents
(everything else) using the same `get_noncausal_mask` the benchmarks use. Counts are intrinsic to a scenario, so they
are computed once (over `num_workers` processes) and cached to `<output_path>/per_scenario_counts.csv`.

The plots are driven entirely by the cached CSVs. On a re-run (with `overwrite=false`), if `causal_distribution.csv`
exists it is loaded directly and the figures are regenerated from it without touching the scenario pkls or the split
JSONs — so you can restyle the plots, or render them on a machine that only has the CSVs. Failing that, the cached
`per_scenario_counts.csv` is reused to rebuild the long-form frame from the splits. Only when neither cache exists (or
`overwrite=true`) are the scenarios loaded and counts recomputed; `overwrite=true` is also needed to pick up changes to
`benchmarks` against already-cached CSVs.

Run the analysis as:
```bash
uv run -m controlledshifts.run_analysis analysis=causal_distribution
```

It writes, under `<output_path>/`: the cached `per_scenario_counts.csv`, the bucketed long-form `causal_distribution.csv`,
a per-benchmark/per-split `summary.csv` (mean/median/std/count), and for each quantity in `quantities`
(`n_causal`, `n_noncausal`, `frac_noncausal`, `n_total`) three side-by-side views with one panel/column per benchmark:
a violin (`<quantity>_violin.png`, splits on the x-axis), a histogram (`<quantity>_histogram.png`, overlaid per-split
density curves), and a ridgeline (`<quantity>_ridge.png`, one overlapping density row per split).

## Ego-SafeShift Score Distribution Analysis

The file `configs/analysis/score_distribution.yaml` configures a data-side comparison of how the ego-safeshift
criticality scores distribute across train/validation/testing. The `ego_safeshift` benchmark ranks scenarios by a
safety score (`score_type`) and sends the hardest (highest-scoring) scenarios to the test set, so its test split should
be visibly shifted toward higher scores; the `uniform` baseline splits the same scenarios randomly, so its
distributions should match across splits.

Like the causal-agents analysis, this reads raw data rather than the combined model-results CSV — but only the scores
CSV at `scores_csv_path` (columns: `scenario_ids` plus one column per score). Each benchmark's split is reproduced
in-script from the scenario scores via the same `split_ids_by_score`/`split_ids_by_ratio` the benchmarks use (with the
configured `score_type`, `split_ratios` and `seed`), so it needs neither the scenario pkls nor a pre-existing split
JSON. Unlike benchmark creation, scenarios are not filtered by on-disk availability — every scored scenario in the CSV
is included. Each benchmark entry names a display name and a split strategy:

```yaml
benchmarks:
  - ego_safeshift:
      name: EgoSafeShift
      split: score
  - uniform:
      name: Uniform (random)
      split: random
```

The plots are driven entirely by the cached `score_distribution.csv`. On a re-run (with `overwrite=false`) it is loaded
directly and the figures are regenerated from it without re-reading the scores CSV. Set `overwrite=true` to rebuild the
frame — also required to pick up changes to `benchmarks` against an already-cached CSV.

Run the analysis as:
```bash
uv run -m controlledshifts.run_analysis analysis=score_distribution
```

It writes, under `<output_path>/`: the long-form `score_distribution.csv`, a per-benchmark/per-split `summary.csv`
(mean/median/std/count), and for each quantity in `quantities` three side-by-side views with one panel/column per
benchmark: a violin (`<quantity>_violin.png`, splits on the x-axis), a histogram (`<quantity>_histogram.png`, overlaid
per-split density curves), and a ridgeline (`<quantity>_ridge.png`, one overlapping density row per split).

## Environments Benchmark Analysis

The file `configs/analysis/environments.yaml` configures a visualization of the NetLSD-descriptor clustering that
defines the environments benchmark. It reads only the artifacts written by benchmark creation under `cache_path`:
`descriptors_cache.pkl` (the NetLSD descriptors), `scaler.pkl` (the `StandardScaler` fit during clustering), and
`{clustering_algorithm}/environment_benchmark.csv` (the per-scenario `cluster_label`, `hardness_score`, `input_set`
and `output_set`). The descriptors are aligned to the CSV and scaled with the saved scaler before TSNE and silhouette,
so the embedding and silhouette scores match the space the benchmark clustered in.

Set `cache_path` and `clustering_algorithm` to match the benchmark run you want to inspect:

```yaml
cache_path: /data/driving/waymo/meta/environments
clustering_algorithm: ward
```

The plots are driven entirely by the cached `environments_embedding.csv` (columns `scenario_id`, `tsne_1`, `tsne_2`,
`cluster_label`, `output_set`, `silhouette`). On a re-run (with `overwrite=false`) it is loaded directly and the
figures are regenerated without recomputing the TSNE embedding or silhouette scores. Set `overwrite=true` to rebuild
it from the descriptor cache.

Run the analysis as:
```bash
uv run -m controlledshifts.run_analysis analysis=environments
```

It writes, under `<output_path>/`: the cached `environments_embedding.csv`, a `tsne.png` (the TSNE embedding stacked
vertically — coloured by assigned cluster on top and by train/validation/testing split on the bottom, each with its
own legend), and a `silhouette.png` (per-cluster silhouette bars with the overall mean marked).

## Scenario Overlap Analysis

The file `configs/analysis/scenario_overlap.yaml` measures how different the benchmarks are by quantifying how many
scenarios their corresponding splits share. It reads only the split JSONs under `splits_path`
(`/data/driving/waymo/splits`), so it is fast and writes no cache. The `benchmarks` list selects which benchmarks to
compare (each entry gives a display `name` and the `split_json` filename), and `splits` selects which splits to
compare (default `[training, validation, testing]`):

```yaml
splits_path: /data/driving/waymo/splits
splits: [training, validation, testing]
benchmarks:
  - uniform:
      name: Uniform
      split_json: uniform
  - ego_safeshift:
      name: EgoSafeShift
      split_json: ego_safeshift
```

For each split it builds a symmetric benchmark x benchmark matrix of the Jaccard index `|A ∩ B| / |A ∪ B|` over the
scenario IDs. Benchmarks built on a shared reference split (e.g. `causal_agents` vs `uniform`) land near 1.0, while
benchmarks that resample the population (e.g. `causal_agents_hard` test set) drop well below.

Run the analysis as:
```bash
uv run -m controlledshifts.run_analysis analysis=scenario_overlap
```

It writes, under `<output_path>/`: a single `scenario_overlap.png` with one annotated Jaccard heatmap per split
(sharing one colorbar), and a tidy `scenario_overlap.csv` (columns `split`, `benchmark_a`, `benchmark_b`, `size_a`,
`size_b`, `intersection`, `union`, `jaccard`) holding the raw intersection counts behind the plotted Jaccard values.


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
