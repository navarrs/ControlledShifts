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
  * `viz_trajpred` (`trajpred`): transform to agent-centric format and render one comparison pane per model. Each pane draws the shared scene context (map, agent history, and the dimmed ground-truth future) plus that model's predicted trajectories, titled with the model name — there is no separate ground-truth-only pane (needs cached model outputs). See `models` below.

  Each visualization config declares a `panes_to_plot` list (values from `SupportedPanes`, e.g. `ALL_AGENTS`, `HIGHLIGHT_RELEVANT`, `CAUSAL_AGENTS_GT`, `CAUSAL_AGENTS_PRED`, `TRAJECTORY_PREDICTION`) that controls which panes are rendered, one window per pane.
* `split_filepath`: path to the benchmark split JSON. The visualized scenarios are taken from this file's `training`/`validation`/`testing` lists, and its `benchmark_name` becomes the `split_type` folder in the output path.
* `splits_to_visualize`: which of `training`/`validation`/`testing` to render (each becomes its own output subfolder).
* `scenarios_root`: directory holding the scenario pickles, organized into `training/`, `validation/`, `testing/` subdirectories of `<scenario_id>.pkl`.
* `num_batches` / `num_scenarios`: for the model-based types (`trajpred`, `model_output`), control how many cached scenarios are loaded; scenarios are sampled if more are available than requested. Cached model outputs live as one pickle per scenario under `batch_cache_path/<split>/<scenario_id>.pkl` (split is `train`/`val`/`test`).
* `model_experiment`: for generic `model_output` visualizations, the tag used as the output `pane_type` folder.
* `models`: for `trajpred`, a list of `{name, batch_cache_path}` entries; each model becomes one pane titled with its `name`. Scenarios are sampled once from the intersection of scenario ids available across all listed models, so the panes stay aligned (here `num_scenarios` governs how many aligned scenarios are drawn and `num_batches` is not applied per model). When `models` is null, a single top-level `batch_cache_path` still works and renders a one-model comparison.

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


## Unshifted-Generalization Analysis

The file `configs/analysis/unshifted_generalization.yaml` configures a complementary study where the training data is
*not* subjected to any artificial shift (it trains/validates on the non-overlapping `mini` subset). Instead of a
seen/unseen pair per benchmark, there is a single in-distribution `reference` split (mini-validation) and a flat list of
evaluation `benchmarks`; every benchmark's gap is measured relative to that one reference:

```yaml
reference:
  name: Mini-Val
  split: "val/waymo-mini-id"
benchmarks:
  - mini_test: {name: Mini-Test, split: "test/waymo-mini-ood"}
  - causal_agents_hard: {name: CausalAgentsHard, split: "test/waymo-remove-noncausal-hard-testing"}
  # ...
```

Run it as:
```bash
uv run -m controlledshifts.run_analysis analysis=unshifted_generalization
```

It writes a vertical per-benchmark LaTeX table to `<output_path>/results.tex` (a reference block of plain absolute
values followed by one block per benchmark whose cells are `value (gap%)` vs the reference, each block ending in a gray
mean row and the table closing with a gray overall-mean row), a grouped value bar chart `benchmark_values.png` (the
reference benchmark is hatched), and a benchmark x model gap heatmap `gap_heatmap.png` (red = worse). The console prints
the per-benchmark mean gap and flags the worst benchmark. As above, the table uses `\rowcolor`, so the consuming LaTeX
document must load `\usepackage[table]{xcolor}` (and `\usepackage{multirow}`).


## Robustness Score Analysis

The file `configs/analysis/robustness.yaml` reduces the same combined results file into comparable *scores*
per model per metric, measured against a reference. Two reference modes are produced:

- `naive_relative` — each model scaled by the **Naive** baseline **within the same benchmark**.
- `uniform_relative` — each model scaled by **its own** performance in the **Uniform** benchmark (e.g. AutoBot on
  EgoSafeShift vs AutoBot on Uniform). The Uniform benchmark is excluded from this mode's aggregation.

All metrics are lower-is-better (errors). Following the MASE / OWA framing of the N-BEATS paper
([arXiv:1905.10437](https://arxiv.org/pdf/1905.10437)), each model is characterized by two reference-relative *score*
axes (the reciprocal MASE skill, so higher is better):

- `id_score = ref_seen / model_seen` — **ID score** (reciprocal MASE on the seen split).
- `ood_score = ref_unseen / model_unseen` — **OOD score** (reciprocal MASE on the unseen split).

Both are dimensionless, **higher == better**, and `1.0` == on par with the reference (the reference's own row is exactly
`1.0`, so the Naive row is a visible baseline in `naive_relative`); `> 1` beats the reference, `< 1` is worse. Because
each split is scaled by the reference *on that same split*, a good model with low absolute OOD error stays high on
`ood_score` regardless of its degradation *factor* — this avoids the **robustness paradox**, where a uniformly-weak
model that multiplies its error by a small factor would otherwise look more robust than a strong model. The division is
NaN-guarded (a non-finite/non-positive numerator or denominator drops out). Each axis is aggregated across benchmarks
(NaN-safe `mean`/`median`, set by `score.aggregate`); the `Combined` column holds the per-model mean across metrics.

### Combined ranking score

To rank models by a single value, the two axes are reduced — within the same reference frame — to a `combined` score
that is the per-metric geometric mean:

```
combined_metric = sqrt(id_score · ood_score)
Combined        = mean(combined_metric across metrics)
```

`Combined` is NaN-safe (a non-positive or missing axis drops the cell). The geometric mean does both jobs at once: its
absolute level rewards quality, and because it punishes ID/OOD imbalance it penalizes shift degradation — so a single
frame captures both, with no second reference. `1.0` means on par with the reference. Under `naive_relative` this
demotes the Naive baseline (pinned at `1.0`, since it is its own reference) and keeps a model genuinely worse than Naive
below it — this is the downstream-performance ranking. Under `uniform_relative` the combined is a **stability** view:
the reference is the model's own Uniform row, so the most *consistent* model (the input-agnostic Naive) ranks high
there, as expected for a self-relative stability score rather than a performance ranking.

Run the analysis as:
```bash
uv run -m controlledshifts.run_analysis analysis=robustness
```

For each reference mode it writes, under `<output_path>/<mode>/`:
- a radar plot, CSV and LaTeX table for each axis (`id_score_*`, `ood_score_*`), with a dashed `1.0`
  reference ring;
- a `score_decomposition.png` scatter — one panel per metric placing each model at
  `(id_score, ood_score)`, with the reference at `(1, 1)`; the upper-right quadrant beats the reference on
  both ID and OOD, and the dashed `y = x` diagonal marks "degrades like the reference" (above it = more shift-robust);
- the combined robustness ranking as a sorted bar chart (`combined_robustness_ranking.png`, best first) plus its CSV and
  LaTeX table (`combined_robustness_scores.*`).


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
own legend), and a `silhouette.png` (per-cluster silhouette bars with the overall mean marked). Set `show_axes: true`
to render the TSNE subplots with axes and labels; by default they are drawn as a bare scatter (legend and title only)
without axes or grid.

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
It also writes an `overlaps/` subdirectory holding the overlapping scenario IDs themselves as JSON: one
`<BenchmarkA>_<BenchmarkB>.json` per benchmark pair plus an `all_benchmarks.json` for the intersection common to every
benchmark. Each file mirrors the split JSONs — a `benchmarks` metadata list naming the group, then one sorted array of
shared scenario IDs per split.


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
