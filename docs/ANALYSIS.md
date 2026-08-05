# Experiment Analysis

## Scenario Visualization

[`scenario_visualization.yaml`](../src/controlledshifts/configs/scenario_visualization.yaml) configures scenario rendering. Scenarios are read from a benchmark split JSON (produced by `create_benchmark`), and the chosen `visualization` config selects the visualizer and what is loaded or computed.

```bash
uv run -m controlledshifts.run_scenario_visualization \
    visualization=[viz_config] \
    split_filepath=[path/to/benchmark_split.json] \
    scenarios_root=[dir/with/training,validation,testing/subdirs] \
    splits_to_visualize=[testing] \
    num_scenarios=[num_scenarios]
```

The available [`visualization`](../src/controlledshifts/configs/visualization/) configs (`viz_type` in parentheses):

| Config | `viz_type` | Renders |
|---|---|---|
| `viz_static` / `viz_animated` | `regular` | The scenarios as-is (static or animated). |
| `viz_scored` | `scored` | Scenario features → scores, with the scene score drawn. Requires `dataset.config.autolabel_agents=true`. |
| `viz_non_background` / `viz_non_background_animated` | `model_output` | Ground-truth and predicted non-background agents. Needs cached model outputs. |
| `viz_non_background_gt` | `non_background_gt` | Only the ground-truth non-background agents, from the label JSONs (`dataset.config.causal_labels_path`). Needs no model outputs. |
| `viz_trajpred` | `trajpred` | One comparison pane per model: the shared scene context (map, agent history, dimmed ground-truth future) plus that model's predictions. Needs cached model outputs; see `models` below. |

Each config declares a `panes_to_plot` list (values from `SupportedPanes`: `ALL_AGENTS`, `HIGHLIGHT_RELEVANT`, `NON_BACKGROUND_AGENTS_GT`, `NON_BACKGROUND_AGENTS_PRED`, `TRAJECTORY_PREDICTION`) controlling which panes are rendered, one window per pane. In the non-background panes the ego agent is blue, background agents are orange and dimmed to `background_alpha`, and non-background agents keep their regular agent-type color.

Key options:

| Option | Purpose |
|---|---|
| `split_filepath` | The benchmark split JSON. Scenarios come from its `training`/`validation`/`testing` lists, and its `benchmark_name` becomes the `split_type` output folder. |
| `splits_to_visualize` | Which of `training`/`validation`/`testing` to render; each becomes its own output subfolder. |
| `scenarios_root` | Directory of scenario pickles, organized into `training/`, `validation/`, `testing/` subdirectories of `<scenario_id>.pkl`. |
| `num_batches` / `num_scenarios` | For `trajpred` and `model_output`, how many cached scenarios to load. Sampled if more are available than requested. |
| `cache_source` | Which `source` to load when a split holds several variants of the same scenario (e.g. `waymo-remove-noncausal-testing`). Leave null for single-source splits; otherwise one arbitrary variant per scenario is loaded and a warning is logged. |
| `model_experiment` | For generic `model_output` visualizations, the tag used as the output `pane_type` folder. |
| `models` | For `trajpred`, a list of `{name, batch_cache_path}` entries, one pane each. Scenarios are sampled from the intersection of IDs available across all models so the panes stay aligned. When null, a single top-level `batch_cache_path` renders one pane. |

Outputs are written under `output_dir/<render>/<split_type>/<split>/<pane_type>`, where `render` is `static`/`animated`, `split_type` is the benchmark name, `split` is `train`/`val`/`test`, and `pane_type` is one of `scenario`, `scenario_scored`, `non_background_scenario`, `non_background_scenario_gt`, `trajectory_prediction` (or the `model_experiment` tag).

<details>
<summary><b>Producing a model-output cache</b> — needed for the <code>trajpred</code> and <code>model_output</code> visualizations.</summary>

The `trajpred` and `model_output` visualizations read cached model outputs, which training does not write by default (`model.config.cache_batch` is `False`). To populate the `val`/`test` caches for an already-trained run, re-run its checkpoint through `eval` with caching enabled. Override `paths.experiment_dir` with the run's own dated directory so the cache lands next to its checkpoint instead of in a fresh `${now:...}` one:

```bash
uv run -m controlledshifts.eval \
  model=wayformer \
  paths=background_agents \
  paths.experiment_dir=background-agents/wayformer/2026-06-12_16-20-11 \
  ckpt_name=epoch_110 \
  model.config.cache_batch=true \
  model.config.cache_every_batch_idx=1
```

`cache_every_batch_idx` gates which batches are written (`batch_idx % cache_every_batch_idx == 0`); it defaults to `100`, so set it to `1` to cache every scenario. Outputs land in `<run_dir>/batch_cache/{val,test}/<source>/<scenario_id>.pkl`, where `source` is the `dataset_name` of the split source that produced it (e.g. `waymo-uniform-testing`) — the source namespaces the file because a split can hold several variants of the same scene. Use a single `trainer.devices`; under DDP every rank writes into the same directory.

To cache many runs at once, `run_model_cache_sweep` does the above for every run listed in a W&B results export, resolving each run's directory and best checkpoint from the CSV:

```bash
# Preview the eval command for every run in the CSV.
uv run -m controlledshifts.run_model_cache_sweep dry_run=true

# Cache them, or resume an interrupted sweep.
uv run -m controlledshifts.run_model_cache_sweep
uv run -m controlledshifts.run_model_cache_sweep skip_existing=true

# Restrict to some models/benchmarks.
uv run -m controlledshifts.run_model_cache_sweep 'models=[wayformer,mtr]' 'benchmarks=[background_agents]'
```

A failing run does not abort the sweep; failures are reported in a summary at the end. See
[`model_cache_sweep.yaml`](../src/controlledshifts/configs/model_cache_sweep.yaml) for all options.

</details>

## Model and Benchmark Analyses

All analyses run through a single entrypoint, selecting a config from [`configs/analysis/`](../src/controlledshifts/configs/analysis/):

```bash
uv run -m controlledshifts.run_analysis analysis=[analysis_name]
```

| `analysis=` | Reads | Produces |
|---|---|---|
| `distribution_shift` | Combined results CSV | Per-benchmark ID vs OOD comparison plots and a LaTeX table. |
| `unshifted_generalization` | Combined results CSV | Gap of every benchmark against one unshifted reference split. |
| `robustness` | Combined results CSV | Reference-relative quality/stability scores, radar plots, and a ranking. |
| `background_agents_distribution` | Scenario pkls + causal labels | How non-background/background agent counts distribute across the background benchmarks' splits. |
| `ego_safeshift_distribution` | Scores CSV | How ego-safeshift criticality scores distribute across splits. |
| `environments_distribution` | Clustering artifacts | TSNE and silhouette plots of the environments benchmark's clustering. |
| `scenario_overlap` | Split JSONs | Jaccard overlap between benchmarks' splits. |

The first three read a single combined results file (`benchmarks_filepath`, e.g. `meta/runs/distribution_shift_results.csv`) whose columns are named `<phase>/waymo-<split>/<metric>`. The rest read raw data or benchmark-creation artifacts directly.

### Distribution Shift

[`analysis/distribution_shift.yaml`](../src/controlledshifts/configs/analysis/distribution_shift.yaml) compares in-distribution (ID) against out-of-distribution (OOD) results. Each benchmark entry names the exact `seen` (ID) and `unseen` (OOD) column prefixes to compare:

```yaml
benchmarks:
  - uniform:
      name: Uniform
      latex: '\uniform'
      seen: "test/waymo-uniform-validation"
      unseen: "test/waymo-uniform-testing"
```

For each model in `models_to_compare`, the seen and unseen values are looked up independently and joined by model name, so a benchmark's seen and unseen splits may come from different training runs. `latex` is the macro the table prints for the benchmark; models use the macros in `MODEL_MACRO_MAP`.

Writes per-benchmark comparison plots under `<output_path>/<benchmark>/` and one combined LaTeX table spanning all benchmarks to `<output_path>/results.tex`. Each benchmark block ends with a mean row (mean seen, mean unseen, mean OOD gap), and the table closes with an overall-mean row. The `table` block holds the paper-specific strings (`caption`, `seen_label`, `unseen_label`, `robustness_label`) and `include_overall_mean`, which defaults to `false` and writes that closing row as commented-out LaTeX.

The table's last two columns are the robustness scores of [Robustness Scores](#robustness-scores), but computed *per benchmark* rather than aggregated across them: quality is scored against that block's Naive row and stability against the model's row in the benchmark named by `table.uniform_key`. Higher is better and the highest per column is bolded; stability inside the Uniform block is `1.000` for every model by construction (it is its own reference), so nothing is bolded there.

### Unshifted Generalization

[`analysis/unshifted_generalization.yaml`](../src/controlledshifts/configs/analysis/unshifted_generalization.yaml) covers the complementary study where training data is *not* artificially shifted (it trains and validates on the non-overlapping `mini` subset). Instead of a seen/unseen pair per benchmark there is a single ID `reference` split and a flat list of evaluation `benchmarks`; every gap is measured against that one reference:

```yaml
reference:
  name: Mini-Val
  split: "val/waymo-mini-id"
benchmarks:
  - mini_test: {name: Mini-Test, split: "test/waymo-mini-ood"}
  - background_agents_hard: {name: BackgroundAgentsHard, split: "test/waymo-remove-noncausal-hard-testing"}
```

Writes a per-benchmark LaTeX table to `<output_path>/results.tex` (cells are `value (gap%)` vs the reference), a grouped bar chart `benchmark_values.png` (the reference is hatched), and a benchmark x model gap heatmap `gap_heatmap.png` (red = worse). The console prints the per-benchmark mean gap and flags the worst benchmark.

### Robustness Scores

[`analysis/robustness.yaml`](../src/controlledshifts/configs/analysis/robustness.yaml) reduces the same results file into comparable *scores* per model, measured against a reference. All metrics are errors, so each model is characterized by two reference-relative axes — dimensionless, **higher is better**, `1.0` == on par with the reference — reduced to a single ranking value by their per-column geometric mean:

```
id_score  = ref_seen   / model_seen
ood_score = ref_unseen / model_unseen
Combined  = mean( sqrt(id_score · ood_score) )
```

A score exists for every `(benchmark, model, metric)` cell, and `aggregate_over` picks which of those two dimensions is collapsed — the surviving one becomes the score columns (and the radar axes):

| `aggregate_over` | Aggregates across | Score columns | Output folder |
|---|---|---|---|
| `benchmark` | Benchmarks | Metrics | `per_metric/` |
| `metric` | Metrics | Benchmark names | `per_benchmark/` |

Radar rims are short labels spelled out in an abbreviation key beside the figure: metrics use the built-in `METRIC_ABBREV_MAP`, benchmarks use the optional per-benchmark `abbrev` field in `robustness.yaml` (`BackgroundAgentsHard` → `BAH`). A benchmark without an `abbrev` is labelled in full.

Two reference modes are produced:

| Mode | Reference | Reads as |
|---|---|---|
| `naive_relative` | The **Naive** baseline, within the same benchmark. | Downstream-performance ranking. |
| `uniform_relative` | The model's **own** Uniform-benchmark row. | Stability. The most consistent model (the input-agnostic Naive) ranks high. Uniform is excluded from this mode's aggregation. |

> [!NOTE]
> The full derivation — the MASE / OWA framing of [N-BEATS](https://arxiv.org/pdf/1905.10437), why the geometric mean is used, and the "robustness paradox" this scaling avoids — lives in the module docstring of [`robustness_scores.py`](../src/controlledshifts/utils/analysis/robustness_scores.py).

For each axis and mode it writes, under `<output_path>/<axis>/<folder>/` (`naive_relative` → `quality_naive`, `uniform_relative` → `stability_uniform`):
- a radar plot, CSV and LaTeX table per score term (`seen_score_*`, `unseen_score_*`), with a dashed `1.0` reference ring;
- `score_decomposition.png` — one panel per score column placing each model at `(id_score, ood_score)`, reference at `(1, 1)`. The upper-right quadrant beats the reference on both; the dashed `y = x` diagonal marks "degrades like the reference" (above it = more shift-robust);
- the combined ranking as a sorted bar chart (`combined_robustness_ranking.png`) plus CSV and LaTeX table.

Each axis folder also gets `<output_path>/<axis>/robustness_summary.png`: a 3x2 grid whose columns are the two modes (Quality / Stability) and whose rows are the Seen radar, Unseen radar and Combined ranking, under a shared model legend.

> [!NOTE]
> The LaTeX tables in these three analyses shade rows with `\rowcolor`, so the consuming document must load `\usepackage[table]{xcolor}` (and `\usepackage{multirow}` for the unshifted-generalization table). The distribution-shift table additionally prints benchmark, model and split *macros* rather than plain names, so the document must define them (`\uniform`, `\naive`, `\seen`, `\unseen`, ...).

### Background Agent Distribution

[`analysis/background_agents_distribution.yaml`](../src/controlledshifts/configs/analysis/background_agents_distribution.yaml) compares how the two background-agents benchmarks distribute agents across their splits. `background_agents` reuses a random reference split, so its per-scenario agent counts should look the same across splits; `background_agents_hard` sends the scenarios with the most background agents to the test set, so its test split should be visibly shifted.

It reads raw data rather than the results CSV: per-scenario `base` pkls (`variants_base_path`), JSON causal labels (`causal_labels_path`), and the benchmark split JSONs (`splits_path`). Each entry names the split JSON to read (`split_json` keeps the legacy on-disk spelling):

```yaml
benchmarks:
  - background_agents: {name: BackgroundAgents, split_json: background_agents}
  - background_agents_hard: {name: BackgroundAgentsHard, split_json: background_agents_hard}
```

Counts (non-background = `causal_ids` + ego; background = everything else, via the same `get_background_mask` the benchmarks use) are intrinsic to a scenario, so they are computed once over `num_workers` processes and cached to `<output_path>/per_scenario_counts.csv`.

Writes, under `<output_path>/` (`outputs/background_agents_distribution_analysis/`): the cached `per_scenario_counts.csv`, the bucketed `background_agents_distribution.csv`, a per-benchmark/per-split `summary.csv`, and the [distribution plots](#shared-conventions) for each quantity in `quantities` (`n_non_background`, `n_background`, `frac_background`, `n_total`).

### Ego-SafeShift Score Distribution

[`analysis/ego_safeshift_distribution.yaml`](../src/controlledshifts/configs/analysis/ego_safeshift_distribution.yaml) compares how the ego-safeshift criticality scores distribute across splits. `ego_safeshift` sends the highest-scoring scenarios to the test set, so its test split should be shifted toward higher scores; the `uniform` baseline splits randomly, so its distributions should match across splits.

It reads only the scores CSV at `scores_csv_path` (columns: `scenario_ids` plus one column per score). Each benchmark's split is reproduced in-script from the scores via the same `split_ids_by_score` / `split_ids_by_ratio` the benchmarks use, so it needs neither the scenario pkls nor a pre-existing split JSON. Unlike benchmark creation, scenarios are not filtered by on-disk availability — every scored scenario in the CSV is included.

```yaml
benchmarks:
  - ego_safeshift: {name: EgoSafeShift, split: score}
  - uniform: {name: Uniform (random), split: random}
```

Writes, under `<output_path>/`: `ego_safeshift_distribution.csv`, a per-benchmark/per-split `summary.csv`, and the
[distribution plots](#shared-conventions) for each quantity in `quantities`.

### Environments Distribution

[`analysis/environments_distribution.yaml`](../src/controlledshifts/configs/analysis/environments_distribution.yaml) visualizes the NetLSD-descriptor clustering that defines the environments benchmark. It reads only the artifacts written by benchmark creation under `cache_path`: `descriptors_cache.pkl`, `scaler.pkl`, and `{clustering_algorithm}/environment_benchmark.csv` (per-scenario `cluster_label`, `hardness_score`, `input_set`, `output_set`). Descriptors are aligned to the CSV and scaled with the saved scaler before TSNE and silhouette, so the embedding matches the space the benchmark clustered in.

Set `cache_path` and `clustering_algorithm` to match the benchmark run you want to inspect:

```yaml
cache_path: /data/driving/waymo/meta/environments
clustering_algorithm: ward
```

```bash
uv run -m controlledshifts.run_analysis analysis=environments_distribution
```

Writes, under `<output_path>/`: `environments_embedding.csv`, `tsne.png` (the embedding stacked vertically — coloured by cluster on top, by train/validation/testing split on the bottom), and `silhouette.png` (per-cluster bars with the overall mean marked). Set `show_axes: true` to render the TSNE subplots with axes and labels; by default they are a bare scatter.

### Scenario Overlap

[`analysis/scenario_overlap.yaml`](../src/controlledshifts/configs/analysis/scenario_overlap.yaml) measures how different the benchmarks are by quantifying how many scenarios their splits share. It reads only the split JSONs under `splits_path`, so it is fast and writes no cache. `benchmarks` selects which to compare and `splits` selects which splits (default `[training, validation, testing]`):

```yaml
splits_path: /data/driving/waymo/splits
splits: [training, validation, testing]
benchmarks:
  - uniform: {name: Uniform, abbrev: UNI, split_json: uniform}
  - ego_safeshift: {name: EgoSafeShift, abbrev: ESS, split_json: ego_safeshift}
```

`abbrev` is the short label used on the heatmap ticks so they fit without rotation; it defaults to the first three letters of `name`. A legend under the panels spells out each abbreviation.

For each split it builds a symmetric benchmark x benchmark matrix of the Jaccard index `|A ∩ B| / |A ∪ B|` over scenario IDs. Benchmarks built on a shared reference split (e.g. `background_agents` vs `uniform`) land near 1.0; benchmarks that resample the population (e.g. `background_agents_hard`) drop well below.

Writes, under `<output_path>/`: `scenario_overlap.png` (one annotated Jaccard heatmap per split, sharing a colorbar), a tidy `scenario_overlap.csv` with the raw intersection counts, and an `overlaps/` subdirectory holding the overlapping scenario IDs as JSON — one `<BenchmarkA>_<BenchmarkB>.json` per pair plus `all_benchmarks.json` for the intersection common to every benchmark.

### Shared conventions

**Distribution plots.** The `background_agents_distribution` and `ego_safeshift_distribution` analyses render, for each quantity listed in `quantities`, the same three side-by-side views with one panel per benchmark: a violin (`<quantity>_violin.png`, splits on the x-axis), a histogram (`<quantity>_histogram.png`, overlaid per-split density curves), and a ridgeline (`<quantity>_ridge.png`, one overlapping density row per split).

**Caching.** The `background_agents_distribution`, `ego_safeshift_distribution` and `environments_distribution` analyses cache their intermediate frames as CSVs and regenerate their figures from those on a re-run, so plots can be restyled without touching the source data. Set `overwrite=true` to rebuild the cache — also required to pick up changes to `benchmarks` against an already-cached CSV.
