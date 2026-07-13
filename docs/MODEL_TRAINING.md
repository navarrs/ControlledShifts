# Training on WOMD

## Training a single experiment

```bash
uv run -m controlledshifts.train model=[model_name]
```
`model` is required; see [Model types](#model-types). Additional config groups (all under [`src/controlledshifts/configs/`](../src/controlledshifts/configs/)):

* `paths`: the benchmark whose split and variant caches to train on — `uniform`, `causal_agents`, `causal_agents_all`, `causal_agents_hard`, `safeshift`, `safeshift_original`, `ego_safeshift`, `environments`, or `mini`. See [BENCHMARKS.md](BENCHMARKS.md). **Default:** `causal_agents`.
* `logger`: `csv`, `mlflow`, `neptune`, `tensorboard`, `wandb`, or `many_loggers` (`csv` + `wandb`). **Default:** `many_loggers`.
* `trainer`: `cpu`, `gpu`, `ddp`, `ddp_sim`, or `mps`. **Default:** `gpu`.
* `scenario`: sets the sequence partition — `waymo` (1.1 s history, 8 s prediction) or `nuscenes` (2.1 s / 6 s). **Default:** `waymo`.
* `dataset`: the input data representation. `waymo` for train/eval; `waymo_analysis` is used by the analysis and visualization entrypoints. See [DATA_PREPARATION.md](DATA_PREPARATION.md).

## Model types

| `model=` | Type | Reference | Description | Notes |
|---|---|---|---|---|
| `cvm` | Baseline (non-learned) | — | Constant-velocity extrapolation of the ego from the velocity over its last two valid history steps. | `map_aware` (**default `true`**) snaps the ego onto nearby lane centerlines and advances along each lane's arc-length at the observed speed, producing one mode per candidate lane; set `false` for plain straight-line extrapolation. Tuned by `lane_search_radius`, `min_speed`, `score_temperature`, `alignment_weight`. |
| `naive` | Baseline (learned) | — | MLP over the ego's own (x, y) history — no map, no other agents, no attention. | Lower bound that isolates the value added by map and social context. |
| `autobot` | Learned | [Girgis et al., ICLR 2022](https://arxiv.org/abs/2104.00563) | Factorized temporal and social attention. | Adapted from [UniTraj](https://github.com/vita-epfl/UniTraj). Uses its own AutoBot criterion. |
| `wayformer` | Learned | [Nayakanti et al., ICRA 2023](https://arxiv.org/abs/2207.05844) | Perceiver-IO scene encoder with a trajectory decoder. | Adapted from [UniTraj](https://github.com/vita-epfl/UniTraj). |
| `scenetransformer` | Learned | [Ngiam et al., ICLR 2022](https://arxiv.org/abs/2106.08417) | Factorized social x temporal attention over a unified scene representation. | Adapted from [AmeliaTF](https://github.com/AmeliaCMU/AmeliaTF/). |
| `mtr` | Learned | [Shi et al., NeurIPS 2022](https://arxiv.org/abs/2209.13508) | PointNet polyline encoder, global transformer, and intention-point-conditioned motion queries. | Intention points are computed from the training data and cached on the first run. |
| `mtr_mini` | Learned | *(as `mtr`)* | Smaller MTR preset: `d_model` 96, 4 attention layers, no local attention. | Hyperparameters mirrored from [SafeShift](https://github.com/cmubig/SafeShift). |

Most models train with the `TrajectoryPrediction` GMM criterion; `autobot` uses its own variant, and the MTR presets compute their loss internally (`criterion: null`).

## Evaluating a single experiment

Pass the checkpoint **name** — the full path is resolved as `${paths.ckpt_path}/${ckpt_name}.ckpt`:
```bash
uv run -m controlledshifts.eval model=[model_name] ckpt_name=[ckpt_name]
```

## Logging

* **CSV** and **W&B** run by default (`many_loggers`). CSV outputs land under `<base_path>/model_cache/<benchmark>/<model>/<date_time>/logs/<task>/csv`, where `<base_path>` is `/data/driving/<dataset>` and `<task>` is `train` or `eval`.
* **MLflow** needs a tracking URI: `uv run -m controlledshifts.train model=wayformer logger=mlflow logger.mlflow.tracking_uri=[uri]`.
* **Tensorboard**: `uv run tensorboard --logdir out/ --host [host] --port [port]`.

## Debugging

Add `debug=[debug_name]`:
* `default`: one epoch on CPU with anomaly detection.
* `fdr`: 1 train, 1 validation and 1 test step.
* `limit`: 3 epochs on 1% of the training data and 5% of val/test.
* `overfit`: 20 epochs overfitting 3 batches.
* `profiler`: a performance profiling run.

```bash
uv run -m controlledshifts.train model=wayformer debug=profiler
```

## Multirun training (parameter sweeps)

Use `-m` and pass comma-separated values for the parameter(s) to sweep:
```bash
uv run -m controlledshifts.train -m model=[model_name] model.config.num_classes=10,20,50,100
```
This launches four sequential experiments. Logs are written under `logs/<task>/multiruns/` instead of `logs/<task>/`.
