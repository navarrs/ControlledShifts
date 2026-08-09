<p align="center">
  <a href="https://github.com/astral-sh/uv">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" /></a>
  <a href="https://github.com/astral-sh/ruff">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>
  <a href="https://github.com/microsoft/pyright">
  <img src="https://microsoft.github.io/pyright/img/pyright_badge.svg" /></a>
  <a href="https://docs.pydantic.dev">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json" /></a>
  <a href="https://hydra.cc">
  <img src="https://img.shields.io/badge/config-Hydra-E87615" /></a>
  <a href="https://lightning.ai">
  <img src="https://img.shields.io/badge/Lightning-2.5-792EE5" /></a>
  <a href="LICENSE">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" /></a>
</p>

# ![WIP](https://img.shields.io/badge/status-WIP-orange) ControlledShifts

A framework for the **standardized evaluation of trajectory prediction models under distribution shift**. Measuring robustness is largely ad hoc — each method commits to its own axis of generalization, making robustness comparison difficult. Rather than collecting new data, ControlledShifts re-splits an existing corpus into **seen** and **unseen** partitions through a shared characterization-and-splitting formulation, instantiated across heterogeneous axes of variation. Three layers implement it:

1. **Characterization and splitting**: a characterization function scores each scenario along an axis of variation; a splitting function partitions the pool into seen (train/validation) and unseen (test) sets. Where the shift is a scene edit rather than a partition, perturbed variants are generated and paired with their originals under the same split.
2. **Training and evaluation**: Hydra-configured, Lightning-based training on the seen partition and evaluation on both.
3. **Robustness scoring and analysis**: a unified scoring scheme condenses seen/unseen performance into a single comparable value, alongside per-benchmark tables, radar plots, and scenario visualization.

Axes of variation and models are independent config groups, so either can be added without touching the other. Three complementary benchmarks target behavioral and topological shifts relevant to the ego agent — Background Agents, Ego-SafeShift, and Environments — alongside an IID control and additional variants; see [BENCHMARKS.md](docs/BENCHMARKS.md). Demonstrated on [Waymo Open Motion](https://waymo.com/open); scenario scoring builds on [SafeShift](https://github.com/cmubig/SafeShift) via [ScenarioCharacterization](https://github.com/navarrs/ScenarioCharacterization).

This project was a collaboration between **LavoroAI** and **StackAV**.

<!-- TODO: replace with the framework diagram -->
<img width="100%" alt="Teaser" src="https://github.com/user-attachments/assets/1f420704-81bd-468e-8eb3-89cc7665d3c7" />
## Installation

Clone the repository and install the package in editable mode:
```bash
git clone git@github.com:navarrs/ControlledShifts.git
cd ControlledShifts
uv run pip install -e .
```

To install with dataset-specific dependencies, use the appropriate optional extra:

```bash
# Waymo Open Motion Dataset (requires Python 3.10)
uv run pip install -e ".[waymo]"
```

If installing with development dependencies, run:
```bash
uv run pip install -e ".[dev]"
uv run pre-commit install
```

## Documentation

| Document | Purpose |
|---|---|
| [DATA_PREPARATION.md](docs/DATA_PREPARATION.md) | Download the raw WOMD data and build dataset cache. |
| [BENCHMARKS.md](docs/BENCHMARKS.md) | The distribution-shift benchmarks: how splits and perturbed variants are created and consumed. |
| [MODEL_TRAINING.md](docs/MODEL_TRAINING.md) | Train and evaluate the supported models. |
| [ANALYSIS.md](docs/ANALYSIS.md) | Visualize scenarios and run model analyses. |


## Development

Run `uv sync --frozen --all-groups` to set up the environment.
Run `pre-commit run --all-files` to run all hooks on all files.
Run `./tests/run_tests.sh` to run the test suite (`-c <category>` for a single category, `-h` for options).

## Citing

```
```
