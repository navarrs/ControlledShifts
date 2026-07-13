# ControlledShifts

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
| [BENCHMARKS.md](docs/BENCHMARKS.md) | The six distribution-shift benchmarks: how splits and perturbed variants are created and consumed. |
| [MODEL_TRAINING.md](docs/MODEL_TRAINING.md) | Train and evaluate the supported models. |
| [ANALYSIS.md](docs/ANALYSIS.md) | Visualize scenarios and run model analyses. |


## Development

Run `uv sync --frozen --all-groups` to set up the environment.
Run `pre-commit run --all-files` to run all hooks on all files.
Run `./tests/run_tests.sh` to run the test suite (`-c <category>` for a single category, `-h` for options).

## Citing

```
```
