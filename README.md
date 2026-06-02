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

# nuScenes dataset (requires Python 3.12)
uv run pip install -e ".[nuscenes]"
```

If installing with development dependencies, run:
```bash
uv run pip install -e ".[dev]"
uv run pre-commit install
```

## Documentation

## Citing

```
```

## Development

Run `uv sync --frozen --all-groups` to set up the environment.
Run `pre-commit run --all-files` to run all hooks on all files.
