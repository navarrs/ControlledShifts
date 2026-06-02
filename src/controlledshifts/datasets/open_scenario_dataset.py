"""OpenScenario dataset loader for pre-processed Scenario objects."""

import pickle
from pathlib import Path

from characterization.schemas import Scenario
from omegaconf import DictConfig

from controlledshifts.datasets.base_dataset import BaseDataset


class OpenScenarioDataset(BaseDataset):
    """Dataset loader for OpenScenario format.

    This dataset expects scenarios to already be saved as Scenario objects in pickle files.
    The preprocessing step should have already converted your source data into the Scenario
    schema format and saved them as .pkl files.

    Directory structure expected:
        /datasets/open_scenario/processed/<variant>/
        ├── training/
        │   ├── scenario_001.pkl  # Each contains a complete Scenario object
        │   ├── scenario_002.pkl
        │   └── ...
        ├── validation/
        │   └── *.pkl
        └── testing/
            └── *.pkl

    The BaseDataset._get_dataset_summary() method automatically discovers all .pkl files
    using rglob, so this works with both flat structures and shard-based structures.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the OpenScenario dataset.

        Args:
            config: Dataset configuration from Hydra config.
        """
        super().__init__(config)

    def load_as_open_scenario(self, path: Path) -> Scenario:
        """Loads a pre-processed Scenario object from a pickle file.

        Unlike WaymoDataset which loads a dict and needs repacking, OpenScenario
        pickles already contain complete Scenario objects. However, to maintain
        compatibility with the base class interface, we return it as a dict-like
        object (the Scenario will be passed to repack_scenario).

        Args:
            path: Full path to the scenario pickle file.

        Returns:
            Scenario: The pre-processed scenario object.
        """
        with path.open("rb") as f:
            return pickle.load(f)
