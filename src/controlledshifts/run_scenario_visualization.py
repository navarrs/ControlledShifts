"""Scenario Visualization Script.

Renders scenarios listed in a benchmark split JSON (produced by ``create_benchmark.py``). The visualization type is
self-described by the chosen visualization config (``visualization=viz_static|viz_animated|viz_scored|viz_causal|
viz_trajpred|...``); each type loads/computes only what it needs:

    * regular       - just draw the scenarios (static or animated).
    * scored        - compute scenario features -> scores and render the scene score.
    * trajpred      - transform to agent-centric format and render one comparison pane per model.
    * model_output  - render other cached model outputs (e.g. causal predictions).

Outputs are written to ``output_dir/<render>/<split_type>/<split>/<pane_type>``.

Example usage:

    # Plain static visualization of the testing split of a benchmark.
    uv run -m controlledshifts.run_scenario_visualization \
        visualization=viz_static \
        split_filepath=/data/driving/waymo/splits/uniform.json \
        scenarios_root=/data/driving/waymo/variants/base \
        splits_to_visualize=[testing] num_scenarios=3

    # Scored visualization (requires the autolabel processors).
    uv run -m controlledshifts.run_scenario_visualization \
        visualization=viz_scored dataset.config.autolabel_agents=true \
        split_filepath=... scenarios_root=...

    # Trajpred visualization: one pane per model, aligned on the scenarios shared across all model caches.
    uv run -m controlledshifts.run_scenario_visualization \
        visualization=viz_trajpred \
        split_filepath=/data/driving/waymo/splits/uniform.json \
        scenarios_root=/data/driving/waymo/variants/base \
        splits_to_visualize=[validation] num_scenarios=3 \
        'models=[{name:wayformer,batch_cache_path:/data/.../wayformer},{name:mtr,batch_cache_path:/data/.../mtr}]'

    # Trajpred with a single model (one pane) via the batch_cache_path fallback.
    uv run -m controlledshifts.run_scenario_visualization \
        visualization=viz_trajpred \
        split_filepath=... scenarios_root=... \
        batch_cache_path=/data/.../wayformer num_scenarios=3

See ``docs/ANALYSIS.md`` for more details.
"""

import random
from collections.abc import Callable
from pathlib import Path
from time import time
from typing import NamedTuple

import hydra
import numpy as np
import pyrootutils
from characterization.schemas import Scenario, ScenarioScores
from numpy.typing import NDArray
from omegaconf import DictConfig

from controlledshifts import benchmarks, utils
from controlledshifts.datasets.agent_centric_processor import AgentCentricProcessor
from controlledshifts.datasets.waymo.repacker import load_scenario
from controlledshifts.schemas import AgentCentricScenario, ModelOutput
from controlledshifts.utils.constants import ModelStatus, VizType
from controlledshifts.utils.plotting import configure_fonts
from controlledshifts.utils.scenario_visualizers.base_visualizer import BaseVisualizer


log = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Maps a benchmark split JSON key to the ``ModelStatus`` tags used in the output path and as the batch-file tag.
SPLIT_TAGS: dict[str, str] = {
    "training": ModelStatus.TRAIN,
    "validation": ModelStatus.VALIDATION,
    "testing": ModelStatus.TEST,
}

# Default output pane folder for each visualization type (model_output uses the model_experiment tag instead).
DEFAULT_PANE_TYPES: dict[VizType, str] = {
    VizType.REGULAR: "scenario",
    VizType.SCORED: "scenario_scored",
    VizType.TRAJPRED: "trajectory_prediction",
    VizType.CAUSAL_GT: "causal_scenario_gt",
}


class PreparedScenario(NamedTuple):
    """A scenario and the optional artifacts to pass to ``visualize_scenario`` for a given visualization type."""

    scenario: Scenario | AgentCentricScenario
    scores: ScenarioScores | None = None
    model_output: ModelOutput | None = None
    causal_gt_ids: NDArray[np.int_] | None = None
    model_outputs: dict[str, ModelOutput] | None = None


def _to_agent_centric(
    processor: AgentCentricProcessor, scenario: Scenario, scores: ScenarioScores | None
) -> AgentCentricScenario | None:
    """Transforms a scenario into agent-centric format, returning None when the transform yields no agents."""
    processed = processor.process_agent_centric_scenario(scenario, scenario_scores=scores)
    if not processed:
        return None
    return AgentCentricScenario(**processed[0])


def prepare_regular(
    processor: AgentCentricProcessor, visualizer: BaseVisualizer, scenario: Scenario, model_output: ModelOutput | None
) -> PreparedScenario | None:
    """Regular visualization: draw the scenario as-is."""
    del processor, visualizer, model_output
    return PreparedScenario(scenario)


def prepare_scored(
    processor: AgentCentricProcessor, visualizer: BaseVisualizer, scenario: Scenario, model_output: ModelOutput | None
) -> PreparedScenario | None:
    """Scored visualization: compute features -> scores and render the scene score."""
    del visualizer, model_output
    scores = processor.compute_scores(scenario)
    return PreparedScenario(scenario, scores=scores)


def prepare_trajpred(
    processor: AgentCentricProcessor, visualizer: BaseVisualizer, scenario: Scenario, model_output: ModelOutput | None
) -> PreparedScenario | None:
    """Trajectory-prediction visualization: transform to agent-centric; per-model outputs are attached by ``main``."""
    del visualizer, model_output
    agent_centric = _to_agent_centric(processor, scenario, scores=None)
    if agent_centric is None:
        return None
    return PreparedScenario(agent_centric)


def prepare_model_output(
    processor: AgentCentricProcessor, visualizer: BaseVisualizer, scenario: Scenario, model_output: ModelOutput | None
) -> PreparedScenario | None:
    """Model-output visualization: render cached model outputs, transforming to agent-centric when required."""
    if visualizer.is_ego_centric:
        agent_centric = _to_agent_centric(processor, scenario, scores=None)
        if agent_centric is None:
            return None
        return PreparedScenario(agent_centric, model_output=model_output)
    return PreparedScenario(scenario, model_output=model_output)


def prepare_causal_gt(
    processor: AgentCentricProcessor, visualizer: BaseVisualizer, scenario: Scenario, model_output: ModelOutput | None
) -> PreparedScenario | None:
    """Causal ground-truth visualization: load causal agent ids from the causal-label files (no model output)."""
    del visualizer, model_output
    causal_labels_path = processor.config.get("causal_labels_path", None)
    if causal_labels_path is None:
        error_message = (
            "viz_type 'causal_gt' needs causal labels; set `dataset.config.causal_labels_path` to the labels directory."
        )
        raise ValueError(error_message)

    causal_ids = utils.load_causal_agent_ids(causal_labels_path, scenario.metadata.scenario_id)
    if causal_ids is None:
        return None

    # The ego agent is always treated as causal (mirrors `causal_idxs[track_index_to_predict] = True` in the dataset).
    ego_index = scenario.metadata.ego_vehicle_index
    ego_id = np.asarray(scenario.agent_data.agent_ids)[ego_index]
    causal_ids = np.unique(np.append(causal_ids, ego_id))
    return PreparedScenario(scenario, causal_gt_ids=causal_ids)


PrepareFn = Callable[[AgentCentricProcessor, BaseVisualizer, Scenario, ModelOutput | None], PreparedScenario | None]

SCENARIO_PREPARER: dict[VizType, PrepareFn] = {
    VizType.REGULAR: prepare_regular,
    VizType.SCORED: prepare_scored,
    VizType.TRAJPRED: prepare_trajpred,
    VizType.MODEL_OUTPUT: prepare_model_output,
    VizType.CAUSAL_GT: prepare_causal_gt,
}


def pane_type_for(viz_type: VizType, config: DictConfig) -> str:
    """Resolves the output pane folder: an explicit config override, else derived from the visualization type."""
    override = config.visualization.get("pane_type", None)
    if override:
        return override

    if viz_type == VizType.MODEL_OUTPUT:
        model_experiment = config.get("model_experiment", None)
        if not model_experiment:
            error_message = (
                "model_experiment must be set for model_output visualizations without an explicit pane_type."
            )
            raise ValueError(error_message)
        return model_experiment

    return DEFAULT_PANE_TYPES[viz_type]


def build_output_dir(output_dir: Path, render: str, split_type: str, split: str, pane_type: str) -> Path:
    """Builds and creates ``output_dir/<render>/<split_type>/<split>/<pane_type>``."""
    resolved = output_dir / render / split_type / split / pane_type
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_trajpred_models(config: DictConfig) -> list[tuple[str, str | Path]]:
    """Resolves the ``(name, cache_path)`` model specs for trajpred, one per comparison pane.

    Uses the ``models`` list when provided; otherwise falls back to the single ``batch_cache_path`` (labelled by
    ``model_experiment``, else the cache directory name) so existing one-model runs keep working. Returns an empty
    list when neither is configured.
    """
    models = config.get("models", None)
    if models:
        return [(model.name, model.batch_cache_path) for model in models]
    if config.batch_cache_path is not None:
        name = config.get("model_experiment", None) or Path(config.batch_cache_path).name or "model"
        return [(name, config.batch_cache_path)]
    return []


def load_split_model_outputs(
    config: DictConfig,
    viz_type: VizType,
    trajpred_models: list[tuple[str, str | Path]],
    scenario_ids: list[str],
    split_tag: str,
) -> tuple[dict[str, ModelOutput] | None, dict[str, dict[str, ModelOutput]] | None, list[str]]:
    """Loads a split's cached model outputs and filters/samples its scenario ids accordingly.

    TRAJPRED loads one cache per model aligned on a shared scenario set; MODEL_OUTPUT loads a single cache; both keep
    only scenarios that have an output. Other viz types load nothing and sample the split's ids down to num_scenarios.

    Returns:
        ``(single-model batches, per-model outputs keyed by scenario then model, filtered scenario ids)``.
    """
    batches: dict[str, ModelOutput] | None = None
    scenario_to_models: dict[str, dict[str, ModelOutput]] | None = None
    if viz_type == VizType.TRAJPRED:
        scenario_to_models = utils.load_batches_per_model(trajpred_models, config.num_scenarios, config.seed, split_tag)
        scenario_ids = [scenario_id for scenario_id in scenario_ids if scenario_id in scenario_to_models]
    elif viz_type == VizType.MODEL_OUTPUT:
        batches = utils.load_batches(
            config.batch_cache_path, config.num_batches, config.num_scenarios, config.seed, split_tag
        )
        scenario_ids = [scenario_id for scenario_id in scenario_ids if scenario_id in batches]
    elif config.num_scenarios is not None and len(scenario_ids) > config.num_scenarios:
        random.seed(config.seed)
        scenario_ids = random.sample(scenario_ids, config.num_scenarios)
    return batches, scenario_to_models, scenario_ids


@hydra.main(version_base="1.3", config_path="configs", config_name="scenario_visualization.yaml")
def main(config: DictConfig) -> None:
    """Hydra entry point for rendering the scenarios of a benchmark split."""
    configure_fonts(log=log)
    utils.print_config_tree(config, resolve=True, save_to_file=False)
    start = time()

    visualizer = hydra.utils.instantiate(config.visualization.visualizer)
    processor = AgentCentricProcessor(config.dataset.config)

    scenarios_root = Path(config.scenarios_root)
    output_root = Path(config.output_dir)

    # The chosen visualization config self-describes the run: its viz_type drives the per-scenario data prep, and the
    # split JSON's benchmark_name becomes the split_type folder in the output path (falling back to the file stem).
    viz_type = VizType(config.visualization.viz_type)
    split = benchmarks.load_benchmark_split(Path(config.split_filepath))
    split_type = split.benchmark_name or Path(config.split_filepath).stem

    # Resolve everything that is constant across all scenarios once: the render style (derived from the visualizer),
    # the output pane folder, the per-type prep function, and whether this type needs cached model outputs.
    render = "animated" if visualizer.is_animated else "static"
    pane_type = pane_type_for(viz_type, config)
    prepare = SCENARIO_PREPARER[viz_type]

    # TRAJPRED renders one pane per model (aligned on a shared scenario set); MODEL_OUTPUT loads a single cache.
    trajpred_models = resolve_trajpred_models(config) if viz_type == VizType.TRAJPRED else []
    if viz_type == VizType.TRAJPRED and not trajpred_models:
        error_message = (
            "viz_type 'trajpred' needs model outputs; set `models` (a list of {name, batch_cache_path}) or the single "
            "`batch_cache_path`."
        )
        raise ValueError(error_message)
    if viz_type == VizType.MODEL_OUTPUT and config.batch_cache_path is None:
        error_message = f"viz_type '{viz_type.value}' needs model outputs; set batch_cache_path to the cached batches."
        raise ValueError(error_message)

    # Render each requested split (train/val/test) into its own subfolder.
    for split_key in config.splits_to_visualize:
        if split_key not in SPLIT_TAGS:
            log.warning("Unknown split '%s'; expected one of %s. Skipping.", split_key, list(SPLIT_TAGS))
            continue
        split_tag = SPLIT_TAGS[split_key]
        scenario_ids = list(getattr(split, split_key))

        # Model-based types load this split's cached outputs and keep only scenarios that have one; the other types
        # load nothing and sample the split's ids down to num_scenarios.
        batches, scenario_to_models, scenario_ids = load_split_model_outputs(
            config, viz_type, trajpred_models, scenario_ids, split_tag
        )

        # Map each scenario id to its pickle in the flat variant store and build the destination folder for this split.
        id_to_path = {scenario_id: scenarios_root / f"{scenario_id}.pkl" for scenario_id in scenario_ids}
        output_dir = build_output_dir(output_root, render, split_type, split_tag, pane_type)
        log.info("Visualizing %d scenarios for split '%s' -> %s", len(scenario_ids), split_key, output_dir)

        for scenario_id in scenario_ids:
            # Skip ids listed in the split whose pickle is missing on disk rather than aborting the whole run.
            scenario_path = id_to_path[scenario_id]
            if not scenario_path.exists():
                log.warning("Scenario %s not found at %s, skipping.", scenario_id, scenario_path)
                continue

            # Load the canonical Scenario, apply the same profile shaping the cache builder uses, then run the per-type
            # prep (scoring, agent-centric transform, etc.). prepare() returns None when the scenario can't be drawn.
            scenario = processor.shape_scenario(load_scenario(scenario_path))
            model_output = batches.get(scenario_id) if batches is not None else None

            prepared = prepare(processor, visualizer, scenario, model_output)
            if prepared is None:
                log.warning("Could not prepare scenario %s for %s, skipping.", scenario_id, viz_type.value)
                continue

            # TRAJPRED attaches this scenario's per-model outputs (guaranteed present: ids were filtered above).
            if scenario_to_models is not None:
                prepared = prepared._replace(model_outputs=scenario_to_models[scenario_id])

            visualizer.visualize_scenario(
                prepared.scenario,
                scores=prepared.scores,
                model_output=prepared.model_output,
                output_dir=str(output_dir),
                causal_gt_ids=prepared.causal_gt_ids,
                model_outputs=prepared.model_outputs,
            )

    log.info("Total time: %.2f seconds", time() - start)


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
