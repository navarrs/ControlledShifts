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

    # Trajpred overlap grid: scenarios shared by two benchmarks, rendered as benchmarks (rows) x models (columns).
    # Each row draws the scene variant its models were evaluated on, so it needs no scenarios_root.
    uv run -m controlledshifts.run_scenario_visualization \
        visualization=viz_trajpred overlap_grid.enabled=true \
        split_filepath=outputs/scenario_overlap_analysis/overlaps/Uniform_CausalAgents.json \
        splits_to_visualize=[validation,testing] num_scenarios=5

See ``docs/ANALYSIS.md`` for more details.
"""

import random
from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import time
from typing import NamedTuple

import hydra
import numpy as np
import pyrootutils
from characterization.schemas import Scenario, ScenarioScores
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from controlledshifts import benchmarks, utils
from controlledshifts.datasets.agent_centric_processor import AgentCentricProcessor
from controlledshifts.datasets.waymo.repacker import load_scenario
from controlledshifts.schemas import AgentCentricScenario, ModelOutput
from controlledshifts.utils import model_runs
from controlledshifts.utils.constants import VizType
from controlledshifts.utils.model_runs import SPLIT_TAGS, ModelCacheSpec
from controlledshifts.utils.plotting import configure_fonts
from controlledshifts.utils.scenario_visualizers.base_visualizer import BaseVisualizer


log = utils.get_pylogger(__name__)

PROJECT_ROOT = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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
    model_grid: dict[str, dict[str, ModelOutput]] | None = None
    row_scenarios: dict[str, AgentCentricScenario] | None = None


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


def resolve_trajpred_specs(config: DictConfig, split_filepath: Path, split_key: str) -> list[ModelCacheSpec]:
    """Resolves one ``ModelCacheSpec`` per trajpred pane, for the split about to be rendered.

    Grid mode (``overlap_grid.enabled``) reads the benchmarks out of the overlap file and pairs each with every
    configured model, resolving that (benchmark, model) run from the results CSV. Each row is pinned to the cache
    source its benchmark declares for this split, so the specs are split-specific.

    Otherwise the flat ``models`` list is used, falling back to the single ``batch_cache_path`` (labelled by
    ``model_experiment``, else the cache directory name) so existing one-model runs keep working. Both flat forms
    render as one unlabelled row. Returns an empty list when nothing is configured.

    Raises:
        ValueError: in grid mode, if the split has no cached outputs or a cell cannot be resolved to a single run.
    """
    grid_config = config.get("overlap_grid", None)
    if grid_config is not None and grid_config.get("enabled", False):
        if SPLIT_TAGS.get(split_key) not in model_runs.CACHE_SPLITS:
            error_message = (
                f"overlap_grid cannot render split '{split_key}': training outputs are never cached. Use "
                f"splits_to_visualize=[validation,testing]."
            )
            raise ValueError(error_message)

        csv_filepath = model_runs.resolve_csv_filepath(grid_config.csv_filepath, PROJECT_ROOT)
        paths_groups = {str(name): str(group) for name, group in grid_config.benchmark_paths_groups.items()}
        runs = model_runs.read_runs(csv_filepath, Path(grid_config.cache_root))
        return model_runs.build_grid_specs(
            benchmarks=benchmarks.load_overlap_benchmarks(split_filepath),
            models=list(grid_config.models),
            split_key=split_key,
            runs=runs,
            paths_groups=paths_groups,
        )

    cache_source = config.get("cache_source", None)
    models = config.get("models", None)
    if models:
        return [
            ModelCacheSpec(name=model.name, cache_path=Path(model.batch_cache_path), source=cache_source)
            for model in models
        ]
    if config.batch_cache_path is not None:
        name = config.get("model_experiment", None) or Path(config.batch_cache_path).name or "model"
        return [ModelCacheSpec(name=name, cache_path=Path(config.batch_cache_path), source=cache_source)]
    return []


def resolve_scenario_roots(config: DictConfig, specs: list[ModelCacheSpec]) -> dict[str, Path]:
    """Maps each scene variant the panes need to the variant store it is loaded from.

    A grid row draws the scene its models were evaluated on, which is not always the unperturbed one --
    causal-agents-hard tests on ``remove_noncausal`` -- so the variants come off the specs and resolve under
    ``overlap_grid.variants_root``. Every other mode draws one scene, from the single ``scenarios_root``.

    Raises:
        ValueError: if ``scenarios_root`` is needed but not set.
    """
    if specs and config.overlap_grid.enabled:
        variants_root = Path(config.overlap_grid.variants_root)
        return {spec.variant: variants_root / spec.variant for spec in specs}

    if config.scenarios_root is None:
        error_message = "scenarios_root must be set: the flat variant store holding the <scenario_id>.pkl scenarios."
        raise ValueError(error_message)
    return {"": Path(config.scenarios_root)}


def prepare_scenario_variants(
    scenario_id: str,
    scenario_roots: dict[str, Path],
    processor: AgentCentricProcessor,
    prepare_one: Callable[[Scenario, ModelOutput | None], PreparedScenario | None],
    model_output: ModelOutput | None,
) -> dict[str, PreparedScenario] | None:
    """Loads and prepares one scenario from each variant store the panes need.

    Returns:
        The prepared scenario per variant, or None when any variant is missing on disk or cannot be drawn -- the panes
        must stay aligned, so a scenario is rendered only if every variant it needs is available.
    """
    prepared_variants: dict[str, PreparedScenario] = {}
    for variant, root in scenario_roots.items():
        scenario_path = root / f"{scenario_id}.pkl"
        if not scenario_path.exists():
            log.warning("Scenario %s not found at %s, skipping.", scenario_id, scenario_path)
            return None

        # Apply the same profile shaping the cache builder uses, then run the per-type prep (scoring, agent-centric
        # transform, etc.). prepare_one() returns None when the scenario cannot be drawn.
        prepared = prepare_one(processor.shape_scenario(load_scenario(scenario_path)), model_output)
        if prepared is None:
            log.warning("Could not prepare scenario %s (variant '%s'), skipping.", scenario_id, variant or "base")
            return None
        prepared_variants[variant] = prepared
    return prepared_variants


def load_split_model_outputs(
    config: DictConfig,
    viz_type: VizType,
    trajpred_specs: list[ModelCacheSpec],
    scenario_ids: list[str],
    split_tag: str,
) -> tuple[dict[str, ModelOutput] | None, dict[str, dict[str, dict[str, ModelOutput]]] | None, list[str]]:
    """Loads a split's cached model outputs and filters/samples its scenario ids accordingly.

    TRAJPRED loads one cache per pane, sampling from the scenarios cached by all of them; MODEL_OUTPUT loads a single
    cache; both keep only scenarios that have an output. Other viz types load nothing and sample the split's ids down
    to num_scenarios.

    Returns:
        ``(single-model batches, per-scenario pane outputs keyed by row then model, filtered scenario ids)``.
    """
    batches: dict[str, ModelOutput] | None = None
    scenario_grid: dict[str, dict[str, dict[str, ModelOutput]]] | None = None
    if viz_type == VizType.TRAJPRED:
        scenario_grid = utils.load_batches_per_model(
            trajpred_specs, scenario_ids, config.num_scenarios, config.seed, split_tag
        )
        scenario_ids = [scenario_id for scenario_id in scenario_ids if scenario_id in scenario_grid]
    elif viz_type == VizType.MODEL_OUTPUT:
        cache_source = config.get("cache_source", None)
        batches = utils.load_batches(
            config.batch_cache_path, config.num_batches, config.num_scenarios, config.seed, split_tag, cache_source
        )
        scenario_ids = [scenario_id for scenario_id in scenario_ids if scenario_id in batches]
    elif config.num_scenarios is not None and len(scenario_ids) > config.num_scenarios:
        random.seed(config.seed)
        scenario_ids = random.sample(scenario_ids, config.num_scenarios)
    return batches, scenario_grid, scenario_ids


@hydra.main(version_base="1.3", config_path="configs", config_name="scenario_visualization.yaml")
def main(config: DictConfig) -> None:
    """Hydra entry point for rendering the scenarios of a benchmark split."""
    configure_fonts(log=log)
    utils.print_config_tree(config, resolve=True, save_to_file=False)
    start = time()

    visualizer = hydra.utils.instantiate(config.visualization.visualizer)

    # A visualization config may carry the dataset params its prep needs (trajpred's agent-centric transform reads the
    # ones the analysis profile nulls, since they normally interpolate ${model.config...}).
    processor_config = config.dataset.config
    dataset_overrides = config.visualization.get("dataset_overrides", None)
    if dataset_overrides is not None:
        processor_config = OmegaConf.merge(processor_config, dataset_overrides)
    processor = AgentCentricProcessor(processor_config)

    output_root = Path(config.output_dir)

    # The chosen visualization config self-describes the run: its viz_type drives the per-scenario data prep, and the
    # split JSON's benchmark_name becomes the split_type folder in the output path (falling back to the file stem).
    viz_type = VizType(config.visualization.viz_type)
    split_filepath = Path(config.split_filepath)
    split = benchmarks.load_benchmark_split(split_filepath)
    split_type = split.benchmark_name or split_filepath.stem

    # Resolve everything that is constant across all scenarios once: the render style (derived from the visualizer),
    # the output pane folder, the per-type prep function, and whether this type needs cached model outputs.
    render = "animated" if visualizer.is_animated else "static"
    pane_type = pane_type_for(viz_type, config)
    prepare_one = partial(SCENARIO_PREPARER[viz_type], processor, visualizer)

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

        # TRAJPRED panes are resolved per split: a benchmark evaluates validation and testing under different cache
        # sources (and sometimes different scene variants), so the specs differ between them.
        trajpred_specs: list[ModelCacheSpec] = []
        if viz_type == VizType.TRAJPRED:
            trajpred_specs = resolve_trajpred_specs(config, split_filepath, split_key)
            if not trajpred_specs:
                error_message = (
                    "viz_type 'trajpred' needs model outputs; enable `overlap_grid`, or set `models` (a list of "
                    "{name, batch_cache_path}) or the single `batch_cache_path`."
                )
                raise ValueError(error_message)

        # Model-based types load this split's cached outputs and keep only scenarios that have one; the other types
        # load nothing and sample the split's ids down to num_scenarios.
        batches, scenario_grid, scenario_ids = load_split_model_outputs(
            config, viz_type, trajpred_specs, scenario_ids, split_tag
        )

        scenario_roots = resolve_scenario_roots(config, trajpred_specs)
        output_dir = build_output_dir(output_root, render, split_type, split_tag, pane_type)
        log.info("Visualizing %d scenarios for split '%s' -> %s", len(scenario_ids), split_key, output_dir)

        for scenario_id in scenario_ids:
            model_output = batches.get(scenario_id) if batches is not None else None
            prepared_variants = prepare_scenario_variants(
                scenario_id, scenario_roots, processor, prepare_one, model_output
            )
            if prepared_variants is None:
                continue
            # Every mode but the grid loads a single scene; the grid's rows share the figure's scenario id anyway.
            prepared = next(iter(prepared_variants.values()))

            # TRAJPRED attaches this scenario's pane outputs (guaranteed present: ids were filtered above). The flat
            # forms produce a single unlabelled row, which renders exactly as it did before the grid existed.
            if scenario_grid is not None:
                grid = scenario_grid[scenario_id]
                if set(grid) == {""}:
                    prepared = prepared._replace(model_outputs=grid[""])
                else:
                    row_scenarios = {spec.group: prepared_variants[spec.variant].scenario for spec in trajpred_specs}
                    prepared = prepared._replace(model_grid=grid, row_scenarios=row_scenarios)

            visualizer.visualize_scenario(
                prepared.scenario,
                scores=prepared.scores,
                model_output=prepared.model_output,
                output_dir=str(output_dir),
                causal_gt_ids=prepared.causal_gt_ids,
                model_outputs=prepared.model_outputs,
                model_grid=prepared.model_grid,
                row_scenarios=prepared.row_scenarios,
            )

    log.info("Total time: %.2f seconds", time() - start)


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
