"""Tests for resolving trained runs from the results export (`utils.model_runs`)."""

import pytest

from controlledshifts.utils.model_runs import build_grid_specs, filter_runs, read_runs, resolve_split_source


_CSV_HEADER = "State,_content.dataset,_content.model,_content.timestamp\n"

# The benchmarks the scenario_overlap analysis compares, mapped onto their Hydra `paths` groups.
_PATHS_GROUPS = {"Uniform": "uniform", "CausalAgents": "causal_agents_hard", "Environments": "environments"}


def _write_csv(tmp_path, rows: list[tuple[str, str, str, str]]):
    csv_filepath = tmp_path / "runs.csv"
    csv_filepath.write_text(_CSV_HEADER + "".join(",".join(row) + "\n" for row in rows))
    return csv_filepath


def test_read_runs_skips_unfinished_rows(tmp_path):
    csv_filepath = _write_csv(
        tmp_path,
        [
            ("finished", "uniform", "wayformer", "2026-06-11_11-05-31"),
            ("crashed", "uniform", "mtr", "2026-06-12_06-01-31"),
        ],
    )

    runs = read_runs(csv_filepath, tmp_path / "cache")

    assert [run.name for run in runs] == ["uniform/wayformer"]
    assert runs[0].run_dir == tmp_path / "cache/uniform/wayformer/2026-06-11_11-05-31"


def test_filter_runs_accepts_the_tag_and_the_paths_group_spelling(tmp_path):
    csv_filepath = _write_csv(
        tmp_path,
        [
            ("finished", "causal-agents-hard", "wayformer", "t0"),
            ("finished", "uniform", "wayformer", "t1"),
        ],
    )
    runs = read_runs(csv_filepath, tmp_path / "cache")

    # The CSV spells the dataset hyphenated; the Hydra `paths` group underscores it. Both must select the run.
    assert [run.name for run in filter_runs(runs, None, ["causal_agents_hard"])] == ["causal-agents-hard/wayformer"]
    assert [run.name for run in filter_runs(runs, None, ["causal-agents-hard"])] == ["causal-agents-hard/wayformer"]


@pytest.mark.parametrize(
    ("paths_group", "split_key", "expected_source", "expected_variant"),
    [
        # A run's test/ cache holds the seen (ID) split alongside the unseen (OOD) one, so the source must be pinned to
        # the split being rendered or the two silently mix.
        ("uniform", "validation", "waymo-uniform-validation", "base"),
        ("uniform", "testing", "waymo-uniform-testing", "base"),
        # causal-agents-hard tests on a perturbed scene, so its panes must draw that variant, not the base one.
        ("causal_agents_hard", "testing", "waymo-remove-noncausal-hard-testing", "remove_noncausal"),
        # environments names its validation source `-id`, so a hand-written source map would be wrong here.
        ("environments", "validation", "waymo-environments-id", "base"),
    ],
)
def test_resolve_split_source_pins_the_split_specific_source(paths_group, split_key, expected_source, expected_variant):
    assert resolve_split_source(paths_group, split_key) == (expected_source, expected_variant)


def test_resolve_split_source_rejects_an_uncached_split():
    with pytest.raises(ValueError, match="no cached model outputs"):
        resolve_split_source("uniform", "training")


def test_build_grid_specs_pairs_every_benchmark_with_every_model(tmp_path):
    rows = []
    for dataset in ("uniform", "environments"):
        for model in ("wayformer", "mtr"):
            rows.append(("finished", dataset, model, "t0"))
            (tmp_path / "cache" / dataset / model / "t0").mkdir(parents=True)
    runs = read_runs(_write_csv(tmp_path, rows), tmp_path / "cache")

    specs = build_grid_specs(
        benchmarks=["Uniform", "Environments"],
        models=["wayformer", "mtr"],
        split_key="validation",
        runs=runs,
        paths_groups=_PATHS_GROUPS,
    )

    assert [(spec.group, spec.name) for spec in specs] == [
        ("Uniform", "wayformer"),
        ("Uniform", "mtr"),
        ("Environments", "wayformer"),
        ("Environments", "mtr"),
    ]
    # Each row is pinned to its own benchmark's source for this split.
    assert {spec.group: spec.source for spec in specs} == {
        "Uniform": "waymo-uniform-validation",
        "Environments": "waymo-environments-id",
    }


def test_build_grid_specs_raises_for_an_unresolvable_cell(tmp_path):
    # environments/mtr is absent from the CSV, so its cell cannot be filled.
    (tmp_path / "cache" / "uniform" / "mtr" / "t0").mkdir(parents=True)
    runs = read_runs(_write_csv(tmp_path, [("finished", "uniform", "mtr", "t0")]), tmp_path / "cache")

    with pytest.raises(ValueError, match=r"Environments/mtr: 0 finished runs"):
        build_grid_specs(
            benchmarks=["Uniform", "Environments"],
            models=["mtr"],
            split_key="validation",
            runs=runs,
            paths_groups=_PATHS_GROUPS,
        )
