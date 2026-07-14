"""Tests for resolving trained runs from the results export (`utils.model_runs`)."""

from controlledshifts.utils.model_runs import filter_runs, read_runs


_CSV_HEADER = "State,_content.dataset,_content.model,_content.timestamp\n"


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
