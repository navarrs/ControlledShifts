"""Tests for the ego-safeshift score distribution analysis."""

import logging

import pandas as pd
from omegaconf import OmegaConf

from controlledshifts.utils.analysis.ego_safeshift_distribution import run_ego_safeshift_distribution_analysis


QUANTITIES = [
    "gt_critical_continuous_individual",
    "gt_critical_continuous_interaction",
    "gt_critical_continuous_safeshift",
]
SCORE_TYPE = "gt_critical_continuous_safeshift"


def _write_scores_csv(csv_path, num_scenarios) -> None:
    """Writes a synthetic scores CSV with .pkl-suffixed ids and one increasing column per quantity."""
    frame = pd.DataFrame({"scenario_ids": [f"scenario_{i:03d}.pkl" for i in range(num_scenarios)]})
    for offset, quantity in enumerate(QUANTITIES):
        frame[quantity] = [float(i + offset) for i in range(num_scenarios)]
    frame.to_csv(csv_path, index=False)


def _make_config(csv_path, output_path):
    return OmegaConf.create(
        {
            "scores_csv_path": str(csv_path),
            "score_type": SCORE_TYPE,
            "split_ratios": [0.70, 0.15, 0.15],
            "seed": 42,
            "overwrite": False,
            "quantities": QUANTITIES,
            "benchmarks": [
                {"ego_safeshift": {"name": "EgoSafeShift", "split": "score"}},
                {"uniform": {"name": "Uniform", "split": "random"}},
            ],
        }
    )


def test_writes_frame_summary_and_plots(tmp_path):
    """A full run writes the long-form frame, the summary, and the three figures per quantity."""
    csv_path = tmp_path / "scene_to_scores_mapping.csv"
    _write_scores_csv(csv_path, num_scenarios=200)
    output_path = tmp_path / "out"

    run_ego_safeshift_distribution_analysis(
        _make_config(csv_path, output_path), logging.getLogger(__name__), output_path
    )

    assert (output_path / "ego_safeshift_distribution.csv").exists()
    assert (output_path / "summary.csv").exists()
    for quantity in QUANTITIES:
        for kind in ("violin", "histogram", "ridge"):
            assert (output_path / f"{quantity}_{kind}.png").exists()


def test_score_split_sends_hardest_to_test(tmp_path):
    """The ego_safeshift (score) panel must place higher scores in test than train; the random panel should not."""
    csv_path = tmp_path / "scene_to_scores_mapping.csv"
    _write_scores_csv(csv_path, num_scenarios=200)
    output_path = tmp_path / "out"

    run_ego_safeshift_distribution_analysis(
        _make_config(csv_path, output_path), logging.getLogger(__name__), output_path
    )

    long_df = pd.read_csv(output_path / "ego_safeshift_distribution.csv")
    ego = long_df[long_df["benchmark"] == "EgoSafeShift"]
    ego_test = ego[ego["split"] == "testing"][SCORE_TYPE].mean()
    ego_train = ego[ego["split"] == "training"][SCORE_TYPE].mean()
    assert ego_test > ego_train  # hardest (highest score) scenarios are routed to the test set

    uniform = long_df[long_df["benchmark"] == "Uniform"]
    uniform_test = uniform[uniform["split"] == "testing"][SCORE_TYPE].mean()
    uniform_train = uniform[uniform["split"] == "training"][SCORE_TYPE].mean()
    # a random split should not systematically shift the test mean far above train
    assert abs(uniform_test - uniform_train) < abs(ego_test - ego_train)
