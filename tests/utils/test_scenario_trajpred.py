"""Tests for the trajpred visualizer's pane layout."""

from controlledshifts.utils.scenario_visualizers.scenario_trajpred import ScenarioTrajpredVisualizer


def test_grid_layout_keeps_row_order_and_collects_columns_first_seen():
    # A row missing a column must not drop it from the layout: that pane is rendered empty instead.
    grid = {"Uniform": {"wayformer": None, "mtr": None}, "CausalAgents": {"mtr": None, "autobot": None}}

    assert ScenarioTrajpredVisualizer._grid_layout(grid) == (
        ["Uniform", "CausalAgents"],
        ["wayformer", "mtr", "autobot"],
    )
