"""Tests for the trajpred visualizer's pane layout and feature decoding."""

import numpy as np

from controlledshifts.utils.scenario_visualizers.scenario_trajpred import (
    _MAP_TYPE_TO_KEY,
    ScenarioTrajpredVisualizer,
)


def test_grid_layout_keeps_row_order_and_collects_columns_first_seen():
    # A row missing a column must not drop it from the layout: that pane is rendered empty instead.
    grid = {"Uniform": {"wayformer": None, "mtr": None}, "CausalAgents": {"mtr": None, "autobot": None}}

    assert ScenarioTrajpredVisualizer._grid_layout(grid) == (
        ["Uniform", "CausalAgents"],
        ["wayformer", "mtr", "autobot"],
    )


def test_map_type_to_key_covers_the_semantic_types_and_skips_padding():
    # 0 is padding/masked and must be absent so the map draw skips it; the real Waymo types map onto map_colors keys.
    assert 0 not in _MAP_TYPE_TO_KEY
    assert _MAP_TYPE_TO_KEY[2] == "lane"
    assert _MAP_TYPE_TO_KEY[15] == "road_edge"
    assert _MAP_TYPE_TO_KEY[18] == "crosswalk"
    assert _MAP_TYPE_TO_KEY[19] == "speed_bump"


def _agent_row(*, type_onehot: list[int], is_ego: int, heading: float, length: float, width: float) -> np.ndarray:
    """Builds a single-timestep obj_trajs row with the fields _decode_agent reads."""
    row = np.zeros((1, 29), dtype=np.float32)
    row[0, 3:6] = (length, width, 1.8)
    row[0, 6:9] = type_onehot
    row[0, 10] = is_ego
    row[0, 23:25] = (np.sin(heading), np.cos(heading))
    return row


def test_decode_agent_reads_type_heading_and_size():
    row = _agent_row(type_onehot=[0, 1, 0], is_ego=0, heading=0.5, length=4.6, width=2.0)

    type_key, heading, length, width, is_ego = ScenarioTrajpredVisualizer._decode_agent(row, valid_step=0)

    assert (type_key, is_ego) == ("TYPE_PEDESTRIAN", False)
    assert heading == np.float32(0.5)
    assert (length, width) == (np.float32(4.6), np.float32(2.0))


def test_decode_agent_flags_the_ego_regardless_of_type_onehot():
    # The ego channel wins over the type one-hot, so the ego is colored as the SDC.
    row = _agent_row(type_onehot=[1, 0, 0], is_ego=1, heading=0.0, length=5.0, width=2.2)

    type_key, _, _, _, is_ego = ScenarioTrajpredVisualizer._decode_agent(row, valid_step=0)

    assert (type_key, is_ego) == ("TYPE_SDC", True)
