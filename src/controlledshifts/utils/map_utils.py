"""Map graph utilities for road topology analysis and visualization.

Provides functions for:
- Converting scenario map data into directed NetworkX graphs (``map_infos_to_graph``).
- Filtering map elements to those within a proximity range of a reference point
  (``transform_map_elements``, ``filter_map_elements_by_proximity``).
- Simplifying graphs by removing isolated nodes and keeping the largest connected component
  (``simplify_graph``).
- Building graphs with matplotlib-compatible position dicts (``build_positioned_graph``).
- Rendering road topology graphs as PNG files (``visualize_scenario_graph``).
"""

import operator
import pickle  # nosec B403
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from controlledshifts.utils.data_utils import find_true_segments


# Color scheme for each map element type in graph and raster visualizations.
_ELEMENT_COLORS: dict[str, str] = {
    "lane": "#4A90D9",
    "road_line": "#F5A623",
    "road_edge": "#7B68EE",
    "crosswalk": "#50C878",
    "speed_bump": "#FF6B6B",
    "stop_sign": "#FF4500",
}
_DEFAULT_NODE_COLOR = "#AAAAAA"
_MAP_ELEMENT_TYPES = ("lane", "road_line", "road_edge", "crosswalk", "speed_bump", "stop_sign")

# Per-type rendering style for raster polyline plots.
_POLYLINE_STYLES: dict[str, dict[str, Any]] = {
    "lane": {"linewidth": 1.0, "alpha": 0.8},
    "road_line": {"linewidth": 0.5, "alpha": 0.6},
    "road_edge": {"linewidth": 1.0, "alpha": 0.8},
    "crosswalk": {"linewidth": 0.5, "alpha": 0.7},
    "speed_bump": {"linewidth": 0.5, "alpha": 0.7},
    "stop_sign": {"marker_size": 16, "alpha": 1.0},  # rendered as scatter
}


def _node_pos(element: dict[str, Any], all_polylines: NDArray[np.float64]) -> dict[str, float]:
    """Returns centroid position attributes for a map element node, or an empty dict if unavailable.

    Args:
        element: Map element dict containing a ``polyline_index`` key with (start, end) row indices.
        all_polylines: Array of shape (N, >=3) from which the centroid is computed.

    Returns:
        Dict with keys ``x``, ``y``, ``z`` (mean of the element's polyline points), or ``{}`` if the element has no
        valid polyline slice.
    """
    start, end = element.get("polyline_index", (0, 0))
    if all_polylines.shape[0] > 0 and end > start:
        centroid = all_polylines[start:end, :3].mean(axis=0)
        return {"x": float(centroid[0]), "y": float(centroid[1]), "z": float(centroid[2])}
    return {}


def _plot_map_raster(ax: Axes, map_infos: dict[str, Any]) -> None:
    """Plots all map polylines onto ax, coloured by element type.

    Each lane, road line, road edge, crosswalk, and speed bump is drawn as a line; stop signs are drawn as scatter
    points. Uses the same colour scheme as the graph visualizer.

    Args:
        ax: Axes to draw on.
        map_infos: Map information dictionary from a scenario pickle file, with ``all_polylines`` and per-type element
            lists already filtered/transformed to the desired coordinate frame.
    """
    all_polylines: NDArray[np.float64] = np.asarray(map_infos.get("all_polylines", np.empty((0, 7))))
    if all_polylines.shape[0] == 0:
        return

    for etype in _MAP_ELEMENT_TYPES:
        color = _ELEMENT_COLORS[etype]
        style = _POLYLINE_STYLES[etype]
        for element in map_infos.get(etype, []):
            start, end = element.get("polyline_index", (0, 0))
            if end <= start:
                continue
            pts = all_polylines[start:end, :2]
            if etype == "stop_sign":
                ax.scatter(pts[:, 0], pts[:, 1], s=style["marker_size"], c=color, marker="H", alpha=style["alpha"])
            else:
                ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=style["linewidth"], alpha=style["alpha"])


def _rotate_points_along_z(points: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """Rotates the (x, y) columns of an array by angle radians around the Z axis.

    Args:
        points: Array whose last dimension is at least 2, with x at index 0 and y at index 1.
        angle: Rotation angle in radians (positive = counter-clockwise).

    Returns:
        Copy of points with (x, y) rotated.
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    result = points.copy()
    result[..., :2] = points[..., :2] @ rot.T
    return result


def transform_map_elements(
    all_polylines: NDArray[np.float64],
    ref_xy: NDArray[np.float64],
    ref_heading: float = 0.0,
) -> NDArray[np.float64]:
    """Transforms polyline points from world coordinates to a local reference frame.

    Args:
        all_polylines: Array of shape (N, >=2) with all polyline points in world coordinates.
        ref_xy: Reference position (x, y) to translate to the origin.
        ref_heading: Reference heading in radians to rotate away from. Defaults to 0.0.

    Returns:
        Copy of all_polylines with (x, y) transformed to the reference frame.
    """
    if all_polylines.shape[0] == 0:
        return all_polylines

    pts_ref: NDArray[np.float64] = all_polylines.copy()
    pts_ref[..., :2] -= ref_xy[:2]
    return _rotate_points_along_z(pts_ref, -ref_heading)


def filter_map_elements_by_proximity(
    map_infos: dict[str, Any],
    polylines: NDArray[np.float64],
    num_map_elements: int = 100,
    map_range: float = 100.0,
) -> dict[str, Any]:
    """Returns a filtered copy of map_infos keeping map elements that are within range of the reference point.

    Mirrors the selection logic in base_dataset.py's ``get_centered_map_data``. Expects ``polylines`` to already be in
    reference-frame coordinates (e.g. the output of ``transform_map_elements``):

    1. A map element passes the **range filter** if any of its points fall within the L∞ box
       ``|x| < map_range AND |y| < map_range``.
    2. Among passing elements, the top-K are kept by ascending **average L2 distance** of their points from the
       reference-frame origin.

    The rebuilt ``all_polylines`` array in the returned dict uses reference-frame coordinates, so every stored point
    satisfies ``|x| < map_range AND |y| < map_range``.

    Args:
        map_infos: Map information dictionary from a scenario pickle file.
        polylines: Array of shape (N, >=2) with all polyline points in reference-frame coordinates.
        num_map_elements: Maximum number of map elements to retain.
        map_range: L∞ half-width of the reference-frame range box in metres. Elements with at least one point inside the
            box are candidates. Defaults to 100.0.

    Returns:
        Shallow copy of map_infos with each element list filtered to the K nearest in-range elements.
    """
    if polylines.shape[0] == 0:
        return map_infos

    # Collect candidates: elements with at least one point inside the L∞ range box. `polylines` is assumed to be in
    # reference-frame coordinates (ego-centered), so the range check and the stored points both use the same coordinate
    # system — x, y values in the output will satisfy |x| < map_range and |y| < map_range by construction.
    candidates: list[tuple[float, str, dict[str, Any], list[int]]] = []
    for etype in _MAP_ELEMENT_TYPES:
        for element in map_infos.get(etype, []):
            start, end = element.get("polyline_index", (0, 0))
            if end <= start:
                continue
            pts = polylines[start:end, :2]  # reference-frame x, y

            in_range_mask = (np.abs(pts[:, 0]) < map_range) & (np.abs(pts[:, 1]) < map_range)
            if not in_range_mask.any():
                continue

            segments = find_true_segments(in_range_mask)
            in_range_indices: list[int] = [idx for seg in segments for idx in seg]

            avg_dist = float(np.linalg.norm(pts[in_range_indices], axis=-1).mean())
            candidates.append((avg_dist, etype, element, in_range_indices))

    # Sort by ascending average distance and keep the top-K.
    candidates.sort(key=operator.itemgetter(0))

    # Rebuild all_polylines from only the in-range points of surviving elements, updating
    # polyline_index so every consumer (e.g. _node_pos) automatically sees range-clipped data.
    filtered: dict[str, list] = {etype: [] for etype in _MAP_ELEMENT_TYPES}
    new_chunks: list[NDArray[np.float64]] = []
    new_offset = 0

    for _, etype, element, in_range_indices in candidates[:num_map_elements]:
        start, end = element["polyline_index"]

        # Use the reference-frame polylines (not the original world-coord array) so that the stored points are the ones
        # actually passed the range check.
        valid_pts = polylines[start:end][in_range_indices]
        n = int(valid_pts.shape[0])
        filtered[etype].append({**element, "polyline_index": (new_offset, new_offset + n)})

        new_chunks.append(valid_pts)
        new_offset += n

    new_all_polylines: NDArray[np.float64] = (
        np.concatenate(new_chunks, axis=0) if new_chunks else np.empty((0, polylines.shape[1]), dtype=np.float64)
    )

    # Build and return a new dict. {**map_infos} copies all metadata keys (e.g. scenario_id) as defaults; the explicit
    # "all_polylines" and **filtered then *override* the original element lists and polyline array with the
    # range-clipped versions.  Later keys win in Python dict merging, so the original "all_polylines" and element-type
    # lists from map_infos are fully replaced — only unrelated metadata is preserved from the spread.
    return {**map_infos, "all_polylines": new_all_polylines, **filtered}


def map_infos_to_graph(
    map_infos: dict[str, Any],
    ref_xy: NDArray[np.float64] | None = None,
    ref_heading: float = 0.0,
    num_map_elements: int | None = None,
    map_range: float = 100.0,
) -> nx.DiGraph:
    """Converts scenario map information into a directed NetworkX graph.

    Nodes represent lanes, road lines, road edges, crosswalks, speed bumps, and stop signs. Directed edges connect lanes
    via their entry/exit lane relationships. Each node stores the centroid (x, y, z) of its polyline as position
    attributes when available.

    Args:
        map_infos: Map information dictionary from a scenario pickle file.
        ref_xy: If provided together with num_map_elements, triggers proximity filtering via
            `filter_map_elements_by_proximity`.
        num_map_elements: Maximum number of map elements to retain when ref_xy is set.
        ref_heading: Reference heading in radians, used for the range check. Defaults to 0.0.
        map_range: L∞ half-width of the reference-frame range box in metres. Defaults to 100.0.

    Returns:
        Directed graph representing the road topology.
    """
    graph = nx.DiGraph()

    if ref_xy is not None and num_map_elements is not None:
        _polylines_pre: NDArray[np.float64] = np.asarray(map_infos.get("all_polylines", np.empty((0, 7))))
        polylines_tf = transform_map_elements(_polylines_pre, ref_xy, ref_heading)
        map_infos = filter_map_elements_by_proximity(map_infos, polylines_tf, num_map_elements, map_range=map_range)

    # If filtering by proximity, the polylines get updated to only include the in-range points.
    all_polylines: NDArray[np.float64] = np.asarray(map_infos.get("all_polylines", np.empty((0, 7))))

    lane_ids: set = {lane["id"] for lane in map_infos.get("lane", [])}
    for lane in map_infos.get("lane", []):
        lane_id = lane["id"]
        graph.add_node(
            lane_id, type="lane", speed_limit_mph=lane.get("speed_limit_mph", 0.0), **_node_pos(lane, all_polylines)
        )
        for entry_id in lane.get("entry_lanes", []):
            if entry_id in lane_ids:
                graph.add_edge(entry_id, lane_id)
        for exit_id in lane.get("exit_lanes", []):
            if exit_id in lane_ids:
                graph.add_edge(lane_id, exit_id)

    # All other road elements
    for etype in _MAP_ELEMENT_TYPES:
        if etype == "lane":
            continue

        for element in map_infos.get(etype, []):
            graph.add_node(element["id"], type=etype, **_node_pos(element, all_polylines))

    return graph


def simplify_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Returns a simplified copy of the road topology graph for more stable encoding.

    Two simplifications are applied in order:

    1. **Remove isolated nodes** — nodes with no edges (degree zero) do not contribute to the graph's Laplacian
        structure but inflate the heat-trace baseline, causing NetLSD descriptors to vary with the number of
        non-connected map elements rather than road topology.
    2. **Keep the largest weakly-connected component** — small disconnected lane fragments at the scene boundary add
        noise without representing the main road network.

    Args:
        graph: Full road topology graph produced by `map_infos_to_graph`.

    Returns:
        Simplified directed graph containing only the largest connected lane subgraph.
    """
    simplified_graph = graph.copy()

    # Remove nodes with degree zero (no edges)
    simplified_graph.remove_nodes_from(list(nx.isolates(simplified_graph)))

    if simplified_graph.number_of_nodes() == 0:
        return simplified_graph

    largest_wcc = max(nx.weakly_connected_components(simplified_graph), key=len)
    return nx.DiGraph(simplified_graph.subgraph(largest_wcc))


def build_positioned_graph(  # noqa: PLR0913
    map_infos: dict[str, Any],
    *,
    ref_xy: NDArray[np.float64] | None = None,
    ref_heading: float = 0.0,
    num_map_elements: int | None = None,
    map_range: float = 100.0,
    simplify: bool = True,
) -> tuple[nx.DiGraph, dict[Any, NDArray[np.float64]]]:
    """Builds a map graph and extracts a matplotlib-compatible position dict from node attributes.

    Reads (x, y) from node attributes set by `map_infos_to_graph`. When ref_xy and num_map_elements are provided the
    graph is filtered using the reference-frame L∞ range box. Nodes without position attributes are placed via a spring
    layout fallback.

    Args:
        map_infos: Map information dictionary from a scenario pickle file.
        ref_xy: Reference position (x, y). When provided together with num_map_elements, triggers proximity filtering.
        num_map_elements: Maximum number of map elements to retain when ref_xy is set.
        ref_heading: Reference heading in radians, forwarded to `map_infos_to_graph`. Defaults to 0.0.
        map_range: L∞ half-width of the reference-frame range box in metres. Defaults to 100.0.
        simplify: If True, runs simplify_graph on the raw graph before returning. Defaults to True.

    Returns:
        The directed graph and a dict mapping node id to (x, y) position.
    """
    graph = map_infos_to_graph(
        map_infos, ref_xy=ref_xy, ref_heading=ref_heading, num_map_elements=num_map_elements, map_range=map_range
    )
    if simplify:
        graph = simplify_graph(graph)

    pos: dict[Any, NDArray[np.float64]] = {
        n: np.array([data["x"], data["y"]]) for n, data in graph.nodes(data=True) if "x" in data and "y" in data
    }

    # Nodes without a polyline-derived centroid (e.g. entry/exit lanes added by edge insertion that were not in
    # map_infos) get positions from a spring layout fallback.
    missing = [n for n in graph.nodes if n not in pos]
    if missing:
        if pos:
            # Anchor already-positioned nodes so the spring layout only places the missing ones without disturbing the
            # real-world coordinates of the rest.
            fallback = nx.spring_layout(graph, pos=dict(pos), fixed=list(pos.keys()), seed=0)
        else:
            # No nodes have position attributes at all; run a full unconstrained spring layout.
            fallback = nx.spring_layout(graph, seed=0)
        pos.update({n: fallback[n] for n in missing})

    return graph, pos


def visualize_scenario_graph(  # noqa: PLR0913
    filepath: Path,
    output_dir: Path,
    *,
    ego_centered: bool = False,
    num_map_elements: int = 100,
    map_range: float = 100.0,
    simplify: bool = True,
) -> None:
    """Renders a side-by-side raster map and road topology graph for a scenario and saves it as a PNG.

    The left panel shows the polyline-level raster map (lanes, road edges, etc.) and the right panel shows the
    corresponding NetworkX topology graph. Both panels use the same filtered/transformed map data so they represent
    the same view. When ego_centered is True the map is cropped to the ego-centric L∞ range box, reproducing the
    view encoded by NetLSD.

    Args:
        filepath: Path to the scenario pickle file.
        output_dir: Directory in which to save the PNG.
        ego_centered: If True, restrict both panels to the ego-centric L∞ range box. Defaults to False.
        num_map_elements: Maximum number of map elements to retain when ego_centered is True. Defaults to 100.
        map_range: L∞ half-width of the ego-centric range box in metres. Defaults to 100.0.
        simplify: If True, runs simplify_graph on the topology graph before visualizing. Defaults to True.
    """
    try:
        with filepath.open("rb") as f:
            scenario = pickle.load(f)  # nosec B301
    except (OSError, pickle.UnpicklingError):
        return

    scenario_id = scenario.get("scenario_id", filepath.stem)
    map_infos = scenario.get("map_infos", {})

    # Apply ego-centric filtering once so both panels see identical data.
    if ego_centered:
        sdc_track_index = scenario["sdc_track_index"]
        curr_time_index = scenario["current_time_index"]
        trajs = scenario["track_infos"]["trajs"][sdc_track_index, curr_time_index]
        ref_xy: NDArray[np.float64] = trajs[:2]
        ref_heading = float(trajs[6])
        polylines_raw: NDArray[np.float64] = np.asarray(map_infos.get("all_polylines", np.empty((0, 7))))
        polylines_tf = transform_map_elements(polylines_raw, ref_xy, ref_heading)
        map_infos = filter_map_elements_by_proximity(map_infos, polylines_tf, num_map_elements, map_range=map_range)

    graph, pos = build_positioned_graph(map_infos, simplify=simplify)

    if graph.number_of_nodes() == 0:
        return

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label)
        for label, color in _ELEMENT_COLORS.items()
    ]

    fig, (ax_raster, ax_graph) = plt.subplots(1, 2, figsize=(16, 8))

    # Left panel: raster map (polylines in map coordinates).
    _plot_map_raster(ax_raster, map_infos)
    ax_raster.set_aspect("equal")
    ax_raster.set_title("Raster map", fontsize=9)
    ax_raster.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)

    # Right panel: topology graph.
    node_colors = [_ELEMENT_COLORS.get(graph.nodes[n].get("type", ""), _DEFAULT_NODE_COLOR) for n in graph.nodes]
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax_graph, node_color=node_colors, node_size=20, hide_ticks=False)
    nx.draw_networkx_edges(graph, pos=pos, ax=ax_graph, edge_color="#CCCCCC", arrowsize=6, width=0.5, hide_ticks=False)
    ax_graph.set_aspect("equal")
    ax_graph.set_title("Topology graph", fontsize=9)
    ax_graph.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)

    fig.suptitle(scenario_id, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(output_dir / f"{scenario_id}.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
