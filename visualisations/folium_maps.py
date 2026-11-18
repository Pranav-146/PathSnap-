"""
Folium interactive map visualisations for PathSnap.

Generates multi-layer interactive maps:
  1. Raw GPS trajectory points (blue circle markers with timestamps and coordinates)
  2. Ground truth route (green polyline following actual road geometry)
  3. HMM Viterbi reconstructed route (red/purple polyline following road network)
  4. Top-1 Baseline route (orange dashed polyline)
  5. Interactive LayerControl toggle to compare reconstructions side-by-side.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import folium
from folium import FeatureGroup, LayerControl
import numpy as np

import config
from hmm.graph import RoadGraph


def create_trip_map(
    trip_data: Dict[str, Any],
    matched_arcs: Optional[List[int]] = None,
    baseline_arcs: Optional[List[int]] = None,
    road_graph: Optional[RoadGraph] = None,
    save_html_path: Optional[Union[str, Path]] = None,
) -> folium.Map:
    """
    Generate an interactive Folium map for a trip.

    Args:
        trip_data: dict from load_trip (track, nodes, arcs, optional route)
        matched_arcs: list of HMM predicted arc_ids
        baseline_arcs: optional list of baseline arc_ids
        road_graph: optional RoadGraph instance (created if not passed)
        save_html_path: optional filepath to save the rendered .html map

    Returns:
        folium.Map instance
    """
    track = trip_data["track"]
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]

    if road_graph is None:
        road_graph = RoadGraph(nodes, arcs)

    # Compute center of the map
    mean_lat = float(track["lat"].mean())
    mean_lon = float(track["lon"].mean())

    fmap = folium.Map(
        location=[mean_lat, mean_lon],
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Layer 1: Raw GPS Points
    gps_group = FeatureGroup(name="Raw GPS Observations (Blue)", show=True)
    # Downsample markers if track is very dense for smooth rendering in browser
    step = 1 if len(track) <= 500 else max(1, len(track) // 500)
    for i in range(0, len(track), step):
        row = track.iloc[i]
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=3,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.8,
            popup=f"Obs {i}<br>t: {row['t']:.1f}s<br>Lat: {row['lat']:.5f}<br>Lon: {row['lon']:.5f}",
        ).add_to(gps_group)
    gps_group.add_to(fmap)

    # Helper function to convert list of arc_ids to lat/lon line segments
    def arcs_to_latlon_coords(arc_list: List[int]) -> List[List[float]]:
        coords = []
        for a in arc_list:
            if a < 0 or a >= road_graph.num_arcs:
                continue
            (lat_u, lon_u), (lat_v, lon_v) = road_graph.get_arc_endpoints_latlon(a)
            if not coords:
                coords.append([lat_u, lon_u])
            coords.append([lat_v, lon_v])
        return coords

    # Layer 2: Ground Truth Route (if available)
    if "route" in trip_data and trip_data["route"] is not None:
        gt_route = trip_data["route"]
        gt_coords = arcs_to_latlon_coords(gt_route)
        if gt_coords:
            gt_group = FeatureGroup(name="Ground Truth Route (Green)", show=True)
            folium.PolyLine(
                gt_coords,
                color="#2ca02c",
                weight=5,
                opacity=0.85,
                tooltip=f"Ground Truth: {len(gt_route)} road arcs",
            ).add_to(gt_group)
            gt_group.add_to(fmap)

    # Layer 3: HMM Viterbi Reconstructed Route
    if matched_arcs is not None:
        # Deduplicate consecutive arcs
        unique_hmm_arcs = []
        for a in matched_arcs:
            if not unique_hmm_arcs or unique_hmm_arcs[-1] != a:
                unique_hmm_arcs.append(a)

        hmm_coords = arcs_to_latlon_coords(unique_hmm_arcs)
        if hmm_coords:
            hmm_group = FeatureGroup(name="HMM Viterbi Reconstructed (Purple)", show=True)
            folium.PolyLine(
                hmm_coords,
                color="#9467bd",
                weight=4,
                opacity=0.9,
                tooltip=f"HMM Predicted: {len(unique_hmm_arcs)} arcs",
            ).add_to(hmm_group)
            hmm_group.add_to(fmap)

    # Layer 4: Top-1 Nearest Candidate Baseline
    if baseline_arcs is not None:
        unique_base_arcs = []
        for a in baseline_arcs:
            if not unique_base_arcs or unique_base_arcs[-1] != a:
                unique_base_arcs.append(a)

        base_coords = arcs_to_latlon_coords(unique_base_arcs)
        if base_coords:
            base_group = FeatureGroup(name="Baseline Top-1 Nearest (Orange dashed)", show=False)
            folium.PolyLine(
                base_coords,
                color="#ff7f0e",
                weight=3,
                dash_array="6, 6",
                opacity=0.8,
                tooltip=f"Baseline: {len(unique_base_arcs)} arcs",
            ).add_to(base_group)
            base_group.add_to(fmap)

    # Add interactive layer controls
    LayerControl(collapsed=False).add_to(fmap)

    if save_html_path is not None:
        out_p = Path(save_html_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(out_p))
        print(f"Map successfully saved to: {out_p}")

    return fmap


if __name__ == "__main__":
    from data_processing.load import load_trip
    from hmm.pipeline import MapMatchingPipeline

    trip_path = Path("Student folder/train/00000001")
    trip = load_trip(trip_path)
    pipeline = MapMatchingPipeline()
    res = pipeline.match_trip(trip)

    out_file = config.RESULTS_DIR / "trip_00000001_map.html"
    create_trip_map(
        trip,
        matched_arcs=res["matched_arcs"],
        baseline_arcs=res["baseline_arcs"],
        road_graph=res["road_graph"],
        save_html_path=out_file,
    )
