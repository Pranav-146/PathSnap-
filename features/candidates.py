"""
Spatial candidate generation and geometric projection onto road network arcs.

Provides:
  - candidates_for_trip: for each GPS observation, retrieves nearby candidate road arcs
    within radius R (default 50 m) and keeps at most k nearest.
  - Computes perpendicular distance, projection ratio, along-arc points, and headings.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

import config


class CandidateGenerator:
    """Fast spatial index for querying road network arcs near GPS observations."""

    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame):
        self.nodes = nodes
        self.arcs = arcs

        # Node coordinates in UTM metres (EPSG:32632)
        node_x = nodes["x"].to_numpy(dtype=np.float64)
        node_y = nodes["y"].to_numpy(dtype=np.float64)

        u = arcs["u"].to_numpy(dtype=np.int64)
        v = arcs["v"].to_numpy(dtype=np.int64)

        self.ax = node_x[u]
        self.ay = node_y[u]
        self.bx = node_x[v]
        self.by = node_y[v]

        self.dx = self.bx - self.ax
        self.dy = self.by - self.ay
        self.len_sq = self.dx ** 2 + self.dy ** 2
        self.len_sq = np.where(self.len_sq == 0, 1e-9, self.len_sq)
        self.arc_lengths = np.sqrt(self.len_sq)

        # Arc midpoints for fast KD-Tree spatial filtering
        mid_x = (self.ax + self.bx) * 0.5
        mid_y = (self.ay + self.by) * 0.5
        self.kdtree = KDTree(np.column_stack([mid_x, mid_y]))

        # Arc headings in radians [-pi, pi]
        self.arc_headings = np.arctan2(self.dy, self.dx)

    def get_candidates_for_point(
        self,
        px: float,
        py: float,
        heading_gps: Optional[float] = None,
        radius: float = config.RADIUS_M,
        k: int = config.K_MAX,
    ) -> List[Dict]:
        """
        Find candidate arcs within search radius from GPS point (px, py).
        Returns list of candidate dicts sorted by perpendicular distance.
        """
        # Search radius plus half max road segment length to ensure coverage
        search_r = radius + 150.0
        candidate_indices = self.kdtree.query_ball_point([px, py], r=search_r)

        if not candidate_indices:
            # Fallback to nearest arc if search ball is empty
            _, nearest_idx = self.kdtree.query([px, py], k=1)
            candidate_indices = [nearest_idx]

        cand_arr = np.array(candidate_indices, dtype=np.int64)

        ax = self.ax[cand_arr]
        ay = self.ay[cand_arr]
        dx = self.dx[cand_arr]
        dy = self.dy[cand_arr]
        len_sq = self.len_sq[cand_arr]

        # Orthogonal projection factor t: clamped to [0, 1]
        t = ((px - ax) * dx + (py - ay) * dy) / len_sq
        t_clamped = np.clip(t, 0.0, 1.0)

        # Projected point coordinates on segment
        proj_x = ax + t_clamped * dx
        proj_y = ay + t_clamped * dy

        # Perpendicular distance in metres
        dists = np.hypot(px - proj_x, py - proj_y)

        # Filter by actual radius
        within_r_mask = dists <= radius
        if not np.any(within_r_mask):
            # If none within radius, keep the single closest candidate
            best_idx = np.argmin(dists)
            valid_indices = np.array([best_idx])
        else:
            valid_indices = np.where(within_r_mask)[0]

        # Sort by distance and take top k
        sorted_order = valid_indices[np.argsort(dists[valid_indices])][:k]

        candidates = []
        for rank, idx in enumerate(sorted_order):
            arc_id = int(cand_arr[idx])
            d = float(dists[idx])
            tau = float(t_clamped[idx])
            arc_heading = float(self.arc_headings[arc_id])

            h_diff = 0.0
            if heading_gps is not None:
                # Heading difference in [-pi, pi]
                diff = arc_heading - heading_gps
                h_diff = (diff + np.pi) % (2 * np.pi) - np.pi

            candidates.append({
                "arc_id": arc_id,
                "dist": d,
                "dist_rank": rank,
                "proj_frac": tau,
                "proj_x": float(proj_x[idx]),
                "proj_y": float(proj_y[idx]),
                "arc_len": float(self.arc_lengths[arc_id]),
                "heading_diff": float(h_diff),
            })

        return candidates


def candidates_for_trip(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    radius: float = config.RADIUS_M,
    k: int = config.K_MAX,
) -> List[List[Dict]]:
    """
    Generate candidate arcs for every GPS point in a trip.

    Returns:
        List of candidate lists, length = len(track).
    """
    generator = CandidateGenerator(nodes, arcs)

    track_x = track["x"].to_numpy(dtype=np.float64)
    track_y = track["y"].to_numpy(dtype=np.float64)
    n_pts = len(track)

    # Estimate GPS heading from successive points
    headings = np.zeros(n_pts, dtype=np.float64)
    if n_pts > 1:
        dx = np.diff(track_x)
        dy = np.diff(track_y)
        h = np.arctan2(dy, dx)
        headings[:-1] = h
        headings[-1] = h[-1]

    candidates_by_point = []
    for i in range(n_pts):
        cands = generator.get_candidates_for_point(
            track_x[i],
            track_y[i],
            heading_gps=headings[i],
            radius=radius,
            k=k,
        )
        candidates_by_point.append(cands)

    return candidates_by_point
