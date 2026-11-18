"""Candidate-arc generation and feature engineering for one trip.

candidates_for_trip(trip) needs NO ground truth, so the API can call it at inference time.
Labels are attached separately (features/build_features.py) from data_processing/labels.py.

Pipeline per trip:
  1. Bounding box of every arc -> STRtree (spatial index; some trips have ~300k arcs).
  2. For each GPS point, query arcs whose box intersects the point +/- radius.
  3. Exact point-to-segment distance in metres; keep arcs within `radius`.
  4. Keep the K nearest per point (candidate set).
  5. Add per-candidate features.
"""
import numpy as np
import pandas as pd
import shapely

import config
from data_processing.geometry import (angle_diff_deg, arc_endpoints, bearing_deg, point_seg)

MAX_SPEED = 60.0  # m/s cap on the GPS speed feature (guards against huge-gap artefacts)


def raw_candidates(trip, radius=config.RADIUS_M, k=config.K_MAX):
    """Nearest-K arcs within `radius` metres for every GPS point.

    Returns DataFrame[obs_idx, arc_id, dist, proj_frac, dist_rank, n_cands, dist_minus_min].
    """
    ax, ay, bx, by = arc_endpoints(trip)
    px, py = trip["track"].x.to_numpy(), trip["track"].y.to_numpy()

    arc_boxes = shapely.box(np.minimum(ax, bx), np.minimum(ay, by),
                            np.maximum(ax, bx), np.maximum(ay, by))
    tree = shapely.STRtree(arc_boxes)
    pt_boxes = shapely.box(px - radius, py - radius, px + radius, py + radius)
    pi, ai = tree.query(pt_boxes)          # (point index, arc index) pairs, boxes intersect

    d, frac = point_seg(px[pi], py[pi], ax[ai], ay[ai], bx[ai], by[ai])
    keep = d <= radius
    df = pd.DataFrame({"obs_idx": pi[keep], "arc_id": ai[keep], "dist": d[keep], "proj_frac": frac[keep]})
    return restrict(df, radius, k)


def restrict(df, radius, k):
    """Re-cut an existing candidate table to a smaller radius / K (recomputes the rank columns)."""
    df = df[df.dist <= radius].sort_values(["obs_idx", "dist", "arc_id"]).copy()
    df["dist_rank"] = df.groupby("obs_idx").cumcount()
    df = df[df.dist_rank < k].copy()
    g = df.groupby("obs_idx")["dist"]
    df["n_cands"] = g.transform("size")
    df["dist_minus_min"] = df["dist"] - g.transform("min")
    return df.reset_index(drop=True)


def add_features(trip, cand):
    """Attach point-level and candidate-level features to a raw candidate table."""
    tr = trip["track"]
    px, py, t = tr.x.to_numpy(), tr.y.to_numpy(), tr.t.to_numpy()
    n = len(tr)

    # --- point-level: GPS heading, speed and time gap, from the neighbouring points ---
    prev_i = np.clip(np.arange(n) - 1, 0, n - 1)
    next_i = np.clip(np.arange(n) + 1, 0, n - 1)
    dx, dy = px[next_i] - px[prev_i], py[next_i] - py[prev_i]
    dt_span = np.maximum(t[next_i] - t[prev_i], 1e-6)
    pts = pd.DataFrame({
        "obs_idx": np.arange(n),
        "gps_heading": bearing_deg(dx, dy),
        "speed": np.minimum(np.hypot(dx, dy) / dt_span, MAX_SPEED),
        "dt_prev": np.r_[0.0, np.diff(t)],
    })

    df = cand.merge(pts, on="obs_idx", how="left")

    # --- arc-level: length, bearing, endpoints ---
    ax, ay, bx, by = arc_endpoints(trip)
    a = df.arc_id.to_numpy()
    df["arc_len"] = np.hypot(bx[a] - ax[a], by[a] - ay[a])
    df["arc_bearing"] = bearing_deg(bx[a] - ax[a], by[a] - ay[a])
    df["heading_diff"] = angle_diff_deg(df.gps_heading.to_numpy(), df.arc_bearing.to_numpy())
    df["u"] = trip["arcs"].u.to_numpy()[a]
    df["v"] = trip["arcs"].v.to_numpy()[a]

    # --- connectivity to the previous point's nearest candidate ---
    nearest = df[df.dist_rank == 0].set_index("obs_idx")[["arc_id", "u", "v"]]
    prev = nearest.reindex(np.arange(n)).shift(1)          # NaN where unavailable
    prev.columns = ["p_arc", "p_u", "p_v"]
    df = df.merge(prev, left_on="obs_idx", right_index=True, how="left")
    shares_node = (df.u == df.p_u) | (df.u == df.p_v) | (df.v == df.p_u) | (df.v == df.p_v)
    df["connected_to_prev_nearest"] = ((df.arc_id == df.p_arc) | shares_node).astype(np.int8)
    df = df.drop(columns=["p_arc", "p_u", "p_v"])

    # --- identifiers (keep first so the parquet reads naturally) ---
    df.insert(0, "trip_id", trip["trip_id"])
    df.insert(2, "t", t[df.obs_idx.to_numpy()])
    df.insert(3, "lon", tr.lon.to_numpy()[df.obs_idx.to_numpy()])
    df.insert(4, "lat", tr.lat.to_numpy()[df.obs_idx.to_numpy()])
    df.insert(5, "x", px[df.obs_idx.to_numpy()])
    df.insert(6, "y", py[df.obs_idx.to_numpy()])
    return df


def candidates_for_trip(trip, radius=config.RADIUS_M, k=config.K_MAX):
    """Full candidate table with features for one trip. Uses no ground-truth information."""
    return add_features(trip, raw_candidates(trip, radius, k))
