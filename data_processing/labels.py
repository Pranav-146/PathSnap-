"""Derive a per-GPS-point reference arc from the trip's ordered .route file.

The dataset gives no label per GPS point, only the ordered list of arcs the vehicle used.
Method: walk along the route with a pointer that only moves forward. For each GPS point,
look at the next WINDOW route arcs (starting at the pointer) and take the nearest one.

Why a forward-only window rather than "nearest route arc overall":
  * the route can pass the same place twice (loops, U-turns), and a global nearest arc
    could jump backwards in time;
  * restricting to a window keeps labels monotonic and consistent with travel order.

Limitations (to state in the report):
  * reference routes are not independently verified (14 route breaks found in the data);
  * at junctions / overlapping parallel arcs a GPS point near two route arcs may take the
    wrong one; points far from the route (dist_to_ref large) are flagged as unreliable;
  * if the pointer lags (e.g. long GPS gaps skipping > WINDOW arcs) labels can be off.
"""
import numpy as np
import pandas as pd

from data_processing.geometry import arc_endpoints, point_seg

WINDOW = 30           # how many upcoming route arcs a point may be matched to
UNRELIABLE_M = 100.0  # dist_to_ref above this => flagged unreliable


def derive_labels(trip, window=WINDOW):
    """Return DataFrame[trip_id, obs_idx, ref_arc, route_pos, dist_to_ref, unreliable]."""
    route = trip["route"]
    ax, ay, bx, by = arc_endpoints(trip)
    rax, ray, rbx, rby = ax[route], ay[route], bx[route], by[route]

    px, py = trip["track"].x.to_numpy(), trip["track"].y.to_numpy()
    n = len(px)
    pos = np.zeros(n, dtype=np.int64)
    dist = np.zeros(n)

    p = 0
    for i in range(n):
        hi = min(p + window, len(route))
        d, _ = point_seg(px[i], py[i], rax[p:hi], ray[p:hi], rbx[p:hi], rby[p:hi])
        j = int(np.argmin(d))
        p += j                    # pointer never moves backwards
        pos[i], dist[i] = p, d[j]

    return pd.DataFrame({
        "trip_id": trip["trip_id"],
        "obs_idx": np.arange(n),
        "ref_arc": route[pos],
        "route_pos": pos,
        "dist_to_ref": dist,
        "unreliable": dist > UNRELIABLE_M,
    })
