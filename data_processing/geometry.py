"""Small numpy geometry helpers. All coordinates are projected metres (see load.project)."""
import numpy as np


def arc_endpoints(trip):
    """Return (ax, ay, bx, by) arrays, one entry per arc: from-node and to-node coordinates.

    Arcs are straight segments between their two nodes, so a point-to-arc distance
    is a point-to-segment distance.
    """
    nx, ny = trip["nodes"].x.to_numpy(), trip["nodes"].y.to_numpy()
    u, v = trip["arcs"].u.to_numpy(), trip["arcs"].v.to_numpy()
    return nx[u], ny[u], nx[v], ny[v]


def point_seg(px, py, ax, ay, bx, by):
    """Distance from points to segments, plus the projection fraction (0 = at A, 1 = at B).

    Works element-wise on equal-length (or broadcastable) arrays. Zero-length segments
    (self-loop arcs) give frac = 0 and the distance to the node.
    """
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(len2 > 0, ((px - ax) * dx + (py - ay) * dy) / len2, 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    cx, cy = ax + frac * dx, ay + frac * dy
    return np.hypot(px - cx, py - cy), frac


def bearing_deg(dx, dy):
    """Compass bearing in degrees (0 = north, clockwise) of a vector (dx = east, dy = north)."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)
