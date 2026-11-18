"""Raw-data loading, validation and coordinate projection.

File formats (whitespace separated, no header):
  .track  lon lat t          one GPS observation per row (~1 Hz)
  .nodes  lon lat            row index = node_id
  .arcs   from_node to_node  row index = arc_id
  .route  arc_id             ground-truth arcs in travel order
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

import config


@lru_cache(maxsize=256)
def _transformer(lon0, lat0):
    # Local azimuthal-equidistant projection centred on the trip: distances from the
    # centre are true metres, and the error is negligible over a trip's extent.
    # (The dataset spans several countries, so one global UTM zone would be wrong.)
    crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def project(lon, lat, center):
    """lon/lat degrees -> local metres (x east, y north) around center=(lon0, lat0)."""
    lon0, lat0 = round(float(center[0]), 4), round(float(center[1]), 4)
    x, y = _transformer(lon0, lat0).transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def _read(path, names, dtype):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def list_trips(split_dir):
    return sorted(p for p in Path(split_dir).iterdir() if p.is_dir())


def load_trip(trip_dir, with_route=True):
    """Load one trip folder. with_route=False skips ground truth (used by the API)."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _read(trip_dir / f"{tid}.track", ["lon", "lat", "t"], float)
    nodes = _read(trip_dir / f"{tid}.nodes", ["lon", "lat"], float)
    arcs = _read(trip_dir / f"{tid}.arcs", ["u", "v"], np.int64)

    # Centre of the local projection comes from the GPS track only (no ground truth used).
    center = (float(track.lon.mean()), float(track.lat.mean()))
    track["x"], track["y"] = project(track.lon, track.lat, center)
    nodes["x"], nodes["y"] = project(nodes.lon, nodes.lat, center)

    trip = {"trip_id": tid, "dir": trip_dir, "track": track, "nodes": nodes, "arcs": arcs,
            "center": center}
    route_path = trip_dir / f"{tid}.route"
    if with_route and route_path.exists():
        trip["route"] = _read(route_path, ["arc"], np.int64)["arc"].to_numpy()
    return trip


def validate_trip(trip):
    """Return a dict of data-quality counts for one trip."""
    tr, nd, ar = trip["track"], trip["nodes"], trip["arcs"]
    dt = np.diff(tr.t.to_numpy())
    q = {
        "trip_id": trip["trip_id"],
        "n_points": len(tr),
        "n_nodes": len(nd),
        "n_arcs": len(ar),
        "track_nan": int(tr[["lon", "lat", "t"]].isna().sum().sum()),
        "nodes_nan": int(nd[["lon", "lat"]].isna().sum().sum()),
        "bad_coords": int((~tr.lon.between(-180, 180) | ~tr.lat.between(-90, 90)).sum()),
        "non_increasing_t": int((dt <= 0).sum()),
        "gaps_gt_1s": int((dt > 1).sum()),
        "max_gap_s": float(dt.max()) if len(dt) else 0.0,
        "pct_dt_eq_1": float((dt == 1).mean() * 100) if len(dt) else 0.0,
        "arc_bad_node": int(((ar.u < 0) | (ar.u >= len(nd)) | (ar.v < 0) | (ar.v >= len(nd))).sum()),
        "self_loops": int((ar.u == ar.v).sum()),
    }
    if "route" in trip:
        r = trip["route"]
        q["route_len"] = len(r)
        q["route_bad_arc"] = int(((r < 0) | (r >= len(ar))).sum())
        # consecutive route arcs should connect: to_node of arc i == from_node of arc i+1
        if len(r) > 1 and q["route_bad_arc"] == 0:
            q["route_breaks"] = int((ar.v.to_numpy()[r[:-1]] != ar.u.to_numpy()[r[1:]]).sum())
    return q


if __name__ == "__main__":
    import time
    rows = []
    t0 = time.time()
    for split, d in [("train", config.TRAIN_DIR), ("student_test", config.STUDTEST_DIR)]:
        for p in list_trips(d):
            q = validate_trip(load_trip(p))
            q["split"] = split
            rows.append(q)
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "data_quality.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} trips OK in {time.time()-t0:.1f}s -> {out}")
    cols = ["track_nan", "nodes_nan", "bad_coords", "non_increasing_t", "arc_bad_node",
            "self_loops", "route_bad_arc", "route_breaks"]
    print("Issue totals:\n", df[cols].sum().to_string())
    print("\nPer-split summary:\n", df.groupby("split")[["n_points", "n_arcs", "route_len", "max_gap_s", "pct_dt_eq_1"]].describe().T.round(1).to_string())
