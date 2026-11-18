"""Interactive Folium map of one trip: raw GPS, road network, reference route.

Usage: python -m visualisations.eda_map [trip_id] [--split studtest|train]
Default: a mid-sized student_test trip. Output: results/maps/eda_<trip>.html
"""
import argparse

import folium
import numpy as np

import config
from data_processing.geometry import arc_endpoints
from data_processing.load import list_trips, load_trip

MAX_ROAD_ARCS = 12000


def build_map(trip, out_path):
    tr, nd, ar = trip["track"], trip["nodes"], trip["arcs"]
    lon_u, lat_u = nd.lon.to_numpy()[ar.u], nd.lat.to_numpy()[ar.u]
    lon_v, lat_v = nd.lon.to_numpy()[ar.v], nd.lat.to_numpy()[ar.v]

    # road arcs inside the GPS bounding box (+ margin), so the map stays light
    m = 0.003
    inside = ((np.maximum(lon_u, lon_v) >= tr.lon.min() - m) & (np.minimum(lon_u, lon_v) <= tr.lon.max() + m) &
              (np.maximum(lat_u, lat_v) >= tr.lat.min() - m) & (np.minimum(lat_u, lat_v) <= tr.lat.max() + m))
    idx = np.flatnonzero(inside)[:MAX_ROAD_ARCS]

    fmap = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    fmap.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])

    roads = folium.FeatureGroup(name=f"Road network ({len(idx)} arcs)", show=True)
    folium.PolyLine([[(lat_u[i], lon_u[i]), (lat_v[i], lon_v[i])] for i in idx],
                    color="#888888", weight=2, opacity=0.7).add_to(roads)
    roads.add_to(fmap)

    route = trip["route"]
    ref = folium.FeatureGroup(name=f"Reference route ({len(route)} arcs)", show=True)
    folium.PolyLine([[(lat_u[a], lon_u[a]), (lat_v[a], lon_v[a])] for a in route],
                    color="#1a8f3c", weight=5, opacity=0.9).add_to(ref)
    ref.add_to(fmap)

    gps = folium.FeatureGroup(name=f"Raw GPS points ({len(tr)})", show=True)
    for i in range(0, len(tr), max(1, len(tr) // 1500)):       # thin to <=1500 markers
        r = tr.iloc[i]
        folium.CircleMarker((r.lat, r.lon), radius=2, color="#1f4fd8", fill=True, fill_opacity=0.9,
                            tooltip=f"obs {i}  t={r.t:.0f}s").add_to(gps)
    gps.add_to(fmap)
    folium.Marker((tr.lat.iloc[0], tr.lon.iloc[0]), tooltip="start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker((tr.lat.iloc[-1], tr.lon.iloc[-1]), tooltip="end", icon=folium.Icon(color="red")).add_to(fmap)

    legend = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 12px;
    border:1px solid #888;border-radius:6px;font-size:13px"><b>Trip %s</b><br>
    <span style="color:#1f4fd8">&#9679;</span> raw GPS points<br>
    <span style="color:#888">&#9473;</span> road network<br>
    <span style="color:#1a8f3c">&#9473;</span> reference route</div>""" % trip["trip_id"]
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(str(out_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trip_id", nargs="?")
    ap.add_argument("--split", default="studtest")
    a = ap.parse_args()
    folder = config.TRAIN_DIR if a.split == "train" else config.STUDTEST_DIR
    if a.trip_id is None:                      # pick a mid-sized, light trip
        best = None
        for p in list_trips(folder):
            n_pts = sum(1 for _ in open(p / f"{p.name}.track"))
            n_arcs = sum(1 for _ in open(p / f"{p.name}.arcs"))
            if 400 <= n_pts <= 1500 and n_arcs <= 15000 and (best is None or n_pts > best[1]):
                best = (p.name, n_pts)
        a.trip_id = best[0]
    out = config.RESULTS_DIR / "maps"
    out.mkdir(parents=True, exist_ok=True)
    trip = load_trip(folder / a.trip_id)
    path = out / f"eda_{a.trip_id}.html"
    build_map(trip, path)
    print("wrote", path)
