"""Exploratory data analysis. Run after features.build_features (needs labels_*.parquet).

Usage: python -m data_processing.eda
Outputs (results/eda/): inventory.csv, summary tables, PNG plots.
Partitions: inventory/gap/network analyses cover train (80) + student_test (10);
the GPS-to-route distance analysis uses the same two partitions. Evaluator data is never used.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import config
from data_processing.geometry import arc_endpoints
from data_processing.load import list_trips, load_trip

OUT = config.RESULTS_DIR / "eda"
SPLITS = {"train": config.TRAIN_DIR, "student_test": config.STUDTEST_DIR}


def network_stats(trip):
    """Weakly connected components and isolated nodes of the road network."""
    n, u, v = len(trip["nodes"]), trip["arcs"].u.to_numpy(), trip["arcs"].v.to_numpy()
    g = coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    ncomp, comp = connected_components(g, directed=False)
    sizes = np.bincount(comp)
    deg = np.bincount(np.r_[u, v], minlength=n)
    return {"components": int(ncomp), "largest_comp_pct": round(100 * sizes.max() / n, 2),
            "isolated_nodes": int((deg == 0).sum())}


def build_inventory():
    rows, gaps = [], []
    for split, folder in SPLITS.items():
        for p in list_trips(folder):
            t = load_trip(p)
            tr = t["track"]
            dt = np.diff(tr.t.to_numpy())
            gaps.append(pd.DataFrame({"split": split, "trip_id": t["trip_id"], "dt": dt}))
            r = {"split": split, "trip_id": t["trip_id"], "n_points": len(tr),
                 "duration_s": float(tr.t.iloc[-1] - tr.t.iloc[0]),
                 "n_nodes": len(t["nodes"]), "n_arcs": len(t["arcs"]), "route_len": len(t["route"]),
                 "center_lon": round(t["center"][0], 3), "center_lat": round(t["center"][1], 3),
                 "extent_w_m": round(float(np.ptp(tr.x))), "extent_h_m": round(float(np.ptp(tr.y))),
                 "max_gap_s": float(dt.max()), "pct_dt_eq_1": round(float((dt == 1).mean() * 100), 2),
                 "n_files": len(list(p.iterdir())),
                 "size_kb": round(sum(f.stat().st_size for f in p.iterdir()) / 1024)}
            r.update(network_stats(t))
            rows.append(r)
    inv = pd.DataFrame(rows)
    return inv, pd.concat(gaps, ignore_index=True)


def plots(inv, gaps, labels):
    # 1) sampling-interval histogram
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    dt = gaps.dt
    ax[0].hist(dt[dt <= 10], bins=np.arange(0.5, 10.6, 1), color="#2a6fbb")
    ax[0].set_yscale("log"); ax[0].set_xlabel("time step between GPS points (s), <=10 s")
    ax[0].set_ylabel("count (log)"); ax[0].set_title(f"Sampling interval: {(dt == 1).mean()*100:.1f}% of steps are exactly 1 s")
    ax[1].hist(np.log10(dt[dt > 1]), bins=40, color="#c2561f")
    ax[1].set_xlabel("log10(gap in s), gaps > 1 s"); ax[1].set_title("Gaps longer than 1 s (train + student_test)")
    fig.tight_layout(); fig.savefig(OUT / "time_gaps.png", dpi=130); plt.close(fig)

    # 2) GPS-to-reference-route distance (GPS noise level; justifies the 50 m candidate radius)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    d = labels.dist_to_ref
    ax.hist(d[d <= 150], bins=75, color="#2a8a4a")
    ax.axvline(config.RADIUS_M, color="k", ls="--", label=f"candidate radius {config.RADIUS_M} m")
    ax.axvline(100, color="r", ls=":", label="unreliable-label threshold 100 m")
    ax.set_xlabel("distance GPS point -> reference route arc (m), <=150 m"); ax.set_ylabel("GPS points")
    ax.set_title(f"median {d.median():.1f} m, p95 {d.quantile(.95):.0f} m (train + student_test)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "gps_to_route_dist.png", dpi=130); plt.close(fig)

    # 3) trip size distributions
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, col, lab in zip(ax, ["n_points", "n_arcs", "route_len"], ["GPS points", "road-network arcs", "reference route arcs"]):
        for sp, c in [("train", "#2a6fbb"), ("student_test", "#c2561f")]:
            a.hist(np.log10(inv[inv.split == sp][col]), bins=15, alpha=.7, label=sp, color=c)
        a.set_xlabel(f"log10({lab})"); a.set_ylabel("trips")
    ax[0].legend(); fig.suptitle("Per-trip sizes"); fig.tight_layout(); fig.savefig(OUT / "trip_sizes.png", dpi=130); plt.close(fig)


def trajectory_plots(trip_ids):
    """GPS track over the reference route for a few trips (projected metres)."""
    fig, axes = plt.subplots(1, len(trip_ids), figsize=(4.6 * len(trip_ids), 4.4))
    for ax, tid in zip(np.atleast_1d(axes), trip_ids):
        folder = config.TRAIN_DIR if (config.TRAIN_DIR / tid).exists() else config.STUDTEST_DIR
        t = load_trip(folder / tid)
        ax_, ay_, bx_, by_ = arc_endpoints(t)
        for a in t["route"]:
            ax.plot([ax_[a], bx_[a]], [ay_[a], by_[a]], color="#2a8a4a", lw=2)
        ax.scatter(t["track"].x, t["track"].y, s=2, color="#2a6fbb")
        ax.set_aspect("equal"); ax.set_title(f"trip {tid}: {len(t['track'])} pts"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.suptitle("GPS track (blue) vs reference route (green)"); fig.tight_layout()
    fig.savefig(OUT / "trajectories.png", dpi=130); plt.close(fig)


def network_plot(trip_id, split_dir):
    """Static road-network view: all arcs near the track (grey), reference route (green), GPS (blue)."""
    from matplotlib.collections import LineCollection
    t = load_trip(split_dir / trip_id)
    ax_, ay_, bx_, by_ = arc_endpoints(t)
    px, py = t["track"].x.to_numpy(), t["track"].y.to_numpy()
    m = 300.0                                         # margin around the GPS bounding box, metres
    near = ((np.maximum(ax_, bx_) >= px.min() - m) & (np.minimum(ax_, bx_) <= px.max() + m) &
            (np.maximum(ay_, by_) >= py.min() - m) & (np.minimum(ay_, by_) <= py.max() + m))
    segs = np.stack([np.column_stack([ax_[near], ay_[near]]), np.column_stack([bx_[near], by_[near]])], axis=1)
    r = t["route"]
    rsegs = np.stack([np.column_stack([ax_[r], ay_[r]]), np.column_stack([bx_[r], by_[r]])], axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.add_collection(LineCollection(segs, colors="#9a9a9a", linewidths=0.6, label=f"road network ({int(near.sum())} arcs)"))
    ax.add_collection(LineCollection(rsegs, colors="#1a8f3c", linewidths=1.3, zorder=4, label=f"reference route ({len(r)} arcs)"))
    ax.scatter(px, py, s=14, color="#1f4fd8", alpha=0.45, label=f"raw GPS ({len(px)} points)", zorder=3)
    ax.set_xlim(px.min() - m, px.max() + m); ax.set_ylim(py.min() - m, py.max() + m)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(loc="best")
    ax.set_title(f"Road network around trip {trip_id} (student_test)")
    fig.tight_layout(); fig.savefig(OUT / "road_network.png", dpi=130); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv, gaps = build_inventory()
    inv.to_csv(OUT / "inventory.csv", index=False)

    labels = pd.concat([pd.read_parquet(config.RESULTS_DIR / "labels_train.parquet"),
                        pd.read_parquet(config.RESULTS_DIR / "labels_studtest.parquet")])
    lab_trip = labels.groupby("trip_id").unreliable.mean().mul(100).round(1).rename("unreliable_label_pct")
    inv = inv.merge(lab_trip, left_on="trip_id", right_index=True, how="left")
    inv.to_csv(OUT / "inventory.csv", index=False)

    summ = inv.groupby("split").agg(trips=("trip_id", "size"), points_total=("n_points", "sum"),
                                    points_median=("n_points", "median"), points_max=("n_points", "max"),
                                    arcs_median=("n_arcs", "median"), arcs_max=("n_arcs", "max"),
                                    route_len_median=("route_len", "median"),
                                    trips_with_gap_gt_60s=("max_gap_s", lambda s: int((s > 60).sum())),
                                    trips_multi_component=("components", lambda s: int((s > 1).sum())),
                                    isolated_nodes_total=("isolated_nodes", "sum")).T
    summ.to_csv(OUT / "summary_by_split.csv")
    print(summ.to_string())
    print(f"\nsteps exactly 1 s apart: {(gaps.dt == 1).mean()*100:.1f}%  |  steps > 1 s: {(gaps.dt > 1).mean()*100:.1f}%")
    print("network: multi-component trips:", int((inv.components > 1).sum()), "of", len(inv),
          "| median largest-component share: %.1f%%" % inv.largest_comp_pct.median())
    print("trips with >20% unreliable labels:", inv[inv.unreliable_label_pct > 20][["trip_id", "split", "unreliable_label_pct"]].to_dict("records"))

    plots(inv, gaps, labels)
    pick = inv[(inv.split == "train")].sort_values("n_points").trip_id.iloc[[len(inv[inv.split == "train"]) // 4, len(inv[inv.split == "train"]) // 2, -30]].tolist()
    trajectory_plots(pick)
    network_plot("00000007", config.STUDTEST_DIR)
    print("trajectory trips:", pick, "\nwrote", OUT)


if __name__ == "__main__":
    main()
