"""Build the candidate datasets for the train and student_test partitions.

Usage:  python -m features.build_features [--radius 50] [--k 8]

Outputs (results/):
  candidates_train.parquet, candidates_studtest.parquet   one row per (GPS point, candidate arc)
  labels_train.parquet, labels_studtest.parquet           one row per GPS point (reference arc)
  candidate_miss_rate.csv                                 miss rate vs search radius
The evaluator partition is never touched here.
"""
import argparse
import time

import pandas as pd

import config
from data_processing.labels import derive_labels
from data_processing.load import list_trips, load_trip
from features.candidates import add_features, raw_candidates, restrict

SWEEP_RADII = [25, 50, 75, 100]
SPLITS = {"train": config.TRAIN_DIR, "studtest": config.STUDTEST_DIR}


def hits_per_point(cand, labels):
    """Boolean per GPS point: is the reference arc inside the candidate set?"""
    m = cand.merge(labels[["obs_idx", "ref_arc"]], on="obs_idx")
    hit = m[m.arc_id == m.ref_arc].obs_idx.unique()
    return labels.obs_idx.isin(hit)


def main(radius, k):
    config.RESULTS_DIR.mkdir(exist_ok=True)
    max_r = max(SWEEP_RADII + [radius])
    miss_rows = []

    for split, folder in SPLITS.items():
        cands, labs = [], []
        miss = {r: [0, 0, 0, 0] for r in SWEEP_RADII}  # radius -> [missed, total, missed_reliable, total_reliable]
        t0 = time.time()
        for path in list_trips(folder):
            trip = load_trip(path)
            lab = derive_labels(trip)
            raw = raw_candidates(trip, radius=max_r, k=k)

            for r in SWEEP_RADII:                    # candidate miss rate at each radius
                h = hits_per_point(restrict(raw, r, k), lab)
                rel = ~lab.unreliable.to_numpy()           # points whose reference arc is trustworthy
                miss[r][0] += int((~h).sum())
                miss[r][1] += len(h)
                miss[r][2] += int((~h[rel]).sum())
                miss[r][3] += int(rel.sum())

            cand = add_features(trip, restrict(raw, radius, k))
            cand = cand.merge(lab[["obs_idx", "ref_arc", "unreliable"]], on="obs_idx", how="left")
            cand["label"] = (cand.arc_id == cand.ref_arc).astype("int8")
            cand = cand.rename(columns={"unreliable": "label_unreliable"}).drop(columns=["ref_arc"])
            cands.append(cand)
            labs.append(lab)

        cand_df, lab_df = pd.concat(cands, ignore_index=True), pd.concat(labs, ignore_index=True)
        cand_df.to_parquet(config.RESULTS_DIR / f"candidates_{split}.parquet", index=False)
        lab_df.to_parquet(config.RESULTS_DIR / f"labels_{split}.parquet", index=False)

        n_pts = len(lab_df)
        print(f"\n[{split}] {len(cands)} trips, {n_pts} GPS points, {len(cand_df)} candidate rows "
              f"({time.time()-t0:.0f}s)")
        print(f"  radius={radius} m, K={k}: positives={int(cand_df.label.sum())} "
              f"({cand_df.label.mean()*100:.1f}% of rows), "
              f"points with no candidates={n_pts - cand_df.groupby(['trip_id','obs_idx']).ngroups}, "
              f"unreliable labels={lab_df.unreliable.mean()*100:.1f}%")
        for r, (m, n, mr_, nr_) in miss.items():
            miss_rows.append({"split": split, "radius_m": r, "k": k, "points": n,
                              "missed": m, "miss_rate_pct": round(100 * m / n, 2),
                              "miss_rate_reliable_pct": round(100 * mr_ / nr_, 2)})

    mr = pd.DataFrame(miss_rows)
    mr.to_csv(config.RESULTS_DIR / "candidate_miss_rate.csv", index=False)
    print("\nCandidate miss rate %, all points (reference arc absent from candidate set):")
    print(mr.pivot(index="radius_m", columns="split", values="miss_rate_pct").to_string())
    print("\nCandidate miss rate %, reliable-label points only:")
    print(mr.pivot(index="radius_m", columns="split", values="miss_rate_reliable_pct").to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=config.RADIUS_M)
    ap.add_argument("--k", type=int, default=config.K_MAX)
    a = ap.parse_args()
    main(a.radius, a.k)
