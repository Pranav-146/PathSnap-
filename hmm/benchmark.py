"""
Multi-trip benchmark for HMM Viterbi Map Matching vs Baseline Top-1.

Evaluates multiple trips across train and student_test partitions:
  - Measures Precision, Recall, F1 on traversed route
  - Measures Route Continuity (% physically connected consecutive edges)
  - Measures In-route observation accuracy
  - Computes Continuity Gain over baseline
Saves results to results/hmm_baseline_comparison.csv
"""

import time
from pathlib import Path
from typing import List
import pandas as pd

import config
from data_processing.load import load_trip
from hmm.evaluate_hmm import evaluate_trip_predictions
from hmm.pipeline import MapMatchingPipeline
from visualisations.folium_maps import create_trip_map


def benchmark_trips(trip_ids: List[str], save_csv_path: Path, generate_maps: bool = True):
    pipeline = MapMatchingPipeline()
    results = []

    print(f"Starting benchmark on {len(trip_ids)} trips...")
    for tid in trip_ids:
        # Locate trip folder in train or student_test
        trip_dir = None
        split_name = "unknown"
        for split, folder in [("train", config.TRAIN_DIR), ("student_test", config.STUDTEST_DIR)]:
            cand = folder / tid
            if cand.exists():
                trip_dir = cand
                split_name = split
                break

        if trip_dir is None:
            print(f"Warning: Trip {tid} not found, skipping.")
            continue

        print(f"\nProcessing Trip {tid} ({split_name})...")
        t0 = time.time()
        trip_data = load_trip(trip_dir, with_route=True)
        res = pipeline.match_trip(trip_data)
        elapsed = time.time() - t0

        if "route" in trip_data and trip_data["route"] is not None:
            metrics = evaluate_trip_predictions(
                res["matched_arcs"],
                trip_data["route"],
                res["candidates_by_step"],
                res["road_graph"],
                baseline_arcs=res["baseline_arcs"],
            )

            row = {
                "trip_id": tid,
                "split": split_name,
                "num_gps_points": len(trip_data["track"]),
                "num_gt_arcs": len(trip_data["route"]),
                "hmm_f1": round(metrics["hmm_route_f1"] * 100, 2),
                "baseline_f1": round(metrics["baseline_route_f1"] * 100, 2),
                "hmm_continuity": round(metrics["hmm_continuity"] * 100, 2),
                "baseline_continuity": round(metrics["baseline_continuity"] * 100, 2),
                "continuity_gain": round(metrics["continuity_gain"] * 100, 2),
                "in_route_acc": round(metrics["hmm_in_route_accuracy"] * 100, 2),
                "miss_rate": round(metrics["candidate_miss_rate"] * 100, 3),
                "inference_time_s": round(elapsed, 2),
            }
            results.append(row)
            print(f"  Done in {elapsed:.2f}s | HMM F1: {row['hmm_f1']}% | Continuity Gain: +{row['continuity_gain']}%")

        if generate_maps:
            map_path = config.RESULTS_DIR / "maps" / f"trip_{tid}_hmm.html"
            create_trip_map(
                trip_data,
                matched_arcs=res["matched_arcs"],
                baseline_arcs=res["baseline_arcs"],
                road_graph=res["road_graph"],
                save_html_path=map_path,
            )

    df = pd.DataFrame(results)
    save_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_csv_path, index=False)
    print(f"\nSaved benchmark comparison table to {save_csv_path}")
    print("\n" + df.to_string(index=False))
    return df


if __name__ == "__main__":
    # Benchmark 3 training trips and 2 student-testing trips
    sample_trips = [
        "00000001",  # train
        "00000002",  # train
        "00000003",  # train
        "00000000",  # student_test
        "00000007",  # student_test
    ]
    out_csv = config.RESULTS_DIR / "hmm_baseline_comparison.csv"
    benchmark_trips(sample_trips, out_csv, generate_maps=True)
