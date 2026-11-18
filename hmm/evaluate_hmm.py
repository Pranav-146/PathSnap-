"""
Evaluation metrics for map matching and route reconstruction.

Calculates:
  1. Pointwise Observation-level Top-1 Accuracy:
     Fraction of GPS observations where predicted arc == ground truth arc.
  2. Candidate Miss Rate:
     Fraction of GPS observations where ground truth arc was not in candidate set.
  3. Route Continuity:
     Fraction of consecutive transitions (arc_i -> arc_{i+1}) that are valid
     directed edges in the road network graph.
  4. Sequence-level Performance:
     Precision, recall, and F1 on unique traversed route arcs.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from hmm.graph import RoadGraph


def compute_route_continuity(arc_sequence: List[int], road_graph: RoadGraph) -> float:
    """
    Fraction of consecutive unique arc transitions that are directly connected in road network.
    (i.e., road_graph.arc_v[arc_t] == road_graph.arc_u[arc_{t+1}]).
    """
    if len(arc_sequence) <= 1:
        return 1.0

    valid_transitions = 0
    total_transitions = 0

    for i in range(len(arc_sequence) - 1):
        a1 = arc_sequence[i]
        a2 = arc_sequence[i + 1]
        if a1 == a2:
            continue
        total_transitions += 1
        # Direct connection check
        if road_graph.arc_v[a1] == road_graph.arc_u[a2]:
            valid_transitions += 1

    if total_transitions == 0:
        return 1.0

    return float(valid_transitions / total_transitions)


def evaluate_trip_predictions(
    matched_arcs: List[int],
    gt_route: np.ndarray,
    candidates_by_step: List[List[Dict]],
    road_graph: RoadGraph,
    baseline_arcs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Evaluate HMM and baseline predictions against ground truth.

    Args:
        matched_arcs: predicted arc_id for each GPS observation (length T)
        gt_route: sequence of ground truth arcs traversed by vehicle
        candidates_by_step: candidates per observation (length T)
        road_graph: RoadGraph instance
        baseline_arcs: optional Top-1 baseline predictions (length T)

    Returns:
        dictionary of metrics comparing HMM vs baseline.
    """
    gt_set = set(gt_route)
    pred_set = set(matched_arcs)

    # 1. Traversed Route Precision / Recall / F1
    true_positives = len(pred_set.intersection(gt_set))
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 2. Candidate Miss Rate (with respect to GT route set)
    misses = 0
    total_steps = len(candidates_by_step)
    for cands in candidates_by_step:
        cand_arcs = {c["arc_id"] for c in cands}
        # Step is a miss if no candidate belongs to the ground-truth route
        if not cand_arcs.intersection(gt_set):
            misses += 1
    miss_rate = misses / total_steps if total_steps > 0 else 0.0

    # 3. Route Continuity
    # Deduplicate consecutive identical arcs for sequence continuity
    unique_hmm_arcs = []
    for a in matched_arcs:
        if not unique_hmm_arcs or unique_hmm_arcs[-1] != a:
            unique_hmm_arcs.append(a)

    continuity_hmm = compute_route_continuity(unique_hmm_arcs, road_graph)

    # 4. In-route observation accuracy:
    # GPS points whose predicted arc is an arc in the true route
    in_route_acc_hmm = float(np.mean([a in gt_set for a in matched_arcs]))

    metrics = {
        "hmm_route_precision": float(precision),
        "hmm_route_recall": float(recall),
        "hmm_route_f1": float(f1),
        "hmm_continuity": float(continuity_hmm),
        "hmm_in_route_accuracy": in_route_acc_hmm,
        "candidate_miss_rate": float(miss_rate),
        "num_gt_arcs": len(gt_route),
        "num_pred_arcs": len(unique_hmm_arcs),
    }

    # Baseline comparison
    if baseline_arcs is not None:
        unique_base_arcs = []
        for a in baseline_arcs:
            if not unique_base_arcs or unique_base_arcs[-1] != a:
                unique_base_arcs.append(a)

        base_set = set(baseline_arcs)
        tp_base = len(base_set.intersection(gt_set))
        prec_base = tp_base / len(base_set) if base_set else 0.0
        rec_base = tp_base / len(gt_set) if gt_set else 0.0
        f1_base = 2 * prec_base * rec_base / (prec_base + rec_base) if (prec_base + rec_base) > 0 else 0.0

        continuity_base = compute_route_continuity(unique_base_arcs, road_graph)
        in_route_acc_base = float(np.mean([a in gt_set for a in baseline_arcs]))

        metrics.update({
            "baseline_route_precision": float(prec_base),
            "baseline_route_recall": float(rec_base),
            "baseline_route_f1": float(f1_base),
            "baseline_continuity": float(continuity_base),
            "baseline_in_route_accuracy": in_route_acc_base,
            "continuity_gain": float(continuity_hmm - continuity_base),
        })

    return metrics


if __name__ == "__main__":
    from pathlib import Path
    from hmm.pipeline import MapMatchingPipeline

    pipeline = MapMatchingPipeline()
    trip_path = Path("Student folder/train/00000001")
    res = pipeline.run_trip(trip_path)

    metrics = evaluate_trip_predictions(
        res["matched_arcs"],
        res["gt_route"],
        res["candidates_by_step"],
        res["road_graph"],
        baseline_arcs=res["baseline_arcs"],
    )

    print("=== Evaluation Report for Trip 00000001 ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
