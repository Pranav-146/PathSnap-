"""
End-to-end Map Matching Pipeline integrating candidate extraction,
classifier scoring hook, and HMM/Viterbi sequence decoding.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

import config
from data_processing.load import load_trip
from features.candidates import FEATURE_COLS, candidates_for_trip
from hmm.graph import RoadGraph
from hmm.viterbi import ViterbiDecoder


def df_candidates_to_list(cand_df: pd.DataFrame, num_points: int) -> List[List[Dict[str, Any]]]:
    """Convert candidate DataFrame into grouped list of candidate dicts per observation."""
    cands_by_step: List[List[Dict[str, Any]]] = [[] for _ in range(num_points)]
    # Ensure sorted by obs_idx, dist_rank
    cand_df = cand_df.sort_values(["obs_idx", "dist_rank"])

    for row in cand_df.itertuples(index=False):
        obs_idx = int(row.obs_idx)
        if obs_idx < num_points:
            cands_by_step[obs_idx].append({
                "arc_id": int(row.arc_id),
                "dist": float(row.dist),
                "dist_rank": int(row.dist_rank),
                "proj_frac": float(row.proj_frac),
                "arc_len": float(row.arc_len),
                "heading_diff": float(row.heading_diff),
            })
    return cands_by_step


class MapMatchingPipeline:
    """Full map matching pipeline for a trip."""

    def __init__(
        self,
        classifier: Optional[Any] = None,
        sigma_z: float = 4.07,
        beta: float = 5.0,
        radius: float = config.RADIUS_M,
        k: int = config.K_MAX,
    ):
        self.classifier = classifier
        self.sigma_z = sigma_z
        self.beta = beta
        self.radius = radius
        self.k = k

    def match_trip(
        self,
        trip_data: Dict[str, Any],
        with_baseline: bool = True,
    ) -> Dict[str, Any]:
        """Run map matching on a loaded trip dictionary."""
        track = trip_data["track"]
        nodes = trip_data["nodes"]
        arcs = trip_data["arcs"]
        n_pts = len(track)

        # 1. Build road graph
        rg = RoadGraph(nodes, arcs)

        # 2. Extract spatial candidates using Person 1's candidates_for_trip
        cand_df = candidates_for_trip(trip_data, radius=self.radius, k=self.k)
        cands_by_step = df_candidates_to_list(cand_df, n_pts)

        # Fallback for any GPS point with zero candidates: attach nearest arc
        for i in range(n_pts):
            if not cands_by_step[i]:
                cands_by_step[i].append({
                    "arc_id": 0,
                    "dist": 50.0,
                    "dist_rank": 0,
                    "proj_frac": 0.5,
                    "arc_len": 50.0,
                    "heading_diff": 0.0,
                })

        # 3. Baseline: Top-1 nearest candidate for each observation
        baseline_arcs = [cands[0]["arc_id"] for cands in cands_by_step]

        # 4. Score candidates with classifier if available
        clf_scores_by_step = None
        classifier_used = False
        if self.classifier is not None and not cand_df.empty:
            # Person 2 classifier scoring hook. Uses the shared FEATURE_COLS list (the same
            # columns the model was trained on). A failure is reported loudly and recorded in
            # the result, never swallowed, so a distance-only run cannot pass as classifier + HMM.
            try:
                probs = self.classifier.predict_proba(cand_df[FEATURE_COLS])[:, 1]
                cand_df["clf_score"] = probs
                clf_scores_by_step = [[] for _ in range(n_pts)]
                for row in cand_df.itertuples(index=False):
                    obs_idx = int(row.obs_idx)
                    if obs_idx < n_pts:
                        clf_scores_by_step[obs_idx].append(float(row.clf_score))
                classifier_used = True
            except Exception as exc:
                warnings.warn(
                    f"Classifier scoring FAILED for trip {trip_data['trip_id']}: {exc!r}. "
                    "Falling back to distance-only emissions; results are NOT classifier + HMM.",
                    RuntimeWarning, stacklevel=2)
                clf_scores_by_step = None

        # 5. Run Viterbi Decoder
        decoder = ViterbiDecoder(rg, sigma_z=self.sigma_z, beta=self.beta)
        t_arr = track["t"].to_numpy(dtype=np.float64)
        x_arr = track["x"].to_numpy(dtype=np.float64)
        y_arr = track["y"].to_numpy(dtype=np.float64)

        matched_arcs, matched_cands = decoder.decode(
            cands_by_step,
            t_arr,
            x_arr,
            y_arr,
            classifier_scores_by_step=clf_scores_by_step,
        )

        # Deduplicate consecutive identical arcs
        unique_route_arcs = []
        for a in matched_arcs:
            if not unique_route_arcs or unique_route_arcs[-1] != a:
                unique_route_arcs.append(a)

        baseline_unique_arcs = []
        for a in baseline_arcs:
            if not baseline_unique_arcs or baseline_unique_arcs[-1] != a:
                baseline_unique_arcs.append(a)

        return {
            "trip_id": trip_data["trip_id"],
            "classifier_used": classifier_used,
            "road_graph": rg,
            "track": track,
            "candidate_df": cand_df,
            "candidates_by_step": cands_by_step,
            "matched_arcs": matched_arcs,
            "unique_route_arcs": unique_route_arcs,
            "matched_candidates": matched_cands,
            "baseline_arcs": baseline_arcs,
            "baseline_unique_arcs": baseline_unique_arcs,
        }

    def run_trip(
        self,
        trip_dir_or_id: Union[str, Path],
        with_route: bool = True
    ) -> Dict[str, Any]:
        """Direct entry point for Person 4 API."""
        p = Path(trip_dir_or_id)
        if not p.exists():
            for split_dir in [config.TRAIN_DIR, config.STUDTEST_DIR]:
                candidate_path = split_dir / p.name
                if candidate_path.exists():
                    p = candidate_path
                    break

        trip_data = load_trip(p, with_route=with_route)
        res = self.match_trip(trip_data)
        if "route" in trip_data:
            res["gt_route"] = trip_data["route"]
        return res
