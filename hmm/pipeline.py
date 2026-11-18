"""
End-to-end Map Matching Pipeline integrating candidate extraction,
classifier scoring hook, and HMM/Viterbi sequence decoding.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

import config
from data_processing.load import load_trip
from features.candidates import candidates_for_trip
from hmm.graph import RoadGraph
from hmm.viterbi import ViterbiDecoder


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
        """
        Args:
            classifier: optional trained classifier model with predict_proba method
            sigma_z: GPS error standard deviation (metres)
            beta: HMM transition parameter
            radius: search radius for candidates (metres)
            k: max candidates per point
        """
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
        """
        Run map matching on a loaded trip dictionary.

        Args:
            trip_data: dict from load_trip (keys: 'trip_id', 'track', 'nodes', 'arcs', optional 'route')
            with_baseline: also compute Top-1 nearest candidate baseline

        Returns:
            dict containing:
                'trip_id': str
                'matched_arcs': List[int] (HMM Viterbi predicted arc_id per GPS observation)
                'unique_route_arcs': List[int] (deduplicated route traversal sequence)
                'matched_candidates': List[Dict] (projection details for each point)
                'baseline_arcs': Optional[List[int]] (Top-1 nearest candidate per point)
                'baseline_unique_arcs': Optional[List[int]]
        """
        track = trip_data["track"]
        nodes = trip_data["nodes"]
        arcs = trip_data["arcs"]

        # 1. Build road graph
        rg = RoadGraph(nodes, arcs)

        # 2. Extract spatial candidates
        cands_by_step = candidates_for_trip(
            track, nodes, arcs, radius=self.radius, k=self.k
        )

        # 3. Baseline: Top-1 nearest candidate for each observation
        baseline_arcs = []
        if with_baseline:
            for step_cands in cands_by_step:
                if step_cands:
                    baseline_arcs.append(step_cands[0]["arc_id"])
                else:
                    baseline_arcs.append(-1)

        # 4. Score candidates with classifier if available
        clf_scores_by_step = None
        if self.classifier is not None:
            # Placeholder hook for feature table classifier inference
            clf_scores_by_step = []
            for step_cands in cands_by_step:
                # Default to None if specific feature vector is not provided
                clf_scores_by_step.append([None] * len(step_cands))

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

        # Deduplicate consecutive identical arcs to obtain the road path traversal
        unique_route_arcs = []
        for a in matched_arcs:
            if not unique_route_arcs or unique_route_arcs[-1] != a:
                unique_route_arcs.append(a)

        baseline_unique_arcs = []
        if baseline_arcs:
            for a in baseline_arcs:
                if not baseline_unique_arcs or baseline_unique_arcs[-1] != a:
                    baseline_unique_arcs.append(a)

        return {
            "trip_id": trip_data["trip_id"],
            "road_graph": rg,
            "track": track,
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
        """
        End-to-end trip matcher given a trip folder path or trip_id.
        Can be directly invoked by Person 4's FastAPI endpoints.
        """
        p = Path(trip_dir_or_id)
        if not p.exists():
            # Check train or student_test splits
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
