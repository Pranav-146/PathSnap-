"""
Viterbi decoder for Hidden Markov Model (HMM) map matching.

Implements the Newson & Krumm (2009) formulation:
  - Emission probability: Gaussian distribution on perpendicular projection distance:
      P(p_t | c_i) = (1 / (sqrt(2*pi) * sigma_z)) * exp(-0.5 * (d_perp / sigma_z)^2)
      * Optionally multiplied or weighted by classifier probability P_clf(c_i)
  - Transition probability: Exponential distribution on absolute difference between
    network shortest path distance and Euclidean distance between GPS observations:
      P(c_j | c_i) = (1 / beta) * exp(-|d_net(c_i, c_j) - d_euclid(p_t, p_{t+1})| / beta)
  - Log-space Viterbi forward dynamic programming with backpointer tracking.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from hmm.graph import RoadGraph


class ViterbiDecoder:
    """Hidden Markov Model sequence decoder using the Viterbi algorithm."""

    def __init__(
        self,
        road_graph: RoadGraph,
        sigma_z: float = 4.07,     # standard deviation of GPS measurement noise (metres)
        beta: float = 5.0,         # transition parameter (scale of network vs euclidean discrepancy)
        max_time_gap: float = 15.0 # seconds threshold to reset HMM when GPS signal drops out
    ):
        self.rg = road_graph
        self.sigma_z = float(sigma_z)
        self.beta = float(beta)
        self.max_time_gap = float(max_time_gap)

        self._log_sigma_const = np.log(1.0 / (np.sqrt(2.0 * np.pi) * self.sigma_z))
        self._log_beta_const = np.log(1.0 / self.beta)

    def log_emission(self, cand: Dict, clf_prob: Optional[float] = None) -> float:
        """
        Compute log emission probability for a candidate arc.
        Combines spatial Gaussian distance with optional trained classifier probability.
        """
        dist = cand["dist"]
        log_p_dist = self._log_sigma_const - 0.5 * (dist / self.sigma_z) ** 2

        if clf_prob is not None:
            # Calibrated probability from classifier
            p = float(np.clip(clf_prob, 1e-6, 1.0 - 1e-6))
            log_p_clf = np.log(p)
            return log_p_dist + log_p_clf

        return log_p_dist

    def log_transition(
        self,
        cand_from: Dict,
        cand_to: Dict,
        d_euclid: float,
        dt: float
    ) -> float:
        """
        Compute log transition probability from cand_from to cand_to.
        Penalizes discrepancies between road network distance and direct GPS displacement.
        """
        # If time gap is large, treat observations as independent
        if dt > self.max_time_gap:
            return 0.0

        arc_from = cand_from["arc_id"]
        arc_to = cand_to["arc_id"]
        frac_from = cand_from["proj_frac"]
        frac_to = cand_to["proj_frac"]

        d_net = self.rg.arc_network_distance(
            arc_from, arc_to, proj_frac_from=frac_from, proj_frac_to=frac_to
        )

        if np.isinf(d_net):
            # No reachable road path exists within cutoff
            return -1e9

        diff = abs(d_net - d_euclid)
        log_prob = self._log_beta_const - (diff / self.beta)
        return float(log_prob)

    def decode(
        self,
        candidates_by_step: List[List[Dict]],
        timestamps: np.ndarray,
        track_x: np.ndarray,
        track_y: np.ndarray,
        classifier_scores_by_step: Optional[List[List[float]]] = None,
    ) -> Tuple[List[int], List[Dict]]:
        """
        Run the Viterbi algorithm to find the most probable sequence of road arcs.

        Args:
            candidates_by_step: candidate arcs at each GPS step t
            timestamps: array of GPS timestamps (seconds)
            track_x: array of GPS x coordinates (UTM metres)
            track_y: array of GPS y coordinates (UTM metres)
            classifier_scores_by_step: optional classifier probability for each candidate

        Returns:
            best_arc_sequence: list of predicted arc_id (one per GPS step)
            matched_candidates: list of winning candidate dicts for each step
        """
        T = len(candidates_by_step)
        if T == 0:
            return [], []

        # trellis[t][i] stores max log probability of path ending at candidate i at step t
        # backpointer[t][i] stores index of previous candidate in step t-1
        V: List[np.ndarray] = []
        backpointer: List[np.ndarray] = []

        # Step 0: Initialization
        cands_0 = candidates_by_step[0]
        K_0 = len(cands_0)
        v0 = np.zeros(K_0, dtype=np.float64)

        for i, c in enumerate(cands_0):
            clf_p = (
                classifier_scores_by_step[0][i]
                if classifier_scores_by_step and len(classifier_scores_by_step[0]) > i
                else None
            )
            v0[i] = self.log_emission(c, clf_p)

        V.append(v0)
        backpointer.append(np.full(K_0, -1, dtype=np.int64))

        # Steps 1 to T-1: Induction
        for t in range(1, T):
            cands_prev = candidates_by_step[t - 1]
            cands_curr = candidates_by_step[t]
            K_prev = len(cands_prev)
            K_curr = len(cands_curr)

            dt = float(timestamps[t] - timestamps[t - 1])
            d_euclid = float(
                np.hypot(track_x[t] - track_x[t - 1], track_y[t] - track_y[t - 1])
            )

            vt = np.full(K_curr, -1e9, dtype=np.float64)
            bp_t = np.full(K_curr, -1, dtype=np.int64)

            # Compute emission probabilities for all current candidates
            log_emissions = np.zeros(K_curr, dtype=np.float64)
            for j, c_j in enumerate(cands_curr):
                clf_p = (
                    classifier_scores_by_step[t][j]
                    if classifier_scores_by_step and len(classifier_scores_by_step[t]) > j
                    else None
                )
                log_emissions[j] = self.log_emission(c_j, clf_p)

            # Check for large time gap
            is_gap = dt > self.max_time_gap

            for j in range(K_curr):
                c_j = cands_curr[j]
                if is_gap:
                    # Reset: independent transition
                    best_prob = np.max(V[t - 1]) + log_emissions[j]
                    best_prev = int(np.argmax(V[t - 1]))
                else:
                    trans_probs = np.zeros(K_prev, dtype=np.float64)
                    for i in range(K_prev):
                        c_i = cands_prev[i]
                        trans_probs[i] = self.log_transition(c_i, c_j, d_euclid, dt)

                    total_candidate_probs = V[t - 1] + trans_probs + log_emissions[j]
                    best_idx = int(np.argmax(total_candidate_probs))
                    best_prob = total_candidate_probs[best_idx]
                    best_prev = best_idx

                    # If all transitions were blocked (-1e9), fall back to nearest transition
                    if best_prob < -1e8:
                        best_idx = int(np.argmax(V[t - 1]))
                        best_prob = V[t - 1][best_idx] + log_emissions[j] - 100.0
                        best_prev = best_idx

                vt[j] = best_prob
                bp_t[j] = best_prev

            V.append(vt)
            backpointer.append(bp_t)

        # Traceback: find most likely sequence
        best_last_idx = int(np.argmax(V[-1]))

        winning_indices = [0] * T
        winning_indices[-1] = best_last_idx

        for t in range(T - 1, 0, -1):
            winning_indices[t - 1] = int(backpointer[t][winning_indices[t]])

        matched_arcs = []
        matched_candidates = []
        for t in range(T):
            idx = winning_indices[t]
            c = candidates_by_step[t][idx]
            matched_arcs.append(c["arc_id"])
            matched_candidates.append(c)

        return matched_arcs, matched_candidates
