"""Shared configuration for PathSnap. Every module imports paths/constants from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Raw data lives outside the repo (not committed). Override with env var PATHSNAP_DATA if needed.
import os
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT.parent / "Student folder" / "Student folder"))
TRAIN_DIR = DATA_DIR / "train"
STUDTEST_DIR = DATA_DIR / "student_test"
# Evaluator trips are dropped in at evaluation time; the API reads from here.
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", STUDTEST_DIR))

RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"

# Coordinates are projected per trip with a local azimuthal-equidistant projection centred
# on the trip's GPS track (see data_processing/load.py). The dataset spans several countries
# (e.g. Germany ~8.6E 49.8N and Vietnam ~106.7E 15.3N), so no single UTM zone fits all trips.

RADIUS_M = 50      # candidate search radius (tuned via candidate miss rate)
K_MAX = 8          # max candidates per GPS point
SEED = 42
