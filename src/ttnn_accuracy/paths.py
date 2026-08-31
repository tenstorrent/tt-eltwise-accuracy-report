import os
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
# Raw CSVs run to hundreds of GB and are never committed, so they may live off-repo.
# One setting for every stage: measure writing where charts cannot read is silent loss.
DATA_DIR = Path(os.environ.get("DATA_DIR") or REPO_ROOT / "data")
STATS_DIR = REPO_ROOT / "stats"
MANIFEST_FILE = STATS_DIR / "ops_manifest.json"
RUNS_DIR = STATS_DIR / "runs"
REPORTS_DIR = REPO_ROOT / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"
INDEX_FILE = REPO_ROOT / "report_index.json"
