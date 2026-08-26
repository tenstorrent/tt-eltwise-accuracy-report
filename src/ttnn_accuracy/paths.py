from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
DATA_DIR = REPO_ROOT / "data"
STATS_DIR = REPO_ROOT / "stats"
MANIFEST_FILE = STATS_DIR / "ops_manifest.json"
RUNS_DIR = STATS_DIR / "runs"
REPORTS_DIR = REPO_ROOT / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"
INDEX_FILE = REPO_ROOT / "report_index.json"
