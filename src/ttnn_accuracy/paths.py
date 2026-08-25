from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"
INDEX_FILE = REPO_ROOT / "report_index.json"
