# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import os
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
# Hundreds of GB, never committed, so it may live off-repo — one setting for every stage.
DATA_DIR = Path(os.environ.get("DATA_DIR") or REPO_ROOT / "data")
STATS_DIR = REPO_ROOT / "stats"
MANIFEST_FILE = STATS_DIR / "ops_manifest.json"
RUNS_DIR = STATS_DIR / "runs"
PERF_DIR = STATS_DIR / "perf"
REPORTS_DIR = REPO_ROOT / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"
INDEX_FILE = REPO_ROOT / "report_index.json"
# Underscored so no arch name collides; here so `compare` stays importable without tt-metal.
RUNS_KEY = "_runs"
ANALYZE_DIR = REPO_ROOT / "analyze-report"
CONTRACT_FILE = ANALYZE_DIR / "contract.md"
ASK_FILE = ANALYZE_DIR / "ask.md"
