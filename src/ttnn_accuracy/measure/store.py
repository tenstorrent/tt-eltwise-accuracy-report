"""Where results and their provenance are written."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from loguru import logger

from ttnn_accuracy.measure.schema import COLUMNS, RunMeta
from ttnn_accuracy.paths import RUNS_DIR

RUN_STAMP = "_run.json"


def write_result(
    df: pd.DataFrame, out_root: Path, arch: str, dtype: str, op: str, variant: str
) -> Path:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{op}/{variant}: result is missing {sorted(missing)}")
    out_dir = out_root / arch / dtype / op
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{variant}.csv"
    df[list(COLUMNS)].to_csv(path, na_rep="NaN", index_label="index")
    return path


def write_run(meta: RunMeta, out_root: Path) -> None:
    """Archive the run, and leave a copy beside the data so charts can attribute it."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{meta.run_id}.json").write_text(
        json.dumps(asdict(meta), indent=2, sort_keys=True) + "\n"
    )
    for dtype in meta.dtypes:
        stamp = out_root / meta.arch / dtype / RUN_STAMP
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(_stamp(stamp, meta))
    logger.info("run provenance → {}", RUNS_DIR / f"{meta.run_id}.json")


def _stamp(stamp: Path, meta: RunMeta) -> str:
    """Accumulate ops across partial runs of one build; a new build supersedes the old.

    Measuring unary then unary_bw must leave both attributed — overwriting made the
    first run's pages read as untracked. `run_id` names the latest run of the build.
    """
    data = asdict(meta)
    if stamp.exists():
        prev = json.loads(stamp.read_text())
        if (prev["tt_metal_commit"], prev["ttnn_version"]) == (
            meta.tt_metal_commit,
            meta.ttnn_version,
        ):
            data["ops"] = sorted(set(prev["ops"]) | set(meta.ops))
    return json.dumps(data, indent=2, sort_keys=True) + "\n"
