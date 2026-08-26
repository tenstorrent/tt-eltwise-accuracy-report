"""Where results and their provenance are written."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from loguru import logger

from ttnn_accuracy.measure.schema import COLUMNS, RunMeta
from ttnn_accuracy.paths import RUNS_DIR


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


def write_run(meta: RunMeta) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{meta.run_id}.json"
    path.write_text(json.dumps(asdict(meta), indent=2, sort_keys=True) + "\n")
    logger.info("run provenance → {}", path)
