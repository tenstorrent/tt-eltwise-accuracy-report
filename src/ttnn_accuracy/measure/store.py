# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

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
    # COLUMNS is the minimum, not the maximum: a pair sweep adds the partner values.
    extra = [c for c in df.columns if c not in COLUMNS]
    df[[*COLUMNS, *extra]].to_csv(path, na_rep="NaN", index_label="index")
    return path


def write_run(meta: RunMeta, out_root: Path, archive: bool = True) -> None:
    """Archive the run and copy it beside the data; `check` is not a run of record."""
    if archive:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        (RUNS_DIR / f"{meta.run_id}.json").write_text(
            json.dumps(asdict(meta), indent=2, sort_keys=True) + "\n"
        )
    for dtype in meta.dtypes:
        stamp = out_root / meta.arch / dtype / RUN_STAMP
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(_stamp(stamp, meta))
    if archive:
        logger.info("run provenance → {}", RUNS_DIR / f"{meta.run_id}.json")


def measured_at(out_root: Path, arch: str, dtypes: list[str], commit: str | None) -> set[tuple]:
    """What this build already measured, keyed on the commit, so a stopped run resumes."""
    done = set()
    for dtype in dtypes:
        stamp = out_root / arch / dtype / RUN_STAMP
        if not stamp.exists() or json.loads(stamp.read_text()).get("tt_metal_commit") != commit:
            continue
        done |= {
            (dtype, csv.parent.name, csv.stem) for csv in (out_root / arch / dtype).glob("*/*.csv")
        }
    return done


def write_failures(failures: dict[str, dict[str, str]], out_root: Path, arch: str) -> None:
    """Which variants produced no data; written even when empty, to clear last run's."""
    for dtype in set(failures) | {p.name for p in (out_root / arch).glob("*") if p.is_dir()}:
        stamp = out_root / arch / dtype / RUN_STAMP
        if not stamp.exists():
            continue
        data = json.loads(stamp.read_text())
        data["failed"] = failures.get(dtype, {})
        stamp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _stamp(stamp: Path, meta: RunMeta) -> str:
    """Accumulate ops across partial runs of one build; a new build supersedes the old."""
    data = asdict(meta)
    if stamp.exists():
        prev = json.loads(stamp.read_text())
        if (prev["tt_metal_commit"], prev["ttnn_version"]) == (
            meta.tt_metal_commit,
            meta.ttnn_version,
        ):
            data["ops"] = sorted(set(prev["ops"]) | set(meta.ops))
    return json.dumps(data, indent=2, sort_keys=True) + "\n"
