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
    # COLUMNS is the minimum every reader can rely on, not the maximum: a two-operand
    # sweep adds the partner values, without which its rows cannot be reproduced.
    extra = [c for c in df.columns if c not in COLUMNS]
    df[[*COLUMNS, *extra]].to_csv(path, na_rep="NaN", index_label="index")
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


def measured_at(out_root: Path, arch: str, dtypes: list[str], commit: str | None) -> set[tuple]:
    """(dtype, op, variant) already measured on this build, so a stopped run can resume.

    Keyed on the tt-metal commit in the stamp: data from another build is not this run's
    to keep, and re-measuring it is the only way the two halves stay comparable.
    """
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
    """Record, beside the data, which variants produced none and why.

    The stamp is the only per-arch-per-dtype artifact the report already reads, so a
    measure-time failure reaches the page by the same route as its provenance. Written
    even when empty, to clear last run's failures once an op starts working again.
    """
    for dtype in set(failures) | {p.name for p in (out_root / arch).glob("*") if p.is_dir()}:
        stamp = out_root / arch / dtype / RUN_STAMP
        if not stamp.exists():
            continue
        data = json.loads(stamp.read_text())
        data["failed"] = failures.get(dtype, {})
        stamp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


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
