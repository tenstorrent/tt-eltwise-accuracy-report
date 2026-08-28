"""Two report indexes → what moved between two tt-metal builds.

Measurement is deterministic: the same inputs on the same silicon through the same
kernels give identical output, so any difference between two indexes is a real change —
never noise, and there is no tolerance to tune. A run exits non-zero when anything got
worse, which is the whole CI gate.

`usable_to` is reported when it moves but never scored: its "—" means both "not within
2 ULP anywhere" and "not applicable to a multi-operand op", and a gate must not guess
which. `mean_ulp`'s "—" is unambiguous — no inexact points — so it scores as 0.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from ttnn_accuracy.report.charts import RUNS_KEY

SCORED = ("max_ulp", "mean_ulp", "ulp_clipped")


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0  # "—": no measurable error


def _variants(index: dict) -> dict[tuple[str, str, str, str], dict]:
    flat = {}
    for arch, dtypes in index.items():
        if arch == RUNS_KEY:
            continue
        for dtype, ops in dtypes.items():
            for op, variants in ops.items():
                for variant, stats in variants.items():
                    flat[arch, dtype, op, variant] = stats
    return flat


def diff(old: dict, new: dict) -> dict[str, list]:
    """Buckets of (key, old_stats, new_stats); `added`/`removed` carry one side only."""
    before, after = _variants(old), _variants(new)
    buckets = {"regressed": [], "improved": [], "changed": [], "added": [], "removed": []}
    for key in sorted(before.keys() | after.keys()):
        a, b = before.get(key), after.get(key)
        if a is None:
            buckets["added"].append((key, None, b))
        elif b is None:
            buckets["removed"].append((key, a, None))
        elif any(_num(a[m]) != _num(b[m]) for m in SCORED) or a["usable_to"] != b["usable_to"]:
            worse = any(_num(b[m]) > _num(a[m]) for m in SCORED)
            better = any(_num(b[m]) < _num(a[m]) for m in SCORED)
            bucket = "regressed" if worse else "improved" if better else "changed"
            buckets[bucket].append((key, a, b))
    return buckets


def _commits(index: dict) -> str:
    runs = index.get(RUNS_KEY, {})
    shas = {run.get("tt_metal_commit") for arch in runs.values() for run in arch.values()}
    return ", ".join(sorted(str(s) for s in shas)) or "unknown"


def compare(old_path: Path, new_path: Path) -> int:
    old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
    buckets = diff(old, new)

    logger.info("baseline {} (tt-metal {})", old_path, _commits(old))
    logger.info("candidate {} (tt-metal {})", new_path, _commits(new))
    for name, rows in buckets.items():
        if not rows:
            continue
        level = "error" if name == "regressed" else "info"
        getattr(logger, level)("{} {}:", len(rows), name)
        for (arch, dtype, op, variant), a, b in rows:
            where = f"  {arch}/{dtype}/{op}/{variant}"
            if a is None or b is None:
                getattr(logger, level)(where)
                continue
            moves = ", ".join(
                f"{m} {a[m]} → {b[m]}" for m in (*SCORED, "usable_to") if str(a[m]) != str(b[m])
            )
            getattr(logger, level)("{}: {}", where, moves)

    unchanged = len(_variants(old).keys() & _variants(new).keys()) - sum(
        len(buckets[k]) for k in ("regressed", "improved", "changed")
    )
    logger.info("{} unchanged", unchanged)
    return 1 if buckets["regressed"] else 0
