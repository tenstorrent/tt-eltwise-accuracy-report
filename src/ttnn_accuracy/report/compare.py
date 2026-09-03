"""Two report indexes → what moved between two tt-metal builds.

Measurement is deterministic, so any difference is real and there is no tolerance to tune.
`usable_to` is reported but never scored: its "—" means both "never within 2 ULP" and
"not applicable here". `mean_ulp`'s "—" is unambiguous, so it scores as 0.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from ttnn_accuracy.paths import RUNS_KEY

# `defects` is scored like an error figure: inf or zero where a value exists is the worst
# answer an op can give, and it is the one the ULP columns cannot see.
SCORED = ("max_ulp", "mean_ulp", "ulp_clipped", "defects")


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
        elif bucket := _bucket(a, b):
            buckets[bucket].append((key, a, b))
    return buckets


def _bucket(a: dict, b: dict) -> str:
    """Where this pair belongs, or "" when nothing worth saying moved.

    Only metrics both sides carry are scored: a field the baseline predates cannot have
    regressed. Treating `defects` as 0 where it was simply absent put 93 entries in
    `regressed` the night it was added, not one of which had moved a ULP. Its arrival is
    still news, so it lands in `changed`.
    """
    scored = [m for m in SCORED if m in a and m in b]
    if any(_num(b[m]) > _num(a[m]) for m in scored):
        return "regressed"
    if any(_num(b[m]) < _num(a[m]) for m in scored):
        return "improved"
    # A metric appearing as 0 says nothing; appearing with a count is the finding itself.
    appeared = [m for m in SCORED if m not in a and _num(b.get(m))]
    return "changed" if appeared or a.get("usable_to") != b.get("usable_to") else ""


def _moves(a: dict, b: dict) -> str:
    return ", ".join(
        f"{m} {a.get(m)} → {b.get(m)}"
        for m in (*SCORED, "usable_to")
        if str(a.get(m)) != str(b.get(m))
    )


# The bucketing is the classification: `changed` is usable_to-only by construction.
SECTIONS = (
    ("regressed", "Regressions", "A scored metric got worse."),
    ("improved", "Improvements", "A scored metric got better."),
    (
        "changed",
        "Expected",
        "Only `usable_to` moved: a different kernel puts the 2 ULP boundary on a "
        "neighbouring group. No scored metric changed.",
    ),
    ("added", "New coverage", "Measured here for the first time."),
    ("removed", "No longer measured", "Present in the baseline, absent now."),
)


def digest(old: dict, new: dict) -> str:
    """The diff as one markdown page — the file a reader opens instead of 754."""
    buckets = diff(old, new)
    out = ["# Findings", "", f"_{_commits(old)} → {_commits(new)}_", ""]
    for key, title, why in SECTIONS:
        if not (rows := buckets[key]):
            continue
        out += [f"## {title} ({len(rows)})", "", why, "", "| Variant | Moved |", "|---|---|"]
        out += [
            f"| `{arch}/{dtype}/{op}/{variant}` | {_moves(a, b) if a and b else '—'} |"
            for (arch, dtype, op, variant), a, b in rows
        ]
        out.append("")
    if not any(buckets.values()):
        out += ["Nothing moved.", ""]
    return "\n".join(out)


def _commits(index: dict) -> str:
    runs = index.get(RUNS_KEY, {})
    shas = {run.get("tt_metal_commit") for arch in runs.values() for run in arch.values()}
    return ", ".join(sorted(str(s) for s in shas)) or "unknown"


def compare(old_path: Path, new_path: Path, findings: Path | None = None) -> int:
    old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
    buckets = diff(old, new)
    if findings:
        findings.parent.mkdir(parents=True, exist_ok=True)
        findings.write_text(digest(old, new))
        logger.info("findings → {}", findings)

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
            getattr(logger, level)("{}: {}", where, _moves(a, b))

    unchanged = len(_variants(old).keys() & _variants(new).keys()) - sum(
        len(buckets[k]) for k in ("regressed", "improved", "changed")
    )
    logger.info("{} unchanged", unchanged)
    return 1 if buckets["regressed"] else 0
