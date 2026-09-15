"""What moved between two indexes, and `history` across every one; no tolerance, measurement is exact."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

from loguru import logger

from ttnn_accuracy.paths import INDEX_FILE, REPO_ROOT, RUNS_KEY

# `defects` and `unflushed` score like an error figure: the answers ULP cannot see.
SCORED = ("max_ulp", "mean_ulp", "ulp_clipped", "defects", "unflushed")


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
    """Which bucket, or "" — a field the baseline predates cannot have regressed."""
    # A different point count means a different domain, so the two are not comparable:
    # narrowing `polygamma` to |x| <= 1024 read as a regression it had nothing to do with.
    if _num(a.get("n_inputs")) != _num(b.get("n_inputs")):
        return "changed"
    scored = [m for m in SCORED if m in a and m in b]
    if any(_num(b[m]) > _num(a[m]) for m in scored):
        return "regressed"
    # Before `improved`: a first count of defects outranks a scored metric ticking down.
    if [m for m in SCORED if m not in a and _num(b.get(m))]:
        return "changed"
    if any(_num(b[m]) < _num(a[m]) for m in scored):
        return "improved"
    return "changed" if a.get("usable_to") != b.get("usable_to") else ""


def _moves(a: dict, b: dict) -> str:
    return ", ".join(
        f"{m} {a.get(m)} → {b.get(m)}"
        for m in (*SCORED, "usable_to", "n_inputs")
        if str(a.get(m)) != str(b.get(m))
    )


# The bucketing is the classification: `changed` is usable_to-only by construction.
SECTIONS = (
    ("regressed", "Regressions", "A scored metric got worse."),
    ("improved", "Improvements", "A scored metric got better."),
    (
        "changed",
        "Expected",
        "No scored metric got worse on comparable measurements. Either `usable_to` moved, "
        "because a different kernel puts the 2 ULP boundary on a neighbouring group; or "
        "`n_inputs` moved, which means the swept domain changed and the two runs measure "
        "different populations; or a metric is reported here for the first time.",
    ),
    ("added", "New coverage", "Measured here for the first time."),
    ("removed", "No longer measured", "Present in the baseline, absent now."),
)


def digest(old: dict, new: dict) -> str:
    """The diff as one markdown page — the file a reader opens instead of every page."""
    buckets = diff(old, new)
    out = [
        "# Findings",
        "",
        "**Status: measured.** Every row below is a published result that changed.",
        "",
        f"_{_commits(old)} → {_commits(new)}_",
        "",
    ]
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


def _verdict(stats: dict) -> str:
    """Indexes older than the verdict column still have to say something."""
    return stats.get("verdict") or f"max_ulp {stats.get('max_ulp', '—')}"


def _committed(path: Path) -> Iterator[tuple[str, str, dict]]:
    """Every committed version of the index, oldest first — the nightly is its own archive."""
    rel = str(path.relative_to(REPO_ROOT))
    log = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "log", "--format=%H %cs", "--reverse", "--", rel],
        capture_output=True,
        text=True,
    )
    for line in log.stdout.splitlines():
        sha, _, date = line.partition(" ")
        blob = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{sha}:{rel}"], capture_output=True, text=True
        )
        if blob.returncode == 0:
            yield sha[:11], date, json.loads(blob.stdout)


def history(op: str | None = None, findings: Path | None = None) -> int:
    """Which build moved each number — where a bisect starts, not just that tonight differs."""
    builds = list(_committed(INDEX_FILE))
    rows = _changes(builds, op)
    title = f"# History — `{op}`" if op else "# History"
    page = [title, "", f"_{len(rows)} changes across {len(builds)} published indexes_", ""]
    page += ["| Build | Date | Variant | tt-metal | Moved |", "|---|---|---|---|---|"]
    page += [
        f"| `{sha}` | {date} | `{'/'.join(key)}` | `{built or '—'}` | {moved} |"
        for date, sha, key, built, moved in rows
    ]
    page.append("")
    text = "\n".join(page)
    logger.info("\n{}", text)
    if findings:
        findings.parent.mkdir(parents=True, exist_ok=True)
        findings.write_text(text)
        logger.success("history → {}", findings)
    return 0


def _changes(builds: list[tuple[str, str, dict]], op: str | None = None) -> list[tuple]:
    """One row per moved metric, oldest first — pure, so git is not needed to test it."""
    seen: dict[tuple, tuple[dict, str | None]] = {}
    rows: list[tuple] = []
    for sha, date, index in builds:
        for key, stats in _variants(index).items():
            if op and key[2] != op:
                continue
            arch, dtype = key[0], key[1]
            built = index.get(RUNS_KEY, {}).get(arch, {}).get(dtype, {}).get("tt_metal_commit")
            before = seen.get(key)
            if before is None:
                rows.append((date, sha, key, built, f"first measured — {_verdict(stats)}"))
            elif moved := _moves(before[0], stats):
                # The number moved on the same tt-metal, so this report changed, not ttnn.
                same = built and built == before[1]
                rows.append((date, sha, key, built, moved + (" — **our change**" if same else "")))
            seen[key] = (stats, built)
    return rows


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
