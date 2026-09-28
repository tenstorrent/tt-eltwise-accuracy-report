# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Orchestration: walk specs × dtypes, sweep each, write one CSV per variant."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ttnn_accuracy.measure.device import open_device, reason, session
from ttnn_accuracy.measure.schema import check_arch, describe_run
from ttnn_accuracy.measure.store import measured_at, write_failures, write_result, write_run
from ttnn_accuracy.measure.sweeps import SWEEPS, specials
from ttnn_accuracy.ops.plan import OpSpec


def measure(
    specs: list[OpSpec],
    dtypes: list[str],
    arch: str,
    out_root: Path,
    device_id: int = 0,
    archive: bool = True,
    device=None,
) -> int:
    """Variants that produced no CSV — each recorded with its reason, measured ones skipped."""
    failures: dict[str, dict[str, str]] = {}
    names = sorted({spec.name for spec in specs})

    with session(device_id, device) as device:
        check_arch(device, arch)
        meta = describe_run(device, arch, names, dtypes)
        done = measured_at(out_root, arch, dtypes, meta.tt_metal_commit)
        write_run(meta, out_root, archive)
        for spec in specs:
            logger.info("{} ({}) [{}]", spec.name, spec.category, spec.variant)
            for dtype in dtypes:
                if dtype not in spec.layouts:  # the probe found this arch rejects it
                    continue
                if (dtype, spec.name, spec.variant) in done:
                    logger.debug("  {} already measured on this build", dtype)
                    continue
                if why := _measure(spec, dtype, arch, out_root, device):
                    failures.setdefault(dtype, {})[f"{spec.name}/{spec.variant}"] = why

    write_failures(failures, out_root, arch)
    failed = sum(len(v) for v in failures.values())
    if failed:
        logger.error("{} variant(s) produced no data", failed)
    return failed


def check(
    specs: list[OpSpec],
    dtypes: list[str],
    arch: str,
    device_id: int = 0,
    perf: bool = False,
    max_ulp: float | None = None,
) -> int:
    """The author's loop: measure these ops and diff them against the baseline read from git."""
    from tempfile import TemporaryDirectory

    from ttnn_accuracy.paths import INDEX_FILE
    from ttnn_accuracy.report.compare import diff
    from ttnn_accuracy.report.score import score_csv

    if not INDEX_FILE.exists():
        raise SystemExit(f"no baseline at {INDEX_FILE} — nothing to check against")
    baseline = json.loads(INDEX_FILE.read_text())

    # One device for both passes: opening it costs about as long as measuring one op.
    with open_device(device_id) as device, TemporaryDirectory(prefix="ttnn-check-") as tmp:
        empty = measure(specs, dtypes, arch, Path(tmp), device_id, archive=False, device=device)
        broken = _check_properties(specs, dtypes, device)
        timing = _check_perf(specs, dtypes, arch, device) if perf else 0
        candidate = {arch: {}}
        for csv in sorted(Path(tmp).glob(f"{arch}/*/*/*.csv")):
            dtype, op, variant = csv.parts[-3], csv.parts[-2], csv.stem
            if scored := score_csv(csv, op, variant):
                candidate[arch].setdefault(dtype, {}).setdefault(op, {})[variant] = scored[0]

    # Only what was just measured: everything else is unchanged by construction.
    measured = {
        (d, o, v) for d, ops in candidate[arch].items() for o, vs in ops.items() for v in vs
    }
    trimmed = {
        arch: {
            dtype: {
                op: {v: s for v, s in vs.items() if (dtype, op, v) in measured}
                for op, vs in ops.items()
            }
            for dtype, ops in baseline.get(arch, {}).items()
        }
    }
    buckets = diff(trimmed, candidate)
    for name, rows in buckets.items():
        for (_, dtype, op, variant), _, now in rows:
            where = f"  {dtype}/{op}/{variant}"
            getattr(logger, "error" if name == "regressed" else "info")(
                "{} {}: {}", name, where, (now or {}).get("verdict", "—")
            )
    over = _over_bar(candidate[arch], max_ulp) if max_ulp is not None else 0
    # One line, last: a kernel author wants the answer, not to read a diff for it.
    moved = ", ".join(f"{len(rows)} {name}" for name, rows in buckets.items() if rows)
    say = logger.error if buckets["regressed"] or over else logger.success
    say("{} variant(s) measured — {}", len(measured), moved or "nothing moved")
    return empty + broken + timing + over + len(buckets["regressed"])


def refine(
    specs: list[OpSpec],
    dtypes: list[str],
    arch: str,
    device_id: int = 0,
    findings: Path | None = None,
) -> int:
    """Sampled bounds that a search around their own worst point proved loose."""
    from ttnn_accuracy.config import SAMPLED
    from ttnn_accuracy.measure.sweeps import refine as refine_point
    from ttnn_accuracy.paths import DATA_DIR

    loosened = 0
    rows_out: list[str] = []
    with open_device(device_id) as device:
        check_arch(device, arch)
        for spec in specs:
            for dtype in (d for d in dtypes if d in spec.layouts):
                if (spec.operands, dtype) not in SAMPLED:  # exhaustive already, nothing to tighten
                    continue
                csv = DATA_DIR / arch / dtype / spec.name / f"{spec.variant}.csv"
                if not csv.exists():
                    logger.warning("no measurement at {} — run `measure` first", csv)
                    continue
                rows = pd.read_csv(csv)
                # Finite: an inf row is a defect carrying no ULP, which `max_ulp` excludes too.
                rows = rows[np.isfinite(rows["ulp_error"])]
                if rows.empty:
                    continue
                row = rows.loc[rows["ulp_error"].idxmax()]
                point = [row[c] for c in ("x", "x2", "x3") if c in rows.columns]
                was = float(row["ulp_error"])
                now, where = refine_point(
                    spec.ttnn_fn, spec.golden_fn, point, dtype, spec.layouts[dtype], device
                )
                tighter = now > was
                loosened += tighter
                at = ", ".join(f"{v:.6g}" for v in where)
                getattr(logger, "error" if tighter else "success")(
                    "{}/{}/{}: sampled {:.6g} → exhaustive {:.6g} at {}",
                    dtype,
                    spec.name,
                    spec.variant,
                    was,
                    now,
                    at,
                )
                if tighter:
                    rows_out.append(
                        f"| `{arch}/{dtype}/{spec.name}/{spec.variant}` | {was:.6g} | "
                        f"{now:.6g} | {now / was:.2f}x | `{at}` |"
                    )

    if findings:
        page = ["# Sampled maxima", ""]
        page += [
            "Binary fp32 and every ternary sweep sample their operands, so the published "
            "`max_ulp` is a lower bound. Each row below is a variant whose worst point holds "
            "a worse answer than the sample drew.",
            "",
        ]
        page += ["| Variant | Published | Exhaustive | Ratio | At |", "|---|---|---|---|---|"]
        page += rows_out or ["| _none — every sampled bound was tight_ | | | | |"]
        page.append("")
        findings.parent.mkdir(parents=True, exist_ok=True)
        findings.write_text("\n".join(page))
        logger.success("findings → {}", findings)
    return loosened


def _over_bar(measured: dict, max_ulp: float) -> int:
    """Variants over an absolute bar — a kernel always wrong moves nothing but still fails."""
    failed = 0
    for dtype, ops in measured.items():
        for op, variants in ops.items():
            for variant, s in variants.items():
                worst = s.get("max_ulp")
                unscorable = s.get("defects", 0) + s.get("unflushed", 0)
                try:
                    exceeds = float(worst) > max_ulp
                except (TypeError, ValueError):
                    exceeds = False  # no scorable point is not a failure to be under a bar
                if exceeds or unscorable:
                    failed += 1
                    logger.error(
                        "over the {} ULP bar  {}/{}/{}: {}",
                        max_ulp,
                        dtype,
                        op,
                        variant,
                        s.get("verdict", worst),
                    )
    return failed


def _properties(spec: OpSpec, dtype: str, device) -> dict[str, int]:
    """What must hold whatever the reference says — wrongness ULP cannot express."""
    from ttnn_accuracy.measure import sweeps

    args = (spec.bounds[dtype], spec.operands, dtype, spec.layouts[dtype], device)
    checks = {"tiling changed the answer": sweeps.shape_invariance(spec.ttnn_fn, *args)}
    # Not in place: `add_(a, a)` is one buffer as destination and source, and hung the device.
    if spec.operands > 1 and not spec.name.endswith("_"):
        checks["aliased operands changed the answer"] = sweeps.aliased_operands(spec.ttnn_fn, *args)
    if spec.name.endswith("_"):
        checks["did not write in place"] = sweeps.writes_in_place(spec.ttnn_fn, *args)
    if spec.operands == 2:
        checks["stopped commuting"] = sweeps.commutes(
            spec.ttnn_fn, spec.golden_fn, spec.bounds[dtype], dtype, spec.layouts[dtype], device
        )
    return {why: n for why, n in checks.items() if n}


def _check_properties(specs: list[OpSpec], dtypes: list[str], device) -> int:
    """Variants violating at least one property, counted once each."""
    failed = 0
    for spec in specs:
        for dtype in (d for d in dtypes if d in spec.layouts):
            if broken := _properties(spec, dtype, device):
                failed += 1
                for why, n in broken.items():
                    logger.error(
                        "{}: {} elements  {}/{}/{}", why, n, dtype, spec.name, spec.variant
                    )
    return failed


def _check_perf(specs: list[OpSpec], dtypes: list[str], arch: str, device) -> int:
    """Time the same ops, diffing only when the published timings came from this host."""
    from tempfile import TemporaryDirectory

    from ttnn_accuracy.measure.perf import measure_perf, perf_diff
    from ttnn_accuracy.paths import PERF_DIR

    with TemporaryDirectory(prefix="ttnn-perf-") as tmp:
        now = Path(tmp) / "now.json"
        untimed = measure_perf(specs, dtypes, arch, out=now, device=device)
        published = PERF_DIR / f"{arch}.json"
        if not published.exists():
            logger.info("no published timings for {} — nothing to compare against", arch)
            return untimed
        try:
            return untimed + perf_diff(published, now)
        except SystemExit as exc:  # different host: the guard in perf_diff, not an error here
            logger.warning("{}", exc)
            return untimed


def _measure(spec: OpSpec, dtype: str, arch: str, out_root: Path, device) -> str:
    """Empty when the CSV was written, else why not — never raises, one variant is not the run."""
    try:
        return _measure_one(spec, dtype, arch, out_root, device)
    except Exception as exc:
        logger.exception("  {}/{} {} failed", spec.name, spec.variant, dtype)
        return f"{type(exc).__name__}: {reason(exc, 160)}"


def _measure_one(spec: OpSpec, dtype: str, arch: str, out_root: Path, device) -> str:
    sweep = SWEEPS.get((spec.operands, dtype))
    if sweep is None:
        logger.error("  no {} sweep for {} operands ({})", dtype, spec.operands, spec.name)
        return f"no {dtype} sweep for {spec.operands} operands"

    lo, hi = spec.bounds[dtype]
    layout = spec.layouts[dtype]
    df = sweep(spec.ttnn_fn, spec.golden_fn, device, lo, hi, layout)
    if df is None:
        logger.error("  no valid {} inputs for {}/{}", dtype, spec.name, spec.variant)
        return f"no valid {dtype} inputs in its domain"

    try:
        extra = specials(spec.ttnn_fn, spec.golden_fn, spec.operands, dtype, layout, device)
        with warnings.catch_warnings():
            # Specials carry no ULP, and pandas deprecation-warns on their all-NA columns.
            warnings.simplefilter("ignore", FutureWarning)
            df = pd.concat([df, extra], ignore_index=True)
    except Exception as exc:  # a third-party golden may balk at inf or NaN; the sweep stands
        logger.warning("  no special-value rows for {}: {}", spec.name, exc)

    df["op"], df["variant"], df["dtype"] = spec.name, spec.variant, dtype
    df["layout"] = layout  # a different layout is a different kernel, so it is a column
    path = write_result(df, out_root, arch, dtype, spec.name, spec.variant)
    logger.success("  {} rows → {}", len(df), path)
    return ""
