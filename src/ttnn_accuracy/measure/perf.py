"""Kernel timing over one resident tensor; every record carries its host, and only one host subtracts."""

from __future__ import annotations

import json
import platform
import statistics
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns

import torch
from loguru import logger

from ttnn_accuracy.config import BATCH, FINITE, NOISE_PCT, REPEATS, SIDE, WARMUP
from ttnn_accuracy.measure.device import session
from ttnn_accuracy.measure.schema import check_arch, describe_run
from ttnn_accuracy.measure.sweeps import DTYPE, LAYOUT
from ttnn_accuracy.ops.plan import OpSpec
from ttnn_accuracy.paths import PERF_DIR, RUNS_KEY

ELEMENTS = SIDE**2


def _host() -> str:
    """The machine, not the booking — a reservation id renames the host under you."""
    return platform.node().split("-special-")[0]


def _stats(samples: list[float]) -> dict[str, float | int]:
    """Median, not mean, and `spread_pct` says how much to trust the rest of the row."""
    ordered = sorted(samples)
    median = statistics.median(ordered)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    return {
        "us_median": round(median, 3),
        "us_min": round(ordered[0], 3),
        "us_p90": round(p90, 3),
        "spread_pct": round(100 * (p90 - ordered[0]) / median, 1) if median else 0.0,
        "melem_per_s": round(ELEMENTS / median, 1) if median else 0.0,  # elements per us
        "elements": ELEMENTS,
        "dispatches": len(samples),
    }


def _operand(bounds: tuple[float, float], dtype: str) -> torch.Tensor:
    """Values across the op's domain — ones would hit a fast path, random would not repeat."""
    lo, hi = bounds
    lo, hi = max(lo, FINITE[0]), min(hi, FINITE[1])
    return torch.linspace(lo, hi, ELEMENTS, dtype=getattr(torch, DTYPE[dtype])).reshape(SIDE, SIDE)


def time_op(spec: OpSpec, dtype: str, device) -> dict[str, float | int]:
    """Time one variant. Raises whatever the op raises; the caller records it."""
    import ttnn

    ttnn_dtype = getattr(ttnn, DTYPE[dtype])
    ttnn_layout = getattr(ttnn, LAYOUT[spec.layouts[dtype]])
    host = _operand(spec.bounds[dtype], dtype)
    tensors = [
        ttnn.from_torch(host, device=device, dtype=ttnn_dtype, layout=ttnn_layout)
        for _ in range(spec.operands)
    ]

    # By buffer, not object: `is` misses an in-place result and the allocator segfaults.
    operands = {t.buffer_address() for t in tensors}

    def release(t) -> None:
        if t.is_allocated() and t.buffer_address() not in operands:
            ttnn.deallocate(t)

    try:
        for _ in range(WARMUP):
            release(spec.ttnn_fn(*tensors))
        ttnn.synchronize_device(device)

        samples = []
        for _ in range(REPEATS):
            start = perf_counter_ns()
            batch = [spec.ttnn_fn(*tensors) for _ in range(BATCH)]
            ttnn.synchronize_device(device)  # dispatch is async; without this it times the queue
            samples.append((perf_counter_ns() - start) / 1000 / BATCH)
            for out in batch:
                release(out)
    finally:
        for t in tensors:
            if t.is_allocated():
                ttnn.deallocate(t)
    return _stats(samples)


def measure_perf(
    specs: list[OpSpec],
    dtypes: list[str],
    arch: str,
    device_id: int = 0,
    out: Path | None = None,
    device=None,
) -> int:
    """Returns the number of variants that produced no timing — the process exit code."""
    results: dict = {}
    failed = 0

    path = out or PERF_DIR / f"{arch}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with session(device_id, device) as device:
        check_arch(device, arch)
        meta = describe_run(device, arch, sorted({s.name for s in specs}), dtypes)
        # The host is part of the measurement: only this field says two rows are comparable.
        results[RUNS_KEY] = asdict(meta) | {"host": _host()}
        for spec in specs:
            for dtype in dtypes:
                if dtype not in spec.layouts:  # the probe found this arch rejects it
                    continue
                try:
                    row = time_op(spec, dtype, device)
                except Exception as exc:
                    logger.warning("  {}/{} {}: {}", spec.name, spec.variant, dtype, exc)
                    failed += 1
                    continue
                results.setdefault(dtype, {}).setdefault(spec.name, {})[spec.variant] = row
                # Rewritten per op: a segfault is outlived, and earlier timings still count.
                path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
                logger.info(
                    "{} [{}] {}: {} us, {} Melem/s (spread {}%)",
                    spec.name,
                    spec.variant,
                    dtype,
                    row["us_median"],
                    row["melem_per_s"],
                    row["spread_pct"],
                )

    logger.success("timings → {}", path)
    if failed:
        logger.error("{} variant(s) produced no timing", failed)
    return failed


def _rows(data: dict) -> dict[tuple[str, str, str], float]:
    """Keyed on the minimum: it moved 3.1% between two runs of one build, the median 7.7%."""
    return {
        (dtype, op, variant): row["us_min"]
        for dtype, ops in data.items()
        if dtype != RUNS_KEY
        for op, variants in ops.items()
        for variant, row in variants.items()
    }


def perf_diff(base_path: Path, cand_path: Path, findings: Path | None = None) -> int:
    """Two timing files → what got slower, beyond the noise. Refuses two hosts."""
    base, cand = json.loads(base_path.read_text()), json.loads(cand_path.read_text())
    hosts = {base[RUNS_KEY]["host"], cand[RUNS_KEY]["host"]}
    if len(hosts) > 1:
        raise SystemExit(f"timings from different hosts are not comparable: {sorted(hosts)}")

    before, after = _rows(base), _rows(cand)
    slower, faster, same = [], [], 0
    for key in sorted(before.keys() & after.keys()):
        was, now = before[key], after[key]
        change = 100 * (now - was) / was if was else 0.0
        if abs(change) <= NOISE_PCT:
            same += 1
        else:
            (slower if change > 0 else faster).append((key, was, now, change))

    page = [
        "# Timings",
        "",
        f"_{base[RUNS_KEY]['tt_metal_commit']} → {cand[RUNS_KEY]['tt_metal_commit']} "
        f"on {cand[RUNS_KEY]['host']}_",
        "",
    ]
    for rows, title in ((slower, "Slower"), (faster, "Faster")):
        if not rows:
            continue
        page += [
            f"## {title} ({len(rows)})",
            "",
            "| Variant | Best was | Best now | Change |",
            "|---|---|---|---|",
        ]
        page += [
            f"| `{dtype}/{op}/{variant}` | {was:.1f} us | {now:.1f} us | {change:+.1f}% |"
            for (dtype, op, variant), was, now, change in sorted(rows, key=lambda r: -abs(r[3]))
        ]
        page.append("")
    page += [f"{same} within ±{NOISE_PCT:g}% — the harness's own reproducibility.", ""]

    text = "\n".join(page)
    logger.info("\n{}", text)
    if findings:
        findings.parent.mkdir(parents=True, exist_ok=True)
        findings.write_text(text)
    return 1 if slower else 0
