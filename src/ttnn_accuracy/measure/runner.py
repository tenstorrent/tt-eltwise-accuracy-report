"""Orchestration: walk specs × dtypes, sweep each, write one CSV per variant."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

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
    """Returns the number of variants that produced no CSV — the process exit code.

    Each is recorded beside the data with its reason, and anything already measured on
    this build is skipped, so a device reset at op 150 costs one op rather than the night.
    """
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
    specs: list[OpSpec], dtypes: list[str], arch: str, device_id: int = 0, perf: bool = False
) -> int:
    """Measure these ops now and say what moved against the published report.

    The kernel author's loop. Accuracy is deterministic, so the baseline is read from git
    rather than re-measured, and only the ops named here go near the device.
    """
    from tempfile import TemporaryDirectory

    from ttnn_accuracy.paths import INDEX_FILE
    from ttnn_accuracy.report.charts import score_csv
    from ttnn_accuracy.report.compare import diff

    if not INDEX_FILE.exists():
        raise SystemExit(f"no baseline at {INDEX_FILE} — nothing to check against")
    baseline = json.loads(INDEX_FILE.read_text())

    # One device for both passes: opening it costs about as long as measuring one op.
    with open_device(device_id) as device, TemporaryDirectory(prefix="ttnn-check-") as tmp:
        empty = measure(specs, dtypes, arch, Path(tmp), device_id, archive=False, device=device)
        timing = _check_perf(specs, dtypes, arch, device) if perf else 0
        candidate = {arch: {}}
        for csv in sorted(Path(tmp).glob(f"{arch}/*/*/*.csv")):
            dtype, op, variant = csv.parts[-3], csv.parts[-2], csv.stem
            if scored := score_csv(csv, op, variant):
                candidate[arch].setdefault(dtype, {}).setdefault(op, {})[variant] = scored[0]

    # Only what was just measured: the baseline holds 766 variants and all but these are
    # unchanged by construction, so diffing the whole thing would bury the answer.
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
    # One line, last: a kernel author wants the answer, not to read a diff for it.
    moved = ", ".join(f"{len(rows)} {name}" for name, rows in buckets.items() if rows)
    say = logger.error if buckets["regressed"] else logger.success
    say("{} variant(s) measured — {}", len(measured), moved or "nothing moved")
    return empty + timing + len(buckets["regressed"])


def _check_perf(specs: list[OpSpec], dtypes: list[str], arch: str, device) -> int:
    """Time the same ops, and diff against the published timings when this is their host.

    Unlike accuracy, a published timing from another machine is not a baseline, so when the
    hosts differ the numbers are printed and nothing is compared.
    """
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
    """Empty when the CSV was written, else why it was not. Never raises: one variant
    must not end the run."""
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
