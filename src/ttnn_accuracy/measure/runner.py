"""Orchestration: walk specs × dtypes, sweep each, write one CSV per variant."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from loguru import logger

from ttnn_accuracy.measure.device import open_device
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
) -> int:
    """Returns the number of variants that produced no CSV — the process exit code.

    Each is also recorded beside the data with its reason: an op that probes but will not
    sweep is a finding, and a reader who cannot see it reads its absence as an oversight.

    A variant already measured on this same tt-metal build is skipped, so an interrupted
    run resumes where it stopped. Hours of measurement should not be lost to a device
    reset at op 150, and a build that differs invalidates everything anyway.
    """
    failures: dict[str, dict[str, str]] = {}
    names = sorted({spec.name for spec in specs})

    with open_device(device_id) as device:
        check_arch(device, arch)
        meta = describe_run(device, arch, names, dtypes)
        done = measured_at(out_root, arch, dtypes, meta.tt_metal_commit)
        write_run(meta, out_root)
        for spec in specs:
            logger.info("{} ({}) [{}]", spec.name, spec.category, spec.variant)
            for dtype in dtypes:
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


def _measure(spec: OpSpec, dtype: str, arch: str, out_root: Path, device) -> str:
    """Empty when the CSV was written, else why it was not. Never raises.

    One variant must not end the run: whatever it does short of killing the process is
    that variant's recorded result, and the other 197 still get measured.
    """
    try:
        return _measure_one(spec, dtype, arch, out_root, device)
    except Exception as exc:
        logger.exception("  {}/{} {} failed", spec.name, spec.variant, dtype)
        return f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:160]}"


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
            # The specials rows carry no ULP by design, and pandas deprecation-warns on
            # concatenating their all-NA columns; the dtypes are float on both sides.
            warnings.simplefilter("ignore", FutureWarning)
            df = pd.concat([df, extra], ignore_index=True)
    except Exception as exc:  # a third-party golden may balk at inf or NaN; the sweep stands
        logger.warning("  no special-value rows for {}: {}", spec.name, exc)

    df["op"], df["variant"], df["dtype"] = spec.name, spec.variant, dtype
    # Layout is chosen per op by `probe`, and a different layout is a different kernel,
    # so it belongs beside dtype rather than being inferable only from the manifest.
    df["layout"] = layout
    path = write_result(df, out_root, arch, dtype, spec.name, spec.variant)
    logger.success("  {} rows → {}", len(df), path)
    return ""
