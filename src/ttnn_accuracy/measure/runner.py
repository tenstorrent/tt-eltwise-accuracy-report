"""Orchestration: walk specs × dtypes, sweep each, write one CSV per variant."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from loguru import logger

from ttnn_accuracy.measure.device import open_device
from ttnn_accuracy.measure.schema import check_arch, describe_run
from ttnn_accuracy.measure.store import write_result, write_run
from ttnn_accuracy.measure.sweeps import SWEEPS, specials
from ttnn_accuracy.ops.plan import OpSpec


def measure(
    specs: list[OpSpec],
    dtypes: list[str],
    arch: str,
    out_root: Path,
    device_id: int = 0,
) -> int:
    """Returns the number of variants that produced no CSV — the process exit code."""
    failed = 0
    names = sorted({spec.name for spec in specs})

    with open_device(device_id) as device:
        check_arch(device, arch)
        write_run(describe_run(device, arch, names, dtypes), out_root)
        for spec in specs:
            logger.info("{} ({}) [{}]", spec.name, spec.category, spec.variant)
            for dtype in dtypes:
                failed += _measure(spec, dtype, arch, out_root, device)

    if failed:
        logger.error("{} variant(s) produced no data", failed)
    return failed


def _measure(spec: OpSpec, dtype: str, arch: str, out_root: Path, device) -> int:
    sweep = SWEEPS.get((spec.operands, dtype))
    if sweep is None:
        logger.error("  no {} sweep for {} operands ({})", dtype, spec.operands, spec.name)
        return 1

    lo, hi = spec.bounds[dtype]
    layout = spec.layouts[dtype]
    try:
        df = sweep(spec.ttnn_fn, spec.golden_fn, device, lo, hi, layout)
    except Exception:
        logger.exception("  {}/{} {} failed", spec.name, spec.variant, dtype)
        return 1

    if df is None:
        logger.error("  no valid {} inputs for {}/{}", dtype, spec.name, spec.variant)
        return 1

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
    return 0
