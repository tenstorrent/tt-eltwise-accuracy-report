"""Orchestration: walk ops × dtypes × variants, sweep each, write one CSV per variant."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from ttnn_accuracy.measure.device import open_device
from ttnn_accuracy.measure.schema import check_arch, describe_run
from ttnn_accuracy.measure.store import write_result, write_run
from ttnn_accuracy.measure.sweeps import SWEEPS
from ttnn_accuracy.ops.registry import get_op, get_registry, variant_slug


def measure(
    op_names: list[str],
    dtypes: list[str],
    arch: str,
    out_root: Path,
    device_id: int = 0,
) -> int:
    """Returns the number of variants that produced no CSV — the process exit code."""
    registry = get_registry()
    # Report every bad name before claiming the device
    known, failed = [], 0
    for name in op_names:
        if name in registry:
            known.append(name)
        else:
            logger.error("unknown op: {}", name)
            failed += 1

    with open_device(device_id) as device:
        check_arch(device, arch)
        write_run(describe_run(device, arch, known, dtypes))
        for name in known:
            entry = get_op(name)
            logger.info("{} ({})", name, entry.category)
            for dtype in dtypes:
                for variant in entry.variants:
                    failed += _measure_variant(entry, variant, dtype, arch, out_root, device)

    if failed:
        logger.error("{} variant(s) produced no data", failed)
    return failed


def _measure_variant(entry, variant, dtype: str, arch: str, out_root: Path, device) -> int:
    slug = variant_slug(variant.params_desc)
    logger.info("  [{}] {}", dtype, slug)
    try:
        df = SWEEPS[dtype](
            variant.ttnn_fn,
            variant.golden_fn,
            device,
            entry.input_range.lo,
            entry.input_range.hi,
        )
    except Exception:
        logger.exception("  {}/{} failed", entry.name, slug)
        return 1

    if df is None:
        logger.error("  no valid {} inputs for {}/{}", dtype, entry.name, slug)
        return 1

    df["op"], df["variant"], df["dtype"] = entry.name, slug, dtype
    path = write_result(df, out_root, arch, dtype, entry.name, slug)
    logger.success("  {} rows → {}", len(df), path)
    return 0
