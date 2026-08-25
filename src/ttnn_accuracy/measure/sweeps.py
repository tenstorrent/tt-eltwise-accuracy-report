"""Input sweeps. One row per bf16 value; one row per group of fp32 values."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from loguru import logger

from ttnn_accuracy.measure import metrics

MIN_NORMAL = 2**-126
TILE_WIDTH = 2**7
FP32_BLOCK = 2**6 * 2**9 * TILE_WIDTH


def _in_domain(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Non-subnormal inputs within [lo, hi]. Hardware flushes subnormals to zero."""
    return ((x == 0.0) | (x.abs() >= MIN_NORMAL)) & (x >= lo) & (x <= hi)


def _tile(x: torch.Tensor) -> torch.Tensor:
    """Reshape to TILE_WIDTH columns, repeating inputs to fill a narrow final row."""
    pad = -x.numel() % TILE_WIDTH
    if pad:
        x = torch.cat([x, x.repeat(pad // x.numel() + 1)[:pad]])
    return x.reshape(-1, TILE_WIDTH)


def _golden(golden_fn: Callable, x: torch.Tensor) -> torch.Tensor:
    x64 = x.to(torch.float64)
    with torch.no_grad():
        return golden_fn(x64, out=torch.zeros_like(x64))


def _on_device(ttnn_fn: Callable, x: torch.Tensor, dtype, device) -> torch.Tensor:
    import ttnn

    tensor = ttnn.from_torch(x, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(ttnn_fn(tensor))


def sweep_bf16(ttnn_fn, golden_fn, device, lo: float, hi: float) -> pd.DataFrame | None:
    """Every bf16 value in [lo, hi] — all 65536 codes, one row each."""
    import ttnn

    codes = np.arange(2**16, dtype=np.uint32).astype(np.uint16).view(np.int16)
    x = torch.from_numpy(codes).view(torch.bfloat16)
    x = x[_in_domain(x.to(torch.float32), lo, hi)]
    if not x.numel():
        return None
    x = _tile(x)
    return metrics.compare(
        x, _golden(golden_fn, x), _on_device(ttnn_fn, x, ttnn.bfloat16, device), group_size=1
    )


def sweep_fp32(ttnn_fn, golden_fn, device, lo: float, hi: float) -> pd.DataFrame | None:
    """The whole fp32 code space in blocks; each row is the worst error of a group."""
    import ttnn

    blocks = 2**32 // FP32_BLOCK
    codes = torch.arange(FP32_BLOCK, dtype=torch.int64)
    frames = []
    for i in range(blocks):
        x = codes.to(torch.int32).view(torch.float32)
        x = x[_in_domain(x, lo, hi)]
        codes += FP32_BLOCK
        if not x.numel():
            continue
        x = _tile(x)
        frames.append(
            metrics.compare(
                x,
                _golden(golden_fn, x),
                _on_device(ttnn_fn, x, ttnn.float32, device),
                group_size=TILE_WIDTH,
            )
        )
        logger.debug("fp32 block {}/{}", i + 1, blocks)
    return pd.concat(frames, ignore_index=True) if frames else None


SWEEPS = {"bf16": sweep_bf16, "fp32": sweep_fp32}
