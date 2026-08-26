"""Input sweeps. One row per bf16 value; one row per group of fp32 values."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from loguru import logger

from ttnn_accuracy.measure import metrics
from ttnn_accuracy.measure.metrics import MIN_NORMAL

TILE_WIDTH = 2**7
FP32_BLOCK = 2**6 * 2**9 * TILE_WIDTH
B_CHUNK = 2**7  # second operands per dispatch, matching ttnn-eltwise-op-tester's batch
TERNARY_STRIDE = 2**9  # keeps a 3-operand sweep the same size as a 2-operand one


def _bf16_values(lo: float, hi: float) -> torch.Tensor:
    """Every bf16 bit pattern the sweep may use, in code order."""
    codes = np.arange(2**16, dtype=np.uint32).astype(np.uint16).view(np.int16)
    values = torch.from_numpy(codes).view(torch.bfloat16)
    return values[_in_domain(values.to(torch.float32), lo, hi)]


def _in_domain(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Non-subnormal inputs within [lo, hi]. Hardware flushes subnormals to zero."""
    return ((x == 0.0) | (x.abs() >= MIN_NORMAL)) & (x >= lo) & (x <= hi)


def _tile(x: torch.Tensor) -> torch.Tensor:
    """Reshape to TILE_WIDTH columns, repeating inputs to fill a narrow final row."""
    pad = -x.numel() % TILE_WIDTH
    if pad:
        x = torch.cat([x, x.repeat(pad // x.numel() + 1)[:pad]])
    return x.reshape(-1, TILE_WIDTH)


def _golden(golden_fn: Callable, *operands: torch.Tensor) -> torch.Tensor:
    """fp64 reference. The unary path passes `out=` so torch writes into a wide buffer."""
    wide = [t.to(torch.float64) for t in operands]
    with torch.no_grad():
        if len(wide) == 1:
            return golden_fn(wide[0], out=torch.zeros_like(wide[0]))
        return golden_fn(*wide)


def _on_device(ttnn_fn: Callable, *operands: torch.Tensor, dtype, device) -> torch.Tensor:
    import ttnn

    tensors = [
        ttnn.from_torch(t, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT) for t in operands
    ]
    return ttnn.to_torch(ttnn_fn(*tensors))


def sweep_bf16(ttnn_fn, golden_fn, device, lo: float, hi: float) -> pd.DataFrame | None:
    """Every bf16 value in [lo, hi] — all 65536 codes, one row each."""
    import ttnn

    x = _bf16_values(lo, hi)
    if not x.numel():
        return None
    measured = x.numel()
    x = _tile(x)
    df = metrics.compare(
        x,
        _golden(golden_fn, x),
        _on_device(ttnn_fn, x, dtype=ttnn.bfloat16, device=device),
        group_size=1,
    )
    # Drop the rows _tile repeated to fill the last tile; they are real inputs measured
    # twice, and counting them twice would skew n_inputs and the mean.
    return df.iloc[:measured]


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
                _on_device(ttnn_fn, x, dtype=ttnn.float32, device=device),
                group_size=TILE_WIDTH,
            )
        )
        logger.debug("fp32 block {}/{}", i + 1, blocks)
    return pd.concat(frames, ignore_index=True) if frames else None


def _multi_operand(ttnn_fn, golden_fn, device, values, batches, total) -> pd.DataFrame:
    """One row per first operand: the pairing of the others that hurt it most.

    Every column of a row describes the same point. Reducing each column independently
    would label a row `overflow` because some far-away operand overflowed, which says
    nothing about the error reported beside it.
    """
    import ttnn

    n = values.numel()
    col = np.arange(n)
    best = np.full(n, -np.inf)  # -inf, not NaN, so the first comparison always takes
    worst: dict[str, np.ndarray] = {}

    for i, (rows, operands) in enumerate(batches, start=1):
        tiled = [_tile(o) for o in operands]
        e = metrics.errors(
            _golden(golden_fn, *tiled),
            _on_device(ttnn_fn, *tiled, dtype=ttnn.bfloat16, device=device),
        )
        e = {k: v[: rows * n].reshape(rows, n) for k, v in e.items()}

        # Rank by ULP, treating undefined as lowest, so a defined point always wins.
        scored = np.nan_to_num(e["ulp_error"], nan=-np.inf)
        pos = scored.argmax(axis=0)
        top = scored[pos, col]
        beats = top > best
        for key, value in e.items():
            picked = value[pos, col]
            worst[key] = picked if key not in worst else np.where(beats, picked, worst[key])
        best = np.where(beats, top, best)
        logger.debug("operand batch {}/{}", i, total)

    return pd.DataFrame(
        {
            "x": values.to(torch.float32).numpy(),
            "y": worst["y"],
            "y_ref": worst["y_ref"],
            "ulp_error": worst["ulp_error"],
            "abs_error": worst["abs_error"],
            "outcome": np.take(metrics.OUTCOMES, worst["outcome"]),
        }
    )


def sweep_binary_bf16(ttnn_fn, golden_fn, device, lo: float, hi: float) -> pd.DataFrame | None:
    """Every bf16 pair in [lo, hi]² — exhaustive, as in ttnn-eltwise-op-tester.

    65536² pairs stream in chunks of second operands. Reducing over b alone, rather than
    over blocks of a too, keeps a row per a-value: a binary op costs a unary op's storage.
    """
    values = _bf16_values(lo, hi)
    n = values.numel()
    if not n:
        return None

    def batches():
        a = values.repeat(B_CHUNK)
        for i in range(0, n, B_CHUNK):
            b = values[i : i + B_CHUNK]
            yield b.numel(), [a[: b.numel() * n], b.repeat_interleave(n)]

    return _multi_operand(ttnn_fn, golden_fn, device, values, batches(), -(-n // B_CHUNK))


def sweep_ternary_bf16(ttnn_fn, golden_fn, device, lo: float, hi: float) -> pd.DataFrame | None:
    """Exhaustive in the first operand, strided in the other two.

    65536³ is 2.8e14 points, so the second and third operands take every TERNARY_STRIDE-th
    bf16 code instead of all of them. The stride is uniform over the code space, so it
    samples every exponent rather than clustering near zero.
    """
    values = _bf16_values(lo, hi)
    n = values.numel()
    if not n:
        return None
    sampled = values[::TERNARY_STRIDE]
    rows = sampled.numel()

    def batches():
        a = values.repeat(rows)
        b = sampled.repeat_interleave(n)
        for c in sampled:
            yield rows, [a, b, c.expand(rows * n)]

    return _multi_operand(ttnn_fn, golden_fn, device, values, batches(), rows)


SWEEPS = {
    (1, "bf16"): sweep_bf16,
    (1, "fp32"): sweep_fp32,
    (2, "bf16"): sweep_binary_bf16,
    (3, "bf16"): sweep_ternary_bf16,
}
