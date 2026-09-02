"""Input sweeps, one per arity and dtype, each returning one row per first operand.

A unary row is a single measured point. A binary or ternary row is the pairing of the
other operands that produced the worst defined ULP for that first operand, so a row is
still one point and every column of it describes the same one.
"""

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
# Worst-of-group per row: keeps the max exact and an fp32 op the weight of a bf16 one.
FP32_GROUP = 2**16
B_CHUNK = 2**7  # second operands per dispatch, matching ttnn-eltwise-op-tester's batch
TERNARY_STRIDE = 2**9  # keeps a 3-operand sweep the same size as a 2-operand one
FP32_SAMPLE_SEED = 0  # fixed: the fp32 pair sample must not move between releases
DTYPE = {"bf16": "bfloat16", "fp32": "float32"}  # ttnn and torch spell these identically
LAYOUT = {"tile": "TILE_LAYOUT", "row_major": "ROW_MAJOR_LAYOUT"}


def _bf16_values(lo: float, hi: float) -> torch.Tensor:
    """Every bf16 bit pattern the sweep may use, in code order."""
    codes = np.arange(2**16, dtype=np.uint32).astype(np.uint16).view(np.int16)
    values = torch.from_numpy(codes).view(torch.bfloat16)
    return values[_in_domain(values.to(torch.float32), lo, hi)]


def _fp32_sample(lo: float, hi: float) -> torch.Tensor:
    """One fp32 value per bf16 cell, low mantissa bits from a fixed seed.

    Striding by 2**16 alone would zero the bits that drive rounding; the seed is fixed so a
    change in the error is the kernel's, not the draw's.
    """
    high = np.arange(2**16, dtype=np.uint32) << 16
    low = np.random.default_rng(FP32_SAMPLE_SEED).integers(0, 2**16, 2**16, dtype=np.uint32)
    values = torch.from_numpy((high | low).view(np.int32)).view(torch.float32)
    return values[_in_domain(values, lo, hi)]


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


def _on_device(
    ttnn_fn: Callable, *operands: torch.Tensor, dtype: str, layout: str = "tile", device
) -> torch.Tensor:
    """Resolves ttnn's dtype and layout here so callers never import ttnn themselves."""
    import ttnn

    ttnn_dtype, ttnn_layout = getattr(ttnn, DTYPE[dtype]), getattr(ttnn, LAYOUT[layout])
    tensors = [
        ttnn.from_torch(t, device=device, dtype=ttnn_dtype, layout=ttnn_layout) for t in operands
    ]
    result = ttnn_fn(*tensors)
    host = ttnn.to_torch(result)
    # Not left to GC: DRAM fragments faster than collection runs, and block times doubled.
    for t in (result, *tensors):
        if t.is_allocated():
            ttnn.deallocate(t)
    return host


def capabilities(ttnn_fn: Callable, operands: int, device) -> tuple[dict[str, str], str]:
    """Which layout each dtype works in, and why nothing did.

    ttnn declares its constraints as TT_FATAL inside the C++ op, so calling is the only
    way to learn them; the answer is recorded rather than rediscovered per run.
    """
    found, why = {}, ""
    for dtype, name in DTYPE.items():
        tile = torch.ones(TILE_WIDTH, TILE_WIDTH, dtype=getattr(torch, name))
        for layout in LAYOUT:
            try:
                _on_device(ttnn_fn, *[tile] * operands, dtype=dtype, layout=layout, device=device)
            except Exception as exc:
                why = why or str(exc).strip().splitlines()[0][:200]
                continue
            found[dtype] = layout
            break
    return found, why


SPECIAL_VALUES = (0.0, -0.0, float("inf"), float("-inf"), float("nan"), MIN_NORMAL, -MIN_NORMAL)


def specials(ttnn_fn, golden_fn, operands: int, dtype: str, layout: str, device) -> pd.DataFrame:
    """The op at the values every sweep filters out, with ones in the other operands.

    No ULP: the distance between two infinities is not a quantity, so the page prints
    device beside golden and the `special` outcome keeps them out of every statistic.
    """
    x = torch.tensor(SPECIAL_VALUES, dtype=getattr(torch, DTYPE[dtype]))
    tiled = [_tile(t) for t in (x, *[torch.ones_like(x)] * (operands - 1))]
    y_ref = _golden(golden_fn, *tiled)
    y = _on_device(ttnn_fn, *tiled, dtype=dtype, layout=layout, device=device)
    k = len(SPECIAL_VALUES)
    return pd.DataFrame(
        {
            "x": np.array(SPECIAL_VALUES, dtype=np.float32),
            "y": y.flatten()[:k].to(torch.float32).numpy(),
            "y_ref": y_ref.flatten()[:k].to(torch.float32).numpy(),
            "ulp_error": np.nan,
            "abs_error": np.nan,
            "outcome": "special",
        }
    )


def sweep_bf16(
    ttnn_fn, golden_fn, device, lo: float, hi: float, layout: str
) -> pd.DataFrame | None:
    """Every bf16 value in [lo, hi] — all 65536 codes, one row each."""
    x = _bf16_values(lo, hi)
    if not x.numel():
        return None
    measured = x.numel()
    x = _tile(x)
    df = metrics.compare(
        x,
        _golden(golden_fn, x),
        _on_device(ttnn_fn, x, dtype="bf16", layout=layout, device=device),
        group_size=1,
    )
    return df.iloc[:measured]  # drop _tile's padding: real inputs, counted twice


def sweep_fp32(
    ttnn_fn, golden_fn, device, lo: float, hi: float, layout: str
) -> pd.DataFrame | None:
    """The whole fp32 code space in blocks; each row is the worst error of a group."""
    blocks = 2**32 // FP32_BLOCK
    codes = torch.arange(FP32_BLOCK, dtype=torch.int64)
    frames = []
    for i in range(blocks):
        x = codes.to(torch.int32).view(torch.float32)
        x = x[_in_domain(x, lo, hi)]
        codes += FP32_BLOCK
        if not x.numel():
            continue
        # Repeats of a real input, which cannot move a worst-of-group row.
        if pad := -x.numel() % FP32_GROUP:
            x = torch.cat([x, x[-1].expand(pad)])
        x = _tile(x)
        frames.append(
            metrics.compare(
                x,
                _golden(golden_fn, x),
                _on_device(ttnn_fn, x, dtype="fp32", layout=layout, device=device),
                group_size=FP32_GROUP,
            )
        )
        logger.debug("fp32 block {}/{}", i + 1, blocks)
    return pd.concat(frames, ignore_index=True) if frames else None


def _multi_operand(
    ttnn_fn, golden_fn, device, values, batches, total, dtype, layout
) -> pd.DataFrame:
    """One row per first operand: the pairing of the others that hurt it most, whole."""
    n = values.numel()
    col = np.arange(n)
    best = np.full(n, -np.inf)  # -inf, not NaN, so the first comparison always takes
    worst: dict[str, np.ndarray] = {}

    for i, (rows, operands) in enumerate(batches, start=1):
        tiled = [_tile(o) for o in operands]
        e = metrics.errors(
            _golden(golden_fn, *tiled),
            _on_device(ttnn_fn, *tiled, dtype=dtype, layout=layout, device=device),
        )
        e = {k: v[: rows * n].reshape(rows, n) for k, v in e.items()}
        # Without the partners `x` alone does not identify the point, so nothing reproduces.
        for k, operand in enumerate(operands[1:], start=2):
            e[f"x{k}"] = operand[: rows * n].to(torch.float32).numpy().reshape(rows, n)

        scored = np.nan_to_num(e["ulp_error"], nan=-np.inf)  # a defined point always wins
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
            **{k: v for k, v in worst.items() if k.startswith("x")},
            "y": worst["y"],
            "y_ref": worst["y_ref"],
            "ulp_error": worst["ulp_error"],
            "abs_error": worst["abs_error"],
            "outcome": np.take(metrics.OUTCOMES, worst["outcome"]),
        }
    )


def _sweep_pairs(ttnn_fn, golden_fn, device, values, dtype, layout) -> pd.DataFrame | None:
    """Cross every first operand with every value in `values`, in chunks of the second."""
    n = values.numel()
    if not n:
        return None

    def batches():
        a = values.repeat(B_CHUNK)
        for i in range(0, n, B_CHUNK):
            b = values[i : i + B_CHUNK]
            yield b.numel(), [a[: b.numel() * n], b.repeat_interleave(n)]

    total = -(-n // B_CHUNK)
    return _multi_operand(ttnn_fn, golden_fn, device, values, batches(), total, dtype, layout)


def sweep_binary_bf16(
    ttnn_fn, golden_fn, device, lo: float, hi: float, layout: str
) -> pd.DataFrame | None:
    """Every bf16 pair in [lo, hi]², streamed in chunks of b — one row per a-value."""
    return _sweep_pairs(ttnn_fn, golden_fn, device, _bf16_values(lo, hi), "bf16", layout)


def sweep_binary_fp32(
    ttnn_fn, golden_fn, device, lo: float, hi: float, layout: str
) -> pd.DataFrame | None:
    """Sampled fp32 pairs, one per bf16 cell: the maximum is a lower bound, and says so."""
    return _sweep_pairs(ttnn_fn, golden_fn, device, _fp32_sample(lo, hi), "fp32", layout)


def _sweep_triples(ttnn_fn, golden_fn, device, values, dtype, layout) -> pd.DataFrame | None:
    """First operand exhaustive, the others strided — values³ is 2.8e14 even in bf16.

    The stride is uniform over the code space, so it samples every exponent.
    """
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

    return _multi_operand(ttnn_fn, golden_fn, device, values, batches(), rows, dtype, layout)


def sweep_ternary_bf16(
    ttnn_fn, golden_fn, device, lo: float, hi: float, layout: str
) -> pd.DataFrame | None:
    return _sweep_triples(ttnn_fn, golden_fn, device, _bf16_values(lo, hi), "bf16", layout)


def sweep_ternary_fp32(
    ttnn_fn, golden_fn, device, lo: float, hi: float, layout: str
) -> pd.DataFrame | None:
    return _sweep_triples(ttnn_fn, golden_fn, device, _fp32_sample(lo, hi), "fp32", layout)


# How each incomplete sweep samples — one count cannot say, the ternaries mix the two.
SAMPLED = {
    (2, "fp32"): "both operands take 65,536 of the 2³² fp32 values, drawn once from a fixed seed",
    (3, "bf16"): (
        f"the first operand is exhaustive; the second and third take every "
        f"{TERNARY_STRIDE}th bf16 code, {2**16 // TERNARY_STRIDE} values each"
    ),
    (3, "fp32"): (
        f"the first operand takes 65,536 of the 2³² fp32 values, drawn once from a fixed "
        f"seed; the second and third take every {TERNARY_STRIDE}th of that sample"
    ),
}

SWEEPS = {
    (1, "bf16"): sweep_bf16,
    (1, "fp32"): sweep_fp32,
    (2, "bf16"): sweep_binary_bf16,
    (2, "fp32"): sweep_binary_fp32,
    (3, "bf16"): sweep_ternary_bf16,
    (3, "fp32"): sweep_ternary_fp32,
}
