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
from ttnn_accuracy.measure.device import reason
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
    ttnn_fn: Callable,
    *operands: torch.Tensor,
    dtype: str,
    layout: str = "tile",
    device,
    read: int | None = None,
    alias: bool = False,
) -> torch.Tensor:
    """Resolves ttnn's dtype and layout here so callers never import ttnn themselves.

    `read` returns that operand after the call rather than the result, which is the only
    way to see whether an in-place op wrote where it promised. `alias` passes one tensor
    as every operand, the overlap no sweep produces.
    """
    import ttnn

    ttnn_dtype, ttnn_layout = getattr(ttnn, DTYPE[dtype]), getattr(ttnn, LAYOUT[layout])
    tensors = [
        ttnn.from_torch(t, device=device, dtype=ttnn_dtype, layout=ttnn_layout) for t in operands
    ]
    if alias:
        tensors = [tensors[0]] * len(tensors)
    result = ttnn_fn(*tensors)
    host = ttnn.to_torch(tensors[read] if read is not None else result)
    # Not left to GC: DRAM fragments faster than collection runs, and block times doubled.
    # By buffer, not by object: an in-place op hands back a fresh wrapper around an
    # operand's buffer, and freeing that buffer twice segfaults the allocator.
    freed: set[int] = set()
    for t in (result, *tensors):
        if t.is_allocated() and t.buffer_address() not in freed:
            freed.add(t.buffer_address())
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
                why = why or reason(exc)
                continue
            # Every layout, not the first: row-major is a different kernel path, and
            # stopping at tile meant nothing ever recorded whether it exists.
            found.setdefault(dtype, []).append(layout)
    return found, why


# Neither dim a multiple of 32, so it pads to 512x160 — 80 tiles over a 64-core grid,
# where every sweep here is 64 tiles, one per core, and splits evenly by construction.
RAGGED = (500, 130)


def _values(bounds: tuple[float, float], dtype: str, n: int) -> torch.Tensor:
    """Finite values across the op's domain — a property holds or it does not, so these
    checks need a spread, not an exhaustive sweep."""
    lo, hi = max(bounds[0], -1e4), min(bounds[1], 1e4)
    return torch.linspace(lo, hi, n, dtype=getattr(torch, DTYPE[dtype]))


def _differing(a: torch.Tensor, b: torch.Tensor) -> int:
    """NaN equals NaN here: two undefined answers agree, and every check below is exact."""
    return int((~((a == b) | (a.isnan() & b.isnan()))).sum())


def shape_invariance(ttnn_fn, bounds, operands: int, dtype: str, layout: str, device) -> int:
    """Elements whose value moved when only the tiling changed. Needs no golden.

    Eltwise has no cross-element interaction, so a result that depends on how the tensor
    was split across cores is a bug in the op, whatever the reference says.
    """
    n = RAGGED[0] * RAGGED[1]
    x = _values(bounds, dtype, n)
    others = [torch.ones_like(x)] * (operands - 1)
    uneven = [t.reshape(*RAGGED) for t in (x, *others)]
    even = [_tile(t) for t in (x, *others)]
    a = _on_device(ttnn_fn, *uneven, dtype=dtype, layout=layout, device=device).flatten()[:n]
    b = _on_device(ttnn_fn, *even, dtype=dtype, layout=layout, device=device).flatten()[:n]
    return _differing(a, b)


def writes_in_place(ttnn_fn, bounds, operands: int, dtype: str, layout: str, device) -> int:
    """Elements an in-place op returned but never wrote into its own first operand.

    The harness scores what the op returns, so one that computed into a fresh tensor and
    left its input untouched reads as bit-exact while every caller gets stale data.
    """
    x = _values(bounds, dtype, TILE_WIDTH * TILE_WIDTH)
    args = [_tile(t) for t in (x, *[torch.ones_like(x)] * (operands - 1))]
    kw = {"dtype": dtype, "layout": layout, "device": device}
    returned = _on_device(ttnn_fn, *args, **kw)
    written = _on_device(ttnn_fn, *args, **kw, read=0)
    return _differing(returned, written)


def commutes(ttnn_fn, golden_fn, bounds, dtype: str, layout: str, device) -> int:
    """Elements where swapping the operands changed the device answer but not the golden's.

    Which binary ops must commute is read off the reference rather than listed by hand, so
    `sub` and `pow` exempt themselves and a new op needs no entry.
    """
    a = _values(bounds, dtype, TILE_WIDTH * TILE_WIDTH)
    b = a.flip(0)
    wide_a, wide_b = _tile(a), _tile(b)
    if _differing(_golden(golden_fn, wide_a, wide_b), _golden(golden_fn, wide_b, wide_a)):
        return 0  # not commutative in the first place
    kw = {"dtype": dtype, "layout": layout, "device": device}
    return _differing(
        _on_device(ttnn_fn, wide_a, wide_b, **kw), _on_device(ttnn_fn, wide_b, wide_a, **kw)
    )


def aliased_operands(ttnn_fn, bounds, operands: int, dtype: str, layout: str, device) -> int:
    """Elements where `f(a, a)` disagreed with `f(a, copy of a)`.

    One buffer read as two operands is a path no sweep produces, and this codebase has
    already been bitten by it once: freeing an in-place result by object identity
    segfaulted the allocator a few ops later.
    """
    x = _values(bounds, dtype, TILE_WIDTH * TILE_WIDTH)
    args = [_tile(x)] * operands
    kw = {"dtype": dtype, "layout": layout, "device": device}
    return _differing(
        _on_device(ttnn_fn, *args, **kw, alias=True), _on_device(ttnn_fn, *args, **kw)
    )


def sampled(operands: int, dtype: str) -> bool:
    """Whether this sweep's maximum is a lower bound rather than the maximum.

    Unary is exhaustive in both dtypes and binary bf16 covers every pair; the rest sample,
    so only those have anything to refine.
    """
    return operands >= 3 or (operands == 2 and dtype == "fp32")


def _neighbours(value: float, dtype: str) -> torch.Tensor:
    """Every candidate the sweep could have drawn for one operand.

    bf16 has only 2**16 codes, so this is the whole space. For fp32 it is every value
    sharing the top 16 bits — exactly the bits `_fp32_sample` fixed with one random draw.
    """
    if dtype == "bf16":
        codes = np.arange(2**16, dtype=np.uint32).astype(np.uint16).view(np.int16)
        return torch.from_numpy(codes).view(torch.bfloat16)
    high = np.float32(value).view(np.uint32) & np.uint32(0xFFFF0000)
    return torch.from_numpy((high | np.arange(2**16, dtype=np.uint32)).view(np.int32)).view(
        torch.float32
    )


def refine(ttnn_fn, golden_fn, point: list[float], dtype: str, layout: str, device):
    """Coordinate-wise exhaustive search through a sampled worst point.

    Holding the other operands and sweeping one across its whole space costs a dispatch per
    operand and can only tighten the bound, never loosen it.
    """
    best, where = -np.inf, list(point)
    for i in range(len(point)):
        candidates = _neighbours(where[i], dtype)
        operands = [
            _tile(candidates if k == i else torch.full_like(candidates, v))
            for k, v in enumerate(where)
        ]
        errors = metrics.errors(
            _golden(golden_fn, *operands),
            _on_device(ttnn_fn, *operands, dtype=dtype, layout=layout, device=device),
        )
        ulp = np.nan_to_num(errors["ulp_error"].flatten(), nan=-np.inf)
        j = int(ulp.argmax())
        if ulp[j] > best:
            best = float(ulp[j])
            where[i] = float(candidates[j])
    return best, where


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
