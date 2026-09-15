"""Input bounds from each golden in float64: domain, representability and subnormals, per dtype."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from ttnn_accuracy.config import FP32_MANTISSA_SAMPLES, MIN_NORMAL

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass(frozen=True, slots=True)
class Domain:
    lo: float
    hi: float
    n_valid: int
    n_undefined: int  # golden is NaN: outside the mathematical domain
    n_overflow: int  # |golden| exceeds what the dtype can hold
    n_flushed: int  # golden is subnormal, so hardware returns zero


def grid(dtype: str) -> torch.Tensor:
    """Every finite normal value of a bf16 grid, or a per-exponent sample of fp32."""
    if dtype == "bf16":
        codes = torch.arange(2**16, dtype=torch.int32).to(torch.int16)
        values = codes.view(torch.bfloat16).to(torch.float64)
    else:
        # One sweep per exponent, sampling mantissas: 2^32 goldens in fp64 is not viable.
        exponents = torch.arange(1, 255, dtype=torch.int32) << 23
        mantissas = torch.linspace(0, 2**23 - 1, FP32_MANTISSA_SAMPLES).to(torch.int32)
        codes = (exponents[:, None] | mantissas[None, :]).flatten()
        positive = codes.view(torch.float32).to(torch.float64)
        values = torch.cat([-positive, torch.zeros(1, dtype=torch.float64), positive])
    finite = values[torch.isfinite(values) & ((values == 0) | (values.abs() >= MIN_NORMAL))]
    return finite.sort().values


def derive(golden, dtype: str) -> dict:
    """Bounds and outcome counts for one op on one dtype."""
    x = grid(dtype)
    with torch.no_grad():
        y = golden(x)
    limit = torch.finfo(DTYPES[dtype]).max
    tiny = torch.finfo(DTYPES[dtype]).tiny

    undefined = y.isnan()  # the grid holds no NaN, so this is the op's own doing
    overflow = ~undefined & (y.abs() > limit)
    flushed = ~undefined & ~overflow & (y != 0) & (y.abs() < tiny)
    valid = ~(undefined | overflow | flushed)

    inside = x[valid]
    return asdict(
        Domain(
            lo=float(inside.min()) if inside.numel() else float("nan"),
            hi=float(inside.max()) if inside.numel() else float("nan"),
            n_valid=int(valid.sum()),
            n_undefined=int(undefined.sum()),
            n_overflow=int(overflow.sum()),
            n_flushed=int(flushed.sum()),
        )
    )
