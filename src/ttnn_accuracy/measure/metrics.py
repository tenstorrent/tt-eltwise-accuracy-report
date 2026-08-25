"""Accuracy metrics"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch

REL_FLOOR = 2**-9  # arbitrary, inherited from the POC
ULP_FLOOR = 1e-45  # fp32 min subnormal: guards the division without clamping bf16 ULPs

MANTISSA_BITS = {torch.bfloat16: 7, torch.float32: 23}


def ulp(x: torch.Tensor) -> torch.Tensor:
    """Unit in the last place of each element, always returned as float32.

    Returning float32 rather than x.dtype prevents bf16 underflow: the ULP of the
    bf16 min-normal is 2^-133, a valid fp32 subnormal that would flush to 0 in bf16.
    """
    mag = x.to(torch.float32).abs().clamp(min=torch.finfo(torch.float32).tiny)
    return torch.pow(2.0, mag.log2().floor() - MANTISSA_BITS[x.dtype])


def flush_subnormals(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t.abs() < torch.finfo(t.dtype).tiny, torch.zeros_like(t), t)


def compare(
    x: torch.Tensor,
    golden: torch.Tensor,
    calculated: torch.Tensor,
    group_size: int,
) -> pd.DataFrame:
    """One row per group of `group_size` consecutive inputs, holding the group's worst error.

    A group whose golden is entirely NaN yields NaN, which is the honest answer and is
    written to the CSV as such; nanmax announcing it on every such group is not useful.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dtype = calculated.dtype
        # Hardware flushes subnormals to zero; flushing the golden the same way stops
        # one side being subnormal and the other exactly 0 from reading as huge error.
        calc = flush_subnormals(calculated.to(dtype)).to(torch.float32)
        gold = flush_subnormals(golden.to(torch.float32))
        unit = ulp(flush_subnormals(golden.to(dtype))).to(torch.float32)

        shape = (x.nelement() // group_size, group_size)

        def grouped(t: torch.Tensor) -> np.ndarray:
            return t.flatten().numpy().reshape(shape)

        x_np, y_np, ref_np = grouped(x.to(torch.float32)), grouped(calc), grouped(gold)

        abs_err = np.abs(ref_np - y_np)
        rel_err = abs_err / np.maximum(np.abs(ref_np), REL_FLOOR)
        ulp_err = abs_err / np.maximum(grouped(unit), ULP_FLOOR)

        # Hardware zeroing a result at the subnormal boundary is a representation
        # limit, not an accuracy error - e.g. tanh(min_normal) via e^(2x)=1.
        boundary = (y_np == 0.0) & (np.abs(ref_np) <= float(torch.finfo(dtype).tiny))
        abs_err, rel_err, ulp_err = (
            np.where(boundary, 0.0, e) for e in (abs_err, rel_err, ulp_err)
        )

        return pd.DataFrame(
            {
                "x": x_np[:, 0],
                "y": y_np[:, 0],
                "y_ref": ref_np[:, 0],
                "ulp_error": np.nanmax(ulp_err, axis=-1),
                "abs_error": np.nanmax(abs_err, axis=-1),
                "rel_error": np.nanmax(rel_err, axis=-1),
            }
        )
