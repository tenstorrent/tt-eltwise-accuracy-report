"""Accuracy metrics, and what each measured point actually demonstrates."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from models.common.utility_functions import ulp as tt_metal_ulp

# One step wider than the measurement, so a correctly rounded answer reads below half a ULP.
WIDER = {torch.bfloat16: torch.float32, torch.float32: torch.float64}

# Ordered by severity: a group is labelled by the worst outcome it contains.
OUTCOMES = (
    "exact",
    "faithful",
    "inexact",
    "flushed",
    "zeroed",
    "unflushed",
    "overflow",
    "undefined",
    "mismatch",
)


def ulp(x: torch.Tensor) -> torch.Tensor:
    """tt-metal's ULP in fp32: callers divide by it, and bf16 would cap the quotient."""
    return tt_metal_ulp(x).to(torch.float32)


def flush_subnormals(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t.abs() < torch.finfo(t.dtype).tiny, torch.zeros_like(t), t)


def adjacent(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """One representable step apart, read off the bit patterns — IEEE neighbours differ by one."""
    ints = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[a.dtype]
    wide = [t.contiguous().view(ints).to(torch.int64) for t in (a, b)]
    return ((wide[0] - wide[1]).abs() == 1).flatten().numpy()


def classify(raw_golden: np.ndarray, gold: np.ndarray, calc: np.ndarray, dtype) -> np.ndarray:
    """Index into OUTCOMES per point. `gold` is the fp64 `raw_golden` as the dtype holds it."""
    limits = torch.finfo(dtype)
    outcome = np.full(gold.shape, OUTCOMES.index("inexact"), dtype=np.int8)

    outcome[calc == gold] = OUTCOMES.index("exact")
    outcome[(gold == 0) & (raw_golden != 0)] = OUTCOMES.index("flushed")  # underflowed
    # Representable, and zeroed anyway: a defect, but 128 ULP in bf16 by construction.
    outcome[(calc == 0) & (gold != 0)] = OUTCOMES.index("zeroed")
    # Subnormals are flushed already, so a survivor is normal: exp(-100) answering -4.3e33.
    outcome[(gold == 0) & (raw_golden != 0) & (calc != 0)] = OUTCOMES.index("unflushed")
    outcome[np.abs(raw_golden) > limits.max] = OUTCOMES.index("overflow")
    outcome[np.isnan(raw_golden)] = OUTCOMES.index("undefined")
    outcome[np.isfinite(gold) != np.isfinite(calc)] = OUTCOMES.index("mismatch")
    return outcome


def errors(golden: torch.Tensor, calculated: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-element error and outcome, flat. Undefined ULP stays NaN, never a flattering 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dtype = calculated.dtype
        device = flush_subnormals(calculated.to(dtype))
        # The reference as the dtype holds it, which is what `exact` can mean at all.
        rounded = flush_subnormals(golden.to(dtype))
        calc = device.to(torch.float32).flatten().numpy()
        gold = rounded.to(torch.float32).flatten().numpy()
        # Measured against a reference wider than the dtype: rounding it first puts both sides
        # on one grid, so every error is a whole multiple of the spacing and a correctly
        # rounded answer cannot be told from an exact one.
        precise = golden.to(WIDER[dtype]).to(torch.float64).flatten().numpy()
        unit = ulp(rounded).flatten().numpy().astype(np.float64)
        raw = golden.to(torch.float64).flatten().numpy()

        outcome = classify(raw, gold, calc, dtype)
        signed = precise - calc.astype(np.float64)
        abs_err = np.abs(signed)
        # Faithful, not wrong: the true value falls strictly between the device's answer and the
        # correctly rounded one, so both are neighbours of it and only the tie-break differs.
        # Strictly, so a representable reference the device missed by a step stays `inexact`.
        straddles = (precise - gold) * (calc.astype(np.float64) - precise) > 0
        outcome[(outcome == OUTCOMES.index("inexact")) & adjacent(device, rounded) & straddles] = (
            OUTCOMES.index("faithful")
        )
        # Spacing around zero is what read 1e24 ULP for a 0.003 absolute error.
        missed = np.isin(outcome, [OUTCOMES.index(k) for k in ("faithful", "inexact")])
        defined = (outcome == OUTCOMES.index("exact")) | (missed & (gold != 0))
        return {
            "y": calc,
            "y_ref": gold,
            "abs_error": abs_err,
            "ulp_error": np.where(defined, abs_err / unit, np.nan),
            # Signed too: a mean that is not zero is a one-sided approximation, which a tuned
            # constant can remove, rather than a precision shortfall needing another step.
            "ulp_signed": np.where(defined, signed / unit, np.nan),
            "rel_error": np.where(defined & (gold != 0), abs_err / np.abs(precise), np.nan),
            "outcome": outcome,
        }


def compare(
    x: torch.Tensor,
    golden: torch.Tensor,
    calculated: torch.Tensor,
    group_size: int,
) -> pd.DataFrame:
    """One row per `group_size` inputs, every column from the worst-defined-ULP point."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        e = errors(golden, calculated)
        rows = x.nelement() // group_size
        shape = (rows, group_size)
        g = {k: v.reshape(shape) for k, v in e.items()}
        col = np.arange(rows)
        pos = np.nan_to_num(g["ulp_error"], nan=-np.inf).argmax(axis=-1)

        return pd.DataFrame(
            {
                "x": x.to(torch.float32).flatten().numpy().reshape(shape)[col, pos],
                "y": g["y"][col, pos],
                "y_ref": g["y_ref"][col, pos],
                # Counted before the argmax throws the group away: a worst point cannot carry
                # a rate, and (1 - p)**65536 reads 0% for any p above about 1e-5.
                "n_defined": np.isfinite(g["ulp_error"]).sum(axis=-1),
                "n_rounded": (g["ulp_error"] <= 0.5).sum(axis=-1),  # NaN compares false
                "ulp_error": g["ulp_error"][col, pos],
                "ulp_signed": g["ulp_signed"][col, pos],
                "abs_error": g["abs_error"][col, pos],
                "rel_error": g["rel_error"][col, pos],
                "outcome": np.take(OUTCOMES, g["outcome"][col, pos]),
            }
        )
