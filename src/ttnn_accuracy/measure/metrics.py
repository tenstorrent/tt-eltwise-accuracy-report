"""Accuracy metrics, and what each measured point actually demonstrates."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from models.common.utility_functions import ulp as tt_metal_ulp

# Ordered by severity: a group is labelled by the worst outcome it contains.
OUTCOMES = (
    "exact",
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
        # The reference is flushed too, so both sides are values the hardware could produce.
        calc = flush_subnormals(calculated.to(dtype)).to(torch.float32).flatten().numpy()
        gold = flush_subnormals(golden.to(dtype)).to(torch.float32).flatten().numpy()
        unit = ulp(flush_subnormals(golden.to(dtype))).flatten().numpy()
        raw = golden.to(torch.float64).flatten().numpy()

        outcome = classify(raw, gold, calc, dtype)
        abs_err = np.abs(gold - calc)
        # Spacing around zero is what read 1e24 ULP for a 0.003 absolute error.
        defined = (outcome == OUTCOMES.index("exact")) | (
            (outcome == OUTCOMES.index("inexact")) & (gold != 0)
        )
        return {
            "y": calc,
            "y_ref": gold,
            "abs_error": abs_err,
            # fp64 for the same reason as rel_error: an error over the subnormal-scale spacing
            # of a min-normal reference exceeds 1e38, and in fp32 that inf is dropped unscored.
            "ulp_error": np.where(defined, abs_err / unit.astype(np.float64), np.nan),
            # fp64: a large error over the smallest normal is 1e76, which fp32 would call inf.
            "rel_error": np.where(
                defined & (gold != 0), abs_err / np.abs(gold).astype(np.float64), np.nan
            ),
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
                "ulp_error": g["ulp_error"][col, pos],
                "abs_error": g["abs_error"][col, pos],
                "rel_error": g["rel_error"][col, pos],
                "outcome": np.take(OUTCOMES, g["outcome"][col, pos]),
            }
        )
