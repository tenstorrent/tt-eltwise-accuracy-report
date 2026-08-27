"""Accuracy metrics, and what each measured point actually demonstrates."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from models.common.utility_functions import ulp as tt_metal_ulp

MIN_NORMAL = 2**-126  # smallest normal bf16 and fp32 value; below it hardware returns zero

# Ordered by severity: a group is labelled by the worst outcome it contains.
OUTCOMES = ("exact", "inexact", "flushed", "zeroed", "overflow", "undefined", "mismatch")


def ulp(x: torch.Tensor) -> torch.Tensor:
    """tt-metal's ULP, widened to float32.

    The definition belongs to tt-metal so the whole org measures the same thing. Only
    the width is ours: tt-metal returns x's dtype, and callers divide by this, so a
    bf16 result would be capped at the bf16 subnormal floor.
    """
    return tt_metal_ulp(x).to(torch.float32)


def flush_subnormals(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t.abs() < torch.finfo(t.dtype).tiny, torch.zeros_like(t), t)


def classify(raw_golden: np.ndarray, gold: np.ndarray, calc: np.ndarray, dtype) -> np.ndarray:
    """Index into OUTCOMES for every point. Bounds come from the dtype, never a literal.

    `raw_golden` is the fp64 reference; `gold` is that reference as the hardware could
    hold it, so a value the dtype cannot represent has already become 0 in `gold`.
    """
    limits = torch.finfo(dtype)
    outcome = np.full(gold.shape, OUTCOMES.index("inexact"), dtype=np.int8)

    outcome[calc == gold] = OUTCOMES.index("exact")
    # gold == 0 with a non-zero reference means the value underflowed the dtype.
    outcome[(gold == 0) & (raw_golden != 0)] = OUTCOMES.index("flushed")
    # The opposite: the dtype could hold this value and the hardware returned zero anyway.
    # Real behaviour worth counting, but ULP cannot express it — a flush at the smallest
    # normal is 128 ULP in bf16 by construction, whatever the absolute error (1e-38).
    outcome[(calc == 0) & (gold != 0)] = OUTCOMES.index("zeroed")
    outcome[np.abs(raw_golden) > limits.max] = OUTCOMES.index("overflow")
    outcome[np.isnan(raw_golden)] = OUTCOMES.index("undefined")
    outcome[np.isfinite(gold) != np.isfinite(calc)] = OUTCOMES.index("mismatch")
    return outcome


def errors(golden: torch.Tensor, calculated: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-element error and outcome, flat. Shared by the unary and binary sweeps.

    ULP is left NaN wherever it is undefined — the reference is zero, so there is no
    spacing to divide by. Those points still report absolute error, and the outcome
    says which case they are, so nothing is silently rewritten to look perfect.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dtype = calculated.dtype
        # Hardware flushes subnormals to zero; flushing the reference the same way keeps
        # the comparison between values the hardware could actually have produced.
        calc = flush_subnormals(calculated.to(dtype)).to(torch.float32).flatten().numpy()
        gold = flush_subnormals(golden.to(dtype)).to(torch.float32).flatten().numpy()
        unit = ulp(flush_subnormals(golden.to(dtype))).flatten().numpy()
        raw = golden.to(torch.float64).flatten().numpy()

        outcome = classify(raw, gold, calc, dtype)
        abs_err = np.abs(gold - calc)
        # Dividing by the spacing around zero is what produced 1e24-ULP readings for a
        # 0.003 absolute error. A ULP is only a distance between two real values: an exact
        # match scores zero, anything else needs a non-zero reference to divide by, and
        # every other outcome — a flush either way, an overflow, a NaN — has no ULP at all.
        defined = (outcome == OUTCOMES.index("exact")) | (
            (outcome == OUTCOMES.index("inexact")) & (gold != 0)
        )
        return {
            "y": calc,
            "y_ref": gold,
            "abs_error": abs_err,
            "ulp_error": np.where(defined, abs_err / unit, np.nan),
            "outcome": outcome,
        }


def compare(
    x: torch.Tensor,
    golden: torch.Tensor,
    calculated: torch.Tensor,
    group_size: int,
) -> pd.DataFrame:
    """One row per group of `group_size` consecutive inputs, holding its worst error."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        e = errors(golden, calculated)
        shape = (x.nelement() // group_size, group_size)
        g = {k: v.reshape(shape) for k, v in e.items()}

        return pd.DataFrame(
            {
                "x": x.to(torch.float32).flatten().numpy().reshape(shape)[:, 0],
                "y": g["y"][:, 0],
                "y_ref": g["y_ref"][:, 0],
                "ulp_error": np.nanmax(g["ulp_error"], axis=-1),
                "abs_error": np.nanmax(g["abs_error"], axis=-1),
                "outcome": np.take(OUTCOMES, g["outcome"].max(axis=-1)),
            }
        )
