# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""One measured CSV → the numbers a page and a diff read. No plotting, so `check` skips matplotlib."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ttnn_accuracy.config import (
    MIN_NORMAL,
    MONOTONIC_TOP,
    NONFINITE_DETAIL,
    OFFENDERS,
    ULP_CLIP,
    USABLE_ULP,
)
from ttnn_accuracy.measure.schema import COLUMNS
from ttnn_accuracy.ops.overrides import OVERRIDES, variant_slug
from ttnn_accuracy.ops.plan import describe


def _defects(df: pd.DataFrame) -> pd.Series:
    """inf or zero where a value exists; a NaN reference is a domain disagreement, not this."""
    return (df["outcome"] == "zeroed") | (
        (df["outcome"] == "mismatch") & df["y_ref"].notna() & np.isfinite(df["y_ref"])
    )


def _finite(s: pd.Series) -> pd.Series:
    return s.replace([float("inf"), float("-inf")], float("nan"))


def _rounded_frac(df: pd.DataFrame) -> float:
    """Correctly rounded per input, not per row: an fp32 row is the worst of 65536 points."""
    defined = df["n_defined"].sum()
    return float(df["n_rounded"].sum() / defined) if defined else 0.0


def _fmt(v) -> str:
    return "—" if v != v else f"{float(v):.3g}"


def _or_exact(value: float, defined: int) -> str:
    """A figure over inexact points: 0 when every defined point was exact, `—` when none was."""
    return _fmt(value if value == value else (0.0 if defined else float("nan")))


def _bits(rel: float) -> str:
    """Bits of precision from a relative error. A zero error has no bits to report, not infinite."""
    return _fmt(-np.log2(rel) if rel > 0 else float("nan"))


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def verdict(
    max_ulp: str,
    mean_ulp: str,
    usable_to: str,
    operands: int | None,
    defects: int,
    points: int,
    unflushed: int,
    faithful: int,
    inexact: int,
) -> str:
    """One of nine fixed phrases; analyze-report/contract.md is their twin and moves with them."""
    mx = _num(max_ulp)
    # First, whatever the ULP says: every other figure here excludes those points.
    if defects:
        return f"{defects} of {points} points returned inf or zero where a value exists"
    # Also unscorable, and otherwise invisible: the ULP of these points would be bit-exact.
    if unflushed:
        return f"{unflushed} of {points} points returned a value where the reference is zero"
    if mx is None:
        return "no scorable points"
    # By outcome: against a wider reference the maximum is half a ULP even when every point
    # carries the dtype's own correctly rounded answer.
    if not (faithful or inexact):
        return "bit-exact"
    # Both neighbours of a tie are within half a ULP, so this is a tie-breaking rule that
    # differs from the reference's, not an accuracy shortfall.
    if not inexact:
        return f"faithfully rounded; {faithful} of {points} points took the other neighbour"
    if mx <= USABLE_ULP:
        return f"within {USABLE_ULP:g} ULP everywhere"
    if operands == 1:
        if _num(usable_to) is not None:
            return f"accurate to |x| <= {usable_to}; up to {max_ulp} ULP beyond"
        return f"never within {USABLE_ULP:g} ULP; mean {mean_ulp}, worst {max_ulp}"
    return f"worst pairing {max_ulp} ULP; mean {mean_ulp}"


def _special_fmt(v: float) -> str:
    """A measured value, at the 9 digits that round-trip fp32: 4 hide a whole fp32 ULP, and a
    row marked `differ` would print the same on both sides."""
    if v != v:
        return "nan"
    if v == 0:
        return "-0" if np.signbit(v) else "0"
    return f"{v:.9g}"


def _agree(y: float, ref: float) -> bool:
    """Two NaNs agree, two zeros of opposite sign do not, and nothing else is a judgement."""
    if y != y and ref != ref:
        return True
    # bool(), or numpy's own returns and the index will not serialise.
    return bool(y == ref and np.signbit(y) == np.signbit(ref))


def _specials_rows(specials: pd.DataFrame) -> list[dict]:
    """Display strings: NaN and ±inf are not JSON numbers, and the sign of zero is the point."""
    return [
        {c: _special_fmt(row[c]) for c in ("x", "y", "y_ref")}
        # Decided on the values, not on their four printed digits, which hide an fp32 ULP.
        | {"agree": _agree(row["y"], row["y_ref"])}
        for _, row in specials.iterrows()
    ]


def _usable_to(df: pd.DataFrame) -> float:
    """Where a cliff starts: `sin` holds 2 ULP to 2.6e5 then collapses. Unary only."""
    if "x2" in df.columns:
        return float("nan")
    finite = df[_finite(df["x"]).notna()]
    ordered = finite.reindex(finite["x"].abs().sort_values().index)
    # NaN ULP is unscorable; 0.0 would let an infinity extend the range it ends.
    ulp = _finite(ordered["ulp_error"]).fillna(0.0)
    within = ulp.mask(_defects(ordered), float("inf")).cummax() <= USABLE_ULP
    # `or nan`: a bound of 0 held only at x=0, which is no range at all.
    return (abs(ordered["x"][within].iloc[-1]) or float("nan")) if within.any() else float("nan")


def _operands(df: pd.DataFrame) -> list[str]:
    """The input columns of a row: `x`, plus the partners a pair or triple sweep recorded."""
    return [c for c in ("x", "x2", "x3") if c in df.columns]


def _offenders(df: pd.DataFrame) -> list[dict]:
    """The worst-ULP points by value. The maximum says how bad; these say at which inputs."""
    ranked = df.assign(ulp=_finite(df["ulp_error"])).nlargest(OFFENDERS, "ulp")
    return [
        {c: _special_fmt(row[c]) for c in _operands(df)}
        | {
            "y_ref": _special_fmt(row["y_ref"]),
            "y": _special_fmt(row["y"]),
            "ulp": _fmt(row["ulp"]),
        }
        for _, row in ranked.iterrows()
        if row["ulp"] > 0
    ]


def _monotonic(df: pd.DataFrame) -> dict:
    """Ordering of the device output wherever the reference is itself ordered, per interval."""
    if "x2" in df.columns:  # x alone does not determine y, so ordering says nothing
        return {}
    ordered = df.sort_values("x")
    sign = np.sign(ordered["x"].values)
    # Only x crossing zero ends one: splitting at holes makes short runs that all look ordered.
    intervals = np.cumsum(np.r_[True, sign[1:] != sign[:-1]])

    pairs = violations = 0
    worst: list[tuple[float, dict]] = []
    for _, block in ordered.assign(_interval=intervals).groupby("_interval", sort=False):
        ok = block["outcome"].isin(("exact", "faithful", "inexact")).values
        if ok.sum() < 2:
            continue
        ref, hw, xs = block["y_ref"].values, block["y"].values, block["x"].values
        # Non-strict: equal neighbours are expected when the dtype cannot separate them.
        step = np.diff(ref[ok])
        if np.all(step >= 0):
            direction = 1.0
        elif np.all(step <= 0):
            direction = -1.0
        else:
            continue  # the reference is not ordered here, so the device need not be either
        # Only neighbours in the sweep, and only where neither end was unscorable.
        comparable = np.flatnonzero(ok[:-1] & ok[1:])
        moved = (np.diff(hw) * direction)[comparable]
        pairs += len(moved)
        for i in comparable[moved < 0]:
            violations += 1
            dy = abs(float(hw[i + 1] - hw[i]))
            # 4 digits: neighbouring bf16 codes are 0.4% apart and would otherwise print alike.
            worst.append(
                (
                    dy,
                    {
                        "x_from": _special_fmt(xs[i]),
                        "x_to": _special_fmt(xs[i + 1]),
                        "y_from": _special_fmt(hw[i]),
                        "y_to": _special_fmt(hw[i + 1]),
                        "dy": _fmt(dy),
                    },
                )
            )
    if not pairs:
        return {}
    worst.sort(key=lambda w: w[0], reverse=True)
    return {
        "pairs": pairs,
        "violations": violations,
        "rate": _fmt(violations / pairs),
        "worst_dy": _fmt(worst[0][0] if worst else 0.0),
        "top": [row for _, row in worst[:MONOTONIC_TOP]],
    }


def _nonfinite(df: pd.DataFrame) -> dict:
    """Which side went non-finite and where; `mismatch` says they differ, this says which way."""
    ref, hw = df["y_ref"].values, df["y"].values
    ref_bad, hw_bad = ~np.isfinite(ref), ~np.isfinite(hw)
    if not (total := int((ref_bad | hw_bad).sum())):
        return {}
    listed = df[ref_bad | hw_bad].head(NONFINITE_DETAIL)
    return {
        "total": total,
        "both": int((ref_bad & hw_bad).sum()),
        "device_only": int((hw_bad & ~ref_bad).sum()),
        "golden_only": int((ref_bad & ~hw_bad).sum()),
        "device_inf": int(np.isinf(hw).sum()),
        "device_nan": int(np.isnan(hw).sum()),
        "golden_inf": int(np.isinf(ref).sum()),
        "golden_nan": int(np.isnan(ref).sum()),
        "detail": [
            {c: _special_fmt(row[c]) for c in _operands(df)}
            | {"y_ref": _special_fmt(row["y_ref"]), "y": _special_fmt(row["y"])}
            for _, row in listed.iterrows()
        ],
    }


def compute_stats(df: pd.DataFrame) -> dict:
    """One variant's stats over defined, non-trivial points; subnormals already removed."""
    ulp = _finite(df["ulp_error"])
    rel = _finite(df["rel_error"])
    # By outcome, not by a non-zero error: against a wider reference a correctly rounded point
    # still carries up to half a ULP, so `ulp > 0` no longer names the points it got wrong.
    wrong = df["outcome"].isin(("faithful", "inexact"))
    defined = ulp.count()  # an unscorable point is not a point the percentiles may average
    outcomes = {k: int(v) for k, v in df["outcome"].value_counts().items()}
    return {
        "max_ulp": _fmt(ulp.max()),
        # Per input in bf16, worst-of-block in fp32: above 0.5 names every block, not every input.
        "min_ulp": _fmt(ulp.min()),
        "mean_ulp": _or_exact(ulp[wrong].mean(), defined),
        # Signed, so a one-sided approximation is visible: a mean near ±mean_ulp is a constant
        # a kernel can absorb, one near zero is a precision shortfall needing another step.
        "bias_ulp": _or_exact(_finite(df["ulp_signed"])[wrong].mean(), defined),
        "p50_ulp": _fmt(ulp.quantile(0.50)),
        "p95_ulp": _fmt(ulp.quantile(0.95)),
        "p99_ulp": _fmt(ulp.quantile(0.99)),
        "exact_frac": _fmt(outcomes.get("exact", 0) / defined if defined else float("nan")),
        "rounded_frac": _fmt(_rounded_frac(df)),
        "usable_to": _fmt(_usable_to(df)),
        "max_abs": _fmt(_finite(df["abs_error"]).max()),
        "max_rel": _fmt(rel.max()),
        "median_rel": _or_exact(rel[wrong].median(), rel.count()),
        "bits_worst": _bits(rel.max()),
        "bits_median": _bits(rel[wrong].median()),
        "ulp_clipped": int((ulp > ULP_CLIP).sum()),
        "defects": int(_defects(df).sum()),
        "unflushed": outcomes.get("unflushed", 0),
        "n_inputs": len(df),
        "outcomes": outcomes,
        "offenders": _offenders(df),
        "monotonic": _monotonic(df),
        "nonfinite": _nonfinite(df),
    }


def score_csv(path: Path, op: str, variant: str) -> tuple[dict, pd.DataFrame] | None:
    """One CSV → its index entry and plottable rows; shared with `check`, so both score alike."""
    raw = pd.read_csv(path, index_col="index")
    if missing := set(COLUMNS) - set(raw.columns):
        logger.error("{} predates the schema, missing {} — re-measure", path, sorted(missing))
        return None
    special = raw[raw["outcome"] == "special"]
    df = raw[(raw["outcome"] != "special") & (raw["x"].abs() >= MIN_NORMAL)]
    if df.empty:
        logger.warning("no normal-range rows in {}", path)
        return None

    info = describe(op)
    stats = compute_stats(df) | {"specials": _specials_rows(special)}
    stats["verdict"] = verdict(
        stats["max_ulp"],
        stats["mean_ulp"],
        stats["usable_to"],
        info.operands if info else None,
        stats["defects"],
        stats["n_inputs"],
        stats["unflushed"],
        stats["outcomes"].get("faithful", 0),
        stats["outcomes"].get("inexact", 0),
    )
    ov = next(
        (o for o in OVERRIDES.get(f"ttnn.{op}", ()) if variant_slug(o.params_desc) == variant),
        None,
    )
    if ov and ov.why:
        stats["rationale"] = ov.why
    return stats, df
