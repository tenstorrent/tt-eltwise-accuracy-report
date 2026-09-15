"""One measured CSV → the numbers a page and a diff read. No plotting, so `check` skips matplotlib."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ttnn_accuracy.config import MIN_NORMAL, ULP_CLIP, USABLE_ULP
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


def _fmt(v) -> str:
    return "—" if v != v else f"{float(v):.3g}"


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
) -> str:
    """One of eight fixed phrases; analyze-report/contract.md is their twin and moves with them."""
    mx = _num(max_ulp)
    # First, whatever the ULP says: every other figure here excludes those points.
    if defects:
        return f"{defects} of {points} points returned inf or zero where a value exists"
    # Also unscorable, and otherwise invisible: the ULP of these points would be bit-exact.
    if unflushed:
        return f"{unflushed} of {points} points returned a value where the reference is zero"
    if mx is None:
        return "no scorable points"
    if mx == 0:
        return "bit-exact"
    if mx <= USABLE_ULP:
        return f"within {USABLE_ULP:g} ULP everywhere"
    if operands == 1:
        if _num(usable_to) is not None:
            return f"accurate to |x| <= {usable_to}; up to {max_ulp} ULP beyond"
        return f"never within {USABLE_ULP:g} ULP; mean {mean_ulp}, worst {max_ulp}"
    return f"worst pairing {max_ulp} ULP; mean {mean_ulp}"


def _special_fmt(v: float) -> str:
    if v != v:
        return "nan"
    if v == 0:
        return "-0" if np.signbit(v) else "0"
    return f"{v:.4g}"


def _specials_rows(specials: pd.DataFrame) -> list[dict]:
    """Display strings: NaN and ±inf are not JSON numbers, and the sign of zero is the point."""
    return [
        {c: _special_fmt(row[c]) for c in ("x", "y", "y_ref")} for _, row in specials.iterrows()
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


def compute_stats(df: pd.DataFrame) -> dict:
    """One variant's stats over defined, non-trivial points; subnormals already removed."""
    ulp = _finite(df["ulp_error"])
    rel = _finite(df["rel_error"])
    inexact = rel[rel > 0]  # like mean_ulp: an exact point has no error to summarise
    defined = ulp.count()  # an unscorable point is not a point the percentiles may average
    outcomes = {k: int(v) for k, v in df["outcome"].value_counts().items()}
    return {
        "max_ulp": _fmt(ulp.max()),
        "mean_ulp": _fmt(ulp[ulp > 0].mean()),
        "p50_ulp": _fmt(ulp.quantile(0.50)),
        "p95_ulp": _fmt(ulp.quantile(0.95)),
        "p99_ulp": _fmt(ulp.quantile(0.99)),
        "exact_frac": _fmt((ulp == 0).sum() / defined if defined else float("nan")),
        "usable_to": _fmt(_usable_to(df)),
        "max_abs": _fmt(_finite(df["abs_error"]).max()),
        "max_rel": _fmt(rel.max()),
        "median_rel": _fmt(inexact.median()),
        "bits_worst": _bits(rel.max()),
        "bits_median": _bits(inexact.median()),
        "ulp_clipped": int((ulp > ULP_CLIP).sum()),
        "defects": int(_defects(df).sum()),
        "unflushed": outcomes.get("unflushed", 0),
        "n_inputs": len(df),
        "outcomes": outcomes,
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
    )
    ov = next(
        (o for o in OVERRIDES.get(f"ttnn.{op}", ()) if variant_slug(o.params_desc) == variant),
        None,
    )
    if ov and ov.why:
        stats["rationale"] = ov.why
    return stats, df
