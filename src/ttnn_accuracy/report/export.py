"""Measured CSVs in the tt-llk SFPU harness's own schema, so its dashboard can render ours.

GitHub strips scripts from the markdown report, so zoom and pan cannot live in `reports/`.
The LLK dashboard already aggregates any harness CSV in the browser, so the cheapest
interactive view of our sweeps is to speak its input format: one file per op, every variant
inside it, keyed by the columns it groups on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ttnn_accuracy.config import MIN_NORMAL
from ttnn_accuracy.ops.plan import describe
from ttnn_accuracy.paths import DATA_DIR, REPO_ROOT

# Their 19 columns, in their order. `to_csv.py` writes floats as %.9g and booleans as T/F.
COLUMNS = (
    "op",
    "input_format",
    "output_format",
    "chip_arch",
    "distribution",
    "intervals",
    "seed",
    "sample_index",
    "test_value",
    "golden_result",
    "hardware_result",
    "approx_mode",
    "fast_mode",
    "dest_acc",
    "signed_error",
    "rel_error",
    "signed_ulp_error",
    "is_finite_hw",
    "is_finite_golden",
)


def _interval(op: str, dtype: str, xs: pd.Series) -> str:
    """`[(lo, hi)]`, which their `parse_intervals` reads as the tested band."""
    info = describe(op)
    lo, hi = info.bounds[dtype] if info and info.bounds else (xs.min(), xs.max())
    return f"[({float(lo):.9g}, {float(hi):.9g})]"


def _variant(df: pd.DataFrame, op: str, arch: str, dtype: str, variant: str) -> pd.DataFrame:
    """One of our measured variants as their rows.

    `fast_approx` is their `approx_mode`; `dest_acc` follows the dtype because that is how
    ttnn derives `fp32_dest_acc_en`, and no eltwise op exposes it separately. `fast_mode` is
    always 0: it is an LLK knob with no ttnn equivalent. Their aggregator trusts the error
    columns, so a point we consider unscorable is written blank for it to skip rather than
    divided by a subnormal-scale spacing and reported as 1e73 ULP."""
    signed = df["y"] - df["y_ref"]
    approx = int("approx" in variant)
    # Their variant key is formats plus three flags, with no field for a scalar parameter, so
    # `relu_max` at 6 and at 1 would collide into one variant and be averaged together. A
    # variant their key cannot hold becomes its own op instead.
    named = op if approx or variant == "default" else f"{op}[{variant}]"
    return pd.DataFrame(
        {
            "op": named,
            "input_format": dtype,
            "output_format": dtype,
            "chip_arch": arch,
            "distribution": "exhaustive" if len(df) >= 2**16 else "sampled",
            "intervals": _interval(op, dtype, df["x"]),
            "seed": "",
            "sample_index": np.arange(len(df)),
            "test_value": df["x"].values,
            "golden_result": df["y_ref"].values,
            "hardware_result": df["y"].values,
            "approx_mode": approx,
            "fast_mode": 0,
            "dest_acc": int(dtype == "fp32"),
            "signed_error": signed.values,
            "rel_error": df["rel_error"].values,
            "signed_ulp_error": (df["ulp_error"] * np.sign(signed)).values,
            "is_finite_hw": np.where(np.isfinite(df["y"]), "T", "F"),
            "is_finite_golden": np.where(np.isfinite(df["y_ref"]), "T", "F"),
        }
    )


def export_llk(out_dir: Path, arch=None, dtype=None, op=None) -> int:
    """data/{arch}/{dtype}/{op}/*.csv → {out}/{arch}/{dtype}/{op}.csv in their schema."""
    if not DATA_DIR.exists():
        logger.error("no data/ directory — run `ttnn-accuracy measure` first")
        return 1
    written = 0
    for op_dir in sorted(DATA_DIR.glob(f"{arch or '*'}/{dtype or '*'}/{op or '*'}")):
        if not op_dir.is_dir():
            continue
        this_arch, this_dtype, this_op = op_dir.parts[-3:]
        frames = []
        for csv in sorted(op_dir.glob("*.csv")):
            raw = pd.read_csv(csv, index_col="index")
            # Their schema carries one `test_value`, so a pair or triple would export with
            # its partner dropped and be drawn as f(x) when it is f(x, worst partner).
            if "x2" in raw.columns:
                logger.info("skipping {}/{}: their schema has no second operand", this_op, csv.stem)
                continue
            # The same population our own pages score, so a reader comparing their summary
            # against ours is looking at one set of points.
            rows = raw[(raw["outcome"] != "special") & (raw["x"].abs() >= MIN_NORMAL)]
            if rows.empty:
                continue
            frames.append(_variant(rows, this_op, this_arch, this_dtype, csv.stem))
        if not frames:
            continue
        out = out_dir / this_arch / this_dtype / f"{this_op}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames)[list(COLUMNS)].to_csv(out, index=False, float_format="%.9g", na_rep="")
        written += 1
        logger.info("{} rows → {}", sum(len(f) for f in frames), out)

    if not written:
        logger.error("nothing matched arch={} dtype={} op={}", arch, dtype, op)
        return 1
    logger.success(
        "{} op file(s) → {}  ·  load them with the dashboard's Load CSV",
        written,
        out_dir if out_dir.is_absolute() else out_dir.relative_to(REPO_ROOT),
    )
    return 0
