#!/usr/bin/env python3
"""
Generate SVG accuracy charts from CSV measurement data and update report_index.json.

Scans data/{arch}/{dtype}/{op}/{variant}.csv, produces
reports/charts/{arch}/{dtype}/{op}_{variant}_ulp.svg, and writes
summary stats into report_index.json (committed to repo as persistent cache).

Usage:
    python generate_charts.py
    python generate_charts.py --arch wh --dtype bf16
    python generate_charts.py --arch wh --dtype bf16 --op exp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
CHARTS_DIR = REPO_ROOT / "reports" / "charts"
INDEX_FILE = REPO_ROOT / "report_index.json"

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["figure.dpi"] = 100

MIN_NORMAL = 2**-126


def remove_subnormals(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["x"].abs() >= MIN_NORMAL]


def load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, index_col="index")
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}")
        return None


def load_index() -> dict:
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text())
    return {}


def save_index(index: dict):
    INDEX_FILE.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def compute_stats(df: pd.DataFrame, display_lo, display_hi) -> dict:
    """Compute summary stats restricted to the display range."""
    d = df.copy()
    d = remove_subnormals(d)
    if display_lo is not None and display_lo not in (None, float("-inf")):
        d = d[d["x"] >= display_lo]
    if display_hi is not None and display_hi not in (None, float("inf")):
        d = d[d["x"] <= display_hi]
    if d.empty:
        return {}
    ulp = d["ulp_error"].replace([float("inf"), float("-inf")], float("nan"))
    return {
        "max_ulp":      _fmt(ulp.max()),
        "mean_ulp":     _fmt(ulp.mean()),
        "max_abs":      _fmt(d["abs_error"].max()),
        "ulp_clipped":  int((ulp > 1000).sum()),
        "n_inputs":     len(d),
    }


def _fmt(v) -> str:
    if v != v:  # nan check
        return "—"
    return f"{float(v):.3g}"


def _aggregate(df: pd.DataFrame, group_size: int = 128) -> tuple:
    """Aggregate per-input rows into exponent groups for lightweight SVG line plots.

    bf16 has 128 mantissa values per exponent, so grouping by 128 consecutive
    sorted values naturally aligns with exponent boundaries. Reduces 65k scatter
    dots to ~500 line segments (~100× smaller SVG).
    """
    df_s = df.sort_values("x").reset_index(drop=True)
    if len(df_s) <= group_size:
        return df_s["x"].values, df_s["ulp_error"].values
    group_idx = np.arange(len(df_s)) // group_size
    agg = df_s.groupby(group_idx, sort=False).agg(
        x=("x", "first"), ulp=("ulp_error", "max")
    )
    return agg["x"].values, agg["ulp"].values


def plot_ulp_chart(
    df: pd.DataFrame,
    op_name: str,
    variant: str,
    arch: str,
    dtype: str,
    out_path: Path,
    display_lo=None,
    display_hi=None,
):
    df = remove_subnormals(df)
    if display_lo is not None:
        df = df[df["x"] >= display_lo]
    if display_hi is not None:
        df = df[df["x"] <= display_hi]
    if df.empty:
        print(f"  Skipping empty data for {op_name}/{variant}")
        return

    ULP_CLIP = 1000.0
    n_clipped = int((df["ulp_error"].values > ULP_CLIP).sum())

    # Aggregate individual per-input rows into exponent groups → lightweight line plot
    x, ulp_raw = _aggregate(df)
    ulp_vals = np.clip(ulp_raw, 0, ULP_CLIP)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, ulp_vals, color="#e67e22", linewidth=1.5, zorder=3)

    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("asinh", linear_width=0.01)
    ax.set_ylim(bottom=0, top=ULP_CLIP * 1.1)
    ax.set_xlabel("Input x", fontsize=11)
    ax.set_ylabel("ULP Error", fontsize=11)

    variant_str = f" [{variant}]" if variant != "default" else ""
    title = f"ttnn.{op_name}{variant_str} — ULP error  [{arch.upper()}, {dtype}]"
    if n_clipped:
        title += f"\n({n_clipped} inputs clipped at {ULP_CLIP:.0f} — see abs error in report)"
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(REPO_ROOT)}")


def generate_charts(arch_filter=None, dtype_filter=None, op_filter=None):
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from ops_registry import get_registry
        registry = get_registry()
    except ImportError:
        registry = {}

    def get_display_bounds(op_name):
        if op_name in registry:
            lo, hi = registry[op_name].input_range.display_bounds()
            return (lo if lo != float("-inf") else None,
                    hi if hi != float("inf") else None)
        return None, None

    if not DATA_DIR.exists():
        print("No data/ directory found. Run measure_accuracy.py first.")
        return

    index = load_index()

    archs = [arch_filter] if arch_filter else sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    for arch in archs:
        arch_dir = DATA_DIR / arch
        if not arch_dir.is_dir():
            continue
        dtypes = [dtype_filter] if dtype_filter else sorted(d.name for d in arch_dir.iterdir() if d.is_dir())
        for dtype in dtypes:
            dtype_dir = arch_dir / dtype
            if not dtype_dir.is_dir():
                continue
            ops = [op_filter] if op_filter else sorted(d.name for d in dtype_dir.iterdir() if d.is_dir())
            for op_name in ops:
                op_dir = dtype_dir / op_name
                if not op_dir.is_dir():
                    continue
                display_lo, display_hi = get_display_bounds(op_name)
                for csv_path in sorted(op_dir.glob("*.csv")):
                    variant = csv_path.stem
                    df = load_csv(csv_path)
                    if df is None:
                        continue
                    print(f"Plotting {arch}/{dtype}/{op_name}/{variant}")
                    out_svg = CHARTS_DIR / arch / dtype / f"{op_name}_{variant}_ulp.svg"
                    plot_ulp_chart(df, op_name, variant, arch, dtype, out_svg,
                                   display_lo=display_lo, display_hi=display_hi)
                    # Update persistent stats index
                    stats = compute_stats(df, display_lo, display_hi)
                    if stats:
                        index.setdefault(arch, {}).setdefault(dtype, {}).setdefault(op_name, {})[variant] = stats

    save_index(index)
    print(f"Updated {INDEX_FILE.relative_to(REPO_ROOT)}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate SVG accuracy charts")
    p.add_argument("--arch", help="Filter by architecture")
    p.add_argument("--dtype", help="Filter by dtype")
    p.add_argument("--op", help="Filter by op name")
    return p.parse_args()


def main():
    args = parse_args()
    generate_charts(arch_filter=args.arch, dtype_filter=args.dtype, op_filter=args.op)


if __name__ == "__main__":
    main()
