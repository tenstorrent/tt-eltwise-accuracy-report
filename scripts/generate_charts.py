#!/usr/bin/env python3
"""
Generate SVG accuracy charts from CSV measurement data.

Scans data/{arch}/{dtype}/{op}/{variant}.csv and produces
reports/charts/{arch}/{dtype}/{op}_{variant}_ulp.svg for each.

Usage:
    python generate_charts.py
    python generate_charts.py --arch wh --dtype bf16
    python generate_charts.py --arch wh --dtype bf16 --op exp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
CHARTS_DIR = REPO_ROOT / "reports" / "charts"

plt.rcParams["svg.fonttype"] = "none"  # keep text as text, not paths
plt.rcParams["figure.dpi"] = 100

ULP_LINES = [
    (1, "1 ULP", "#2ecc71"),
    (3, "3 ULP", "#f39c12"),
    (10, "10 ULP", "#e74c3c"),
]

MIN_NORMAL_F32 = 2**-126
MIN_NORMAL_BF16 = 2**-126


def remove_subnormals(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["x"].abs() >= MIN_NORMAL_F32
    return df[mask]


def load_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, index_col="index")
        return df
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}")
        return None


def plot_ulp_chart(
    df: pd.DataFrame,
    op_name: str,
    variant: str,
    arch: str,
    dtype: str,
    out_path: Path,
    display_lo: float | None = None,
    display_hi: float | None = None,
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

    fig, ax = plt.subplots(figsize=(10, 5))

    x = df["x"].values
    ulp_vals = np.clip(df["ulp_error"].values, 0, ULP_CLIP)
    n_clipped = int((df["ulp_error"].values > ULP_CLIP).sum())

    # Individual data points (bf16 exhaustive, group_size=1): scatter — one dot per input.
    # Grouped data (fp32): line — one point per batch of inputs (worst-case ULP shown).
    individual = len(df) > 5000
    if individual:
        ax.scatter(x, ulp_vals, s=1.5, alpha=0.4, color="#3498db",
                   label="ULP error", zorder=3, linewidths=0)
    else:
        ax.plot(x, ulp_vals, color="#3498db", linewidth=0.8,
                label="ULP error (worst case per batch)", zorder=3)

    for level, label, color in ULP_LINES:
        ax.axhline(y=level, color=color, linewidth=1.2, linestyle=":", alpha=0.9, label=label, zorder=2)

    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("asinh", linear_width=0.01)
    ax.set_ylim(bottom=0, top=ULP_CLIP * 1.1)

    ax.set_xlabel("Input x", fontsize=11)
    ax.set_ylabel("ULP Error", fontsize=11)
    variant_str = f" [{variant}]" if variant != "default" else ""
    title = f"ttnn.{op_name}{variant_str} — ULP error  [{arch.upper()}, {dtype}]"
    if n_clipped:
        title += f"\n({n_clipped} inputs clipped at {ULP_CLIP:.0f} ULP — real hardware behaviour, see abs error)"
    ax.set_title(title, fontsize=11)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
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

    def get_display_bounds(op_name: str):
        if op_name in registry:
            entry = registry[op_name]
            lo, hi = entry.input_range.display_bounds()
            return lo if lo != float("-inf") else None, hi if hi != float("inf") else None
        return None, None

    archs = [arch_filter] if arch_filter else sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    for arch in archs:
        arch_dir = DATA_DIR / arch
        if not arch_dir.is_dir():
            print(f"No data directory for arch={arch}")
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

                    out_svg = CHARTS_DIR / arch / dtype / f"{op_name}_{variant}_ulp.svg"
                    print(f"Plotting {arch}/{dtype}/{op_name}/{variant}")
                    plot_ulp_chart(df, op_name, variant, arch, dtype, out_svg,
                                   display_lo=display_lo, display_hi=display_hi)


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
