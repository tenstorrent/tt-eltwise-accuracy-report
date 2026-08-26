"""SVG accuracy charts from measured CSVs, plus the report_index.json summary cache.

Scans data/{arch}/{dtype}/{op}/{variant}.csv and produces
reports/charts/{arch}/{dtype}/{op}_{variant}_ulp.svg.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from ttnn_accuracy.measure.metrics import MIN_NORMAL
from ttnn_accuracy.paths import CHARTS_DIR, DATA_DIR, INDEX_FILE, REPO_ROOT

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["figure.dpi"] = 100

ULP_CLIP = 1000.0
NEAR_ZERO_X = 0.01
NEAR_ZERO_MAX_ABS = 0.5


def _finite(s: pd.Series) -> pd.Series:
    return s.replace([float("inf"), float("-inf")], float("nan"))


def _fmt(v) -> str:
    return "—" if v != v else f"{float(v):.3g}"


def _subdirs(parent: Path, only: str | None) -> list[Path]:
    if only:
        child = parent / only
        return [child] if child.is_dir() else []
    return sorted(d for d in parent.iterdir() if d.is_dir())


def compute_stats(df: pd.DataFrame) -> dict:
    """Summary stats for one variant. `df` must already have subnormals removed."""
    ulp = _finite(df["ulp_error"])
    stats = {
        "max_ulp": _fmt(ulp.max()),
        "mean_ulp": _fmt(ulp.mean()),
        "max_abs": _fmt(_finite(df["abs_error"]).max()),
        "ulp_clipped": int((ulp > ULP_CLIP).sum()),
        "n_inputs": len(df),
    }
    if nz := _near_zero_atol(df):
        stats["near_zero_atol"] = nz
    return stats


def _near_zero_atol(df: pd.DataFrame) -> dict | None:
    """Absolute tolerance for the region where ULP is inflated because the output is ~0.

    Some ops (silu, elu, gelu_fast_approx) produce near-zero outputs for inputs close to
    x=0, where ULP is ill-defined. A small absolute error there confirms ULP inflation
    rather than a real defect, so a large one is reported as ULP instead.
    """
    near0 = df[(df["x"].abs() < NEAR_ZERO_X) & (df["ulp_error"] > ULP_CLIP)]
    if near0.empty:
        return None
    max_abs = _finite(near0["abs_error"]).dropna().max()
    if not (max_abs <= NEAR_ZERO_MAX_ABS):  # also rejects an all-NaN region
        return None
    return {
        "x_lo": _fmt(near0["x"].min()),
        "x_hi": _fmt(near0["x"].max()),
        "max_abs": _fmt(max_abs),
        "n": len(near0),
    }


def _aggregate(df: pd.DataFrame, group_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-input rows into exponent groups for a lightweight SVG.

    bf16 has 128 mantissa values per exponent, so grouping 128 consecutive sorted values
    aligns with exponent boundaries. Turns 65k scatter dots into ~500 line segments.
    """
    ordered = df.sort_values("x").reset_index(drop=True)
    if len(ordered) <= group_size:
        return ordered["x"].values, ordered["ulp_error"].values
    groups = np.arange(len(ordered)) // group_size
    agg = ordered.groupby(groups, sort=False).agg(x=("x", "first"), ulp=("ulp_error", "max"))
    return agg["x"].values, agg["ulp"].values


def plot_ulp_chart(df: pd.DataFrame, op: str, variant: str, arch: str, dtype: str, out: Path):
    n_clipped = int((df["ulp_error"].values > ULP_CLIP).sum())
    x, ulp = _aggregate(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, np.clip(ulp, 0, ULP_CLIP), color="#e67e22", linewidth=1.5, zorder=3)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("asinh", linear_width=0.01)
    ax.set_ylim(bottom=0, top=ULP_CLIP * 1.1)
    ax.set_xlabel("Input x", fontsize=11)
    ax.set_ylabel("ULP Error", fontsize=11)

    suffix = "" if variant == "default" else f" [{variant}]"
    title = f"ttnn.{op}{suffix} — ULP error  [{arch.upper()}, {dtype}]"
    if n_clipped:
        title += f"\n({n_clipped} inputs clipped at {ULP_CLIP:.0f} — see abs error in report)"
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    logger.success("{}", out.relative_to(REPO_ROOT))


def generate_charts(arch_filter=None, dtype_filter=None, op_filter=None) -> int:
    if not DATA_DIR.exists():
        logger.error("no data/ directory — run `ttnn-accuracy measure` first")
        return 1

    index = json.loads(INDEX_FILE.read_text()) if INDEX_FILE.exists() else {}
    plotted = 0

    for arch_dir in _subdirs(DATA_DIR, arch_filter):
        for dtype_dir in _subdirs(arch_dir, dtype_filter):
            for op_dir in _subdirs(dtype_dir, op_filter):
                arch, dtype, op = arch_dir.name, dtype_dir.name, op_dir.name
                for csv_path in sorted(op_dir.glob("*.csv")):
                    variant = csv_path.stem
                    raw = pd.read_csv(csv_path, index_col="index")
                    df = raw[raw["x"].abs() >= MIN_NORMAL]
                    if df.empty:
                        logger.warning("no normal-range rows in {}", csv_path)
                        continue
                    logger.info("plotting {}/{}/{}/{}", arch, dtype, op, variant)
                    svg = CHARTS_DIR / arch / dtype / f"{op}_{variant}_ulp.svg"
                    plot_ulp_chart(df, op, variant, arch, dtype, svg)
                    index.setdefault(arch, {}).setdefault(dtype, {}).setdefault(op, {})[variant] = (
                        compute_stats(df)
                    )
                    plotted += 1

    if not plotted:
        logger.error("no CSVs matched arch={} dtype={} op={}", arch_filter, dtype_filter, op_filter)
        return 1

    INDEX_FILE.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    logger.success("{} charts, updated {}", plotted, INDEX_FILE.relative_to(REPO_ROOT))
    return 0
