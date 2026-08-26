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
from ttnn_accuracy.measure.schema import COLUMNS
from ttnn_accuracy.measure.store import RUN_STAMP
from ttnn_accuracy.paths import CHARTS_DIR, DATA_DIR, INDEX_FILE, REPO_ROOT

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["figure.dpi"] = 100

# Underscored so it cannot collide with an arch name when the report walks the index.
RUNS_KEY = "_runs"
ULP_CLIP = 1000.0


def _finite(s: pd.Series) -> pd.Series:
    return s.replace([float("inf"), float("-inf")], float("nan"))


def _fmt(v) -> str:
    return "—" if v != v else f"{float(v):.3g}"


def _subdirs(parent: Path, only: str | None) -> list[Path]:
    if only:
        child = parent / only
        return [child] if child.is_dir() else []
    return sorted(d for d in parent.iterdir() if d.is_dir())


def _record_run(index: dict, arch: str, dtype: str, stamp: Path) -> None:
    """Carry the measuring run into the index, so a page can say what produced its numbers."""
    if not stamp.exists():
        return
    run = json.loads(stamp.read_text())
    index.setdefault(RUNS_KEY, {}).setdefault(arch, {})[dtype] = {
        k: run[k] for k in ("run_id", "tt_metal_commit", "ttnn_version", "device_arch", "ops")
    }


def compute_stats(df: pd.DataFrame) -> dict:
    """Summary stats for one variant. `df` must already have subnormals removed.

    ULP figures cover only the points where ULP is defined and non-trivial: the mean over
    exact points would be diluted by ops that return zero across most of their range, and
    the max over undefined points is what produced 1e24 readings.
    """
    ulp = _finite(df["ulp_error"])
    inexact = ulp[df["outcome"] == "inexact"]
    return {
        "max_ulp": _fmt(ulp.max()),
        "mean_ulp": _fmt(inexact.mean()),
        "max_abs": _fmt(_finite(df["abs_error"]).max()),
        "ulp_clipped": int((ulp > ULP_CLIP).sum()),
        "n_inputs": len(df),
        "outcomes": {k: int(v) for k, v in df["outcome"].value_counts().items()},
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
    plotted = stale = 0

    for arch_dir in _subdirs(DATA_DIR, arch_filter):
        for dtype_dir in _subdirs(arch_dir, dtype_filter):
            _record_run(index, arch_dir.name, dtype_dir.name, dtype_dir / RUN_STAMP)
            for op_dir in _subdirs(dtype_dir, op_filter):
                arch, dtype, op = arch_dir.name, dtype_dir.name, op_dir.name
                for csv_path in sorted(op_dir.glob("*.csv")):
                    variant = csv_path.stem
                    raw = pd.read_csv(csv_path, index_col="index")
                    if missing := set(COLUMNS) - set(raw.columns):
                        logger.error(
                            "{} predates the current schema, missing {} — re-measure it",
                            csv_path,
                            ", ".join(sorted(missing)),
                        )
                        stale += 1
                        continue
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
    if stale:
        logger.error("{} CSV(s) skipped as stale", stale)
    return 1 if stale else 0
