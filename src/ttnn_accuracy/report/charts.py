"""data/{arch}/{dtype}/{op}/{variant}.csv → SVG charts and the report_index.json summary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from ttnn_accuracy.config import (
    CDF_POINTS,
    MIN_BIN,
    N_BINS,
    NOISE_PCT,
    ULP_CLIP,
    ULP_LINES,
)
from ttnn_accuracy.measure.store import RUN_STAMP
from ttnn_accuracy.ops.plan import resolve
from ttnn_accuracy.paths import CHARTS_DIR, DATA_DIR, INDEX_FILE, PERF_DIR, REPO_ROOT, RUNS_KEY
from ttnn_accuracy.report.score import _finite, score_csv

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["figure.dpi"] = 100
plt.rcParams["svg.hashsalt"] = "ttnn-accuracy"  # else ids are random and every chart churns


def _subdirs(parent: Path, only: str | None) -> list[Path]:
    if only:
        child = parent / only
        return [child] if child.is_dir() else []
    return sorted(d for d in parent.iterdir() if d.is_dir())


def _record_perf(index: dict, arch: str) -> None:
    """Attach `stats/perf/{arch}.json` to its entries, with the host that took it."""
    path = PERF_DIR / f"{arch}.json"
    if not path.exists():
        return
    timings = json.loads(path.read_text())
    host = timings.get(RUNS_KEY, {}).get("host")
    attached = 0
    for dtype, ops in timings.items():
        if dtype == RUNS_KEY:
            continue
        # On the run record too: "Measured against" names the host, and could not.
        index.setdefault(RUNS_KEY, {}).setdefault(arch, {}).setdefault(dtype, {})["host"] = host
        for op, variants in ops.items():
            for variant, row in variants.items():
                entry = index.get(arch, {}).get(dtype, {}).get(op, {}).get(variant)
                if entry is not None:
                    # What a page shows; us_min and us_p90 stay in stats/perf for perf-diff.
                    entry["perf"] = {k: row[k] for k in ("us_median", "melem_per_s")}
                    entry["perf"]["host"] = host
                    if row["spread_pct"] > NOISE_PCT:
                        entry["perf"]["spread_pct"] = row["spread_pct"]
                    attached += 1
    logger.info("{} timings from {} attached to the index", attached, path.name)


def _record_run(index: dict, arch: str, dtype: str, stamp: Path) -> None:
    """Carry the measuring run into the index, so a page can say what produced its numbers."""
    if not stamp.exists():
        return
    run = json.loads(stamp.read_text())
    index.setdefault(RUNS_KEY, {}).setdefault(arch, {})[dtype] = {
        k: run[k] for k in ("run_id", "tt_metal_commit", "ttnn_version", "device_arch", "ops")
    } | {"failed": run.get("failed", {})}


def _aggregate(df: pd.DataFrame, group_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Group 128 sorted values, one bf16 exponent, turning 65k dots into ~500 segments."""
    ordered = df.sort_values("x").reset_index(drop=True)
    if len(ordered) <= group_size:
        return ordered["x"].values, ordered["ulp_error"].values
    groups = np.arange(len(ordered)) // group_size
    agg = ordered.groupby(groups, sort=False).agg(x=("x", "first"), ulp=("ulp_error", "max"))
    return agg["x"].values, agg["ulp"].values


def _scored(df: pd.DataFrame) -> pd.DataFrame:
    """x against |ULP|, unscorable points dropped — what every distribution panel reads."""
    scored = pd.DataFrame({"x": _finite(df["x"]), "ulp": _finite(df["ulp_error"])}).dropna()
    return scored[scored["x"].abs() > 0]


def _cdf(ulp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of points at or below each threshold. Zero cannot sit on a log axis, so the
    curve starts at the exact-match fraction instead of at 0."""
    positive = np.sort(ulp[ulp > 0])
    if not positive.size:
        return np.array([]), np.array([])
    grid = np.geomspace(positive[0], positive[-1], CDF_POINTS)
    below = ulp.size - positive.size + np.searchsorted(positive, grid, side="right")
    return grid, below / ulp.size


def _bins(scored: pd.DataFrame) -> pd.DataFrame:
    """Bins uniform in asinh(x), not in x: an exhaustive sweep is uniform over exponents,
    so 32 bins of equal width in x would put every point in the one containing zero."""
    spaced = np.arcsinh(scored["x"].values)
    edges = np.linspace(spaced.min(), spaced.max(), N_BINS + 1)
    at = np.clip(np.searchsorted(edges, spaced, side="right") - 1, 0, N_BINS - 1)
    binned = scored.groupby(at).agg(
        x=("x", "min"),
        n=("ulp", "size"),
        p50=("ulp", lambda s: s.quantile(0.50)),
        p95=("ulp", lambda s: s.quantile(0.95)),
        p99=("ulp", lambda s: s.quantile(0.99)),
        worst=("ulp", "max"),
    )
    return binned[binned["n"] >= MIN_BIN]


def _reference_lines(ax, reach: float, horizontal: bool) -> None:
    """A 100-ULP rule under data that never leaves 1 ULP only compresses the interesting part."""
    draw = ax.axhline if horizontal else ax.axvline
    for line, colour in zip(ULP_LINES, ("#27ae60", "#e67e22", "#c0392b", "#8e44ad"), strict=True):
        if reach >= line:
            draw(line, color=colour, linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)


def _plot_error(ax, df: pd.DataFrame, op: str, variant: str, arch: str, dtype: str) -> None:
    """ULP against x, clamped: a 1e36 outlier would flatten every other point to the axis."""
    x, ulp = _aggregate(df)
    ax.plot(x, np.clip(ulp, 0, ULP_CLIP), color="#e67e22", linewidth=1.5, zorder=3)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("asinh", linear_width=0.01)
    ax.set_ylim(bottom=0, top=ULP_CLIP * 1.1)
    ax.set_xlabel("Input x", fontsize=11)
    ax.set_ylabel("ULP Error", fontsize=11)
    suffix = "" if variant == "default" else f" [{variant}]"
    title = f"ttnn.{op}{suffix} — ULP error  [{arch.upper()}, {dtype}]"
    if n_clipped := int((df["ulp_error"].values > ULP_CLIP).sum()):
        title += f"\n({n_clipped} inputs clipped at {ULP_CLIP:.0f} — see abs error in report)"
    ax.set_title(title, fontsize=11)


def _plot_cdf(ax, scored: pd.DataFrame, reach: float) -> None:
    """What fraction of points sit within N ULP — the question a max cannot answer."""
    thresholds, fractions = _cdf(scored["ulp"].values)
    ax.step(thresholds, fractions, where="post", color="#2980b9", linewidth=1.5, zorder=3)
    if len(thresholds):
        ax.set_xscale("log")
        # Every chart stops at the chart clamp: a worst point of 1e36 would otherwise spread
        # 37 decades and hide the bands anyone reads, and no two charts would compare.
        ax.set_xlim(min(thresholds[0], 1.0), ULP_CLIP)
    _reference_lines(ax, min(reach, ULP_CLIP), horizontal=False)
    ax.set_ylim(bottom=0, top=1.02)
    ax.set_xlabel("|ULP| threshold", fontsize=11)
    ax.set_ylabel("Fraction of points within", fontsize=11)
    exact = float((scored["ulp"] == 0).mean()) if len(scored) else 0.0
    ax.set_title(f"Within N ULP — {exact:.1%} of points are exact, worst {reach:.3g}", fontsize=11)


def _plot_bins(ax, scored: pd.DataFrame) -> None:
    """Where in the domain the error lives, which the whole-sweep percentiles cannot say."""
    binned = _bins(scored)
    for column, colour, marker in (
        ("p50", "#27ae60", "o"),
        ("p95", "#e67e22", "s"),
        ("p99", "#c0392b", "^"),
        ("worst", "#8e44ad", "v"),
    ):
        ax.plot(
            binned["x"], binned[column], marker, color=colour, markersize=4, label=column, zorder=3
        )
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Input x", fontsize=11)
    ax.set_ylabel("|ULP| per bin", fontsize=11)
    hidden = N_BINS - len(binned)
    ax.set_title(
        f"Percentiles over {N_BINS} bins uniform in asinh(x)"
        + (f" — {hidden} under {MIN_BIN} samples, hidden" if hidden else ""),
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")


def plot_ulp_chart(df: pd.DataFrame, op: str, variant: str, arch: str, dtype: str, out: Path):
    scored = _scored(df)
    reach = scored["ulp"].max() if len(scored) else 0.0

    fig, (error_ax, cdf_ax, bin_ax) = plt.subplots(3, 1, figsize=(10, 13))
    _plot_error(error_ax, df, op, variant, arch, dtype)
    _plot_cdf(cdf_ax, scored, reach)
    _plot_bins(bin_ax, scored)
    for panel in (error_ax, cdf_ax, bin_ax):
        panel.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Date suppressed: matplotlib stamps one, and it rewrote every chart nightly.
    fig.savefig(out, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    logger.success("{}", os.path.relpath(out, REPO_ROOT))  # relative_to raises off-tree


def generate_charts(arch_filter=None, dtype_filter=None, op_filter=None) -> int:
    if not DATA_DIR.exists():
        logger.error("no data/ directory — run `ttnn-accuracy measure` first")
        return 1

    index = json.loads(INDEX_FILE.read_text()) if INDEX_FILE.exists() else {}
    # An op that left scope keeps its CSV, so the index would republish its last measurement.
    # By op, not by variant: `custom-report` measures at a parameter the plan does not carry.
    scope = {
        arch: {s.name for s in resolve(None, None, arch)[0]}
        for arch in sorted({d.name for d in DATA_DIR.iterdir() if d.is_dir()} | index.keys())
        if arch != RUNS_KEY
    }
    dropped = []
    for arch, dtypes in index.items():
        if arch == RUNS_KEY:
            continue
        for dtype, ops in dtypes.items():
            for gone in [op for op in ops if op not in scope[arch]]:
                del ops[gone]
                dropped.append(f"{arch}/{dtype}/{gone}")
    if dropped:
        logger.info("dropped, no longer in scope: {}", ", ".join(dropped))
    plotted = stale = 0

    for arch_dir in _subdirs(DATA_DIR, arch_filter):
        for dtype_dir in _subdirs(arch_dir, dtype_filter):
            _record_run(index, arch_dir.name, dtype_dir.name, dtype_dir / RUN_STAMP)
            for op_dir in _subdirs(dtype_dir, op_filter):
                arch, dtype, op = arch_dir.name, dtype_dir.name, op_dir.name
                for csv_path in sorted(op_dir.glob("*.csv")):
                    variant = csv_path.stem
                    if op not in scope[arch]:  # else the purge above is undone
                        continue
                    scored = score_csv(csv_path, op, variant)
                    if scored is None:
                        stale += 1
                        continue
                    stats, df = scored
                    logger.info("plotting {}/{}/{}/{}", arch, dtype, op, variant)
                    plot_ulp_chart(
                        df,
                        op,
                        variant,
                        arch,
                        dtype,
                        CHARTS_DIR / arch / dtype / f"{op}_{variant}_ulp.svg",
                    )
                    index.setdefault(arch, {}).setdefault(dtype, {}).setdefault(op, {})[variant] = (
                        stats
                    )
                    plotted += 1
        # After the dtypes, so every entry this arch owns exists to attach a timing to.
        _record_perf(index, arch_dir.name)

    if not plotted:
        logger.error("no CSVs matched arch={} dtype={} op={}", arch_filter, dtype_filter, op_filter)
        return 1

    INDEX_FILE.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    logger.success("{} charts, updated {}", plotted, os.path.relpath(INDEX_FILE, REPO_ROOT))
    if stale:
        logger.error("{} CSV(s) skipped as stale", stale)
    return 1 if stale else 0
