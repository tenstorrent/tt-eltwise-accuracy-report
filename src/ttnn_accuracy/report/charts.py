# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""data/{arch}/{dtype}/{op}/{variant}.csv → SVG charts and the report_index.json summary."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from ttnn_accuracy.config import (
    CDF_POINTS,
    MAX_FINITE,
    MIN_BIN,
    N_BINS,
    NOISE_PCT,
    ULP_CLIP,
    ULP_LINES,
)
from ttnn_accuracy.measure.store import RUN_STAMP
from ttnn_accuracy.ops.plan import describe, resolve
from ttnn_accuracy.paths import CHARTS_DIR, DATA_DIR, INDEX_FILE, PERF_DIR, REPO_ROOT, RUNS_KEY
from ttnn_accuracy.report.score import _finite, _rounded_frac, score_csv

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["figure.dpi"] = 100
plt.rcParams["svg.hashsalt"] = "ttnn-accuracy"  # else ids are random and every chart churns

RULE_COLOURS = ("#16a085", "#27ae60", "#e67e22", "#c0392b", "#8e44ad")  # 0.5, 1, 3, 10, 100 ULP
SERIES_COLOURS = ("#27ae60", "#e67e22", "#c0392b", "#8e44ad")  # p50, p95, p99, worst
UNDEFINED = "#c0392b"  # beyond the op's domain
UNSAMPLED = "#7f8c8d"  # inside it, and never measured


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
    # `ci_run` is the whole sweep, where `run_id` is one shard of it. A sharded sweep writes
    # a commit per shard, so this is the only field that says which commits belong together.
    ci = os.environ.get("GITHUB_RUN_ID")
    index.setdefault(RUNS_KEY, {}).setdefault(arch, {})[dtype] = (
        {k: run[k] for k in ("run_id", "tt_metal_commit", "ttnn_version", "device_arch", "ops")}
        | {"failed": run.get("failed", {})}
        | ({"ci_run": ci} if ci else {})
    )


@dataclass(frozen=True, slots=True)
class _Chart:
    """One variant's rows and everything the three panels read, derived once."""

    df: pd.DataFrame
    scored: pd.DataFrame  # x against |ULP|, unscorable points dropped
    op: str
    variant: str
    arch: str
    dtype: str
    reach: float  # worst scorable |ULP|
    rounded: float  # share of inputs within half a ULP, counted before the group reduction
    covered: tuple[float, float]  # the least and greatest x the sweep wrote


def _chart(df: pd.DataFrame, op: str, variant: str, arch: str, dtype: str) -> _Chart:
    scored = pd.DataFrame({"x": _finite(df["x"]), "ulp": _finite(df["ulp_error"])}).dropna()
    scored = scored[scored["x"].abs() > 0]
    swept = _finite(df["x"]).dropna()  # every row: the drawn series keeps one x per 128
    return _Chart(
        df=df,
        scored=scored,
        op=op,
        variant=variant,
        arch=arch,
        dtype=dtype,
        reach=float(scored["ulp"].max()) if len(scored) else 0.0,
        rounded=_rounded_frac(df),
        covered=(float(swept.min()), float(swept.max())),
    )


def _aggregate(df: pd.DataFrame, group_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Group 128 sorted values, one bf16 exponent, turning 65k dots into ~500 segments."""
    ordered = df.sort_values("x").reset_index(drop=True)
    if len(ordered) <= group_size:
        return ordered["x"].values, ordered["ulp_error"].values
    groups = np.arange(len(ordered)) // group_size
    agg = ordered.groupby(groups, sort=False).agg(x=("x", "first"), ulp=("ulp_error", "max"))
    # Each group is labelled by its first x, so the line would stop short of the sweep's edge.
    return (
        np.append(agg["x"].values, ordered["x"].values[-1]),
        np.append(agg["ulp"].values, ordered["ulp_error"].values[-1]),
    )


def _cdf(positive: np.ndarray, total: int, baseline: float) -> tuple[np.ndarray, np.ndarray]:
    """Fraction within each threshold, opening at the exact-match level so one step is visible."""
    grid = np.geomspace(positive[0], positive[-1], CDF_POINTS)
    exact = total - positive.size
    below = exact + np.searchsorted(positive, grid, side="right")
    return np.insert(grid, 0, baseline), np.insert(below / total, 0, exact / total)


def _bins(scored: pd.DataFrame) -> pd.DataFrame:
    """Bins uniform in asinh(x): equal width in x would hold every point in the one about zero."""
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
    """Each rule appears once the data reaches the one before, leaving one band of headroom."""
    draw = ax.axhline if horizontal else ax.axvline
    shown = 1 + sum(reach >= line for line in ULP_LINES)
    for line, colour in zip(ULP_LINES[:shown], RULE_COLOURS, strict=False):
        draw(line, color=colour, linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)


def _domain_bands(ax, chart: _Chart) -> None:
    """Red beyond the op's domain, grey where the domain reaches further than the sweep did."""
    info = describe(chart.op)
    if not info or not info.bounds:
        return
    lo, hi = info.bounds[chart.dtype]
    limit = MAX_FINITE[chart.dtype]
    # An edge at the format's own limit means the dtype ran out, not that the function ended.
    show_lo = np.isfinite(lo) and abs(lo) < limit
    show_hi = np.isfinite(hi) and abs(hi) < limit
    view_lo, view_hi = ax.get_xlim()
    if show_lo or show_hi:
        # Display space, so one margin rule holds on this panel's symlog axis.
        forward, back = ax.transData.transform, ax.transData.inverted().transform
        margin = 0.12 * (forward((view_hi, 0))[0] - forward((view_lo, 0))[0])
        if show_lo:
            view_lo = back((forward((lo, 0))[0] - margin, 0))[0]
        if show_hi:
            view_hi = back((forward((hi, 0))[0] + margin, 0))[0]
        ax.set_xlim(view_lo, view_hi)
    if show_lo:
        ax.axvspan(view_lo, lo, color=UNDEFINED, alpha=0.07, zorder=0)
    if show_hi:
        ax.axvspan(hi, view_hi, color=UNDEFINED, alpha=0.07, zorder=0)
    for edge, measured in zip((lo, hi), chart.covered, strict=True):
        if np.isfinite(edge) and edge != measured:
            ax.axvspan(*sorted((edge, measured)), color=UNSAMPLED, alpha=0.10, zorder=0)


def _plot_error(ax, chart: _Chart) -> None:
    """ULP against x, clamped: a 1e36 outlier would flatten every other point to the axis."""
    x, ulp = _aggregate(chart.df)
    ax.plot(x, np.clip(ulp, 0, ULP_CLIP), color="#e67e22", linewidth=1.5, zorder=3)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("asinh", linear_width=0.01)
    ax.set_ylim(bottom=0, top=ULP_CLIP * 1.1)
    _reference_lines(ax, min(chart.reach, ULP_CLIP), horizontal=True)
    _domain_bands(ax, chart)
    ax.set_xlabel("Input x", fontsize=11)
    ax.set_ylabel("ULP Error", fontsize=11)
    suffix = "" if chart.variant == "default" else f" [{chart.variant}]"
    title = f"ttnn.{chart.op}{suffix} — ULP error  [{chart.arch.upper()}, {chart.dtype}]"
    if clipped := int((chart.df["ulp_error"].values > ULP_CLIP).sum()):
        title += f"\n({clipped} inputs clipped at {ULP_CLIP:.0f} — see abs error in report)"
    ax.set_title(title, fontsize=11)


def _plot_cdf(ax, chart: _Chart) -> None:
    """What fraction of points sit within N ULP — the question a max cannot answer."""
    values = chart.scored["ulp"].values
    positive = np.sort(values[values > 0])
    if positive.size:
        # The clamp, so charts compare — unless every error is past it and the curve would not fit.
        first = positive[0]
        lo, hi = (
            (first / 2, positive[-1] * 2) if first > ULP_CLIP else (min(first, 1.0) / 2, ULP_CLIP)
        )
        thresholds, fractions = _cdf(positive, values.size, lo)
        ax.step(thresholds, fractions, where="post", color="#2980b9", linewidth=1.5, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        _reference_lines(ax, min(chart.reach, ULP_CLIP), horizontal=False)
    else:
        # Nothing to plot: a 1-ULP rule and a 0-to-1 threshold axis would both read as a bug.
        ax.set_xticks([])
        ax.text(
            0.5,
            0.5,
            "every measured point matches the reference exactly",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=UNSAMPLED,
        )
    ax.set_ylim(bottom=0, top=1.02)
    ax.set_xlabel("|ULP| threshold", fontsize=11)
    ax.set_ylabel("Fraction of points within", fontsize=11)
    ax.set_title(
        f"Within N ULP — {chart.rounded:.1%} correctly rounded, worst {chart.reach:.3g}",
        fontsize=11,
    )


def _plot_bins(ax, chart: _Chart) -> None:
    """Where in the domain the error lives, which the whole-sweep percentiles cannot say."""
    binned = _bins(chart.scored)
    for (column, marker), colour in zip(
        (("p50", "o"), ("p95", "s"), ("p99", "^"), ("worst", "v")),
        SERIES_COLOURS,
        strict=True,
    ):
        ax.plot(
            binned["x"], binned[column], marker, color=colour, markersize=4, label=column, zorder=3
        )
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("symlog", linthresh=1e-3)
    _reference_lines(ax, chart.reach, horizontal=True)
    _domain_bands(ax, chart)
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
    """One variant as three panels: error against x, the distribution, and where it lives."""
    chart = _chart(df, op, variant, arch, dtype)
    fig, panels = plt.subplots(3, 1, figsize=(10, 13))
    for draw, panel in zip((_plot_error, _plot_cdf, _plot_bins), panels, strict=True):
        draw(panel, chart)
    for panel in panels:
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
    # By op, not variant: a dispatch may measure a parameter the plan does not carry.
    scope = {
        arch: {s.name for s in resolve(None, None, arch)[0]}
        for arch in sorted({d.name for d in DATA_DIR.iterdir() if d.is_dir()} | index.keys())
        if arch != RUNS_KEY
    }
    # An op that left scope keeps its CSV, so the index would republish its last measurement.
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


def plot_ulp_comparison(
    was: pd.DataFrame, now: pd.DataFrame, op: str, variant: str, arch: str, dtype: str, out: Path
) -> None:
    """Both kernels' ULP against x on one panel: the table says how much, this says where."""
    chart = _chart(now, op, variant, arch, dtype)
    fig, ax = plt.subplots(figsize=(10, 5))
    # main drawn wide and under, so a branch that changed nothing still shows it as a halo
    # rather than hiding it completely.
    for df, colour, label, width in (
        (was, "#95a5a6", "main", 3.0),
        (now, "#e67e22", "branch", 1.2),
    ):
        x, ulp = _aggregate(df)
        ax.plot(x, np.clip(ulp, 0, ULP_CLIP), color=colour, linewidth=width, label=label, zorder=3)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("asinh", linear_width=0.01)
    ax.set_ylim(bottom=0, top=ULP_CLIP * 1.1)
    _reference_lines(ax, min(chart.reach, ULP_CLIP), horizontal=True)
    _domain_bands(ax, chart)
    ax.set_xlabel("Input x", fontsize=11)
    ax.set_ylabel("ULP Error", fontsize=11)
    suffix = "" if variant == "default" else f" [{variant}]"
    ax.set_title(f"ttnn.{op}{suffix} — ULP error  [{arch.upper()}, {dtype}]", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    # PNG, not SVG: GitHub's image proxy will not render an SVG inside a comment.
    fig.savefig(out, format="png", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
