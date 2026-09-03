"""SVG accuracy charts from measured CSVs, plus the report_index.json summary cache.

Scans data/{arch}/{dtype}/{op}/{variant}.csv and produces
reports/charts/{arch}/{dtype}/{op}_{variant}_ulp.svg.
"""

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

from ttnn_accuracy.measure.metrics import MIN_NORMAL
from ttnn_accuracy.measure.schema import COLUMNS
from ttnn_accuracy.measure.store import RUN_STAMP
from ttnn_accuracy.ops.overrides import EXCLUDED, OVERRIDES, variant_slug
from ttnn_accuracy.ops.plan import describe
from ttnn_accuracy.paths import CHARTS_DIR, DATA_DIR, INDEX_FILE, PERF_DIR, REPO_ROOT, RUNS_KEY

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["figure.dpi"] = 100
plt.rcParams["svg.hashsalt"] = "ttnn-accuracy"  # else ids are random and every chart churns

ULP_CLIP = 1000.0
USABLE_ULP = 2.0  # what "still accurate here" means for the usable-range figure


def _defects(df: pd.DataFrame) -> pd.Series:
    """Wrong answers that carry no ULP: inf or zero where a representable value exists.

    `mismatch` also covers the device answering where the reference is NaN — a disagreement
    about the domain, not about precision — so it counts only where y_ref is finite.
    """
    return (df["outcome"] == "zeroed") | (
        (df["outcome"] == "mismatch") & df["y_ref"].notna() & np.isfinite(df["y_ref"])
    )


def _finite(s: pd.Series) -> pd.Series:
    return s.replace([float("inf"), float("-inf")], float("nan"))


def _fmt(v) -> str:
    return "—" if v != v else f"{float(v):.3g}"


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def verdict(
    max_ulp: str, mean_ulp: str, usable_to: str, operands: int | None, defects: int, points: int
) -> str:
    """One of seven fixed phrases, so no reader invents the judgment.

    The rules live here alone; analyze-report/contract.md is their twin and moves with them.
    """
    mx = _num(max_ulp)
    # First, and whatever the ULP says: an op that returns inf or zero where a value
    # exists is not accurate to anything, and every other figure here excludes those points.
    if defects:
        return f"{defects} of {points} points returned inf or zero where a value exists"
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
    """Largest finite |x| below which no point yet exceeds USABLE_ULP; NaN when none does.

    One maximum hides a cliff: `sin` holds to 2 ULP out to 2.6e5 and collapses past it.
    Unary only — with a sampled partner one bad pair would latch the running maximum.
    """
    if "x2" in df.columns:
        return float("nan")
    finite = df[_finite(df["x"]).notna()]
    ordered = finite.reindex(finite["x"].abs().sort_values().index)
    # NaN ULP means unscorable, which a flush is and a defect is not: 0.0 would let an
    # infinity extend the usable range it ends.
    ulp = _finite(ordered["ulp_error"]).fillna(0.0)
    within = ulp.mask(_defects(ordered), float("inf")).cummax() <= USABLE_ULP
    # `or nan`: a bound of 0 means only x=0 held, and "accurate to |x| <= 0" reads as a
    # range where there is none — the no-usable-range phrase is the honest one.
    return (abs(ordered["x"][within].iloc[-1]) or float("nan")) if within.any() else float("nan")


def _subdirs(parent: Path, only: str | None) -> list[Path]:
    if only:
        child = parent / only
        return [child] if child.is_dir() else []
    return sorted(d for d in parent.iterdir() if d.is_dir())


def _record_perf(index: dict, arch: str) -> None:
    """Attach `stats/perf/{arch}.json` to its entries, with the host that took it."""
    from ttnn_accuracy.measure.perf import NOISE_PCT  # here so `charts` need not import ttnn

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


def compute_stats(df: pd.DataFrame) -> dict:
    """Summary stats for one variant; `df` must already have subnormals removed.

    ULP covers only defined, non-trivial points: the mean would be diluted by ops that
    return zero across most of their range, the max produced 1e24 readings.
    """
    ulp = _finite(df["ulp_error"])
    return {
        "max_ulp": _fmt(ulp.max()),
        "mean_ulp": _fmt(ulp[ulp > 0].mean()),
        "usable_to": _fmt(_usable_to(df)),
        "max_abs": _fmt(_finite(df["abs_error"]).max()),
        "ulp_clipped": int((ulp > ULP_CLIP).sum()),
        "defects": int(_defects(df).sum()),
        "n_inputs": len(df),
        "outcomes": {k: int(v) for k, v in df["outcome"].value_counts().items()},
    }


def score_csv(path: Path, op: str, variant: str) -> tuple[dict, pd.DataFrame] | None:
    """One measured CSV → its index entry and the rows worth plotting, or None.

    Shared with `check`, so a developer's numbers are produced exactly as the report's are.
    """
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
    )
    ov = next(
        (o for o in OVERRIDES.get(f"ttnn.{op}", ()) if variant_slug(o.params_desc) == variant),
        None,
    )
    if ov and ov.why:
        stats["rationale"] = ov.why
    return stats, df


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
    # Date suppressed: matplotlib stamps one, and it rewrote every chart nightly.
    fig.savefig(out, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    logger.success("{}", os.path.relpath(out, REPO_ROOT))  # relative_to raises off-tree


def generate_charts(arch_filter=None, dtype_filter=None, op_filter=None) -> int:
    if not DATA_DIR.exists():
        logger.error("no data/ directory — run `ttnn-accuracy measure` first")
        return 1

    index = json.loads(INDEX_FILE.read_text()) if INDEX_FILE.exists() else {}
    # charts only visits ops with CSVs, so an op that left scope would keep its entry.
    for arch, dtypes in index.items():
        if arch == RUNS_KEY:
            continue
        for ops in dtypes.values():
            for gone in [op for op in ops if f"ttnn.{op}" in EXCLUDED]:
                del ops[gone]
                logger.info("dropped {} from the index — excluded from scope", gone)
    plotted = stale = 0

    for arch_dir in _subdirs(DATA_DIR, arch_filter):
        for dtype_dir in _subdirs(arch_dir, dtype_filter):
            _record_run(index, arch_dir.name, dtype_dir.name, dtype_dir / RUN_STAMP)
            for op_dir in _subdirs(dtype_dir, op_filter):
                arch, dtype, op = arch_dir.name, dtype_dir.name, op_dir.name
                for csv_path in sorted(op_dir.glob("*.csv")):
                    variant = csv_path.stem
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
    logger.success("{} charts, updated {}", plotted, INDEX_FILE.relative_to(REPO_ROOT))
    if stale:
        logger.error("{} CSV(s) skipped as stale", stale)
    return 1 if stale else 0
