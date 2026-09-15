"""One op's page: every figure measured for it, on one arch and dtype, variants side by side."""

from __future__ import annotations

from ttnn_accuracy.config import MIN_NORMAL
from ttnn_accuracy.measure.sweeps import SAMPLED
from ttnn_accuracy.ops.plan import describe, params_desc
from ttnn_accuracy.paths import REPORTS_DIR, RUNS_KEY
from ttnn_accuracy.report.index import (
    ARCH_DISPLAY,
    DTYPE_DISPLAY,
    GENERATED_NOTE,
    _get_index,
    _usable_text,
    chart_rel_path,
    load_summary,
)

STATS_ROWS = (
    ("Max ULP", "max_ulp"),
    ("Mean ULP", "mean_ulp"),
    ("p50 ULP", "p50_ulp"),
    ("p95 ULP", "p95_ulp"),
    ("p99 ULP", "p99_ulp"),
    ("Exact fraction", "exact_frac"),
    ("Accurate to \\|x\\|", "usable_to"),
    ("Max abs error", "max_abs"),
    ("Max rel error", "max_rel"),
    ("Median rel error", "median_rel"),
    ("Bits of precision (worst)", "bits_worst"),
    ("Bits of precision (median)", "bits_median"),
)


def _input_range_text(op_key: str, dtype: str) -> str:
    """The range this op was measured over. Derived bounds differ per dtype."""
    e = describe(op_key)
    if not e or not e.bounds:
        return "unknown"
    lo, hi = e.bounds[dtype]

    def _s(v: float) -> str:
        if abs(v) <= MIN_NORMAL * 2:
            return "0" if v == 0 else (r"min\_normal" if v > 0 else r"\-min\_normal")
        return f"{v:.4g}"

    match lo == float("-inf"), hi == float("inf"):
        case True, True:
            return "all normal values"
        case True, False:
            return f"x ≤ {_s(hi)}"
        case False, True:
            return f"x ≥ {_s(lo)}"
        case _:
            return f"x ∈ [{_s(lo)}, {_s(hi)}]"


def _cell(summary: dict, key: str) -> str:
    if key == "usable_to":
        return _usable_text(summary)
    value = summary.get(key, "—")
    return value + " ⚠" if key == "max_ulp" and summary.get("ulp_clipped") else value


def _stats_table(arch: str, dtype: str, entries: list[tuple[str, str]], unary: bool) -> str:
    """A column per parameter set: twelve figures against three variants does not fit as rows.

    A row absent from every summary is dropped, so an index measured before a figure existed
    reads as a shorter table rather than a wall of em dashes."""
    summaries = [(params_desc(k, v), load_summary(arch, dtype, k, v) or {}) for k, v in entries]
    rows = [
        f"| {label} | " + " | ".join(_cell(s, key) for _, s in summaries) + " |\n"
        for label, key in STATS_ROWS
        if (unary or key != "usable_to") and any(key in s for _, s in summaries)
    ]
    head = "| | " + " | ".join(f"`{p}`" for p, _ in summaries) + " |\n"
    rule = "|---" * (len(summaries) + 1) + "|\n"
    return head + rule + "".join(rows) + "\n"


def _specials_table(arch: str, dtype: str, entries: list[tuple[str, str]]) -> str:
    """Device beside golden at ±0, ±inf, NaN and the smallest normals; no ULP exists there."""
    rows = []
    for op_key, variant in entries:
        summary = load_summary(arch, dtype, op_key, variant)
        for r in (summary or {}).get("specials", []):
            mark = "agree" if r["y"] == r["y_ref"] else "**differ**"
            rows.append(
                f"| `{params_desc(op_key, variant)}` | {r['x']} | {r['y']} | {r['y_ref']} | {mark} |\n"
            )
    if not rows:
        return ""
    return (
        "### Special values\n\n"
        "| Variant | x | device | golden | |\n|---|---|---|---|---|\n" + "".join(rows) + "\n"
    )


def _outcome_table(arch: str, dtype: str, entries: list[tuple[str, str]]) -> str:
    """What each measured point demonstrated. Only `exact` and `inexact` carry a ULP."""
    counted = [(k, v, load_summary(arch, dtype, k, v)) for k, v in entries]
    counted = [(k, v, s) for k, v, s in counted if s and s.get("outcomes")]
    if not counted:
        return ""
    kinds = sorted({k for _, _, s in counted for k in s["outcomes"]})
    head = "| Parameters | " + " | ".join(kinds) + " |\n"
    rule = "|------------" + "|------" * len(kinds) + "|\n"
    rows = [
        f"| `{params_desc(k, v)}` | "
        + " | ".join(f"{s['outcomes'].get(kind, 0):,}" for kind in kinds)
        + " |\n"
        for k, v, s in counted
    ]
    return "**Outcomes**\n\n" + head + rule + "".join(rows) + "\n"


def _rows_of(arch: str, dtype: str, entries: list[tuple[str, str]], field: str) -> list[tuple]:
    """(params, value) for every variant that carries `field`, so an empty section is dropped."""
    found = [
        (params_desc(k, v), (load_summary(arch, dtype, k, v) or {}).get(field)) for k, v in entries
    ]
    return [(params, value) for params, value in found if value]


def _operand_headers(rows: list[dict]) -> list[str]:
    """`x`, plus the partner columns a pair or triple sweep recorded in the same row."""
    return [c for c in ("x", "x2", "x3") if any(c in row for row in rows)]


def _offenders_table(arch: str, dtype: str, entries: list[tuple[str, str]]) -> str:
    """The worst points by value. A maximum tells you how bad; this tells you where to look."""
    found = _rows_of(arch, dtype, entries, "offenders")
    if not found:
        return ""
    inputs = _operand_headers([row for _, rows in found for row in rows])
    head = "| Parameters | " + " | ".join(inputs) + " | golden | device | ULP |\n"
    rule = "|---" * (len(inputs) + 4) + "|\n"
    body = "".join(
        f"| `{params}` | " + " | ".join(row.get(c, "—") for c in inputs) + f" | {row['y_ref']} "
        f"| {row['y']} | {row['ulp']} |\n"
        for params, rows in found
        for row in rows
    )
    return "**Worst points**\n\n" + head + rule + body + "\n"


def _monotonic_table(arch: str, dtype: str, entries: list[tuple[str, str]]) -> str:
    """Ordering is not accuracy: an op can sit within 1 ULP and still step backwards."""
    found = _rows_of(arch, dtype, entries, "monotonic")
    if not found:
        return ""
    head = "| Parameters | Pairs | Violations | Rate | Worst \\|Δy\\| |\n|---|---|---|---|---|\n"
    body = "".join(
        f"| `{params}` | {m['pairs']:,} | {m['violations']:,} | {m['rate']} | {m['worst_dy']} |\n"
        for params, m in found
    )
    worst = "".join(
        f"| `{params}` | {row['x_from']} → {row['x_to']} | {row['y_from']} → {row['y_to']} "
        f"| {row['dy']} |\n"
        for params, m in found
        for row in m["top"]
    )
    if worst:
        worst = (
            "\n_Worst violations — the device output moved against the reference's own "
            "direction between these neighbouring inputs._\n\n"
            "| Parameters | x | device | \\|Δy\\| |\n|---|---|---|---|\n" + worst
        )
    return (
        "**Monotonicity**\n\n_Checked only where the reference is itself ordered, and never "
        "across a discontinuity._\n\n" + head + body + worst + "\n"
    )


def _nonfinite_table(arch: str, dtype: str, entries: list[tuple[str, str]]) -> str:
    """Which side went non-finite. These points carry no ULP, so no other figure counts them."""
    found = _rows_of(arch, dtype, entries, "nonfinite")
    if not found:
        return ""
    head = (
        "| Parameters | Points | Both | Device only | Golden only | Device inf | Device nan "
        "| Golden inf | Golden nan |\n" + "|---" * 9 + "|\n"
    )
    body = "".join(
        f"| `{params}` | {n['total']:,} | {n['both']:,} | {n['device_only']:,} "
        f"| {n['golden_only']:,} | {n['device_inf']:,} | {n['device_nan']:,} "
        f"| {n['golden_inf']:,} | {n['golden_nan']:,} |\n"
        for params, n in found
    )
    inputs = _operand_headers([row for _, n in found for row in n["detail"]])
    detail = "".join(
        f"| `{params}` | "
        + " | ".join(row.get(c, "—") for c in inputs)
        + f" | {row['y_ref']} | {row['y']} |\n"
        for params, n in found
        for row in n["detail"]
    )
    return (
        "**Non-finite outputs**\n\n"
        + head
        + body
        + "\n| Parameters | "
        + " | ".join(inputs)
        + " | golden | device |\n"
        + "|---" * (len(inputs) + 3)
        + "|\n"
        + detail
        + "\n"
    )


def _sampling_note(dtype: str, entries: list[tuple[str, str]]) -> str:
    """A sampled maximum is a lower bound; arity and dtype say which sweeps sample."""
    info = describe(entries[0][0])
    how = SAMPLED.get((info.operands, dtype)) if info else None
    if not how:
        return ""
    return (
        f"> **Sampled, not exhaustive.** For this op {how}. The figures above are a lower "
        "bound on the error, and comparable between releases, but they are not a bound "
        "over the dtype.\n\n"
    )


def _provenance(arch: str, dtype: str, entries: list[tuple[str, str]]) -> str:
    """Only claim a run for ops that run actually covered — silence would read as fresh."""
    run = _get_index().get(RUNS_KEY, {}).get(arch, {}).get(dtype)
    if not run or not all(op_key in run["ops"] for op_key, _ in entries):
        return "> **Provenance unknown.** These figures predate run tracking. Re-measure to attribute them.\n\n"
    return (
        f"_Measured on {run['device_arch']} · tt-metal `{run['tt_metal_commit']}` · "
        f"ttnn `{run['ttnn_version']}` · run `{run['run_id']}`_\n\n"
    )


def op_detail_page(arch: str, dtype: str, display_name: str, entries: list[tuple[str, str]]) -> str:
    """reports/by_arch/{arch}/{dtype}/{name}.md — `entries` is every parameter set for the op."""
    page_path = REPORTS_DIR / "by_arch" / arch / dtype / f"{display_name}.md"

    lines = [
        f"# {display_name} — {ARCH_DISPLAY.get(arch, arch)}, {DTYPE_DISPLAY.get(dtype, dtype)}\n",
        GENERATED_NOTE,
        f"**Architecture:** {ARCH_DISPLAY.get(arch, arch)}  \n",
        f"**Data type:** {DTYPE_DISPLAY.get(dtype, dtype)}  \n",
    ]
    lines.append("\n---\n\n")
    lines.append(
        f"[← {ARCH_DISPLAY.get(arch, arch)} {DTYPE_DISPLAY.get(dtype, dtype)} ops](README.md) | "
        f"[All archs for {display_name}](../../../by_op/{display_name}/README.md) | "
        f"[Top](../../../README.md)\n\n"
    )

    # Any entry will do: everything on this page shares a display_name, so one range
    range_text = _input_range_text(entries[0][0], dtype)
    lines.append(f"**Measured input range:** {range_text}  \n\n")

    # Unary only: elsewhere a bound on x would describe the sampled partner.
    info = describe(entries[0][0])
    lines.append(_stats_table(arch, dtype, entries, bool(info and info.operands == 1)))
    for op_key, variant in entries:
        s = load_summary(arch, dtype, op_key, variant)
        if s and s.get("verdict"):
            why = f" · _{s['rationale']}_" if s.get("rationale") else ""
            lines.append(f"**{params_desc(op_key, variant)}** — {s['verdict']}{why}  \n")
    lines.append("\n")
    lines.append(_outcome_table(arch, dtype, entries))
    lines.append(_offenders_table(arch, dtype, entries))
    lines.append(_monotonic_table(arch, dtype, entries))
    lines.append(_nonfinite_table(arch, dtype, entries))
    lines.append(_specials_table(arch, dtype, entries))
    lines.append(_sampling_note(dtype, entries))
    lines.append(_provenance(arch, dtype, entries))

    # One chart per parameter set
    multi = len(entries) > 1
    for op_key, variant in entries:
        params = params_desc(op_key, variant)
        if multi:
            lines.append(f"### `{params}`\n\n")
        chart = chart_rel_path(arch, dtype, op_key, variant, page_path.parent)
        lines.append(f"![ULP error — {display_name} {params}]({chart})\n\n")

    return "".join(lines)
