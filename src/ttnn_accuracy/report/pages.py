"""One op's page: every figure measured for it, on one arch and dtype, variants side by side."""

from __future__ import annotations

from ttnn_accuracy.config import MIN_NORMAL, SAMPLED
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
    ("Min ULP", "min_ulp"),
    ("Mean ULP", "mean_ulp"),
    ("Signed bias (ULP)", "bias_ulp"),
    ("p50 ULP", "p50_ulp"),
    ("p95 ULP", "p95_ulp"),
    ("p99 ULP", "p99_ulp"),
    ("Exact fraction", "exact_frac"),
    ("Correctly rounded", "rounded_frac"),
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


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """A markdown table, so no builder writes its own header rule and miscounts the columns."""
    return (
        "| "
        + " | ".join(headers)
        + " |\n"
        + "|---" * len(headers)
        + "|\n"
        + "".join("| " + " | ".join(cells) + " |\n" for cells in rows)
    )


def _summaries(arch: str, dtype: str, entries: list[tuple[str, str]]) -> list[tuple[str, dict]]:
    """(parameters, stats) per variant on this page, in the order the page lists them."""
    return [(params_desc(k, v), load_summary(arch, dtype, k, v) or {}) for k, v in entries]


def _carrying(summaries: list[tuple[str, dict]], field: str) -> list[tuple[str, dict]]:
    """Only the variants holding `field`, so a section absent from the index is dropped whole."""
    return [(params, s[field]) for params, s in summaries if s.get(field)]


def _operand_headers(rows: list[dict]) -> list[str]:
    """`x`, plus the partner columns a pair or triple sweep recorded in the same row."""
    return [c for c in ("x", "x2", "x3") if any(c in row for row in rows)]


def _point_rows(listed: list[tuple[str, dict]], tail: tuple[tuple[str, str], ...] = ()) -> tuple:
    """Headers and cells for stored points: parameters, the inputs, both answers, then `tail`."""
    inputs = _operand_headers([row for _, row in listed])
    headers = ["Parameters", *inputs, "golden", "device", *(header for _, header in tail)]
    return headers, [
        [f"`{params}`", *(row.get(c, "—") for c in inputs), row["y_ref"], row["y"]]
        + [row[key] for key, _ in tail]
        for params, row in listed
    ]


def _stats_table(arch: str, dtype: str, entries: list[tuple[str, str]], unary: bool) -> str:
    """A column per parameter set: twelve figures against three variants does not fit as rows."""
    summaries = _summaries(arch, dtype, entries)
    # A row no summary carries is dropped: an older index reads short, not as em dashes.
    rows = [
        [label, *(_cell(s, key) for _, s in summaries)]
        for label, key in STATS_ROWS
        if (unary or key != "usable_to") and any(key in s for _, s in summaries)
    ]
    return _table(["", *(f"`{p}`" for p, _ in summaries)], rows) + "\n"


def _specials_table(arch: str, dtype: str, entries: list[tuple[str, str]]) -> str:
    """Device beside golden at ±0, ±inf, NaN and the smallest normals; no ULP exists there."""
    rows = [
        [
            f"`{params}`",
            r["x"],
            r["y"],
            r["y_ref"],
            # An index older than the stored verdict can only be re-read from its own digits.
            "agree" if r.get("agree", r["y"] == r["y_ref"]) else "**differ**",
        ]
        for params, specials in _carrying(_summaries(arch, dtype, entries), "specials")
        for r in specials
    ]
    if not rows:
        return ""
    return "### Special values\n\n" + _table(["Variant", "x", "device", "golden", ""], rows) + "\n"


def _outcome_table(arch: str, dtype: str, entries: list[tuple[str, str]]) -> str:
    """What each measured point demonstrated. Only `exact`, `faithful` and `inexact` carry a ULP."""
    counted = _carrying(_summaries(arch, dtype, entries), "outcomes")
    if not counted:
        return ""
    kinds = sorted({kind for _, outcomes in counted for kind in outcomes})
    rows = [
        [f"`{params}`", *(f"{outcomes.get(kind, 0):,}" for kind in kinds)]
        for params, outcomes in counted
    ]
    return "**Outcomes**\n\n" + _table(["Parameters", *kinds], rows) + "\n"


def _offenders_table(arch: str, dtype: str, entries: list[tuple[str, str]]) -> str:
    """The worst points by value. A maximum tells you how bad; this tells you where to look."""
    found = _carrying(_summaries(arch, dtype, entries), "offenders")
    if not found:
        return ""
    listed = [(params, row) for params, rows in found for row in rows]
    return "**Worst points**\n\n" + _table(*_point_rows(listed, (("ulp", "ULP"),))) + "\n"


def _monotonic_table(arch: str, dtype: str, entries: list[tuple[str, str]]) -> str:
    """Ordering is not accuracy: an op can sit within 1 ULP and still step backwards."""
    found = _carrying(_summaries(arch, dtype, entries), "monotonic")
    if not found:
        return ""
    summary = _table(
        ["Parameters", "Pairs", "Violations", "Rate", "Worst \\|Δy\\|"],
        [
            [f"`{params}`", f"{m['pairs']:,}", f"{m['violations']:,}", m["rate"], m["worst_dy"]]
            for params, m in found
        ],
    )
    violations = [
        [
            f"`{params}`",
            f"{row['x_from']} → {row['x_to']}",
            f"{row['y_from']} → {row['y_to']}",
            row["dy"],
        ]
        for params, m in found
        for row in m["top"]
    ]
    worst = (
        "\n_Worst violations — the device output moved against the reference's own direction "
        "between these neighbouring inputs._\n\n"
        + _table(["Parameters", "x", "device", "\\|Δy\\|"], violations)
        if violations
        else ""
    )
    return (
        "**Monotonicity**\n\n_Checked only where the reference is itself ordered, and never "
        "across a discontinuity._\n\n" + summary + worst + "\n"
    )


def _nonfinite_table(arch: str, dtype: str, entries: list[tuple[str, str]]) -> str:
    """Which side went non-finite. These points carry no ULP, so no other figure counts them."""
    found = _carrying(_summaries(arch, dtype, entries), "nonfinite")
    if not found:
        return ""
    sides = ("total", "both", "device_only", "golden_only")
    kinds = ("device_inf", "device_nan", "golden_inf", "golden_nan")
    summary = _table(
        [
            "Parameters",
            "Points",
            "Both",
            "Device only",
            "Golden only",
            *(k.replace("_", " ").capitalize() for k in kinds),
        ],
        [[f"`{params}`", *(f"{n[k]:,}" for k in (*sides, *kinds))] for params, n in found],
    )
    listed = [(params, row) for params, n in found for row in n["detail"]]
    return "**Non-finite outputs**\n\n" + summary + "\n" + _table(*_point_rows(listed, ())) + "\n"


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
