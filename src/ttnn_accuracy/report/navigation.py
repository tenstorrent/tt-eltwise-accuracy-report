# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The index pages: every way into the tree — by arch, by op, by dtype — and the top README."""

from __future__ import annotations

from ttnn_accuracy.ops.manifest import load as load_manifest
from ttnn_accuracy.ops.plan import describe, params_desc
from ttnn_accuracy.paths import REPORTS_DIR, RUNS_KEY
from ttnn_accuracy.report.index import (
    ARCH_DISPLAY,
    ARCHS,
    DTYPE_DISPLAY,
    DTYPES,
    GENERATED_NOTE,
    _get_index,
    _us,
    _usable_text,
    chart_rel_path,
    get_op_category,
    group_by_display,
    load_summary,
)


def _unmeasured_table(arch: str, dtype: str) -> str:
    """Ops in scope that produced no data, and why — otherwise they read as never attempted."""
    rows = {
        q.rsplit(".", 1)[-1]: why for q, why in load_manifest()["rejected"].get(arch, {}).items()
    }
    rows |= _get_index().get(RUNS_KEY, {}).get(arch, {}).get(dtype, {}).get("failed", {})
    if not rows:
        return ""
    body = "".join(f"| `{op}` | {why} |\n" for op, why in sorted(rows.items()))
    return (
        f"## Not measurable on {ARCH_DISPLAY.get(arch, arch)}, "
        f"{DTYPE_DISPLAY.get(dtype, dtype)}\n\nIn scope, but produced no data — each with "
        f"the reason recorded when it was probed or swept.\n\n"
        f"| Op | Why |\n|----|-----|\n{body}\n"
    )


def arch_dtype_index(arch: str, dtype: str, ops: dict[str, list[str]]) -> str:
    """reports/by_arch/{arch}/{dtype}/README.md"""
    page_path = REPORTS_DIR / "by_arch" / arch / dtype / "README.md"
    groups = group_by_display(ops)

    lines = [
        f"# {ARCH_DISPLAY.get(arch, arch)} — {DTYPE_DISPLAY.get(dtype, dtype)}\n",
        GENERATED_NOTE,
        f"[← {ARCH_DISPLAY.get(arch, arch)}](../README.md) | [Top](../../../README.md)\n\n",
        "## Operations Summary\n\n",
        "_Accurate to \\|x\\|: the largest \\|x\\| within 2 ULP. `n/a` for ops of more than one "
        "operand, where a bound on x would describe its sampled partner instead._\n\n",
        "| Op | Parameters | Max ULP | Mean ULP | Accurate to \\|x\\| | Max abs error | µs |\n",
        "|----|------------|---------|----------|-----------------|---------------|----|\n",
    ]

    for display_name, entries in groups.items():
        for i, (op_key, variant) in enumerate(entries):
            params = params_desc(op_key, variant)
            summary = load_summary(arch, dtype, op_key, variant)
            info = describe(op_key)
            max_ulp = (summary["max_ulp"] if summary else "—") + (
                " ⚠" if summary and summary.get("ulp_clipped") else ""
            )
            mean_ulp = summary["mean_ulp"] if summary else "—"
            if not summary:
                usable = "—"
            elif info and info.operands > 1:
                usable = "n/a"
            else:
                usable = _usable_text(summary)
            max_abs = summary["max_abs"] if summary else "—"
            us = _us((summary or {}).get("perf"))
            op_cell = f"[{display_name}]({display_name}.md)" if i == 0 else ""
            lines.append(
                f"| {op_cell} | `{params}` | {max_ulp} | {mean_ulp} | {usable} | {max_abs} "
                f"| {us} |\n"
            )

    lines.append("\n")
    lines.append(_unmeasured_table(arch, dtype))
    lines.append("\n---\n\n## Charts Preview\n\n")
    for display_name, entries in groups.items():
        lines.append(f"### [{display_name}]({display_name}.md)\n\n")
        for op_key, variant in entries:
            chart = chart_rel_path(arch, dtype, op_key, variant, page_path.parent)
            params = params_desc(op_key, variant)
            lines.append(f"![{display_name} {params}]({chart})\n\n")

    return "".join(lines)


def arch_index(arch: str, data_for_arch: dict) -> str:
    """reports/by_arch/{arch}/README.md"""
    groups = {d: group_by_display(data_for_arch.get(d, {})) for d in DTYPES}
    available_dtypes = [d for d in DTYPES if d in data_for_arch]

    lines = [
        f"# {ARCH_DISPLAY.get(arch, arch)}\n",
        GENERATED_NOTE,
        "[← Architectures](../README.md) | [Top](../../README.md)\n\n",
        "## Data Types\n\n",
    ]
    lines += [
        f"- [{DTYPE_DISPLAY.get(d, d)}]({d}/README.md) — {len(groups[d])} ops\n"
        for d in available_dtypes
    ]

    lines.append("\n## Quick Summary\n\n")
    lines.append(
        "| Op | "
        + " | ".join(f"[{DTYPE_DISPLAY.get(d, d)}]({d}/README.md)" for d in available_dtypes)
        + " |\n"
    )
    lines.append("|----" + "|------" * len(available_dtypes) + "|\n")

    for display_name in sorted(set().union(*(g.keys() for g in groups.values()))):
        cells = []
        for dtype in available_dtypes:
            group = groups[dtype].get(display_name)
            if not group:
                cells.append("—")
                continue
            summary = load_summary(arch, dtype, *group[0])
            label = f"{summary['max_ulp']} ULP" if summary else "data"
            cells.append(f"[{label}]({dtype}/{display_name}.md)")
        lines.append(f"| `{display_name}` | " + " | ".join(cells) + " |\n")

    return "".join(lines)


def arch_list_index(data: dict) -> str:
    lines = [
        "# By Architecture\n",
        GENERATED_NOTE,
        "[← Top](../README.md)\n\n",
        "| Architecture | bf16 ops | fp32 ops |\n",
        "|-------------|---------|----------|\n",
    ]
    for arch in sorted(data.keys()):
        bf16_n = len(group_by_display(data[arch].get("bf16", {})))
        fp32_n = len(group_by_display(data[arch].get("fp32", {})))
        lines.append(
            f"| [{ARCH_DISPLAY.get(arch, arch)}]({arch}/README.md) | {bf16_n} | {fp32_n} |\n"
        )
    return "".join(lines)


def op_cross_arch_page(display_name: str, data: dict) -> str:
    """reports/by_op/{display_name}/README.md — op across all arch/dtype"""
    page_path = REPORTS_DIR / "by_op" / display_name / "README.md"

    groups = {
        (a, d): group_by_display(data.get(a, {}).get(d, {})).get(display_name, [])
        for a in ARCHS
        for d in DTYPES
    }
    # (arch, dtype, op_key, variant) — display_name comes from this same data, so never empty.
    all_entries = [(a, d, k, v) for (a, d), g in groups.items() for k, v in g]

    lines = [
        f"# `{display_name}` — All Architectures & Data Types\n",
        GENERATED_NOTE,
        "[← All ops](../README.md) | [Top](../../README.md)\n\n",
    ]
    lines.append("\n")

    all_params = sorted(
        dict.fromkeys(params_desc(op_key, variant) for _, _, op_key, variant in all_entries)
    )
    available_archs = sorted(dict.fromkeys(a for a, _, _, _ in all_entries))
    available_dtypes = sorted(dict.fromkeys(d for _, d, _, _ in all_entries))

    header = (
        "| Parameters | Arch | "
        + " | ".join(DTYPE_DISPLAY.get(d, d) for d in available_dtypes)
        + " |\n"
    )
    sep = "|------------|------" + "|------" * len(available_dtypes) + "|\n"
    lines.extend([header, sep])

    for params in all_params:
        for arch in available_archs:
            cells = []
            for dtype in available_dtypes:
                entry = next(
                    ((k, v) for k, v in groups[arch, dtype] if params_desc(k, v) == params),
                    None,
                )
                if not entry:
                    cells.append("—")
                    continue
                summary = load_summary(arch, dtype, *entry)
                link = f"../../by_arch/{arch}/{dtype}/{display_name}.md"
                label = f"{summary['max_ulp']} ULP" if summary else "data"
                cells.append(f"[{label}]({link})")
            arch_label = ARCH_DISPLAY.get(arch, arch)
            lines.append(f"| `{params}` | {arch_label} | " + " | ".join(cells) + " |\n")

    lines.append("\n---\n\n## Charts\n\n")
    for arch in available_archs:
        for dtype in available_dtypes:
            group = groups[arch, dtype]
            if not group:
                continue
            lines.append(
                f"### {ARCH_DISPLAY.get(arch, arch)}, {DTYPE_DISPLAY.get(dtype, dtype)}\n\n"
            )
            for op_key, variant in group:
                params = params_desc(op_key, variant)
                chart = chart_rel_path(arch, dtype, op_key, variant, page_path.parent)
                lines.append(
                    f"**`{params}`**\n\n![{display_name} {params} {arch} {dtype}]({chart})\n\n"
                )

    return "".join(lines)


def op_list_index(data: dict) -> str:
    groups = {(a, d): group_by_display(data.get(a, {}).get(d, {})) for a in ARCHS for d in DTYPES}
    all_display = sorted(set().union(*(g.keys() for g in groups.values())))

    lines = [
        "# By Operation\n",
        GENERATED_NOTE,
        "[← Top](../README.md)\n\n",
        f"Total: {len(all_display)} ops measured\n\n",
        "| Op | Category | WH bf16 | WH fp32 | BH bf16 | BH fp32 |\n",
        "|----|----------|---------|---------|---------|----------|\n",
    ]

    for display_name in all_display:
        category = "—"
        cells = []
        for arch in ARCHS:
            for dtype in DTYPES:
                group = groups[arch, dtype].get(display_name)
                if not group:
                    cells.append("—")
                    continue
                op_key, variant = group[0]
                if category == "—":
                    category = get_op_category(op_key)
                summary = load_summary(arch, dtype, op_key, variant)
                link = f"{display_name}/README.md"
                cells.append(f"[{summary['max_ulp']}]({link})" if summary else f"[✓]({link})")
        lines.append(
            f"| [{display_name}]({display_name}/README.md) | {category} | "
            + " | ".join(cells)
            + " |\n"
        )

    return "".join(lines)


def dtype_index(dtype: str, data: dict) -> str:
    available_archs = sorted(arch for arch in data if dtype in data[arch])
    groups = {a: group_by_display(data[a][dtype]) for a in available_archs}
    all_display = sorted(set().union(*(g.keys() for g in groups.values())))

    lines = [
        f"# {DTYPE_DISPLAY.get(dtype, dtype)}\n",
        GENERATED_NOTE,
        "[← Dtypes](../README.md) | [Top](../../README.md)\n\n",
        "| Op | " + " | ".join(ARCH_DISPLAY.get(a, a) for a in available_archs) + " |\n",
        "|----" + "|------" * len(available_archs) + "|\n",
    ]

    for display_name in all_display:
        cells = []
        for arch in available_archs:
            group = groups[arch].get(display_name)
            if not group:
                cells.append("—")
                continue
            summary = load_summary(arch, dtype, *group[0])
            link = f"../../by_arch/{arch}/{dtype}/{display_name}.md"
            cells.append(f"[{summary['max_ulp']} ULP]({link})" if summary else f"[✓]({link})")
        lines.append(f"| `{display_name}` | " + " | ".join(cells) + " |\n")

    return "".join(lines)


def dtype_list_index(data: dict) -> str:
    lines = [
        "# By Data Type\n",
        GENERATED_NOTE,
        "[← Top](../README.md)\n\n",
    ]
    for dtype in DTYPES:
        n_wh = len(group_by_display(data.get("wh", {}).get(dtype, {})))
        n_bh = len(group_by_display(data.get("bh", {}).get(dtype, {})))
        lines.append(
            f"- [{DTYPE_DISPLAY.get(dtype, dtype)}]({dtype}/README.md)"
            f" — WH: {n_wh} ops, BH: {n_bh} ops\n"
        )
    return "".join(lines)


def _defect_table() -> str:
    """Every variant returning inf or zero where a value exists — otherwise one row in 827."""
    rows = sorted(
        (
            -s["defects"] / s["n_inputs"],
            op,
            f"| [`{op}`](by_op/{op}/README.md) | `{params_desc(op, variant)}` | {arch} {dtype} "
            f"| {s['defects']:,} of {s['n_inputs']:,} | {100 * s['defects'] / s['n_inputs']:.1f}% |",
        )
        for arch, dtypes in _get_index().items()
        if arch != RUNS_KEY
        for dtype, ops in dtypes.items()
        for op, variants in ops.items()
        for variant, s in variants.items()
        if s.get("defects")
    )
    if not rows:
        return ""
    return (
        f"\n---\n\n## Returning inf or zero where a value exists\n\n"
        f"{len(rows)} variants across {len({op for _, op, _ in rows})} ops. These carry no ULP — "
        "the reference is a number the dtype can hold and the device returned `inf` or `0`, so "
        "every accuracy figure beside them excludes the point. Worst share first.\n\n"
        "| Op | Parameters | Where | Points | Share |\n|---|---|---|---|---|\n"
        + "\n".join(row for _, _, row in rows)
        + "\n"
    )


def top_readme(data: dict) -> str:
    groups = {(a, d): group_by_display(data.get(a, {}).get(d, {})) for a in data for d in DTYPES}
    total_archs = len(data)
    total_ops = len(set().union(*(g.keys() for g in groups.values())))

    lines = [
        "# TT-Metal Eltwise Op Accuracy Reports\n\n",
        GENERATED_NOTE,
        "Accuracy measurements for TT-Metal elementwise operations vs PyTorch golden reference.\n",
        "Charts show ULP (units in last place) error across the full input range.\n\n",
        "---\n\n",
        "## Browse Reports\n\n",
        "| View | Description |\n",
        "|------|-------------|\n",
        "| [By Architecture](by_arch/README.md) | Start from hardware platform (WH, BH), then dtype, then op |\n",
        "| [By Operation](by_op/README.md) | Start from op name, compare across architectures and dtypes |\n",
        "| [By Data Type](by_dtype/README.md) | Start from bf16 or fp32, see all ops on all archs |\n",
        "\n---\n\n",
        "## Quick Stats\n\n",
        "| Metric | Count |\n",
        "|--------|-------|\n",
        f"| Architectures measured | {total_archs} |\n",
        f"| Unique ops measured | {total_ops} |\n",
    ]
    for arch in sorted(data):
        for dtype in DTYPES:
            if n := len(groups[arch, dtype]):
                lines.append(
                    f"| {ARCH_DISPLAY.get(arch, arch)} {DTYPE_DISPLAY.get(dtype, dtype)} ops | {n} |\n"
                )

    lines.append(_defect_table())
    lines.extend(
        [
            "\n---\n\n",
            "## Data Collection\n\n",
            "```bash\n",
            "# Collect measurements (run on target hardware)\n",
            "ttnn-accuracy measure --arch wh --category unary --dtype both\n",
            "\n",
            "# Generate SVG charts and update report_index.json\n",
            "ttnn-accuracy charts\n",
            "\n",
            "# Regenerate this report tree\n",
            "ttnn-accuracy report\n",
            "```\n",
        ]
    )
    return "".join(lines)
