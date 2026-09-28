# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Writes the whole reports/ tree: which pages exist, and in what order they are produced."""

from __future__ import annotations

from loguru import logger

from ttnn_accuracy.paths import ASK_FILE, REPORTS_DIR
from ttnn_accuracy.report.ask import ask_page
from ttnn_accuracy.report.index import DTYPES, discover_ops, group_by_display, write
from ttnn_accuracy.report.navigation import (
    arch_dtype_index,
    arch_index,
    arch_list_index,
    dtype_index,
    dtype_list_index,
    op_cross_arch_page,
    op_list_index,
    top_readme,
)
from ttnn_accuracy.report.pages import op_detail_page


def generate_reports(arch_filter=None, dtype_filter=None, categories=None) -> int:
    if categories is None:
        categories = ["unary"]

    data = discover_ops(categories=categories)

    # data carries an entry per arch/dtype directory even when every op was filtered out.
    if not any(ops for dtypes in data.values() for ops in dtypes.values()):
        logger.error(
            "no charts matched categories={} — run `ttnn-accuracy charts` first", categories
        )
        return 1

    archs = [arch_filter] if arch_filter else sorted(data)
    dtypes = [dtype_filter] if dtype_filter else DTYPES
    grouped = {(a, d): group_by_display(data.get(a, {}).get(d, {})) for a in archs for d in dtypes}

    write(REPORTS_DIR / "README.md", top_readme(data))
    write(ASK_FILE, ask_page())

    write(REPORTS_DIR / "by_arch" / "README.md", arch_list_index(data))
    for arch in archs:
        write(REPORTS_DIR / "by_arch" / arch / "README.md", arch_index(arch, data.get(arch, {})))
        for dtype in dtypes:
            write(
                REPORTS_DIR / "by_arch" / arch / dtype / "README.md",
                arch_dtype_index(arch, dtype, data.get(arch, {}).get(dtype, {})),
            )
            for display_name, entries in grouped[arch, dtype].items():
                write(
                    REPORTS_DIR / "by_arch" / arch / dtype / f"{display_name}.md",
                    op_detail_page(arch, dtype, display_name, entries),
                )

    write(REPORTS_DIR / "by_op" / "README.md", op_list_index(data))
    for display_name in sorted(set().union(*(g.keys() for g in grouped.values()))):
        write(
            REPORTS_DIR / "by_op" / display_name / "README.md",
            op_cross_arch_page(display_name, data),
        )

    write(REPORTS_DIR / "by_dtype" / "README.md", dtype_list_index(data))
    for dtype in dtypes:
        write(REPORTS_DIR / "by_dtype" / dtype / "README.md", dtype_index(dtype, data))

    logger.success("report tree regenerated")
    return 0
