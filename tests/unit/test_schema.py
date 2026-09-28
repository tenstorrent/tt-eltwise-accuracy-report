# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Provenance parsing and the arity rules, both device-free."""

from __future__ import annotations

import pytest

from ttnn_accuracy.measure.schema import _commit_from_version
from ttnn_accuracy.ops import arity
from ttnn_accuracy.ops.introspect import DiscoveredOp


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.75.0rc10.dev817+g2adf8c65887", "2adf8c65887"),
        ("0.75.0", None),
    ],
)
def test_tt_metal_commit_is_read_from_the_ttnn_version(version, expected):
    assert _commit_from_version(version) == expected


def _op(name: str, required: int | None) -> DiscoveredOp:
    return DiscoveredOp(
        name=name,
        qualified_name=f"ttnn.{name}",
        has_golden=required is not None,
        signature=None,
        required_args=required,
        is_backward=name.endswith("_bw"),
        is_cpp=True,
        is_experimental=False,
        golden=None,
    )


@pytest.mark.parametrize(
    ("name", "required", "operands", "category"),
    [
        ("exp", 1, 1, "unary"),
        ("add", 2, 2, "binary"),
        ("where", 3, 3, "ternary"),
        ("exp_bw", 2, 1, "unary_bw"),  # the gradient is not an operand under test
        ("add_bw", 3, 2, "binary_bw"),
        ("zeros", 0, None, None),
        ("mystery", None, None, None),
    ],
)
def test_operand_count_and_category(name, required, operands, category):
    op = _op(name, required)
    assert arity.operands(op) == operands
    assert arity.category(op) == category
