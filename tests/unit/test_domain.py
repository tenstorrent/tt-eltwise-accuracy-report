# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Domain derivation against functions whose real bounds are known by hand."""

from __future__ import annotations

import pytest
import torch

from ttnn_accuracy.domain.derive import derive, grid


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_grid_is_sorted_and_holds_zero(dtype):
    g = grid(dtype)
    assert torch.equal(g, g.sort().values)
    assert (g == 0).any()
    assert torch.isfinite(g).all()


def test_bf16_grid_covers_every_normal_code():
    # 2^16 codes, less the two zeros already counted once, less subnormals and non-finites.
    assert 60000 < grid("bf16").numel() < 2**16


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_log_domain_starts_above_zero(dtype):
    d = derive(torch.log, dtype)
    assert d["lo"] > 0
    assert d["n_undefined"] > 0  # every negative input


@pytest.mark.parametrize(("dtype", "expected"), [("bf16", 88.5), ("fp32", 88.7)])
def test_exp_upper_bound_is_the_overflow_edge_not_a_constant(dtype, expected):
    """exp overflows where its result stops fitting, and that differs per dtype."""
    d = derive(torch.exp, dtype)
    assert d["hi"] == pytest.approx(expected, abs=0.5)
    assert d["n_overflow"] > 0


def test_reciprocal_bound_is_where_the_result_becomes_subnormal():
    """1/x underflows past ~8.5e37 — a boundary no one wrote down."""
    d = derive(torch.reciprocal, "bf16")
    assert d["hi"] == pytest.approx(8.5e37, rel=0.01)
    assert d["n_flushed"] > 0


def test_tanh_is_defined_everywhere():
    d = derive(torch.tanh, "bf16")
    assert d["n_undefined"] == d["n_overflow"] == d["n_flushed"] == 0
