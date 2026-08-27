"""Hand-checked ULP values. These pin the definition every later change is measured against."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ttnn_accuracy.measure import metrics


@pytest.mark.parametrize(
    ("value", "dtype", "expected"),
    [
        (1.0, torch.bfloat16, 2**-7),  # exponent 0, 7 mantissa bits
        (2.0, torch.bfloat16, 2**-6),
        (0.5, torch.bfloat16, 2**-8),
        (1.0, torch.float32, 2**-23),  # exponent 0, 23 mantissa bits
        (256.0, torch.float32, 2**-15),  # exponent 8 → 2^(8-23)
    ],
)
def test_ulp_matches_hand_calculation(value, dtype, expected):
    assert metrics.ulp(torch.tensor([value], dtype=dtype)).item() == pytest.approx(expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("value", [1.0, 0.5, 3.7, 1e-3, 1024.0, 2**-126])
def test_ulp_matches_the_nextafter_definition(value, dtype):
    """tt-metal defines ULP as nextafter(|x|) - |x|; we compute it from the exponent.

    Kept as a test rather than an import: models.common.utility_functions is not in the
    ttnn wheel, and its dtype-native subtraction underflows for small bf16 inputs.
    """
    x = torch.tensor([value], dtype=dtype)
    reference = torch.nextafter(x.abs(), torch.tensor(float("inf"), dtype=dtype)) - x.abs()
    assert metrics.ulp(x).item() == pytest.approx(reference.to(torch.float32).item())


def test_ulp_is_sign_symmetric():
    pos = metrics.ulp(torch.tensor([3.5], dtype=torch.bfloat16))
    neg = metrics.ulp(torch.tensor([-3.5], dtype=torch.bfloat16))
    assert pos.item() == neg.item()


def test_ulp_of_bf16_min_normal_survives_as_float32():
    """2^-133 is a valid fp32 subnormal but flushes to 0 in bf16 — it must not be lost."""
    u = metrics.ulp(torch.tensor([2**-126], dtype=torch.bfloat16))
    assert u.dtype is torch.float32
    assert u.item() > 0.0


def test_flush_subnormals_zeroes_below_min_normal():
    t = torch.tensor([2**-130, 2**-126, 0.0, -(2**-130)], dtype=torch.float32)
    assert metrics.flush_subnormals(t).tolist() == [0.0, 2**-126, 0.0, 0.0]


def _compare(x, golden, calculated, dtype=torch.bfloat16, group_size=1):
    return metrics.compare(
        torch.tensor(x, dtype=dtype),
        torch.tensor(golden, dtype=torch.float64),
        torch.tensor(calculated, dtype=dtype),
        group_size=group_size,
    )


def test_exact_match_is_zero_error():
    df = _compare([1.0, 2.0], [1.0, 2.0], [1.0, 2.0])
    assert df["ulp_error"].tolist() == [0.0, 0.0]
    assert df["abs_error"].tolist() == [0.0, 0.0]


def test_one_ulp_off_reads_as_one_ulp():
    df = _compare([1.0], [1.0], [1.0 + 2**-7])
    assert df["ulp_error"].item() == pytest.approx(1.0)


def test_underflowed_golden_is_reported_as_flushed_not_as_zero_error():
    """The dtype cannot hold the reference, so there is nothing to score — say so."""
    df = _compare([1.0], [float(torch.finfo(torch.bfloat16).tiny) / 2], [0.0])
    assert df["outcome"].item() == "flushed"
    assert np.isnan(df["ulp_error"].item())
    assert df["abs_error"].item() == 0.0


def test_hardware_zeroing_a_representable_value_is_not_a_128_ulp_error():
    """What made atan2 read 128 ULP for an absolute error of 1e-38.

    The reference is the smallest normal, which the dtype holds perfectly well, so this
    is not `flushed`. The hardware returned zero anyway. ULP cannot express that: a flush
    at the smallest normal is the exponent range itself, 2**7 in bf16, whatever the
    absolute error — so it is counted as its own outcome and left out of the ULP figures.
    """
    tiny = float(torch.finfo(torch.bfloat16).tiny)
    df = _compare([1.0], [tiny], [0.0])
    assert df["outcome"].item() == "zeroed"
    assert np.isnan(df["ulp_error"].item())
    assert df["abs_error"].item() == pytest.approx(tiny)


def test_ulp_is_undefined_when_the_reference_is_zero():
    """A non-zero result against a zero reference is what produced 1e24-ULP readings."""
    df = _compare([1.0], [0.0], [2**-100])
    assert np.isnan(df["ulp_error"].item())
    assert df["abs_error"].item() == pytest.approx(2**-100)


def test_outcomes_are_labelled():
    nan = float("nan")
    assert _compare([1.0], [1.0], [1.0])["outcome"].item() == "exact"
    assert _compare([1.0], [1.0], [1.0 + 2**-7])["outcome"].item() == "inexact"
    assert _compare([1.0], [nan], [nan])["outcome"].item() == "undefined"
    assert _compare([1.0], [1e40], [float("inf")])["outcome"].item() == "overflow"
    # A finite answer where the reference is NaN is a defect, not an unscorable point.
    assert _compare([1.0], [nan], [1.0])["outcome"].item() == "mismatch"


def test_group_size_keeps_first_x_and_worst_error():
    df = _compare([1.0, 1.0], [1.0, 1.0], [1.0, 1.0 + 2**-6], group_size=2)
    assert len(df) == 1
    assert df["x"].item() == 1.0
    assert df["ulp_error"].item() == pytest.approx(2.0)
