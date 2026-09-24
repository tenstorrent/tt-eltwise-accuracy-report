# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The timing statistics. The dispatch loop needs silicon; the arithmetic does not."""

from __future__ import annotations

from ttnn_accuracy.measure.perf import ELEMENTS, _operand, _stats


def test_median_ignores_one_outlier():
    """One scheduler hiccup must not move the reported figure — that is why it is a median."""
    clean = _stats([10.0] * 29 + [10.0])
    spiked = _stats([10.0] * 29 + [900.0])
    assert spiked["us_median"] == clean["us_median"]
    assert spiked["us_p90"] == clean["us_p90"]


def test_spread_reports_an_unquiet_machine():
    quiet = _stats([10.0] * 30)
    noisy = _stats([10.0 + i for i in range(30)])
    assert quiet["spread_pct"] == 0.0
    assert noisy["spread_pct"] > 100


def test_throughput_matches_the_median():
    assert _stats([100.0] * 10)["melem_per_s"] == round(ELEMENTS / 100.0, 1)


def test_operand_stays_inside_a_bounded_domain():
    x = _operand((-1.0, 1.0), "fp32")
    assert x.numel() == ELEMENTS
    assert x.min() >= -1.0 and x.max() <= 1.0


def test_operand_is_finite_when_the_domain_is_not():
    x = _operand((float("-inf"), float("inf")), "bf16")
    assert x.isfinite().all()


def test_the_host_is_the_machine_not_the_reservation(monkeypatch):
    """A re-reservation renamed the box mid-project and every timing stopped comparing."""
    from ttnn_accuracy.measure import perf

    for booked in (
        "wh-glx6u-02-special-ijankowski-for-reservation-207113",
        "wh-glx6u-02-special-ijankowski-for-reservation-208357",
    ):
        monkeypatch.setattr(perf.platform, "node", lambda b=booked: b)
        assert perf._host() == "wh-glx6u-02"
    monkeypatch.setattr(perf.platform, "node", lambda: "plain-hostname")
    assert perf._host() == "plain-hostname"
