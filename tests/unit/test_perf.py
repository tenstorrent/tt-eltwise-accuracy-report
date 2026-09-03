"""The timing statistics and how a page renders them. The dispatch loop needs silicon;
the arithmetic does not."""

from __future__ import annotations

from ttnn_accuracy.measure.perf import ELEMENTS, _operand, _stats
from ttnn_accuracy.report.pages import _us


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


def test_a_page_flags_the_timings_it_should_not_trust():
    """charts attaches spread_pct past NOISE_PCT only: relu bf16 came back 351 us at 46%,
    and a bare 351 beside a quiet row would read as the same measurement."""
    quiet = {"us_median": 686.401, "melem_per_s": 24442.3}
    assert _us(quiet) == "686"
    assert _us(quiet | {"spread_pct": 46.2}) == "686 ±46%"
    assert _us(None) == "—"
