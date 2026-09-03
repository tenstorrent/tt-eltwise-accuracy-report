"""What the report renders, and that it renders the same thing twice.

The pages are committed nightly, so output that differs from itself on unchanged data
buries the night's findings in a diff nobody can read.
"""

from __future__ import annotations

import pandas as pd

from ttnn_accuracy.report.charts import plot_ulp_chart
from ttnn_accuracy.report.pages import GENERATED_NOTE, _us


def test_a_chart_is_byte_identical_on_identical_data(tmp_path):
    """matplotlib stamps a date and randomises every element id."""
    df = pd.DataFrame({"x": [0.5, 1.0, 2.0], "ulp_error": [0.0, 1.5, 3.0]})
    first, second = tmp_path / "a.svg", tmp_path / "b.svg"
    for out in (first, second):
        plot_ulp_chart(df, "exp", "default", "wh", "bf16", out)
    assert first.read_bytes() == second.read_bytes()


def test_a_page_carries_no_clock():
    """1,972 of one night's 3,657 changed lines were the generation time."""
    assert "Generated" not in GENERATED_NOTE


def test_a_page_flags_the_timings_it_should_not_trust():
    """charts attaches spread_pct past NOISE_PCT only."""
    quiet = {"us_median": 686.401, "melem_per_s": 24442.3}
    assert _us(quiet) == "686"
    assert _us(quiet | {"spread_pct": 46.2}) == "686 ±46%"
    assert _us(None) == "—"
