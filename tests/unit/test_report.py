"""What the report renders, and that it renders the same thing twice.

The pages are committed nightly, so output that differs from itself on unchanged data
buries the night's findings in a diff nobody can read.
"""

from __future__ import annotations

import pandas as pd

from ttnn_accuracy.report.charts import plot_ulp_chart
from ttnn_accuracy.report.index import GENERATED_NOTE, _us


def test_a_chart_is_byte_identical_on_identical_data(tmp_path):
    """matplotlib stamps a date and randomises every element id."""
    df = pd.DataFrame(
        {
            "x": [0.5, 1.0, 2.0],
            "ulp_error": [0.0, 1.5, 3.0],
            "n_defined": [1, 1, 1],
            "n_rounded": [1, 0, 0],
        }
    )
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


def test_ask_page_inlines_the_live_contract_and_a_rounded_column():
    from ttnn_accuracy.report.ask import ask_page

    text = ask_page()
    assert "rounded_frac" in text
    assert "| Rounded |" in text
    assert "faithful" in text
    assert "Do not subtract `max_ulp`" in text


def test_a_custom_parameter_variant_still_charts(tmp_path, monkeypatch):
    """A dispatch measures at a parameter the plan lacks; a per-variant scope dropped
    every one and `charts` exited 1 with "no CSVs matched"."""
    from ttnn_accuracy.ops.plan import OpSpec
    from ttnn_accuracy.report import charts, score

    csv = tmp_path / "data" / "wh" / "bf16" / "relu_max" / "upper_limit6.0.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text(
        "index,x,y,y_ref,n_defined,n_rounded,ulp_error,ulp_signed,abs_error,rel_error,"
        "outcome,op,variant,dtype,layout\n"
        + "".join(
            f"{i},{i + 1.0},{i + 1.0},{i + 1.0},1,1,0.0,0.0,0.0,0.0,"
            "exact,relu_max,upper_limit6.0,bf16,tile\n"
            for i in range(4)
        )
    )
    planned = OpSpec("relu_max", "upper_limit1.0", "unary", 1, None, None, {}, {})
    monkeypatch.setattr(charts, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(charts, "CHARTS_DIR", tmp_path / "charts")
    monkeypatch.setattr(charts, "INDEX_FILE", tmp_path / "index.json")
    monkeypatch.setattr(charts, "resolve", lambda *a, **k: ([planned], []))
    monkeypatch.setattr(score, "describe", lambda op: None)

    assert charts.generate_charts(None, None, None) == 0
    assert (tmp_path / "charts" / "wh" / "bf16" / "relu_max_upper_limit6.0_ulp.svg").exists()
