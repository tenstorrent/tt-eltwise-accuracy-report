"""The diff between two report indexes, device-free.

Mateusz's harness once computed bf16 ULP with the fp32 mantissa and needed a repair
script; the spacing pin in test_metrics guards ours. This file guards the other half of
regression detection: an index change must land in the right bucket, because CI gates on
`regressed` alone.
"""

from __future__ import annotations

import pytest

from ttnn_accuracy.report.compare import diff

STATS = {
    "max_ulp": "1",
    "mean_ulp": "1",
    "ulp_clipped": 0,
    "usable_to": "3.39e+38",
    "max_abs": "0.25",
    "n_inputs": 65024,
    "outcomes": {"inexact": 65024},
}


def _index(**stats) -> dict:
    return {
        "_runs": {"wh": {"bf16": {"tt_metal_commit": "abc"}}},
        "wh": {"bf16": {"exp": {"default": STATS | stats}}},
    }


@pytest.mark.parametrize(
    ("change", "bucket"),
    [
        ({"max_ulp": "254"}, "regressed"),
        ({"ulp_clipped": 3}, "regressed"),
        ({"mean_ulp": "—"}, "improved"),  # no inexact points left: errors vanished
        ({"usable_to": "1"}, "changed"),  # reported, never scored: its "—" is ambiguous
        ({}, None),
    ],
)
def test_a_moved_metric_lands_in_the_right_bucket(change, bucket):
    buckets = diff(_index(), _index(**change))
    assert [name for name, rows in buckets.items() if rows] == ([bucket] if bucket else [])


@pytest.mark.parametrize(
    ("max_ulp", "mean_ulp", "usable_to", "operands", "expected"),
    [
        ("0", "—", "3.39e+38", 1, "bit-exact"),
        ("2", "1.1", "3.39e+38", 1, "within 2 ULP everywhere"),
        (
            "3.38e+38",
            "2.5e+36",
            "2.62e+05",
            1,
            "accurate to |x| <= 2.62e+05; up to 3.38e+38 ULP beyond",
        ),
        ("1.14e+36", "4.5e+34", "—", 1, "never within 2 ULP; mean 4.5e+34, worst 1.14e+36"),
        ("254", "2.42", "—", 2, "worst pairing 254 ULP; mean 2.42"),
        ("—", "—", "—", 1, "no scorable points"),
    ],
)
def test_verdicts_follow_the_contract(max_ulp, mean_ulp, usable_to, operands, expected):
    from ttnn_accuracy.report.charts import verdict

    assert verdict(max_ulp, mean_ulp, usable_to, operands) == expected


def test_coverage_changes_are_named_not_scored():
    old, new = _index(), _index()
    new["wh"]["bf16"]["sinh"] = {"default": dict(STATS)}
    del new["wh"]["bf16"]["exp"]
    buckets = diff(old, new)
    assert [k for k, *_ in buckets["added"]] == [("wh", "bf16", "sinh", "default")]
    assert [k for k, *_ in buckets["removed"]] == [("wh", "bf16", "exp", "default")]
    assert not buckets["regressed"]
