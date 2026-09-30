# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

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
    "defects": 0,
    "unflushed": 0,
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
        # The logaddexp class: no ULP moves, because these points never had one.
        ({"defects": 15566}, "regressed"),
        # Nor here: every one of these points reads bit-exact on the ULP it does not have.
        ({"unflushed": 396}, "regressed"),
        ({"mean_ulp": "—"}, "improved"),  # no inexact points left: errors vanished
        ({"usable_to": "1"}, "changed"),  # reported, never scored: its "—" is ambiguous
        ({}, None),
    ],
)
def test_a_moved_metric_lands_in_the_right_bucket(change, bucket):
    buckets = diff(_index(), _index(**change))
    assert [name for name, rows in buckets.items() if rows] == ([bucket] if bucket else [])


@pytest.mark.parametrize(
    ("max_ulp", "usable_to", "operands", "defects", "unflushed", "inexact", "expected"),
    [
        # Half a ULP is what a correctly rounded answer costs against a wider reference, so
        # `bit-exact` is decided by the outcome counts and never by the maximum.
        ("0.5", "3.39e+38", 1, 0, 0, 0, "bit-exact"),
        ("2", "3.39e+38", 1, 0, 0, 1153, "within 2 ULP"),
        ("3.38e+38", "2.62e+05", 1, 0, 0, 9364, "accurate to |x| <= 2.62e+05"),
        ("1.14e+36", "—", 1, 0, 0, 33679, "never within 2 ULP"),
        ("254", "—", 2, 0, 0, 65024, "worst sampled pairing"),
        ("—", "—", 1, 0, 0, 0, "no scorable points"),
        # rpow_bw at exponent 2.0: every scorable point exact, half of them infinite.
        ("0.5", "3.39e+38", 1, 32384, 0, 0, "32384/64777 defects; rest 0.5 ULP"),
        # The quasar exp case: bit-exact everywhere the reference is a number.
        ("0.5", "3.39e+38", 1, 0, 396, 0, "396/64777 unflushed; rest 0.5 ULP"),
    ],
)
def test_verdicts_follow_the_contract(
    max_ulp, usable_to, operands, defects, unflushed, inexact, expected
):
    from ttnn_accuracy.report.score import verdict

    assert verdict(max_ulp, usable_to, operands, defects, 64777, unflushed, 0, inexact) == expected


def test_a_tie_broken_the_other_way_is_not_reported_as_an_error():
    """WH breaks ties away from zero, torch to even, so bf16 add misses 4.2 billion pairings."""
    from ttnn_accuracy.report.score import verdict

    assert verdict("0.621", "—", 2, 0, 64777, 0, 64000, 0) == "faithful; 64000/64777 tie-breaks"


def test_coverage_changes_are_named_not_scored():
    old, new = _index(), _index()
    new["wh"]["bf16"]["sinh"] = {"default": dict(STATS)}
    del new["wh"]["bf16"]["exp"]
    buckets = diff(old, new)
    assert [k for k, *_ in buckets["added"]] == [("wh", "bf16", "sinh", "default")]
    assert [k for k, *_ in buckets["removed"]] == [("wh", "bf16", "exp", "default")]
    assert not buckets["regressed"]


def test_a_metric_the_baseline_predates_is_not_a_regression():
    """`defects` arriving put 93 entries in `regressed`; none had moved a ULP."""
    old = _index()
    del old["wh"]["bf16"]["exp"]["default"]["defects"]
    buckets = diff(old, _index(defects=15566))
    assert not buckets["regressed"]
    assert [k for k, *_ in buckets["changed"]] == [("wh", "bf16", "exp", "default")]


def test_a_metric_both_sides_carry_still_scores():
    assert diff(_index(defects=0), _index(defects=7))["regressed"]
    assert diff(_index(defects=7), _index(defects=0))["improved"]


@pytest.mark.parametrize(
    ("stats", "bar", "failures"),
    [
        ({"max_ulp": "1", "defects": 0}, 2.0, 0),
        ({"max_ulp": "3", "defects": 0}, 2.0, 1),
        ({"max_ulp": "2", "defects": 0}, 2.0, 0),  # the bar is inclusive
        ({"max_ulp": "0", "defects": 329}, 2.0, 1),  # a defect is over any bar
        ({"max_ulp": "—", "defects": 0}, 2.0, 0),  # nothing scorable is not a failure
    ],
)
def test_the_absolute_bar_is_independent_of_the_baseline(stats, bar, failures):
    from ttnn_accuracy.measure.runner import _over_bar

    assert _over_bar({"bf16": {"exp": {"default": stats}}}, bar) == failures


def test_a_new_defect_count_outranks_a_metric_that_improved():
    """multigammaln gained 7,422 defects while ulp_clipped fell by two."""
    old = _index(ulp_clipped=2989)
    del old["wh"]["bf16"]["exp"]["default"]["defects"]
    buckets = diff(old, _index(ulp_clipped=2987, defects=7422))
    assert not buckets["improved"]
    assert [k for k, *_ in buckets["changed"]] == [("wh", "bf16", "exp", "default")]


def test_an_improvement_with_no_new_metric_is_still_an_improvement():
    assert diff(_index(ulp_clipped=5), _index(ulp_clipped=2))["improved"]


def test_a_drop_in_correctly_rounded_share_is_a_regression():
    assert diff(_index(rounded_frac="0.997"), _index(rounded_frac="0.990"))["regressed"]
    assert diff(_index(rounded_frac="0.990"), _index(rounded_frac="0.997"))["improved"]


def test_two_pre_fractional_indexes_still_score_ulp():
    """A nightly against last week's published report, both pre-dating rounded_frac."""
    assert diff(_index(max_ulp="1"), _index(max_ulp="254"))["regressed"]


def test_ulp_against_a_pre_fractional_baseline_is_not_a_kernel_regression():
    """Published max_ulp=0 meant bit-identical to a same-width golden. Against a wider
    reference the same kernel reads 0.5 ULP — a definition change, not a worse kernel."""
    old = _index()
    buckets = diff(old, _index(max_ulp="0.5", rounded_frac="0.983"))
    assert not buckets["regressed"]
    assert [k for k, *_ in buckets["changed"]] == [("wh", "bf16", "exp", "default")]


def test_ulp_still_scores_once_both_indexes_use_the_wider_reference():
    old = _index(rounded_frac="0.983", max_ulp="0.5")
    assert diff(old, _index(rounded_frac="0.983", max_ulp="2"))["regressed"]
    assert diff(old, _index(rounded_frac="0.983", max_ulp="0.4"))["improved"]


def _build(sha: str, commit: str, **stats) -> tuple[str, str, dict]:
    index = _index(**stats)
    index["_runs"]["wh"]["bf16"]["tt_metal_commit"] = commit
    return sha, "2026-09-04", index


def test_history_separates_a_kernel_change_from_a_measurement_change():
    """`sin` jumped to 3.38e+38 the night the bf16 baseline grew to 180 ops — the sweep
    widened, the kernel did not. Reading that as a regression sends someone bisecting ttnn."""
    from ttnn_accuracy.report.compare import _changes

    ours = _changes([_build("a", "tt1"), _build("b", "tt1", max_ulp="9")])
    theirs = _changes([_build("a", "tt1"), _build("b", "tt2", max_ulp="9")])
    assert "our change" in ours[-1][4]
    assert "our change" not in theirs[-1][4]


def test_history_says_what_a_variant_looked_like_when_it_first_appeared():
    from ttnn_accuracy.report.compare import _changes

    (row,) = _changes([_build("a", "tt1")])
    assert row[4].startswith("first measured")
    assert row[3] == "tt1"


def test_a_defect_does_not_hide_how_wrong_the_scorable_points_are():
    """bh/fp32/log_bw read "2 of 64514 points returned inf or zero" and nothing else, while
    the other 64,512 were 9.02e+04 ULP out. The defect leads; it must not be the whole line."""
    from ttnn_accuracy.report.score import verdict

    said = verdict("9.02e+04", "—", 1, 2, 64514, 0, 0, 64512)
    assert said.startswith("2/64514 defects")
    assert "9.02e+04 ULP" in said
