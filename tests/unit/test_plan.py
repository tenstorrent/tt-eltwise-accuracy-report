"""Every exclusion the manifest records must keep the op out of the plan.

Three times a stage has written an exclusion the planner never read — the `outcome`
column, the `sampled` column, and `rejected`, whose ops were swept anyway and failed one
by one on the device. Lint and the rest of the suite passed through all three, because
nothing asserted that the two halves agree. These cases do.
"""

from __future__ import annotations

import pytest

from ttnn_accuracy.ops import plan

MEASURABLE = {
    "name": "exp",
    "category": "unary",
    "operands": 1,
    "elementwise": True,
    "real_valued": True,
    "depends_on_input": True,
    "has_golden": True,
    "signature": None,
    "is_cpp": True,
    "is_experimental": False,
}


def _manifest(op_changes: dict, rejected: dict) -> dict:
    return {
        "ops": {"ttnn.exp": MEASURABLE | op_changes},
        "domains": {"ttnn.exp": {d: [0.0, 1.0] for d in ("bf16", "fp32")}},
        "refused": {},
        "layouts": {},
        "rejected": rejected,
    }


@pytest.mark.parametrize(
    ("op_changes", "rejected", "expected"),
    [
        ({"elementwise": False}, {}, "not elementwise"),
        ({"real_valued": False}, {}, "not real-valued"),
        ({"depends_on_input": False}, {}, "does not depend on its input"),
        ({"elementwise": None}, {}, "never classified"),
        ({}, {"ttnn.exp": "TT_FATAL: only rank-4"}, "rejected every dtype and layout"),
    ],
)
def test_an_excluded_op_never_reaches_the_plan(monkeypatch, op_changes, rejected, expected):
    manifest = _manifest(op_changes, rejected)
    monkeypatch.setattr(plan, "_manifest", lambda: manifest)

    specs, _ = plan.resolve("manifest", None, "unary")
    assert specs == []

    # Naming it explicitly must still say why, rather than calling it unknown.
    assert expected in plan._why_missing(manifest, "exp")
