"""Exclusions must keep an op out of the plan, and naming it must return the reason.

Three times a stage has written an exclusion the planner never read — the `outcome`
column, the `sampled` column, and `rejected`, whose ops were swept anyway and failed one
by one on the device. These cases assert the two halves agree.
"""

from __future__ import annotations

import pytest

from ttnn_accuracy.ops import plan

EXP = {
    "name": "exp",
    "category": "unary",
    "operands": 1,
    "signature": None,
    "is_cpp": True,
    "is_experimental": False,
}


def _manifest(rejected: dict | None = None, unprobeable: dict | None = None) -> dict:
    return {
        "ops": {"ttnn.exp": dict(EXP)},
        "unprobeable": unprobeable or {},
        "domains": {"ttnn.exp": {d: {"lo": 0.0, "hi": 1.0} for d in ("bf16", "fp32")}},
        "refused": {},
        "layouts": {},
        "rejected": {"wh": rejected or {}},
    }


def test_a_device_rejected_op_never_reaches_the_plan(monkeypatch):
    manifest = _manifest(rejected={"ttnn.exp": "TT_FATAL: only rank-4"})
    monkeypatch.setattr(plan, "_manifest", lambda: manifest)
    specs, _ = plan.resolve(None, "unary", "wh")
    assert specs == []


@pytest.mark.parametrize(
    ("name", "manifest", "expected"),
    [
        ("clone", _manifest(), "excluded — moves data"),
        (
            "conv2d",
            _manifest(unprobeable={"ttnn.conv2d": "TypeError: reshape()"}),
            "refused the probe",
        ),
        ("exp", _manifest(rejected={"ttnn.exp": "TT_FATAL"}), "rejected every dtype"),
        ("exp", _manifest(), "category filter"),
        ("matmul", _manifest(), "ttnn has it, this manifest does not"),
    ],
)
def test_a_missing_op_is_answered_with_its_recorded_reason(name, manifest, expected):
    assert expected in plan._why_missing(manifest, name, "wh")


def test_a_typo_is_named_as_one_rather_than_called_not_eltwise():
    """The two things a newcomer hits — a typo and a genuinely new op — need opposite answers."""
    assert "does not register it" in plan._why_missing(_manifest(), "reul", "wh")
