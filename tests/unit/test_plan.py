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


def test_a_predicate_golden_is_scoreable():
    """`logical_xor_` was measured and `logical_xor` was not, because the in-place form
    writes into a float tensor and the other returns bool. 21 ops sat outside the report."""
    import torch

    from ttnn_accuracy.ops.arity import call_golden

    out = call_golden(
        lambda a, b: a > b, [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])], False
    )
    assert out.is_floating_point()
    assert out.tolist() == [1.0, 0.0]


def test_a_predicate_is_seen_to_depend_on_its_input():
    """Every probe range was strictly positive, so `gtz` answered the same everywhere and
    read as a constant like `zeros_like`. Zero is the only place it changes its mind."""
    import torch

    from ttnn_accuracy.ops.arity import _same

    gtz = lambda a: (a > 0).to(a.dtype)  # noqa: E731
    base = torch.rand(8, dtype=torch.float64) * 0.8 + 0.1
    assert all(_same(gtz(base), gtz(f(base))) for f in (lambda a: a + 10.0, lambda a: -a * 0 + 1))
    assert not _same(gtz(base), gtz(base * 0.0))


def _entry(**over) -> dict:
    base = {"name": "exp", "category": "unary", "operands": 1, "signature": "(x, **_)"}
    return {"ops": {"ttnn.exp": base | over}}


def test_a_cosmetic_signature_change_does_not_invalidate_a_probe():
    """ttnn rendering `**_` as `*args, **_` cost 69 derived domains and 85 layout records."""
    from ttnn_accuracy.ops.manifest import diff

    assert diff(_entry(), _entry(signature="(x, *args, **_)"))[2] == []


def test_a_changed_arity_does_invalidate_it():
    from ttnn_accuracy.ops.manifest import diff

    assert diff(_entry(), _entry(operands=2))[2] == ["ttnn.exp"]
    assert diff(_entry(), _entry(category="unary_bw"))[2] == ["ttnn.exp"]


def test_an_integer_scalar_lands_on_the_op_s_own_spelling():
    """`--params relu_max=6` published a second `upper_limit6` beside `upper_limit=6.0`."""
    from ttnn_accuracy.ops.overrides import OVERRIDES, variant_slug, with_value

    base = OVERRIDES["ttnn.relu_max"][0]
    assert with_value(base, 6).params_desc == "upper_limit=6.0"
    assert variant_slug(with_value(base, 6).params_desc) == "upper_limit6.0"
    assert with_value(OVERRIDES["ttnn.polygamma"][0], 2).ttnn_kwargs == {"k": 2}


def test_a_value_measured_only_in_this_run_still_reads_as_a_parameter():
    from ttnn_accuracy.ops.plan import params_desc

    assert params_desc("relu_max", "upper_limit3.0") == "upper_limit=3.0"
    assert params_desc("relu_max", "upper_limit6.0") == "upper_limit=6.0"
