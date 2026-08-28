"""Every editorial decision the pipeline cannot derive, in one table.

Discovery learns what ttnn exposes and what each golden refuses; it cannot choose the
slope leaky_relu is measured at, supply the reference ttnn never attached to divide, or
rule that `clone` computes nothing worth scoring. Those calls are made here and nowhere
else — an op is covered, parameterised or excluded by adding one entry, which is also
the entire surface a future automated pipeline needs to write to.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


def bw_fn(ttnn_bw_op):
    """Backward call: gradient of ones, take the gradient w.r.t. the first operand.

    The gradient tensor and the grads not under test are freed before returning — an
    fp32 sweep makes a thousand of each per op, faster than GC keeps up with.
    """
    import ttnn

    def call(*operands, **params):
        ones = ttnn.ones_like(operands[0])
        grads = ttnn_bw_op(ones, *operands, **params)
        for t in (ones, *grads[1:]):
            if t.is_allocated():
                ttnn.deallocate(t)
        return grads[0]

    return call


@dataclass(frozen=True, slots=True)
class Override:
    """Scalar parameters an op needs before it is measurable.

    The values are editorial: discovery can learn that leaky_relu refuses two tensors,
    but not that it should be measured at torch's default slope. Binding them with
    functools.partial makes the bound argument stop counting as an operand, so the op
    reclassifies (leaky_relu binary → unary) and probe, derive and sweep run unchanged.
    `golden` replaces the attached golden outright — for a condition the sweep cannot
    send in the attached golden's dtype (`where`), scalars keyword binding cannot reach
    (`threshold_bw`), or an op upstream never gave a reference at all (`divide`).
    """

    ttnn_kwargs: dict
    golden_kwargs: dict
    params_desc: str
    golden: Callable | None = None


def _where_golden(condition, x, y):
    # Bitwise-nonzero is the device's truth test, != 0 is IEEE's: they part company only
    # at -0.0, and that disagreement is a finding to measure, not to paper over.
    import torch

    return torch.where(condition != 0, x, y)


def _divide_golden(x, y):
    # ttnn.divide never got a golden upstream; the mathematics is not in doubt.
    import torch

    return torch.divide(x, y)


def _threshold_bw_golden(grad, x):
    # ttnn binds the scalars as min/max but they are threshold and value; the replaced
    # branch has zero slope, so value never appears in the gradient. Written out rather
    # than bound because the attached golden takes its scalars as unnamed *args.
    import torch

    return torch.where(x > 0.5, grad, torch.zeros_like(grad))


def _override(
    ttnn_kwargs: dict, golden_kwargs: dict | None = None, golden=None, desc: str | None = None
) -> Override:
    """golden_kwargs defaults to ttnn_kwargs; six goldens name the same scalar differently."""
    desc = desc or ",".join(f"{k}={v}" for k, v in ttnn_kwargs.items()) or "default"
    return Override(
        ttnn_kwargs, ttnn_kwargs if golden_kwargs is None else golden_kwargs, desc, golden
    )


_FAST = _override({"fast_and_approximate_mode": True}, {}, desc="fast_approx")

OVERRIDES: dict[str, tuple[Override, ...]] = {
    "ttnn.divide": (_override({}, {}, golden=_divide_golden),),
    "ttnn.exp": (_override({}), _FAST),
    "ttnn.gelu": (_override({}), _FAST),
    "ttnn.leaky_relu": (_override({"negative_slope": 0.01}),),
    "ttnn.heaviside": (_override({"value": 0.5}),),
    "ttnn.relu_max": (_override({"upper_limit": 1.0}),),
    "ttnn.relu_min": (_override({"lower_limit": 1.0}),),
    "ttnn.rpow": (_override({"exponent": 2.0}, {"dim": 2.0}),),
    "ttnn.rpow_bw": (_override({"exponent": 2.0}, {"alpha": 2.0}),),
    "ttnn.softcap": (_override({"beta": 50.0}),),
    "ttnn.div_no_nan_bw": (_override({"scalar": 2.0}, {"alpha": 2.0}),),
    "ttnn.clamp": (_override({"min": -1.0, "max": 1.0}),),
    "ttnn.clip": (_override({"min": -1.0, "max": 1.0}),),
    "ttnn.threshold": (_override({"threshold": 0.5, "value": 0.0}),),
    "ttnn.threshold_bw": (_override({"min": 0.5, "max": 0.0}, {}, golden=_threshold_bw_golden),),
    "ttnn.addalpha": (_override({"alpha": 2.0}),),
    "ttnn.subalpha": (_override({"alpha": 2.0}),),
    "ttnn.addcdiv_bw": (_override({"alpha": 1.0}, {"value": 1.0}),),
    "ttnn.addcmul_bw": (_override({"alpha": 1.0}, {"value": 1.0}),),
    "ttnn.rdiv": (_override({"value": 2.0}),),
    "ttnn.rdiv_bw": (_override({"scalar": 2.0}, {"value": 2.0}),),
    "ttnn.polygamma": (_override({"k": 1}),),
    "ttnn.polygamma_bw": (_override({"n": 1}),),
    "ttnn.pow_bw": (_override({"exponent": 2.0}),),
    "ttnn.prelu": (_override({"weight": 0.25}, {"input_tensor_b": 0.25}),),
    "ttnn.where": (_override({}, {}, golden=_where_golden),),
}

# Elementwise by every probe, but not eltwise mathematics this report can score.
# `identity` stays, as the deliberate control for the pass-through path itself.
EXCLUDED = (
    dict.fromkeys(
        (
            "ttnn.clone",
            "ttnn.move",
            "ttnn.from_device",
            "ttnn.from_torch",
            "ttnn.reallocate",
            "ttnn.reshard",
            "ttnn.interleaved_to_sharded",
            "ttnn.sharded_to_interleaved",
            "ttnn.squeeze",
            "ttnn.tilize",
            "ttnn.tilize_with_val_padding",
            "ttnn.tilize_with_zero_padding",
            "ttnn.untilize",
            "ttnn.to_device",
            "ttnn.to_dtype",
            "ttnn.to_layout",
            "ttnn.to_memory_config",
            "ttnn.to_torch",
            "ttnn.typecast",
            "ttnn.fill_implicit_tile_padding",
            "ttnn.indexed_fill",
        ),
        "moves data without computing on it",
    )
    | dict.fromkeys(
        ("ttnn.angle", "ttnn.conj", "ttnn.real"), "complex-valued — the sweep sends real floats"
    )
    | {"ttnn.plus_one": "integer-only — ULP needs floats"}
)


def variant_slug(params_desc: str) -> str:
    """Filename form of a variant. Sole definition — measure and report must agree on it."""
    return params_desc.replace(" ", "_").replace(",", "_").replace("=", "")
