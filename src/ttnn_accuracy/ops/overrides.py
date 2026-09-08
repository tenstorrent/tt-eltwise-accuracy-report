"""Every editorial decision discovery cannot derive: scalars, supplied goldens, exclusions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


def bw_fn(ttnn_bw_op):
    """Gradient of ones, w.r.t. the first operand; the rest are freed faster than GC would."""
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
    """Scalars an op needs to be measurable; binding them stops the argument counting as an operand."""

    ttnn_kwargs: dict
    golden_kwargs: dict
    params_desc: str
    golden: Callable | None = None
    # Where the reference stops being computable: polygamma_bw took 1.8h at |x| ~ 1e8.
    bounds: dict[str, tuple[float, float]] | None = None
    why: str = ""  # carried into the index, so "why 0.01?" is answered from the record


def _where_golden(condition, x, y):
    # Device truth is bitwise-nonzero, IEEE's is != 0: they differ at -0.0, which is a finding.
    import torch

    return torch.where(condition != 0, x, y)


def _divide_golden(x, y):
    # ttnn.divide never got a golden upstream; the mathematics is not in doubt.
    import torch

    return torch.divide(x, y)


def _threshold_bw_golden(grad, x):
    # ttnn calls the scalars min/max; they are threshold and value, and value has no slope.
    import torch

    return torch.where(x > 0.5, grad, torch.zeros_like(grad))


def _override(
    ttnn_kwargs: dict,
    golden_kwargs: dict | None = None,
    golden=None,
    desc: str | None = None,
    why: str = "",
) -> Override:
    """golden_kwargs defaults to ttnn_kwargs; six goldens name the same scalar differently."""
    desc = desc or ",".join(f"{k}={v}" for k, v in ttnn_kwargs.items()) or "default"
    return Override(
        ttnn_kwargs, ttnn_kwargs if golden_kwargs is None else golden_kwargs, desc, golden, why=why
    )


_FAST = _override(
    {"fast_and_approximate_mode": True},
    {},
    desc="fast_approx",
    why="the approximation mode models opt into for speed",
)

OVERRIDES: dict[str, tuple[Override, ...]] = {
    "ttnn.divide": (
        _override({}, {}, golden=_divide_golden, why="upstream never attached a golden"),
    ),
    "ttnn.exp": (_override({}), _FAST),
    "ttnn.gelu": (_override({}), _FAST),
    # A scalar's boundaries are where the op degenerates and a kernel special-cases.
    "ttnn.leaky_relu": (
        _override({"negative_slope": 0.01}, why="torch's default slope"),
        _override({"negative_slope": 0.0}, why="degenerates to relu"),
        _override({"negative_slope": 1.0}, why="degenerates to identity"),
    ),
    "ttnn.heaviside": (
        _override({"value": 0.5}, why="torch's convention at x = 0"),
        _override({"value": 0.0}, why="the other convention at x = 0"),
        _override({"value": 1.0}, why="the third convention at x = 0"),
    ),
    "ttnn.relu_max": (
        _override({"upper_limit": 1.0}, why="unit clamp, the registry's midpoint"),
        _override({"upper_limit": 6.0}, why="relu6, the clamp models actually use"),
        _override({"upper_limit": 0.0}, why="degenerates to a constant zero"),
    ),
    "ttnn.relu_min": (
        _override({"lower_limit": 1.0}, why="unit clamp, the registry's midpoint"),
        _override({"lower_limit": 0.0}, why="degenerates to relu"),
    ),
    "ttnn.rpow": (
        _override({"exponent": 2.0}, {"dim": 2.0}, why="squaring"),
        _override({"exponent": 0.5}, {"dim": 0.5}, why="the square-root path, a half-integer"),
        _override({"exponent": 1.0}, {"dim": 1.0}, why="degenerates to identity"),
    ),
    "ttnn.rpow_bw": (_override({"exponent": 2.0}, {"alpha": 2.0}, why="squaring"),),
    "ttnn.softcap": (_override({"beta": 50.0}, why="Gemma's logit soft-cap"),),
    "ttnn.div_no_nan_bw": (
        _override({"scalar": 2.0}, {"alpha": 2.0}, why="2 keeps the op distinct from assign"),
    ),
    # Corners, not a grid: a midpoint never reaches an empty or collapsed interval.
    "ttnn.clamp": (
        _override({"min": -1.0, "max": 1.0}, why="the symmetric unit interval"),
        _override({"min": 0.0, "max": 0.0}, why="collapsed interval — every output is 0"),
        _override({"min": 1.0, "max": -1.0}, why="empty interval, min above max"),
    ),
    "ttnn.clip": (
        _override({"min": -1.0, "max": 1.0}, why="the symmetric unit interval"),
        _override({"min": 0.0, "max": 0.0}, why="collapsed interval — every output is 0"),
        _override({"min": 1.0, "max": -1.0}, why="empty interval, min above max"),
    ),
    "ttnn.threshold": (
        _override({"threshold": 0.5, "value": 0.0}, why="domain midpoint, torch's zero fill"),
        _override({"threshold": 0.0, "value": 1.0}, why="threshold at zero, non-zero fill"),
    ),
    "ttnn.threshold_bw": (
        _override(
            {"min": 0.5, "max": 0.0},
            {},
            golden=_threshold_bw_golden,
            why="ttnn binds threshold and value as min/max; the golden's *args are unbindable",
        ),
    ),
    "ttnn.addalpha": (_override({"alpha": 2.0}, why="2 keeps the op distinct from add"),),
    "ttnn.subalpha": (_override({"alpha": 2.0}, why="2 keeps the op distinct from subtract"),),
    "ttnn.addcdiv_bw": (_override({"alpha": 1.0}, {"value": 1.0}, why="torch's default value"),),
    "ttnn.addcmul_bw": (_override({"alpha": 1.0}, {"value": 1.0}, why="torch's default value"),),
    "ttnn.rdiv": (_override({"value": 2.0}, why="2 keeps the op distinct from reciprocal"),),
    "ttnn.rdiv_bw": (
        _override({"scalar": 2.0}, {"value": 2.0}, why="2 keeps the op distinct from reciprocal"),
    ),
    "ttnn.polygamma": (
        _override({"k": 1}, why="trigamma, the first order in use"),
        _override({"k": 2}, why="tetragamma; the order changes the series, not just a constant"),
        _override({"k": 4}, why="high enough that the reference and the kernel can diverge"),
    ),
    "ttnn.polygamma_bw": (
        Override(
            {"n": 1},
            {"n": 1},
            "n=1",
            bounds=dict.fromkeys(("bf16", "fp32"), (-1024.0, float("inf"))),
            why="trigamma; swept above -1024, where torch can still compute the reference",
        ),
    ),
    "ttnn.pow_bw": (_override({"exponent": 2.0}, why="squaring"),),
    "ttnn.prelu": (
        _override({"weight": 0.25}, {"input_tensor_b": 0.25}, why="torch's channel-weight init"),
    ),
    "ttnn.where": (
        _override({}, {}, golden=_where_golden, why="the attached golden demands a bool condition"),
    ),
}

# Elementwise by probe, not scorable mathematics. `identity` stays as the control.
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


def with_value(ov: Override, value: float | int) -> Override:
    """The same override at another value; the table keeps both spellings, which differ for six goldens."""
    if len(ov.ttnn_kwargs) != 1:
        takes = "no scalar to set" if not ov.ttnn_kwargs else f"more than one: {ov.params_desc}"
        raise ValueError(f"takes {takes} — edit overrides.py for this one")
    return Override(
        dict.fromkeys(ov.ttnn_kwargs, value),
        dict.fromkeys(ov.golden_kwargs, value),
        ",".join(f"{k}={value}" for k in ov.ttnn_kwargs),
        ov.golden,
        ov.bounds,
        f"{ov.why} (measured here at {value})" if ov.why else f"measured at {value}",
    )


def variant_slug(params_desc: str) -> str:
    """Filename form of a variant. Sole definition — measure and report must agree on it."""
    return params_desc.replace(" ", "_").replace(",", "_").replace("=", "")
