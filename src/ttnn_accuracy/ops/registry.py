"""
Op registry: defines all eltwise ops with variants, valid input ranges, and golden functions.

Calling conventions:
  unary/unary_bw  ttnn_fn: callable(x: ttnn.Tensor) -> ttnn.Tensor
  binary          ttnn_fn: callable(x: ttnn.Tensor, y: ttnn.Tensor) -> ttnn.Tensor
  golden_fn:      callable(x: torch.Tensor, out=None) -> torch.Tensor
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger


@dataclass(slots=True)
class InputRange:
    """Domain bounds an op is measured over. float('+-inf') means unbounded."""

    lo: float = float("-inf")
    hi: float = float("inf")
    note: str = ""


@dataclass
class OpVariant:
    """One variant of an op.

    ttnn_fn: unary/unary_bw → callable(x) → ttnn.Tensor
             binary         → callable(x, y) → ttnn.Tensor
    golden_fn: callable(x: torch.Tensor, out=None) → torch.Tensor
    params_desc: param string, e.g. "alpha=1.0" or "default"
    """

    ttnn_fn: Callable
    golden_fn: Callable
    params_desc: str = "default"


@dataclass
class OpEntry:
    name: str
    category: str  # "unary" | "binary" | "unary_bw"
    variants: list[OpVariant]
    input_range: InputRange = field(default_factory=InputRange)
    display_name: str = ""
    description: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name


def _build_registry() -> dict[str, OpEntry]:
    """Build the op registry.

    ttnn and torch are imported here rather than at module scope so importing this
    module — for `variant_slug`, or for `ttnn-accuracy --help` — costs nothing.
    """
    import torch
    import ttnn

    entries: dict[str, OpEntry] = {}

    def add(entry: OpEntry):
        entries[entry.name] = entry

    def bw_golden(ttnn_bw_op):
        """Wrap ttnn backward golden (grad, x) → [grad_tensor] into (x, out=None)."""
        bw_fn = ttnn.get_golden_function(ttnn_bw_op)

        def golden(x, out=None):
            with torch.enable_grad():
                grad = torch.ones_like(x)
                x_req = x.detach().requires_grad_(True)
                result = bw_fn(grad, x_req)[0]
            if out is not None:
                out.copy_(result)
                return out
            return result

        return golden

    def bw_golden_torch(torch_op):
        """Backward golden via torch autograd."""

        def golden(x, out=None):
            with torch.enable_grad():
                grad = torch.ones_like(x)
                x_req = x.detach().requires_grad_(True)
                y = torch_op(x_req)
                y.backward(gradient=grad)
                result = x_req.grad
            if out is not None:
                out.copy_(result)
                return out
            return result

        return golden

    def bw_fn(ttnn_bw_op):
        """Standard unary backward: grad=ones, returns first gradient."""
        return lambda x: ttnn_bw_op(ttnn.ones_like(x), x)[0]

    add(
        OpEntry(
            name="abs",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.abs(x), golden_fn=torch.abs)],
        )
    )

    add(
        OpEntry(
            name="identity",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.identity(x), golden_fn=lambda x, out=None: x.clone()
                )
            ],
        )
    )

    add(
        OpEntry(
            name="exp",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.exp(x), golden_fn=torch.exp, params_desc="default"
                ),
                OpVariant(
                    ttnn_fn=lambda x: ttnn.exp(x, fast_and_approximate_mode=True),
                    golden_fn=torch.exp,
                    params_desc="fast_approx",
                ),
            ],
            input_range=InputRange(hi=88.0, note="x > ~88 overflows fp32"),
        )
    )

    add(
        OpEntry(
            name="exp2",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.exp2(x), golden_fn=torch.exp2)],
        )
    )

    add(
        OpEntry(
            name="expm1",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.expm1(x), golden_fn=torch.expm1)],
        )
    )

    add(
        OpEntry(
            name="log",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.log(x), golden_fn=torch.log)],
            input_range=InputRange(lo=1.18e-38, note="x > 0"),
        )
    )

    add(
        OpEntry(
            name="log2",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.log2(x), golden_fn=torch.log2)],
            input_range=InputRange(lo=1.18e-38, note="x > 0"),
        )
    )

    add(
        OpEntry(
            name="log10",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.log10(x), golden_fn=torch.log10)],
            input_range=InputRange(lo=1.18e-38, note="x > 0"),
        )
    )

    add(
        OpEntry(
            name="log1p",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.log1p(x), golden_fn=torch.log1p)],
            input_range=InputRange(lo=-1.0 + 1e-7, note="x > -1"),
        )
    )

    add(
        OpEntry(
            name="sqrt",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.sqrt(x), golden_fn=torch.sqrt)],
            input_range=InputRange(lo=0.0, note="x ≥ 0"),
        )
    )

    add(
        OpEntry(
            name="rsqrt",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.rsqrt(x), golden_fn=torch.rsqrt)],
            input_range=InputRange(lo=1.18e-38, note="x > 0"),
        )
    )

    add(
        OpEntry(
            name="cbrt",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.cbrt(x),
                    # torch.pow(negative, 1/3) returns NaN; use sign * |x|^(1/3) for real cbrt
                    golden_fn=lambda x, out=None: (
                        x.sign() * x.abs().to(torch.float64).pow(1 / 3)
                    ).to(x.dtype),
                )
            ],
        )
    )

    add(
        OpEntry(
            name="reciprocal",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.reciprocal(x),
                    golden_fn=lambda x, out=None: torch.reciprocal(x),
                )
            ],
            input_range=InputRange(note="avoid x ≈ 0"),
        )
    )

    add(
        OpEntry(
            name="tanh",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.tanh(x), golden_fn=torch.tanh)],
        )
    )

    add(
        OpEntry(
            name="sinh",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.sinh(x), golden_fn=torch.sinh)],
        )
    )

    add(
        OpEntry(
            name="cosh",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.cosh(x), golden_fn=torch.cosh)],
        )
    )

    # BF16 argument reduction: ULP_bf16(128) = 1, so sin/cos become meaningless for |x| > ~128.
    add(
        OpEntry(
            name="sin",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.sin(x), golden_fn=torch.sin)],
            input_range=InputRange(
                lo=-128.0,
                hi=128.0,
                note="BF16 argument reduction is undefined for |x| > ~128",
            ),
        )
    )

    add(
        OpEntry(
            name="cos",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.cos(x), golden_fn=torch.cos)],
            input_range=InputRange(
                lo=-128.0,
                hi=128.0,
                note="BF16 argument reduction is undefined for |x| > ~128",
            ),
        )
    )

    add(
        OpEntry(
            name="tan",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.tan(x), golden_fn=torch.tan)],
            input_range=InputRange(
                lo=-math.pi / 2 * 0.99,
                hi=math.pi / 2 * 0.99,
                note="avoid poles at (n+0.5)π; BF16 argument reduction undefined for |x| > ~128",
            ),
        )
    )

    add(
        OpEntry(
            name="asin",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.asin(x), golden_fn=torch.asin)],
            input_range=InputRange(lo=-1.0, hi=1.0, note="x ∈ [-1, 1]"),
        )
    )

    add(
        OpEntry(
            name="acos",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.acos(x), golden_fn=torch.acos)],
            input_range=InputRange(lo=-1.0, hi=1.0, note="x ∈ [-1, 1]"),
        )
    )

    add(
        OpEntry(
            name="atan",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.atan(x), golden_fn=torch.atan)],
        )
    )

    add(
        OpEntry(
            name="erf",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.erf(x),
                    golden_fn=lambda x, out=None: torch.special.erf(x),
                )
            ],
        )
    )

    add(
        OpEntry(
            name="erfinv",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.erfinv(x),
                    golden_fn=lambda x, out=None: torch.special.erfinv(x),
                )
            ],
            input_range=InputRange(lo=-1.0 + 1e-6, hi=1.0 - 1e-6, note="x ∈ (-1, 1)"),
        )
    )

    add(
        OpEntry(
            name="sigmoid",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.sigmoid(x), golden_fn=torch.sigmoid)],
        )
    )

    add(
        OpEntry(
            name="log_sigmoid",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.log_sigmoid(x),
                    golden_fn=lambda x, out=None: torch.nn.functional.logsigmoid(x),
                )
            ],
        )
    )

    add(
        OpEntry(
            name="silu",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.silu(x),
                    golden_fn=lambda x, out=None: torch.nn.functional.silu(x),
                )
            ],
            # sigmoid(x) becomes subnormal (flushed to 0) for x < -ln(2^126) ≈ -87.3,
            # causing silu = x*0 = 0 while golden gives a small normal value.
            input_range=InputRange(
                lo=-87.0,
            ),
        )
    )

    add(
        OpEntry(
            name="gelu",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.gelu(x),
                    golden_fn=lambda x, out=None: torch.nn.functional.gelu(x),
                    params_desc="default",
                ),
                OpVariant(
                    ttnn_fn=lambda x: ttnn.gelu(x, fast_and_approximate_mode=True),
                    golden_fn=lambda x, out=None: torch.nn.functional.gelu(x),
                    params_desc="fast_approx",
                ),
            ],
        )
    )

    add(
        OpEntry(
            name="mish",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.mish(x),
                    golden_fn=lambda x, out=None: torch.nn.functional.mish(x),
                )
            ],
        )
    )

    add(
        OpEntry(
            name="hardmish",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.hardmish(x),
                    golden_fn=lambda x, out=None: x * torch.clamp(x + 2, 0, 2) / 2,
                )
            ],
            description="hardmish: x * clamp(x+2, 0, 2) / 2",
        )
    )

    add(
        OpEntry(
            name="relu",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.relu(x),
                    golden_fn=lambda x, out=None: torch.nn.functional.relu(x),
                )
            ],
        )
    )

    for upper in [0.5, 1.0, 2.0]:
        add(
            OpEntry(
                name=f"relu_max_ul{upper}".replace(".", "p"),
                category="unary",
                display_name="relu_max",
                variants=[
                    OpVariant(
                        ttnn_fn=lambda x, u=upper: ttnn.relu_max(x, upper_limit=u),
                        golden_fn=lambda x, out=None, u=upper: torch.minimum(
                            torch.nn.functional.relu(x), torch.full_like(x, u)
                        ),
                        params_desc=f"upper_limit={upper}",
                    )
                ],
                description=f"relu_max upper_limit={upper}",
            )
        )

    for lower in [0.5, 1.0, 2.0]:
        add(
            OpEntry(
                name=f"relu_min_ll{lower}".replace(".", "p"),
                category="unary",
                display_name="relu_min",
                variants=[
                    OpVariant(
                        ttnn_fn=lambda x, lim=lower: ttnn.relu_min(x, lower_limit=lim),
                        golden_fn=lambda x, out=None, lim=lower: torch.maximum(
                            torch.nn.functional.relu(x), torch.full_like(x, lim)
                        ),
                        params_desc=f"lower_limit={lower}",
                    )
                ],
                description=f"relu_min lower_limit={lower}",
            )
        )

    add(
        OpEntry(
            name="hardtanh",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.hardtanh(x),
                    golden_fn=lambda x, out=None: torch.nn.functional.hardtanh(x, -1, 1),
                    params_desc="min=-1,max=1",
                ),
                OpVariant(
                    ttnn_fn=lambda x: ttnn.hardtanh(x, min_val=-2, max_val=2),
                    golden_fn=lambda x, out=None: torch.nn.functional.hardtanh(x, -2, 2),
                    params_desc="min=-2,max=2",
                ),
            ],
        )
    )

    for alpha in [0.5, 1.0, 2.0]:
        add(
            OpEntry(
                name=f"elu_alpha{alpha}".replace(".", "p"),
                category="unary",
                display_name="elu",
                variants=[
                    OpVariant(
                        ttnn_fn=lambda x, a=alpha: ttnn.elu(x, alpha=a),
                        golden_fn=lambda x, out=None, a=alpha: torch.nn.functional.elu(x, alpha=a),
                        params_desc=f"alpha={alpha}",
                    )
                ],
                description=f"ELU alpha={alpha}",
            )
        )

    for alpha in [0.5, 1.0, 2.0]:
        add(
            OpEntry(
                name=f"celu_alpha{alpha}".replace(".", "p"),
                category="unary",
                display_name="celu",
                variants=[
                    OpVariant(
                        ttnn_fn=lambda x, a=alpha: ttnn.celu(x, alpha=a),
                        golden_fn=lambda x, out=None, a=alpha: torch.nn.functional.celu(x, alpha=a),
                        params_desc=f"alpha={alpha}",
                    )
                ],
                description=f"CELU alpha={alpha}",
            )
        )

    add(
        OpEntry(
            name="selu",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.selu(x),
                    golden_fn=lambda x, out=None: torch.nn.functional.selu(x),
                )
            ],
        )
    )

    # For softplus(x, beta) = log(1+e^(beta*x))/beta, e^(beta*x) becomes subnormal when
    # beta*x < -ln(2^126) ≈ -87.3, i.e. x < -87.3/beta.  Hardware flushes e^(beta*x)→0
    # giving softplus→0, while float64 golden gives log(1+tiny)/beta ≈ tiny/beta which
    # can be a normal fp32 value (especially for fractional beta), creating false ULP spikes.
    for beta, lo_bound in [(0.5, -174.0), (1.0, -87.0), (2.0, -43.0)]:
        add(
            OpEntry(
                name=f"softplus_beta{beta}".replace(".", "p"),
                category="unary",
                display_name="softplus",
                variants=[
                    OpVariant(
                        ttnn_fn=lambda x, b=beta: ttnn.softplus(x, beta=b),
                        golden_fn=lambda x, out=None, b=beta: torch.nn.functional.softplus(
                            x, beta=b
                        ),
                        params_desc=f"beta={beta}",
                    )
                ],
                input_range=InputRange(lo=lo_bound),
                description=f"Softplus beta={beta}",
            )
        )

    add(
        OpEntry(
            name="softsign",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.softsign(x),
                    golden_fn=lambda x, out=None: torch.nn.functional.softsign(x),
                )
            ],
            # For |x| >= 2^126 ≈ 8.507e37 the hardware's reciprocal of (1+|x|) underflows
            # to subnormal and is flushed to 0, so x*(1/(1+|x|)) → 0 instead of ±1.
            input_range=InputRange(
                lo=-8.5e37,
                hi=8.5e37,
            ),
        )
    )

    add(
        OpEntry(
            name="tanhshrink",
            category="unary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x: ttnn.tanhshrink(x),
                    golden_fn=lambda x, out=None: torch.nn.functional.tanhshrink(x),
                )
            ],
        )
    )

    add(
        OpEntry(
            name="logit",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.logit(x), golden_fn=torch.logit)],
            input_range=InputRange(lo=1e-6, hi=1.0 - 1e-6, note="x ∈ (0, 1)"),
        )
    )

    add(
        OpEntry(
            name="digamma",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.digamma(x), golden_fn=torch.digamma)],
            input_range=InputRange(lo=1e-6, note="x > 0 (poles at non-positive integers)"),
        )
    )

    add(
        OpEntry(
            name="lgamma",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.lgamma(x), golden_fn=torch.lgamma)],
            input_range=InputRange(lo=1e-6, note="x > 0"),
        )
    )

    add(
        OpEntry(
            name="round",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.round(x), golden_fn=torch.round)],
        )
    )

    add(
        OpEntry(
            name="ceil",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.ceil(x), golden_fn=torch.ceil)],
        )
    )

    add(
        OpEntry(
            name="floor",
            category="unary",
            variants=[OpVariant(ttnn_fn=lambda x: ttnn.floor(x), golden_fn=torch.floor)],
        )
    )

    add(
        OpEntry(
            name="add",
            category="binary",
            variants=[OpVariant(ttnn_fn=lambda x, y: ttnn.add(x, y), golden_fn=torch.add)],
        )
    )

    add(
        OpEntry(
            name="subtract",
            category="binary",
            variants=[
                OpVariant(ttnn_fn=lambda x, y: ttnn.subtract(x, y), golden_fn=torch.subtract)
            ],
        )
    )

    add(
        OpEntry(
            name="multiply",
            category="binary",
            variants=[
                OpVariant(ttnn_fn=lambda x, y: ttnn.multiply(x, y), golden_fn=torch.multiply)
            ],
        )
    )

    add(
        OpEntry(
            name="divide",
            category="binary",
            variants=[
                OpVariant(
                    ttnn_fn=lambda x, y: ttnn.divide(x, y),
                    golden_fn=torch.div,
                    params_desc="default",
                ),
                OpVariant(
                    ttnn_fn=lambda x, y: ttnn.div(x, y, accurate_mode=True),
                    golden_fn=torch.div,
                    params_desc="accurate",
                ),
            ],
            input_range=InputRange(note="avoid y ≈ 0"),
        )
    )

    add(
        OpEntry(
            name="hypot",
            category="binary",
            variants=[OpVariant(ttnn_fn=lambda x, y: ttnn.hypot(x, y), golden_fn=torch.hypot)],
        )
    )

    add(
        OpEntry(
            name="pow",
            category="binary",
            variants=[OpVariant(ttnn_fn=lambda x, y: ttnn.pow(x, y), golden_fn=torch.pow)],
            input_range=InputRange(lo=0.0, note="x ≥ 0 for real output"),
        )
    )

    add(
        OpEntry(
            name="atan2",
            category="binary",
            variants=[OpVariant(ttnn_fn=lambda x, y: ttnn.atan2(x, y), golden_fn=torch.atan2)],
        )
    )

    add(
        OpEntry(
            name="rsub",
            category="binary",
            variants=[OpVariant(ttnn_fn=lambda x, y: ttnn.rsub(x, y), golden_fn=torch.rsub)],
        )
    )

    # Domains are unrestricted here: derive.py will compute each op's real bounds
    # from its fp64 golden. Until then a bw op is swept over every input.
    missing = []
    for bw_name in [
        "abs_bw",
        "floor_bw",
        "exp_bw",
        "exp2_bw",
        "expm1_bw",
        "log_bw",
        "log10_bw",
        "log2_bw",
        "log1p_bw",
        "sqrt_bw",
        "rsqrt_bw",
        "sin_bw",
        "cos_bw",
        "tan_bw",
        "asin_bw",
        "acos_bw",
        "atan_bw",
        "sinh_bw",
        "cosh_bw",
        "tanh_bw",
        "asinh_bw",
        "atanh_bw",
        "tanhshrink_bw",
        "hardtanh_bw",
        "digamma_bw",
        "lgamma_bw",
        "erfinv_bw",
        "sigmoid_bw",
        "silu_bw",
        "gelu_bw",
        "celu_bw",
        "elu_bw",
        "selu_bw",
        "softplus_bw",
        "softsign_bw",
    ]:
        _op = getattr(ttnn, bw_name, None)
        if _op is None:
            missing.append(bw_name)
        else:
            add(
                OpEntry(
                    name=bw_name,
                    category="unary_bw",
                    variants=[OpVariant(ttnn_fn=bw_fn(_op), golden_fn=bw_golden(_op))],
                )
            )

    # acosh_bw: ttnn golden needs device kwargs → use torch autograd instead
    if acosh_bw := getattr(ttnn, "acosh_bw", None):
        add(
            OpEntry(
                name="acosh_bw",
                category="unary_bw",
                variants=[
                    OpVariant(ttnn_fn=bw_fn(acosh_bw), golden_fn=bw_golden_torch(torch.acosh))
                ],
                input_range=InputRange(lo=1.0, note="x ≥ 1"),
            )
        )
    else:
        missing.append("acosh_bw")

    if missing:
        logger.warning("not in this ttnn build, skipped: {}", ", ".join(missing))

    return entries


_REGISTRY: dict[str, OpEntry] | None = None


def get_registry() -> dict[str, OpEntry]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def list_op_names(category: str | None = None) -> list[str]:
    reg = get_registry()
    if category:
        return [k for k, v in reg.items() if v.category == category]
    return list(reg.keys())


def get_op(name: str) -> OpEntry:
    return get_registry()[name]


def variant_slug(params_desc: str) -> str:
    """Filename form of a variant. Sole definition — measure and report must agree on it."""
    return params_desc.replace(" ", "_").replace(",", "_").replace("=", "")
