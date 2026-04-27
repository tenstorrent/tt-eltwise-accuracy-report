"""
Op registry: defines all eltwise ops with variants, valid input ranges, and golden functions.
This is the single source of truth for what gets measured and reported.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InputRange:
    """Valid/interesting input range for an op.

    lo/hi may be float('-inf')/float('inf') for unbounded domains.
    display_lo/display_hi narrow what gets shown on the x-axis of plots.
    """
    lo: float = float("-inf")
    hi: float = float("inf")
    display_lo: Optional[float] = None
    display_hi: Optional[float] = None
    note: str = ""

    def display_bounds(self):
        lo = self.display_lo if self.display_lo is not None else self.lo
        hi = self.display_hi if self.display_hi is not None else self.hi
        return lo, hi


@dataclass
class OpVariant:
    """One concrete variant of an op (specific parameter combination).

    ttnn_fn: callable(x_ttnn, output_tensor) -> ttnn.Tensor  (for unary / unary_bw)
             callable(x_ttnn, y_ttnn) -> ttnn.Tensor          (for binary)
    golden_fn: callable(x_torch, out=None) -> torch.Tensor
    params_desc: human-readable string describing the parameters, e.g. "alpha=1.0"
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


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def _build_registry() -> dict[str, OpEntry]:
    """Build and return the full op registry.

    Imports ttnn and torch lazily so the registry module can be imported
    without a live TT device (e.g., for report generation scripts).
    """
    import torch
    import ttnn

    entries: dict[str, OpEntry] = {}

    def add(entry: OpEntry):
        entries[entry.name] = entry

    def bw_golden(ttnn_bw_op):
        """Wrap ttnn backward golden (grad, x) -> [grad_tensor] into (x, out=None)."""
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
        """Backward golden via torch autograd when ttnn golden needs device kwargs."""
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

    def bw_impl(ttnn_bw_op):
        return lambda x, output_tensor: ttnn_bw_op(ttnn.ones_like(x), x)[0]

    # -----------------------------------------------------------------------
    # UNARY OPERATIONS
    # -----------------------------------------------------------------------

    add(OpEntry(
        name="abs",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=ttnn.abs,
            golden_fn=torch.abs,
        )],
        input_range=InputRange(display_lo=-200, display_hi=200),
    ))

    add(OpEntry(
        name="identity",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=ttnn.identity,
            golden_fn=lambda x, out=None: x.clone(),
        )],
        input_range=InputRange(display_lo=-200, display_hi=200),
    ))

    add(OpEntry(
        name="exp",
        category="unary",
        variants=[
            OpVariant(
                ttnn_fn=ttnn.exp,
                golden_fn=torch.exp,
                params_desc="default",
            ),
            OpVariant(
                ttnn_fn=lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
                golden_fn=torch.exp,
                params_desc="fast_approx",
            ),
        ],
        input_range=InputRange(lo=float("-inf"), hi=88.0, display_lo=-10, display_hi=88,
                               note="x > ~88 overflows f32"),
    ))

    add(OpEntry(
        name="exp2",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.exp2, golden_fn=torch.exp2)],
        input_range=InputRange(display_lo=-20, display_hi=127),
    ))

    add(OpEntry(
        name="expm1",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.expm1, golden_fn=torch.expm1)],
        input_range=InputRange(display_lo=-10, display_hi=88),
    ))

    add(OpEntry(
        name="log",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.log, golden_fn=torch.log)],
        input_range=InputRange(lo=1.18e-38, display_lo=1e-7, display_hi=1e7,
                               note="x > 0"),
    ))

    add(OpEntry(
        name="log2",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.log2, golden_fn=torch.log2)],
        input_range=InputRange(lo=1.18e-38, display_lo=1e-7, display_hi=1e7,
                               note="x > 0"),
    ))

    add(OpEntry(
        name="log10",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.log10, golden_fn=torch.log10)],
        input_range=InputRange(lo=1.18e-38, display_lo=1e-7, display_hi=1e7,
                               note="x > 0"),
    ))

    add(OpEntry(
        name="log1p",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.log1p, golden_fn=torch.log1p)],
        input_range=InputRange(lo=-1.0 + 1e-7, display_lo=-1.0, display_hi=100,
                               note="x > -1"),
    ))

    add(OpEntry(
        name="sqrt",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.sqrt, golden_fn=torch.sqrt)],
        input_range=InputRange(lo=0.0, display_lo=0, display_hi=1e6,
                               note="x >= 0"),
    ))

    add(OpEntry(
        name="rsqrt",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.rsqrt, golden_fn=torch.rsqrt)],
        input_range=InputRange(lo=1.18e-38, display_lo=1e-7, display_hi=1e6,
                               note="x > 0"),
    ))

    add(OpEntry(
        name="cbrt",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.cbrt(x),
            golden_fn=lambda x, out=None: torch.pow(x.to(torch.float64), 1/3).to(x.dtype),
        )],
        input_range=InputRange(display_lo=-1e6, display_hi=1e6),
    ))

    add(OpEntry(
        name="reciprocal",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=ttnn.reciprocal,
            golden_fn=lambda x, out=None: torch.reciprocal(x),
        )],
        input_range=InputRange(display_lo=-100, display_hi=100,
                               note="avoid x ≈ 0"),
    ))

    add(OpEntry(
        name="tanh",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.tanh, golden_fn=torch.tanh)],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="sinh",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.sinh(x),
            golden_fn=torch.sinh,
        )],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="cosh",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.cosh(x),
            golden_fn=torch.cosh,
        )],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="sin",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.sin, golden_fn=torch.sin)],
        input_range=InputRange(display_lo=-math.pi * 4, display_hi=math.pi * 4),
    ))

    add(OpEntry(
        name="cos",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.cos, golden_fn=torch.cos)],
        input_range=InputRange(display_lo=-math.pi * 4, display_hi=math.pi * 4),
    ))

    add(OpEntry(
        name="tan",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.tan, golden_fn=torch.tan)],
        input_range=InputRange(display_lo=-math.pi / 2 * 0.99, display_hi=math.pi / 2 * 0.99,
                               note="avoid poles at (n+0.5)π"),
    ))

    add(OpEntry(
        name="asin",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.asin, golden_fn=torch.asin)],
        input_range=InputRange(lo=-1.0, hi=1.0, display_lo=-1, display_hi=1,
                               note="x ∈ [-1, 1]"),
    ))

    add(OpEntry(
        name="acos",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.acos, golden_fn=torch.acos)],
        input_range=InputRange(lo=-1.0, hi=1.0, display_lo=-1, display_hi=1,
                               note="x ∈ [-1, 1]"),
    ))

    add(OpEntry(
        name="atan",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.atan, golden_fn=torch.atan)],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    add(OpEntry(
        name="erf",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.erf(x),
            golden_fn=lambda x, out=None: torch.special.erf(x),
        )],
        input_range=InputRange(display_lo=-5, display_hi=5),
    ))

    add(OpEntry(
        name="erfinv",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.erfinv(x),
            golden_fn=lambda x, out=None: torch.special.erfinv(x),
        )],
        input_range=InputRange(lo=-1.0 + 1e-6, hi=1.0 - 1e-6,
                               display_lo=-1, display_hi=1,
                               note="x ∈ (-1, 1)"),
    ))

    add(OpEntry(
        name="sigmoid",
        category="unary",
        variants=[OpVariant(ttnn_fn=ttnn.sigmoid, golden_fn=torch.sigmoid)],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="log_sigmoid",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=ttnn.log_sigmoid,
            golden_fn=lambda x, out=None: torch.nn.functional.logsigmoid(x),
        )],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="silu",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=ttnn.silu,
            golden_fn=lambda x, out=None: torch.nn.functional.silu(x),
        )],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="gelu",
        category="unary",
        variants=[
            OpVariant(
                ttnn_fn=ttnn.gelu,
                golden_fn=lambda x, out=None: torch.nn.functional.gelu(x),
                params_desc="default",
            ),
            OpVariant(
                ttnn_fn=lambda x, output_tensor: ttnn.gelu(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
                golden_fn=lambda x, out=None: torch.nn.functional.gelu(x),
                params_desc="fast_approx",
            ),
        ],
        input_range=InputRange(display_lo=-5, display_hi=5),
    ))

    add(OpEntry(
        name="mish",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=ttnn.mish,
            golden_fn=lambda x, out=None: torch.nn.functional.mish(x),
        )],
        input_range=InputRange(display_lo=-5, display_hi=5),
    ))

    add(OpEntry(
        name="hardmish",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=ttnn.hardmish,
            golden_fn=lambda x, out=None: torch.nn.functional.hardswish(x),
        )],
        input_range=InputRange(display_lo=-5, display_hi=5),
    ))

    add(OpEntry(
        name="relu",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=ttnn.relu,
            golden_fn=lambda x, out=None: torch.nn.functional.relu(x),
        )],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    # relu_max with multiple upper_limit values
    for upper in [0.5, 1.0, 2.0]:
        _u = upper  # capture
        add(OpEntry(
            name=f"relu_max_ul{upper}".replace(".", "p"),
            category="unary",
            display_name="relu_max",
            variants=[OpVariant(
                ttnn_fn=lambda x, output_tensor, u=_u: ttnn.relu_max(x, output_tensor=output_tensor, upper_limit=u),
                golden_fn=lambda x, out=None, u=_u: torch.minimum(
                    torch.nn.functional.relu(x), torch.full_like(x, u)
                ),
                params_desc=f"upper_limit={upper}",
            )],
            input_range=InputRange(display_lo=-5, display_hi=5),
            description=f"relu_max with upper_limit={upper}",
        ))

    # relu_min with multiple lower_limit values
    for lower in [0.5, 1.0, 2.0]:
        _l = lower
        add(OpEntry(
            name=f"relu_min_ll{lower}".replace(".", "p"),
            category="unary",
            display_name="relu_min",
            variants=[OpVariant(
                ttnn_fn=lambda x, output_tensor, l=_l: ttnn.relu_min(x, output_tensor=output_tensor, lower_limit=l),
                golden_fn=lambda x, out=None, l=_l: torch.maximum(
                    torch.nn.functional.relu(x), torch.full_like(x, l)
                ),
                params_desc=f"lower_limit={lower}",
            )],
            input_range=InputRange(display_lo=-5, display_hi=5),
            description=f"relu_min with lower_limit={lower}",
        ))

    add(OpEntry(
        name="hardtanh",
        category="unary",
        variants=[
            OpVariant(
                ttnn_fn=lambda x, output_tensor: ttnn.hardtanh(x),
                golden_fn=lambda x, out=None: torch.nn.functional.hardtanh(x, min_val=-1, max_val=1),
                params_desc="min=-1,max=1",
            ),
            OpVariant(
                ttnn_fn=lambda x, output_tensor: ttnn.hardtanh(x, min_val=-2, max_val=2),
                golden_fn=lambda x, out=None: torch.nn.functional.hardtanh(x, min_val=-2, max_val=2),
                params_desc="min=-2,max=2",
            ),
        ],
        input_range=InputRange(display_lo=-5, display_hi=5),
    ))

    # elu with multiple alpha values
    for alpha in [0.5, 1.0, 2.0]:
        _a = alpha
        add(OpEntry(
            name=f"elu_alpha{alpha}".replace(".", "p"),
            category="unary",
            display_name="elu",
            variants=[OpVariant(
                ttnn_fn=lambda x, output_tensor, a=_a: ttnn.elu(x, output_tensor=output_tensor, alpha=a),
                golden_fn=lambda x, out=None, a=_a: torch.nn.functional.elu(x, alpha=a),
                params_desc=f"alpha={alpha}",
            )],
            input_range=InputRange(display_lo=-5, display_hi=5),
            description=f"ELU activation with alpha={alpha}",
        ))

    # celu with multiple alpha values
    for alpha in [0.5, 1.0, 2.0]:
        _a = alpha
        add(OpEntry(
            name=f"celu_alpha{alpha}".replace(".", "p"),
            category="unary",
            display_name="celu",
            variants=[OpVariant(
                ttnn_fn=lambda x, output_tensor, a=_a: ttnn.celu(x, output_tensor=output_tensor, alpha=a),
                golden_fn=lambda x, out=None, a=_a: torch.nn.functional.celu(x, alpha=a),
                params_desc=f"alpha={alpha}",
            )],
            input_range=InputRange(display_lo=-5, display_hi=5),
            description=f"CELU activation with alpha={alpha}",
        ))

    add(OpEntry(
        name="selu",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.selu(x),
            golden_fn=lambda x, out=None: torch.nn.functional.selu(x),
        )],
        input_range=InputRange(display_lo=-5, display_hi=5),
    ))

    # softplus with multiple beta values
    for beta in [0.5, 1.0, 2.0]:
        _b = beta
        add(OpEntry(
            name=f"softplus_beta{beta}".replace(".", "p"),
            category="unary",
            display_name="softplus",
            variants=[OpVariant(
                ttnn_fn=lambda x, output_tensor, b=_b: ttnn.softplus(x, beta=b),
                golden_fn=lambda x, out=None, b=_b: torch.nn.functional.softplus(x, beta=b),
                params_desc=f"beta={beta}",
            )],
            input_range=InputRange(display_lo=-5, display_hi=5),
            description=f"Softplus with beta={beta}",
        ))

    add(OpEntry(
        name="softsign",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.softsign(x),
            golden_fn=lambda x, out=None: torch.nn.functional.softsign(x),
        )],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="tanhshrink",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.tanhshrink(x),
            golden_fn=lambda x, out=None: torch.nn.functional.tanhshrink(x),
        )],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="logit",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.logit(x),
            golden_fn=torch.logit,
        )],
        input_range=InputRange(lo=1e-6, hi=1.0 - 1e-6,
                               display_lo=0, display_hi=1,
                               note="x ∈ (0, 1)"),
    ))

    add(OpEntry(
        name="digamma",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.digamma(x),
            golden_fn=torch.digamma,
        )],
        input_range=InputRange(lo=1e-6, display_lo=0, display_hi=10,
                               note="x > 0 (avoid poles at non-positive integers)"),
    ))

    add(OpEntry(
        name="lgamma",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.lgamma(x),
            golden_fn=torch.lgamma,
        )],
        input_range=InputRange(lo=1e-6, display_lo=0, display_hi=20,
                               note="x > 0"),
    ))

    add(OpEntry(
        name="round",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.round(x, output_tensor=output_tensor),
            golden_fn=torch.round,
        )],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="ceil",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.ceil(x, output_tensor=output_tensor),
            golden_fn=torch.ceil,
        )],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="floor",
        category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x, output_tensor: ttnn.floor(x, output_tensor=output_tensor),
            golden_fn=torch.floor,
        )],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    # -----------------------------------------------------------------------
    # BINARY OPERATIONS
    # -----------------------------------------------------------------------

    add(OpEntry(
        name="add",
        category="binary",
        variants=[OpVariant(ttnn_fn=ttnn.add, golden_fn=torch.add)],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    add(OpEntry(
        name="subtract",
        category="binary",
        variants=[OpVariant(ttnn_fn=ttnn.subtract, golden_fn=torch.subtract)],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    add(OpEntry(
        name="multiply",
        category="binary",
        variants=[
            OpVariant(ttnn_fn=ttnn.multiply, golden_fn=torch.multiply, params_desc="default"),
        ],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    add(OpEntry(
        name="divide",
        category="binary",
        variants=[
            OpVariant(ttnn_fn=ttnn.divide, golden_fn=torch.div, params_desc="default"),
            OpVariant(
                ttnn_fn=lambda x, y: ttnn.div(x, y, accurate_mode=True),
                golden_fn=torch.div,
                params_desc="accurate",
            ),
        ],
        input_range=InputRange(display_lo=-100, display_hi=100,
                               note="avoid y ≈ 0"),
    ))

    add(OpEntry(
        name="hypot",
        category="binary",
        variants=[OpVariant(ttnn_fn=ttnn.hypot, golden_fn=torch.hypot)],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    add(OpEntry(
        name="pow",
        category="binary",
        variants=[OpVariant(ttnn_fn=ttnn.pow, golden_fn=torch.pow)],
        input_range=InputRange(lo=0.0, display_lo=0, display_hi=10,
                               note="x >= 0 for real output"),
    ))

    add(OpEntry(
        name="atan2",
        category="binary",
        variants=[OpVariant(ttnn_fn=ttnn.atan2, golden_fn=torch.atan2)],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    add(OpEntry(
        name="rsub",
        category="binary",
        variants=[OpVariant(ttnn_fn=ttnn.rsub, golden_fn=torch.rsub)],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    # -----------------------------------------------------------------------
    # UNARY BACKWARD OPERATIONS
    # -----------------------------------------------------------------------

    add(OpEntry(
        name="abs_bw",
        category="unary_bw",
        variants=[OpVariant(ttnn_fn=bw_impl(ttnn.abs_bw), golden_fn=bw_golden(ttnn.abs_bw))],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="floor_bw",
        category="unary_bw",
        variants=[OpVariant(ttnn_fn=bw_impl(ttnn.floor_bw), golden_fn=bw_golden(ttnn.floor_bw))],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    for bw_name, bw_op, display_lo, display_hi, golden_kwargs in [
        ("exp_bw",   ttnn.exp_bw,   -10,  88,   {}),
        ("exp2_bw",  ttnn.exp2_bw,  -20, 127,   {}),
        ("expm1_bw", ttnn.expm1_bw, -10,  88,   {}),
        ("log_bw",   ttnn.log_bw,   1e-7, 1e7,  {}),
        ("log10_bw", ttnn.log10_bw, 1e-7, 1e7,  {}),
        ("log2_bw",  ttnn.log2_bw,  1e-7, 1e7,  {}),
        ("log1p_bw", ttnn.log1p_bw, -1,   100,  {}),
        ("sqrt_bw",  ttnn.sqrt_bw,  0,    1e6,  {}),
        ("rsqrt_bw", ttnn.rsqrt_bw, 1e-7, 1e6,  {}),
        ("sin_bw",   ttnn.sin_bw,   -math.pi*4, math.pi*4, {}),
        ("cos_bw",   ttnn.cos_bw,   -math.pi*4, math.pi*4, {}),
        ("tan_bw",   ttnn.tan_bw,   -math.pi/2*0.99, math.pi/2*0.99, {}),
        ("asin_bw",  ttnn.asin_bw,  -1,   1,    {}),
        ("acos_bw",  ttnn.acos_bw,  -1,   1,    {}),
        ("atan_bw",  ttnn.atan_bw,  -100, 100,  {}),
        ("sinh_bw",  ttnn.sinh_bw,  -10,  10,   {}),
        ("cosh_bw",  ttnn.cosh_bw,  -10,  10,   {}),
        ("tanh_bw",  ttnn.tanh_bw,  -10,  10,   {}),
        ("asinh_bw", ttnn.asinh_bw, -10,  10,   {}),
        ("atanh_bw", ttnn.atanh_bw, -1,   1,    {}),
        ("tanhshrink_bw", ttnn.tanhshrink_bw, -10, 10, {}),
        ("hardtanh_bw",   ttnn.hardtanh_bw,   -5,  5,  {}),
        ("digamma_bw",    ttnn.digamma_bw,     1e-6, 10, {}),
        ("lgamma_bw",     ttnn.lgamma_bw,      1e-6, 20, {}),
        ("erfinv_bw",     ttnn.erfinv_bw,      -1,   1,  {}),
        ("sigmoid_bw",    ttnn.sigmoid_bw,     -10,  10, {}),
        ("silu_bw",       ttnn.silu_bw,        -10,  10, {}),
        ("gelu_bw",       ttnn.gelu_bw,        -5,   5,  {}),
        ("celu_bw",       ttnn.celu_bw,        -5,   5,  {}),
        ("elu_bw",        ttnn.elu_bw,         -5,   5,  {}),
        ("selu_bw",       ttnn.selu_bw,        -5,   5,  {}),
        ("softplus_bw",   ttnn.softplus_bw,    -5,   5,  {}),
        ("softsign_bw",   ttnn.softsign_bw,    -10,  10, {}),
    ]:
        _op = bw_op
        add(OpEntry(
            name=bw_name,
            category="unary_bw",
            variants=[OpVariant(
                ttnn_fn=bw_impl(_op),
                golden_fn=bw_golden(_op),
            )],
            input_range=InputRange(display_lo=display_lo, display_hi=display_hi),
        ))

    # acosh_bw uses torch autograd as its ttnn golden needs device kwargs
    add(OpEntry(
        name="acosh_bw",
        category="unary_bw",
        variants=[OpVariant(
            ttnn_fn=bw_impl(ttnn.acosh_bw),
            golden_fn=bw_golden_torch(torch.acosh),
        )],
        input_range=InputRange(lo=1.0, display_lo=1, display_hi=100,
                               note="x >= 1"),
    ))

    return entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_REGISTRY: Optional[dict[str, OpEntry]] = None


def get_registry() -> dict[str, OpEntry]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def list_op_names(category: Optional[str] = None) -> list[str]:
    reg = get_registry()
    if category:
        return [k for k, v in reg.items() if v.category == category]
    return list(reg.keys())


def get_op(name: str) -> OpEntry:
    return get_registry()[name]
