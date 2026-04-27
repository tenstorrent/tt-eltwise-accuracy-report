"""
Op registry: defines all eltwise ops with variants, valid input ranges, and golden functions.
This is the single source of truth for what gets measured and reported.

Calling conventions:
  unary/unary_bw  ttnn_fn: callable(x: ttnn.Tensor) -> ttnn.Tensor
  binary          ttnn_fn: callable(x: ttnn.Tensor, y: ttnn.Tensor) -> ttnn.Tensor
  golden_fn:      callable(x: torch.Tensor, out=None) -> torch.Tensor
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

    lo/hi: domain bounds (float('-inf')/float('inf') for unbounded).
    display_lo/display_hi: narrower x-axis range for plots.
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

    ttnn_fn: unary/unary_bw → callable(x) → ttnn.Tensor
             binary         → callable(x, y) → ttnn.Tensor
    golden_fn: callable(x: torch.Tensor, out=None) → torch.Tensor
    params_desc: human-readable param string, e.g. "alpha=1.0" or "default"
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

    ttnn/torch are imported lazily so the module can be imported without a
    live TT device (e.g., in generate_reports.py / generate_charts.py).
    """
    import torch
    import ttnn

    entries: dict[str, OpEntry] = {}

    def add(entry: OpEntry):
        entries[entry.name] = entry

    # ------------------------------------------------------------------
    # Backward op helpers
    # ------------------------------------------------------------------

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
        """Backward golden via torch autograd (for bw ops whose ttnn golden needs device kwargs)."""
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

    # -----------------------------------------------------------------------
    # UNARY OPERATIONS
    # -----------------------------------------------------------------------

    add(OpEntry(
        name="abs", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.abs(x), golden_fn=torch.abs)],
        input_range=InputRange(display_lo=-200, display_hi=200),
    ))

    add(OpEntry(
        name="identity", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.identity(x),
                            golden_fn=lambda x, out=None: x.clone())],
        input_range=InputRange(display_lo=-200, display_hi=200),
    ))

    add(OpEntry(
        name="exp", category="unary",
        variants=[
            OpVariant(ttnn_fn=lambda x: ttnn.exp(x),
                      golden_fn=torch.exp, params_desc="default"),
            OpVariant(ttnn_fn=lambda x: ttnn.exp(x, fast_and_approximate_mode=True),
                      golden_fn=torch.exp, params_desc="fast_approx"),
        ],
        input_range=InputRange(hi=88.0, display_lo=-10, display_hi=88,
                               note="x > ~88 overflows fp32"),
    ))

    add(OpEntry(
        name="exp2", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.exp2(x), golden_fn=torch.exp2)],
        input_range=InputRange(display_lo=-20, display_hi=127),
    ))

    add(OpEntry(
        name="expm1", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.expm1(x), golden_fn=torch.expm1)],
        input_range=InputRange(display_lo=-10, display_hi=88),
    ))

    add(OpEntry(
        name="log", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.log(x), golden_fn=torch.log)],
        input_range=InputRange(lo=1.18e-38, display_lo=1e-7, display_hi=1e7, note="x > 0"),
    ))

    add(OpEntry(
        name="log2", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.log2(x), golden_fn=torch.log2)],
        input_range=InputRange(lo=1.18e-38, display_lo=1e-7, display_hi=1e7, note="x > 0"),
    ))

    add(OpEntry(
        name="log10", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.log10(x), golden_fn=torch.log10)],
        input_range=InputRange(lo=1.18e-38, display_lo=1e-7, display_hi=1e7, note="x > 0"),
    ))

    add(OpEntry(
        name="log1p", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.log1p(x), golden_fn=torch.log1p)],
        input_range=InputRange(lo=-1.0 + 1e-7, display_lo=-1.0, display_hi=100, note="x > -1"),
    ))

    add(OpEntry(
        name="sqrt", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.sqrt(x), golden_fn=torch.sqrt)],
        input_range=InputRange(lo=0.0, display_lo=0, display_hi=1e6, note="x ≥ 0"),
    ))

    add(OpEntry(
        name="rsqrt", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.rsqrt(x), golden_fn=torch.rsqrt)],
        input_range=InputRange(lo=1.18e-38, display_lo=1e-7, display_hi=1e6, note="x > 0"),
    ))

    add(OpEntry(
        name="cbrt", category="unary",
        variants=[OpVariant(
            ttnn_fn=lambda x: ttnn.cbrt(x),
            golden_fn=lambda x, out=None: torch.pow(x.to(torch.float64), 1/3).to(x.dtype),
        )],
        input_range=InputRange(display_lo=-1e6, display_hi=1e6),
    ))

    add(OpEntry(
        name="reciprocal", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.reciprocal(x),
                            golden_fn=lambda x, out=None: torch.reciprocal(x))],
        input_range=InputRange(display_lo=-100, display_hi=100, note="avoid x ≈ 0"),
    ))

    add(OpEntry(
        name="tanh", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.tanh(x), golden_fn=torch.tanh)],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="sinh", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.sinh(x), golden_fn=torch.sinh)],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="cosh", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.cosh(x), golden_fn=torch.cosh)],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="sin", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.sin(x), golden_fn=torch.sin)],
        # BF16 has only 7 mantissa bits; argument reduction loses all precision for |x| >> 2π.
        # ULP_bf16(x) >= 1 for |x| >= 128, making sin(x) effectively random beyond ~±128.
        input_range=InputRange(display_lo=-128, display_hi=128,
                               note="BF16 argument reduction loses precision for |x| > ~128"),
    ))

    add(OpEntry(
        name="cos", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.cos(x), golden_fn=torch.cos)],
        input_range=InputRange(display_lo=-128, display_hi=128,
                               note="BF16 argument reduction loses precision for |x| > ~128"),
    ))

    add(OpEntry(
        name="tan", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.tan(x), golden_fn=torch.tan)],
        input_range=InputRange(display_lo=-math.pi / 2 * 0.99, display_hi=math.pi / 2 * 0.99,
                               note="avoid poles at (n+0.5)π; BF16 precision lost for |x| > ~128"),
    ))

    add(OpEntry(
        name="asin", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.asin(x), golden_fn=torch.asin)],
        input_range=InputRange(lo=-1.0, hi=1.0, display_lo=-1, display_hi=1, note="x ∈ [-1, 1]"),
    ))

    add(OpEntry(
        name="acos", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.acos(x), golden_fn=torch.acos)],
        input_range=InputRange(lo=-1.0, hi=1.0, display_lo=-1, display_hi=1, note="x ∈ [-1, 1]"),
    ))

    add(OpEntry(
        name="atan", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.atan(x), golden_fn=torch.atan)],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    add(OpEntry(
        name="erf", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.erf(x),
                            golden_fn=lambda x, out=None: torch.special.erf(x))],
        input_range=InputRange(display_lo=-5, display_hi=5),
    ))

    add(OpEntry(
        name="erfinv", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.erfinv(x),
                            golden_fn=lambda x, out=None: torch.special.erfinv(x))],
        input_range=InputRange(lo=-1.0 + 1e-6, hi=1.0 - 1e-6,
                               display_lo=-1, display_hi=1, note="x ∈ (-1, 1)"),
    ))

    add(OpEntry(
        name="sigmoid", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.sigmoid(x), golden_fn=torch.sigmoid)],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="log_sigmoid", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.log_sigmoid(x),
                            golden_fn=lambda x, out=None: torch.nn.functional.logsigmoid(x))],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="silu", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.silu(x),
                            golden_fn=lambda x, out=None: torch.nn.functional.silu(x))],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="gelu", category="unary",
        variants=[
            OpVariant(ttnn_fn=lambda x: ttnn.gelu(x),
                      golden_fn=lambda x, out=None: torch.nn.functional.gelu(x),
                      params_desc="default"),
            OpVariant(ttnn_fn=lambda x: ttnn.gelu(x, fast_and_approximate_mode=True),
                      golden_fn=lambda x, out=None: torch.nn.functional.gelu(x),
                      params_desc="fast_approx"),
        ],
        input_range=InputRange(display_lo=-5, display_hi=5),
    ))

    add(OpEntry(
        name="mish", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.mish(x),
                            golden_fn=lambda x, out=None: torch.nn.functional.mish(x))],
        input_range=InputRange(display_lo=-5, display_hi=5),
    ))

    add(OpEntry(
        name="hardmish", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.hardmish(x),
                            golden_fn=lambda x, out=None: torch.nn.functional.hardswish(x))],
        input_range=InputRange(display_lo=-5, display_hi=5),
        description="uses hardswish as golden (hardmish ≈ hardswish)",
    ))

    add(OpEntry(
        name="relu", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.relu(x),
                            golden_fn=lambda x, out=None: torch.nn.functional.relu(x))],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    for upper in [0.5, 1.0, 2.0]:
        _u = upper
        add(OpEntry(
            name=f"relu_max_ul{upper}".replace(".", "p"),
            category="unary", display_name="relu_max",
            variants=[OpVariant(
                ttnn_fn=lambda x, u=_u: ttnn.relu_max(x, upper_limit=u),
                golden_fn=lambda x, out=None, u=_u: torch.minimum(
                    torch.nn.functional.relu(x), torch.full_like(x, u)),
                params_desc=f"upper_limit={upper}",
            )],
            input_range=InputRange(display_lo=-5, display_hi=5),
            description=f"relu_max upper_limit={upper}",
        ))

    for lower in [0.5, 1.0, 2.0]:
        _l = lower
        add(OpEntry(
            name=f"relu_min_ll{lower}".replace(".", "p"),
            category="unary", display_name="relu_min",
            variants=[OpVariant(
                ttnn_fn=lambda x, l=_l: ttnn.relu_min(x, lower_limit=l),
                golden_fn=lambda x, out=None, l=_l: torch.maximum(
                    torch.nn.functional.relu(x), torch.full_like(x, l)),
                params_desc=f"lower_limit={lower}",
            )],
            input_range=InputRange(display_lo=-5, display_hi=5),
            description=f"relu_min lower_limit={lower}",
        ))

    add(OpEntry(
        name="hardtanh", category="unary",
        variants=[
            OpVariant(ttnn_fn=lambda x: ttnn.hardtanh(x),
                      golden_fn=lambda x, out=None: torch.nn.functional.hardtanh(x, -1, 1),
                      params_desc="min=-1,max=1"),
            OpVariant(ttnn_fn=lambda x: ttnn.hardtanh(x, min_val=-2, max_val=2),
                      golden_fn=lambda x, out=None: torch.nn.functional.hardtanh(x, -2, 2),
                      params_desc="min=-2,max=2"),
        ],
        input_range=InputRange(display_lo=-5, display_hi=5),
    ))

    for alpha in [0.5, 1.0, 2.0]:
        _a = alpha
        add(OpEntry(
            name=f"elu_alpha{alpha}".replace(".", "p"),
            category="unary", display_name="elu",
            variants=[OpVariant(
                ttnn_fn=lambda x, a=_a: ttnn.elu(x, alpha=a),
                golden_fn=lambda x, out=None, a=_a: torch.nn.functional.elu(x, alpha=a),
                params_desc=f"alpha={alpha}",
            )],
            input_range=InputRange(display_lo=-5, display_hi=5),
            description=f"ELU alpha={alpha}",
        ))

    for alpha in [0.5, 1.0, 2.0]:
        _a = alpha
        add(OpEntry(
            name=f"celu_alpha{alpha}".replace(".", "p"),
            category="unary", display_name="celu",
            variants=[OpVariant(
                ttnn_fn=lambda x, a=_a: ttnn.celu(x, alpha=a),
                golden_fn=lambda x, out=None, a=_a: torch.nn.functional.celu(x, alpha=a),
                params_desc=f"alpha={alpha}",
            )],
            input_range=InputRange(display_lo=-5, display_hi=5),
            description=f"CELU alpha={alpha}",
        ))

    add(OpEntry(
        name="selu", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.selu(x),
                            golden_fn=lambda x, out=None: torch.nn.functional.selu(x))],
        input_range=InputRange(display_lo=-5, display_hi=5),
    ))

    for beta in [0.5, 1.0, 2.0]:
        _b = beta
        add(OpEntry(
            name=f"softplus_beta{beta}".replace(".", "p"),
            category="unary", display_name="softplus",
            variants=[OpVariant(
                ttnn_fn=lambda x, b=_b: ttnn.softplus(x, beta=b),
                golden_fn=lambda x, out=None, b=_b: torch.nn.functional.softplus(x, beta=b),
                params_desc=f"beta={beta}",
            )],
            input_range=InputRange(display_lo=-5, display_hi=5),
            description=f"Softplus beta={beta}",
        ))

    add(OpEntry(
        name="softsign", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.softsign(x),
                            golden_fn=lambda x, out=None: torch.nn.functional.softsign(x))],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="tanhshrink", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.tanhshrink(x),
                            golden_fn=lambda x, out=None: torch.nn.functional.tanhshrink(x))],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="logit", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.logit(x), golden_fn=torch.logit)],
        input_range=InputRange(lo=1e-6, hi=1.0 - 1e-6,
                               display_lo=0, display_hi=1, note="x ∈ (0, 1)"),
    ))

    add(OpEntry(
        name="digamma", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.digamma(x), golden_fn=torch.digamma)],
        input_range=InputRange(lo=1e-6, display_lo=0, display_hi=10,
                               note="x > 0 (poles at non-positive integers)"),
    ))

    add(OpEntry(
        name="lgamma", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.lgamma(x), golden_fn=torch.lgamma)],
        input_range=InputRange(lo=1e-6, display_lo=0, display_hi=20, note="x > 0"),
    ))

    add(OpEntry(
        name="round", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.round(x), golden_fn=torch.round)],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="ceil", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.ceil(x), golden_fn=torch.ceil)],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    add(OpEntry(
        name="floor", category="unary",
        variants=[OpVariant(ttnn_fn=lambda x: ttnn.floor(x), golden_fn=torch.floor)],
        input_range=InputRange(display_lo=-10, display_hi=10),
    ))

    # -----------------------------------------------------------------------
    # BINARY OPERATIONS
    # -----------------------------------------------------------------------

    add(OpEntry(
        name="add", category="binary",
        variants=[OpVariant(ttnn_fn=lambda x, y: ttnn.add(x, y), golden_fn=torch.add)],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    add(OpEntry(
        name="subtract", category="binary",
        variants=[OpVariant(ttnn_fn=lambda x, y: ttnn.subtract(x, y), golden_fn=torch.subtract)],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    add(OpEntry(
        name="multiply", category="binary",
        variants=[OpVariant(ttnn_fn=lambda x, y: ttnn.multiply(x, y), golden_fn=torch.multiply)],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    add(OpEntry(
        name="divide", category="binary",
        variants=[
            OpVariant(ttnn_fn=lambda x, y: ttnn.divide(x, y), golden_fn=torch.div,
                      params_desc="default"),
            OpVariant(ttnn_fn=lambda x, y: ttnn.div(x, y, accurate_mode=True), golden_fn=torch.div,
                      params_desc="accurate"),
        ],
        input_range=InputRange(display_lo=-100, display_hi=100, note="avoid y ≈ 0"),
    ))

    add(OpEntry(
        name="hypot", category="binary",
        variants=[OpVariant(ttnn_fn=lambda x, y: ttnn.hypot(x, y), golden_fn=torch.hypot)],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    add(OpEntry(
        name="pow", category="binary",
        variants=[OpVariant(ttnn_fn=lambda x, y: ttnn.pow(x, y), golden_fn=torch.pow)],
        input_range=InputRange(lo=0.0, display_lo=0, display_hi=10, note="x ≥ 0 for real output"),
    ))

    add(OpEntry(
        name="atan2", category="binary",
        variants=[OpVariant(ttnn_fn=lambda x, y: ttnn.atan2(x, y), golden_fn=torch.atan2)],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    add(OpEntry(
        name="rsub", category="binary",
        variants=[OpVariant(ttnn_fn=lambda x, y: ttnn.rsub(x, y), golden_fn=torch.rsub)],
        input_range=InputRange(display_lo=-100, display_hi=100),
    ))

    # -----------------------------------------------------------------------
    # UNARY BACKWARD OPERATIONS
    # -----------------------------------------------------------------------

    if hasattr(ttnn, 'abs_bw'):
        add(OpEntry(
            name="abs_bw", category="unary_bw",
            variants=[OpVariant(ttnn_fn=bw_fn(ttnn.abs_bw), golden_fn=bw_golden(ttnn.abs_bw))],
            input_range=InputRange(display_lo=-10, display_hi=10),
        ))

    if hasattr(ttnn, 'floor_bw'):
        add(OpEntry(
            name="floor_bw", category="unary_bw",
            variants=[OpVariant(ttnn_fn=bw_fn(ttnn.floor_bw), golden_fn=bw_golden(ttnn.floor_bw))],
            input_range=InputRange(display_lo=-10, display_hi=10),
        ))

    for bw_name, bw_op_name, d_lo, d_hi in [
        ("exp_bw",        "exp_bw",        -10,          88           ),
        ("exp2_bw",       "exp2_bw",        -20,         127          ),
        ("expm1_bw",      "expm1_bw",       -10,          88          ),
        ("log_bw",        "log_bw",          1e-7,        1e7         ),
        ("log10_bw",      "log10_bw",        1e-7,        1e7         ),
        ("log2_bw",       "log2_bw",         1e-7,        1e7         ),
        ("log1p_bw",      "log1p_bw",        -1,          100         ),
        ("sqrt_bw",       "sqrt_bw",         0,           1e6         ),
        ("rsqrt_bw",      "rsqrt_bw",        1e-7,        1e6         ),
        ("sin_bw",        "sin_bw",          -math.pi*4,  math.pi*4   ),
        ("cos_bw",        "cos_bw",          -math.pi*4,  math.pi*4   ),
        ("tan_bw",        "tan_bw",          -math.pi/2*.99, math.pi/2*.99),
        ("asin_bw",       "asin_bw",         -1,           1          ),
        ("acos_bw",       "acos_bw",         -1,           1          ),
        ("atan_bw",       "atan_bw",         -100,         100        ),
        ("sinh_bw",       "sinh_bw",         -10,          10         ),
        ("cosh_bw",       "cosh_bw",         -10,          10         ),
        ("tanh_bw",       "tanh_bw",         -10,          10         ),
        ("asinh_bw",      "asinh_bw",        -10,          10         ),
        ("atanh_bw",      "atanh_bw",        -1,           1          ),
        ("tanhshrink_bw", "tanhshrink_bw",   -10,          10         ),
        ("hardtanh_bw",   "hardtanh_bw",     -5,           5          ),
        ("digamma_bw",    "digamma_bw",       1e-6,        10         ),
        ("lgamma_bw",     "lgamma_bw",        1e-6,        20         ),
        ("erfinv_bw",     "erfinv_bw",        -1,           1         ),
        ("sigmoid_bw",    "sigmoid_bw",       -10,          10        ),
        ("silu_bw",       "silu_bw",          -10,          10        ),
        ("gelu_bw",       "gelu_bw",          -5,           5         ),
        ("celu_bw",       "celu_bw",          -5,           5         ),
        ("elu_bw",        "elu_bw",           -5,           5         ),
        ("selu_bw",       "selu_bw",          -5,           5         ),
        ("softplus_bw",   "softplus_bw",      -5,           5         ),
        ("softsign_bw",   "softsign_bw",      -10,          10        ),
    ]:
        if hasattr(ttnn, bw_op_name):
            _op = getattr(ttnn, bw_op_name)
            add(OpEntry(
                name=bw_name, category="unary_bw",
                variants=[OpVariant(ttnn_fn=bw_fn(_op), golden_fn=bw_golden(_op))],
                input_range=InputRange(display_lo=d_lo, display_hi=d_hi),
            ))

    # acosh_bw: ttnn golden needs device kwargs → use torch autograd instead
    if hasattr(ttnn, 'acosh_bw'):
        add(OpEntry(
            name="acosh_bw", category="unary_bw",
            variants=[OpVariant(ttnn_fn=bw_fn(ttnn.acosh_bw),
                                golden_fn=bw_golden_torch(torch.acosh))],
            input_range=InputRange(lo=1.0, display_lo=1, display_hi=100, note="x ≥ 1"),
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
