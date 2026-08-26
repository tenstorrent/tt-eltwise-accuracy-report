"""Operand count and elementwise-ness, decided by asking the golden rather than by name.

A backward golden takes the incoming gradient first, so its measurable operand count is
one lower than its signature suggests: `exp_bw(grad, x)` is a unary op under test.
"""

from __future__ import annotations

import torch

from ttnn_accuracy.ops.introspect import DiscoveredOp

CATEGORY = {1: "unary", 2: "binary", 3: "ternary"}
PROBE_SIZE = 64


def operands(op: DiscoveredOp) -> int | None:
    if op.required_args is None:
        return None
    n = op.required_args - 1 if op.is_backward else op.required_args
    return n if n > 0 else None


def category(op: DiscoveredOp) -> str | None:
    """`unary`/`binary`/`ternary`, suffixed `_bw` for backward. None when unmeasurable."""
    name = CATEGORY.get(operands(op))
    if name is None:
        return None
    return f"{name}_bw" if op.is_backward else name


def call_golden(golden, args: list[torch.Tensor], backward: bool) -> torch.Tensor:
    if backward:
        # A backward golden differentiates its operands, so they need to carry grad.
        with torch.enable_grad():
            tracked = [a.detach().requires_grad_(True) for a in args]
            out = golden(torch.ones_like(tracked[0]), *tracked)
    else:
        out = golden(*args)
    out = out[0] if isinstance(out, (list, tuple)) else out
    return out.detach() if isinstance(out, torch.Tensor) else out


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(((a == b) | (a.isnan() & b.isnan())).all())


def is_elementwise(op: DiscoveredOp) -> bool | None:
    """Does output[i] depend only on input[i]? None when the golden refuses the probe.

    Perturbing the first element must leave every other output untouched. A permutation
    test would let softmax through — it is permutation-equivariant without being
    elementwise — whereas this excludes it, along with matmul and every reduction.
    """
    n = operands(op)
    if n is None:
        return None
    # (0.1, 0.9) keeps log, sqrt, asin, atanh and logit inside their domains. Seeded so the
    # manifest is reproducible: an unchanged ttnn must produce an unchanged file.
    rng = torch.Generator().manual_seed(0)
    base = [
        torch.rand(PROBE_SIZE, dtype=torch.float64, generator=rng) * 0.8 + 0.1 for _ in range(n)
    ]
    bumped = [a.clone() for a in base]
    for a in bumped:
        a[0] *= 0.5
    try:
        before = call_golden(op.golden, base, op.is_backward)
        after = call_golden(op.golden, bumped, op.is_backward)
    except Exception:
        return None
    if not isinstance(before, torch.Tensor) or before.shape != base[0].shape:
        return False
    return _same(before[1:], after[1:])
