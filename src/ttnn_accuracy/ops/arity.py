"""Operand count and elementwise-ness, decided by asking the golden rather than by name.

A backward golden takes the incoming gradient first, so its measurable operand count is
one lower than its signature suggests: `exp_bw(grad, x)` is a unary op under test.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class Probe:
    """What a handful of golden calls reveal about an op."""

    elementwise: bool
    real_valued: bool
    depends_on_input: bool


def probe(op: DiscoveredOp) -> Probe | None:
    """Probe the golden. None when it refuses to be called.

    `elementwise`: does output[i] depend only on input[i]? Perturbing the first element
    must leave every other output untouched. A permutation test would let softmax
    through — it is permutation-equivariant without being elementwise — whereas this
    excludes it, along with matmul and every reduction.

    `real_valued`: a predicate like `isnan` returns bool, and the distance between two
    booleans is not measured in ULP. Asked of the golden's own output dtype rather than
    listed by name, so a predicate added to ttnn later is excluded without a code change.
    Catching it here keeps those ops from failing at measurement with a torch error about
    `abs_cpu`, which describes the symptom rather than the cause.

    `depends_on_input`: `zeros_like` is elementwise and real-valued and returns the same
    thing whatever it is given, so measuring its accuracy yields a flawless score for an
    op that computes nothing. Two further inputs settle it, because one cannot: `sign` is
    flat across (0.1, 0.9) and (2, 3) alike, so only flipping the sign as well
    distinguishes an op that ignores its input from one that is merely flat here.
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
        elsewhere = [
            call_golden(op.golden, [f(a) for a in base], op.is_backward)
            # +10 clears the usual saturation points — hardtanh at 1, hardsigmoid at 3,
            # relu6 at 6 — which a smaller shift leaves inside, making a piecewise op
            # look constant. Negation is the second probe because `sign` is flat across
            # any two positive ranges.
            for f in (lambda a: a + 10.0, lambda a: -a)
        ]
    except Exception:
        return None
    if not isinstance(before, torch.Tensor) or before.shape != base[0].shape:
        return Probe(elementwise=False, real_valued=False, depends_on_input=False)
    return Probe(
        elementwise=_same(before[1:], after[1:]),
        real_valued=before.is_floating_point(),
        depends_on_input=not all(_same(before, other) for other in elsewhere),
    )
