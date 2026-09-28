# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Operand count and elementwise-ness asked of the golden; a backward one eats the gradient first."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from ttnn_accuracy.ops.introspect import DiscoveredOp

CATEGORY = {1: "unary", 2: "binary", 3: "ternary"}
PROBE_SIZE = 64

# Six goldens want a `device` to substitute SFPU sentinels; the reference must stay IEEE.
_IEEE_DEVICE = SimpleNamespace(sfpu_nan=lambda: float("nan"), sfpu_inf=lambda: float("inf"))


def _extra_kwargs(golden) -> dict:
    try:
        params = inspect.signature(golden).parameters.values()
    except (TypeError, ValueError):  # torch builtins used directly have no signature
        return {}
    needs = any(
        p.kind is p.KEYWORD_ONLY and p.default is p.empty and p.name == "device" for p in params
    )
    return {"device": _IEEE_DEVICE} if needs else {}


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
    kwargs = _extra_kwargs(golden)
    if backward:
        # A backward golden differentiates its operands, so they need to carry grad.
        with torch.enable_grad():
            tracked = [a.detach().requires_grad_(True) for a in args]
            out = golden(torch.ones_like(tracked[0]), *tracked, **kwargs)
    else:
        out = golden(*args, **kwargs)
    out = out[0] if isinstance(out, (list, tuple)) else out
    if isinstance(out, torch.Tensor) and out.dtype == torch.bool:
        # A predicate answers 0 or 1 exactly, so a float needs no second scoring path.
        out = out.to(args[0].dtype)
    return out.detach() if isinstance(out, torch.Tensor) else out


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(((a == b) | (a.isnan() & b.isnan())).all())


@dataclass(frozen=True, slots=True)
class Probe:
    """What a handful of golden calls reveal about an op."""

    elementwise: bool
    real_valued: bool
    depends_on_input: bool


def probe(op: DiscoveredOp) -> tuple[Probe | None, str]:
    """Probe the golden for elementwise-ness, a real output and input dependence; say why it refused."""
    n = operands(op)
    if n is None:
        return None, "takes no tensor operands"
    # (0.1, 0.9) keeps log, asin and logit in domain; seeded so the manifest reproduces.
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
            # +10 clears relu6, negation catches `sign`, and zero/inf/NaN catch predicates.
            for f in (
                lambda a: a + 10.0,
                lambda a: -a,
                lambda a: a * 0.0,
                lambda a: a * float("inf"),
                lambda a: a * float("-inf"),  # the base is positive, so +inf never makes one
                lambda a: a * float("nan"),
            )
        ]
    except Exception as exc:
        msg = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        return None, f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    if not isinstance(before, torch.Tensor) or before.shape != base[0].shape:
        return Probe(elementwise=False, real_valued=False, depends_on_input=False), ""
    return Probe(
        elementwise=_same(before[1:], after[1:]),
        real_valued=before.is_floating_point(),
        depends_on_input=not all(_same(before, other) for other in elsewhere),
    ), ""
