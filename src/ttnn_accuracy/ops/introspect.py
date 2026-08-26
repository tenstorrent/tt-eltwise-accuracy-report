"""What ttnn actually exposes, asked of ttnn itself.

`query_registered_operations` is ttnn's own registry, so an op added upstream appears
here without anyone editing a list. Nothing is read from the tt-metal source tree:
that is test layout, not API, and it is absent from a wheel install.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiscoveredOp:
    name: str
    qualified_name: str
    has_golden: bool
    signature: str | None
    required_args: int | None
    is_backward: bool
    is_cpp: bool
    is_experimental: bool
    golden: Callable | None  # live reference, never serialised into the manifest


def _required_args(golden) -> tuple[str | None, int | None]:
    """Signature of the golden and how many operands it takes before defaults."""
    try:
        sig = inspect.signature(golden)
    except (TypeError, ValueError):
        return None, None
    required = sum(
        p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        for p in sig.parameters.values()
    )
    return str(sig), required


def discover(include_experimental: bool = False) -> list[DiscoveredOp]:
    import ttnn

    found = []
    for op in ttnn.decorators.query_registered_operations(include_experimental):
        qualified = op.python_fully_qualified_name
        name = qualified.rsplit(".", 1)[-1]
        signature, required = (
            _required_args(op.golden_function) if op.golden_function else (None, None)
        )
        found.append(
            DiscoveredOp(
                name=name,
                qualified_name=qualified,
                has_golden=op.golden_function is not None,
                signature=signature,
                required_args=required,
                is_backward=name.endswith("_bw"),
                is_cpp=op.is_cpp_operation,
                is_experimental=op.is_experimental,
                golden=op.golden_function,
            )
        )
    return sorted(found, key=lambda o: o.qualified_name)
