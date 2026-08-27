"""What ttnn actually exposes, asked of ttnn itself.

`query_registered_operations` is ttnn's own registry, so an op added upstream appears
here without anyone editing a list. Nothing is read from the tt-metal source tree:
that is test layout, not API, and it is absent from a wheel install.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass


def resolve(qualified_name: str):
    """The live ttnn callable behind a manifest key, e.g. `ttnn.experimental.plus_one`."""
    import ttnn

    obj = ttnn
    for part in qualified_name.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


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


POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _signature(golden) -> tuple[str | None, int | None]:
    """The golden's signature, and how many operands it takes.

    Only positional parameters count. Several ttnn goldens declare a keyword-only
    `device` with no default — `asin` is `(input_tensor_a, *args, device, **kwargs)` —
    and counting that as an operand made unary ops look binary.
    """
    try:
        sig = inspect.signature(golden)
    except (TypeError, ValueError):
        return None, None
    operands = sum(
        p.default is inspect.Parameter.empty and p.kind in POSITIONAL
        for p in sig.parameters.values()
    )
    return str(sig), operands


def discover(include_experimental: bool = False) -> list[DiscoveredOp]:
    import ttnn

    found = []
    for op in ttnn.decorators.query_registered_operations(include_experimental):
        qualified = op.python_fully_qualified_name
        name = qualified.rsplit(".", 1)[-1]
        signature, required = _signature(op.golden_function) if op.golden_function else (None, None)
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
