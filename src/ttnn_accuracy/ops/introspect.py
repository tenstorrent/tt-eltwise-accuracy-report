"""ttnn's own registry, so an op added upstream appears without anyone editing a list."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from ttnn_accuracy.ops.overrides import OVERRIDES


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
    """The golden's signature and its positional count; a keyword-only `device` is not an operand."""
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
        golden = op.golden_function
        # Bound before the signature is read, so a bound scalar stops counting as an operand.
        if variants := OVERRIDES.get(qualified):
            ov = variants[0]
            if ov.golden or golden is not None:
                golden = ov.golden or partial(golden, **ov.golden_kwargs)
        signature, required = _signature(golden) if golden else (None, None)
        found.append(
            DiscoveredOp(
                name=name,
                qualified_name=qualified,
                has_golden=golden is not None,
                signature=signature,
                required_args=required,
                is_backward=name.endswith("_bw"),
                is_cpp=op.is_cpp_operation,
                is_experimental=op.is_experimental,
                golden=golden,
            )
        )
    return sorted(found, key=lambda o: o.qualified_name)
