"""What to measure, resolved from the manifest or from the legacy registry.

Both sources produce the same OpSpec, so the runner never learns which it got. Bounds
are per dtype because that is what derivation produces: exp overflows at 88.5 in bf16
and 88.67 in fp32, a distinction the registry's single constant could not express.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from math import isnan

from ttnn_accuracy.domain.derive import DTYPES
from ttnn_accuracy.ops import arity, introspect
from ttnn_accuracy.ops.manifest import eltwise, load
from ttnn_accuracy.ops.registry import bw_fn, get_registry, list_op_names, variant_slug


@dataclass(frozen=True, slots=True)
class OpSpec:
    name: str
    variant: str
    category: str
    operands: int
    ttnn_fn: Callable
    golden_fn: Callable
    bounds: dict[str, tuple[float, float]]
    layouts: dict[str, str]  # per dtype, like bounds: an op may want row-major in one only


@dataclass(frozen=True, slots=True)
class OpInfo:
    """What the report needs to describe an op, from whichever source knows it."""

    category: str
    display_name: str
    operands: int
    note: str
    description: str
    bounds: dict[str, tuple[float, float]] | None


UNBOUNDED = dict.fromkeys(DTYPES, (-float("inf"), float("inf")))
TILED = dict.fromkeys(DTYPES, "tile")  # the default until `probe` finds otherwise


@cache
def _manifest() -> dict:
    """Read once: describe() is called per op per page and the file is large."""
    return load()


@cache
def describe(op_key: str) -> OpInfo | None:
    """Metadata for a measured op, from whichever source defines it.

    A registry op keeps the registry's own bounds — that is what its committed data was
    measured with, and substituting derived bounds here would misdescribe it.
    """
    entry = get_registry().get(op_key)
    if entry:
        return OpInfo(
            category=entry.category,
            display_name=entry.display_name,
            operands=2 if entry.category.startswith("binary") else 1,
            note=entry.input_range.note,
            description=entry.description,
            bounds=dict.fromkeys(DTYPES, (entry.input_range.lo, entry.input_range.hi)),
        )
    manifest = _manifest()
    op = manifest["ops"].get(f"ttnn.{op_key}")
    if not op:
        return None
    bounds = _bounds(manifest["domains"].get(f"ttnn.{op_key}"))
    return OpInfo(
        category=op["category"],
        display_name=op["name"],
        operands=op["operands"],
        note="",
        description="",
        # Two operands are never derived, and the sweep covers everything — matching
        # _from_manifest so the page states the range the data was actually measured over.
        bounds=bounds or (UNBOUNDED if op["operands"] > 1 else None),
    )


def params_desc(op_key: str, variant: str) -> str:
    """The human-readable form of a variant slug. Manifest ops have only `default`."""
    entry = get_registry().get(op_key)
    if not entry:
        return variant
    return next(
        (v.params_desc for v in entry.variants if variant_slug(v.params_desc) == variant),
        variant,
    )


def resolve(
    source: str, names: list[str] | None, category: str | None
) -> tuple[list[OpSpec], list[str]]:
    if source == "manifest":
        return _from_manifest(names, category)
    return _from_registry(names or list_op_names(category=category))


def _from_registry(names: list[str]) -> tuple[list[OpSpec], list[str]]:
    registry = get_registry()
    specs, problems = [], []
    for name in names:
        entry = registry.get(name)
        if entry is None:
            problems.append(f"unknown op: {name}")
            continue
        bounds = dict.fromkeys(DTYPES, (entry.input_range.lo, entry.input_range.hi))
        specs += [
            OpSpec(
                name=entry.name,
                variant=variant_slug(v.params_desc),
                category=entry.category,
                operands=2 if entry.category.startswith("binary") else 1,
                ttnn_fn=v.ttnn_fn,
                golden_fn=v.golden_fn,
                bounds=bounds,
                layouts=TILED,  # the registry predates layout discovery; its ops are all tiled
            )
            for v in entry.variants
        ]
    return specs, problems


def _from_manifest(names: list[str] | None, category: str | None) -> tuple[list[OpSpec], list[str]]:
    """Every eltwise arity the sweeps can build operands for."""
    specs, problems = [], []
    wanted = set(names or [])
    manifest = _manifest()
    considered = set()

    for qualified, entry in eltwise(manifest).items():
        operands = entry["operands"]
        if wanted and entry["name"] not in wanted:
            continue
        if category and entry["category"] != category:
            continue

        considered.add(entry["name"])
        bounds = _bounds(manifest["domains"].get(qualified))
        if bounds is None and operands == 1:  # only a single axis is ever derived
            refused = manifest["refused"].get(qualified)
            problems.append(
                f"{qualified}: {refused}"
                if refused
                else f"{qualified}: no derived domain — run `ttnn-accuracy derive`"
            )
            continue
        # Derivation is a single-axis bisection, so it never runs for two operands.
        # Classification marks the invalid pairs, so the full range is safe to sweep.
        bounds = bounds or UNBOUNDED
        try:
            op = introspect.resolve(qualified)
        except AttributeError:
            problems.append(f"{qualified}: not reachable on the ttnn module")
            continue

        accepts = manifest["layouts"].get(qualified)
        if accepts == {}:  # probed, runs in no dtype; None means never probed
            continue

        backward = entry["category"].endswith("_bw")
        specs.append(
            OpSpec(
                name=entry["name"],
                variant="default",
                category=entry["category"],
                operands=operands,
                ttnn_fn=bw_fn(op) if backward else op,
                golden_fn=_golden(op.golden_function, backward, operands),
                bounds=bounds,
                layouts={**TILED, **(accepts or {})},
            )
        )

    problems += [_why_missing(manifest, n) for n in sorted(wanted - considered)]
    return specs, problems


def _why_missing(manifest: dict, name: str) -> str:
    """Say which of the four reasons applies, rather than calling everything unknown."""
    qualified = f"ttnn.{name}"
    op = manifest["ops"].get(qualified)
    if op is None:
        return f"unknown op: {name}"
    if op["elementwise"] is None:
        return f"{qualified}: golden refused the probe, so it was never classified"
    if not op["elementwise"]:
        return f"{qualified}: not elementwise — output depends on more than its own input"
    if not op["real_valued"]:
        return f"{qualified}: golden is not real-valued — ULP needs a real value to measure"
    return f"{qualified}: excluded by the category filter"


def _golden(golden: Callable, backward: bool, operands: int) -> Callable:
    """Adapt a ttnn golden to the shape the sweeps call: (x, out=None), or one per operand."""
    if operands == 1:
        return lambda x, out=None: arity.call_golden(golden, [x], backward)
    return lambda *args: arity.call_golden(golden, list(args), backward)


def _bounds(domains: dict | None) -> dict[str, tuple[float, float]] | None:
    if not domains:
        return None
    bounds = {}
    for dtype in DTYPES:
        d = domains.get(dtype)
        if not d or isnan(d["lo"]) or isnan(d["hi"]):
            return None
        bounds[dtype] = (d["lo"], d["hi"])
    return bounds
