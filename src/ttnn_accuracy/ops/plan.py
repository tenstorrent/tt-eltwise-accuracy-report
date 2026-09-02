"""What to measure, resolved from the manifest into one OpSpec per variant.

Bounds are per dtype because that is what derivation produces: exp overflows at 88.5 in
bf16 and 88.67 in fp32.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, partial
from math import isnan

from ttnn_accuracy.domain.derive import DTYPES
from ttnn_accuracy.ops import arity, introspect
from ttnn_accuracy.ops.manifest import load
from ttnn_accuracy.ops.overrides import EXCLUDED, OVERRIDES, bw_fn, variant_slug, with_value


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
    """What the report needs to describe an op, straight from the manifest."""

    category: str
    operands: int
    bounds: dict[str, tuple[float, float]] | None


UNBOUNDED = dict.fromkeys(DTYPES, (-float("inf"), float("inf")))
TILED = dict.fromkeys(DTYPES, "tile")  # the default until `probe` finds otherwise


@cache
def _manifest() -> dict:
    """Read once: describe() is called per op per page and the file is large."""
    return load()


@cache
def describe(op_key: str) -> OpInfo | None:
    """Metadata for a measured op, straight from the manifest."""
    manifest = _manifest()
    op = manifest["ops"].get(f"ttnn.{op_key}")
    if not op:
        return None
    variants = OVERRIDES.get(f"ttnn.{op_key}", ())
    bounds = (variants[0].bounds if variants and variants[0].bounds else None) or _bounds(
        manifest["domains"].get(f"ttnn.{op_key}")
    )
    return OpInfo(
        category=op["category"],
        operands=op["operands"],
        # Two operands are never derived and the sweep covers everything.
        bounds=bounds or (UNBOUNDED if op["operands"] > 1 else None),
    )


def params_desc(op_key: str, variant: str) -> str:
    """The human-readable form of a variant slug, recovered from the overrides table."""
    for ov in OVERRIDES.get(f"ttnn.{op_key}", ()):
        if variant_slug(ov.params_desc) == variant:
            return ov.params_desc
    return variant


def resolve(
    names: list[str] | None,
    category: str | None,
    arch: str = "wh",
    values: dict[str, float] | None = None,
) -> tuple[list[OpSpec], list[str]]:
    """Every eltwise arity the sweeps can build operands for on this architecture."""
    specs, problems = [], []
    wanted = set(names or [])
    manifest = _manifest()
    layouts, rejected = manifest["layouts"].get(arch, {}), manifest["rejected"].get(arch, {})
    considered = set()

    for qualified, entry in manifest["ops"].items():
        operands = entry["operands"]
        if wanted and entry["name"] not in wanted:
            continue
        if category and entry["category"] != category:
            continue

        # Before `considered`: a category sweep skips it, naming it still gets the reason.
        if qualified in rejected:
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
        # Bisection is single-axis, and classification marks the invalid pairs anyway.
        bounds = bounds or UNBOUNDED
        try:
            op = introspect.resolve(qualified)
        except AttributeError:
            problems.append(f"{qualified}: not reachable on the ttnn module")
            continue

        accepts = layouts.get(qualified)
        backward = entry["category"].endswith("_bw")
        variants = OVERRIDES.get(qualified, (None,))
        if values and (value := values.get(entry["name"])) is not None:
            if variants[0] is None:
                problems.append(f"{qualified}: takes no scalar parameter to set")
                continue
            try:
                variants = (with_value(variants[0], value),)
            except ValueError as exc:
                problems.append(f"{qualified}: {exc}")
                continue
        for ov in variants:
            ttnn_fn, golden = bw_fn(op) if backward else op, op.golden_function
            if ov:
                ttnn_fn = partial(ttnn_fn, **ov.ttnn_kwargs)
                golden = ov.golden or partial(golden, **ov.golden_kwargs)
            specs.append(
                OpSpec(
                    name=entry["name"],
                    variant=variant_slug(ov.params_desc) if ov else "default",
                    category=entry["category"],
                    operands=operands,
                    ttnn_fn=ttnn_fn,
                    golden_fn=_golden(golden, backward, operands),
                    bounds=ov.bounds if ov and ov.bounds else bounds,
                    # The probe's record wins: a dtype it rejected must not reach the device.
                    layouts=accepts or TILED,
                )
            )

    problems += [_why_missing(manifest, n, arch) for n in sorted(wanted - considered)]
    return specs, problems


def _why_missing(manifest: dict, name: str, arch: str) -> str:
    """Which reason applies — the exclusion table and the refusals answer before "unknown"."""
    qualified = f"ttnn.{name}"
    if why := EXCLUDED.get(qualified):
        return f"{qualified}: excluded — {why}"
    if why := manifest["unprobeable"].get(qualified):
        return f"{qualified}: golden refused the probe — {why}"
    if why := manifest["rejected"].get(arch, {}).get(qualified):
        return f"{qualified}: {arch} rejected every dtype and layout probed — {why}"
    if qualified in manifest["ops"]:
        return f"{qualified}: excluded by the category filter"
    return f"unknown op: {name} — not eltwise, or ttnn does not register it"


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
