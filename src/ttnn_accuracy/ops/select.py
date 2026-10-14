# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Changed tt-metal paths to the ops and architectures a run should measure."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ttnn_accuracy.config import SELECT_CAP
from ttnn_accuracy.ops.manifest import load

# How tt-metal spells each architecture in a path. `common` serves both; quasar is real,
# but no runner here measures it, so a quasar-only change selects nothing.
ARCHES_OF_DIR = {"wormhole_b0": ("wh",), "blackhole": ("bh",), "common": ("wh", "bh")}

# An SFPU kernel, in any of the three trees that hold them.
KERNEL = re.compile(
    r"tt_metal/(?:hw/ckernels/(?P<dir>\w+)/(?:metal/llk_api/)?llk_sfpu"
    r"|tt-llk/tt_llk_(?P<llk>\w+)/common/inc/sfpu)/ckernel_sfpu_(?P<stem>\w+)\.h$"
)
# Anything else under those trees is a header every kernel may include, so it takes the
# architecture whole. This is the blind spot that let a ckernel_defs.h change ship unmeasured.
SHARED = re.compile(r"tt_metal/(?:hw/ckernels/(?P<dir>\w+)/|tt-llk/tt_llk_(?P<llk>\w+)/)")
# The compute-kernel API a ttnn op calls. Architecture-independent.
COMPUTE_API = re.compile(r"tt_metal/hw/inc/api/compute/(?:eltwise_\w+/)?(?P<stem>\w+)\.h$")
# A ttnn eltwise family. Touching one selects its category, never a single op.
FAMILY = re.compile(r"ttnn/cpp/ttnn/operations/eltwise/(?P<family>\w+)/")

# A kernel written for one dtype or one algorithm still implements its base op.
VARIANT = re.compile(r"_(?:bf16|fp32|int32|custom|appx|compat|derivative)$")

FAMILY_CATEGORY = {
    "unary": "unary",
    "unary_ng": "unary",
    "unary_backward": "unary_bw",
    "binary": "binary",
    "binary_ng": "binary",
    "binary_backward": "binary_bw",
    "ternary": "ternary",
    "ternary_backward": "ternary_bw",
}

# One header, several ops, where the stem names none of them.
ALIASES = {
    "isinf_isnan": ("isinf", "isnan", "isfinite", "isneginf", "isposinf"),
    "trigonometry": ("sin", "cos", "tan", "asin", "acos", "atan"),
    "activations": ("celu", "hardshrink", "hardsigmoid", "softshrink", "softsign"),
    "comp": ("eq", "ne", "gt", "ge", "lt", "le"),
    "binary_bitwise": ("bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not"),
}


@dataclass(slots=True)
class Selection:
    """What to measure and the path that asked for it — a reason per op, never a bare list."""

    ops: list[str] = field(default_factory=list)
    arches: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    capped: bool = False


def _known(manifest: dict) -> dict[str, dict]:
    """Op name to its manifest entry, unqualified — a changed path never carries `ttnn.`."""
    return {e["name"]: e for e in manifest["ops"].values()}


def _supported(manifest: dict, name: str, arch: str) -> bool:
    return f"ttnn.{name}" in manifest.get("layouts", {}).get(arch, {})


def _from_stem(stem: str, known: dict[str, dict]) -> tuple[str, ...]:
    """The ops a kernel stem implements: an alias set, the op itself, or its base op."""
    if stem in ALIASES:
        return tuple(n for n in ALIASES[stem] if n in known)
    if found := tuple(n for n in (stem, f"{stem}_bw") if n in known):
        return found
    base = VARIANT.sub("", stem)
    return tuple(n for n in (base, f"{base}_bw") if n in known) if base != stem else ()


def _named_in(patch: str, names: tuple[str, ...]) -> tuple[str, ...]:
    """The ops a diff's own changed lines name. Underscores are part of a name here, so
    `calculate_softsign` names softsign while `celu` does not name `elu`."""
    changed = "\n".join(
        line
        for line in patch.splitlines()
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    )
    return tuple(
        n for n in names if re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", changed)
    )


def _from_ttnn_kernel(path: str, category: str, known: dict[str, dict]) -> tuple[str, ...]:
    """The op a ttnn kernel computes, named between its family prefix and any algorithm
    suffix: `eltwise_bw_gelu_tanh.cpp` is gelu_bw written with tanh, not tanh_bw. The diff
    cannot say this — it is full of the primitives the derivative is built from."""
    stem = re.sub(r"^eltwise_(?:bw_)?", "", Path(path).stem)
    while stem:
        for name in (stem, f"{stem}_bw"):
            if known.get(name, {}).get("category") == category:
                return (name,)
        if "_" not in stem:
            return ()
        stem = stem.rsplit("_", 1)[0]
    return ()


def _includers(tree: Path, header: str) -> set[str]:
    """Stems of SFPU kernels that include this header — a helper names no op of its own."""
    name = Path(header).name
    found = set()
    for kernels in tree.glob("tt_metal/hw/ckernels/*/**/llk_sfpu"):
        for path in kernels.glob("ckernel_sfpu_*.h"):
            if f'#include "{name}"' in path.read_text(errors="ignore"):
                found.add(path.stem.removeprefix("ckernel_sfpu_"))
    return found


def select(
    paths: list[str],
    manifest: dict | None = None,
    tree: Path | None = None,
    hunks: dict[str, str] | None = None,
) -> Selection:
    """Ops and arches worth measuring for these changed paths. Empty is a valid answer."""
    manifest = manifest if manifest is not None else load()
    known = _known(manifest)
    picked = Selection()
    arches: set[str] = set()
    categories: set[str] = set()

    def take(stem: str, path: str) -> bool:
        names = _from_stem(stem, known)
        # An alias header declares several ops and a diff usually touches one, so take the
        # ops its own lines name — drawn from every op, since a header may declare one the
        # alias predates. Naming none is not evidence of none: a shared macro moves them all
        # and mentions no op, and that case keeps the whole set.
        if stem in ALIASES and (named := _named_in((hunks or {}).get(path, ""), tuple(known))):
            names = named
        for name in names:
            picked.reasons.setdefault(name, path)
        return bool(names)

    for path in paths:
        if found := KERNEL.search(path):
            for arch in ARCHES_OF_DIR.get(found["dir"] or found["llk"], ()):
                arches.add(arch)
            if not ARCHES_OF_DIR.get(found["dir"] or found["llk"]):
                continue
            if take(found["stem"], path):
                continue
            # A helper naming no op. Resolve it through whoever includes it, or take the
            # architecture whole rather than let an unmeasured kernel through.
            resolved = any(take(stem, path) for stem in _includers(tree, path)) if tree else False
            if not resolved:
                picked.unresolved.append(path)
                categories.update(FAMILY_CATEGORY.values())
        elif found := SHARED.search(path):
            hit = ARCHES_OF_DIR.get(found["dir"] or found["llk"], ())
            arches.update(hit)
            if hit:
                categories.update(FAMILY_CATEGORY.values())
        elif found := COMPUTE_API.search(path):
            arches.update(("wh", "bh"))
            take(found["stem"], path)
        elif found := FAMILY.search(path):
            if category := FAMILY_CATEGORY.get(found["family"]):
                arches.update(("wh", "bh"))
                # A kernel names its op; anything else in the family names only its scope.
                named = _from_ttnn_kernel(path, category, known)
                for name in named:
                    picked.reasons.setdefault(name, path)
                if not named:
                    categories.add(category)

    # An op no selected architecture supports is not worth a dispatch.
    picked.ops = sorted(
        n for n in picked.reasons if any(_supported(manifest, n, a) for a in arches)
    )
    picked.reasons = {n: picked.reasons[n] for n in picked.ops}
    picked.arches = sorted(arches)
    picked.categories = sorted(categories)

    # Both are reported. `ops` is the narrowest thing that covers the change and is what a
    # run should measure; `categories` is the wider scope a shared header or a ttnn family
    # implies, for a human deciding whether the narrow run was enough.
    if len(picked.ops) > SELECT_CAP:
        picked.capped = True
        picked.categories = sorted(
            set(picked.categories) | {known[n]["category"] for n in picked.ops}
        )
        picked.ops = []
    return picked


def select_paths(
    paths: list[str], tree: Path | None = None, hunks: dict[str, str] | None = None
) -> int:
    """The selection as JSON on stdout — data a workflow pipes into `jq`, not a log line, so
    it goes to the stream and not to the logger. Nothing matched is still success."""
    selection = select(paths, tree=tree, hunks=hunks)
    sys.stdout.write(json.dumps(asdict(selection), indent=2, sort_keys=True) + "\n")
    return 0
