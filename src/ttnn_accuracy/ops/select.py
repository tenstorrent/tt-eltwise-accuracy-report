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
    "activations": ("celu", "elu", "hardsigmoid", "hardswish", "selu", "silu", "swish"),
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


def _includers(tree: Path, header: str) -> set[str]:
    """Stems of SFPU kernels that include this header — a helper names no op of its own."""
    name = Path(header).name
    found = set()
    for kernels in tree.glob("tt_metal/hw/ckernels/*/**/llk_sfpu"):
        for path in kernels.glob("ckernel_sfpu_*.h"):
            if f'#include "{name}"' in path.read_text(errors="ignore"):
                found.add(path.stem.removeprefix("ckernel_sfpu_"))
    return found


def select(paths: list[str], manifest: dict | None = None, tree: Path | None = None) -> Selection:
    """Ops and arches worth measuring for these changed paths. Empty is a valid answer."""
    manifest = manifest if manifest is not None else load()
    known = _known(manifest)
    picked = Selection()
    arches: set[str] = set()
    categories: set[str] = set()

    def take(stem: str, path: str) -> bool:
        names = _from_stem(stem, known)
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


def select_paths(paths: list[str], tree: Path | None = None) -> int:
    """The selection as JSON on stdout — data a workflow pipes into `jq`, not a log line, so
    it goes to the stream and not to the logger. Nothing matched is still success."""
    sys.stdout.write(json.dumps(asdict(select(paths, tree=tree)), indent=2, sort_keys=True) + "\n")
    return 0
