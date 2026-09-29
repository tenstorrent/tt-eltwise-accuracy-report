# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The committed eltwise manifest; a rebuild keeps derivation and probing for unchanged ops."""

from __future__ import annotations

import json
from collections import Counter
from functools import partial

from loguru import logger

from ttnn_accuracy.paths import MANIFEST_FILE

# `derive`, `arity` and `overrides` reach torch and ttnn, which building the manifest needs
# and reading it does not. Imported where they are used so `load` costs a JSON parse.


def build(include_experimental: bool = False) -> dict:
    """Eltwise facts only; `unprobeable` keeps every refusal, which is the coverage queue."""
    from ttnn_accuracy.ops import arity, introspect
    from ttnn_accuracy.ops.overrides import EXCLUDED

    ops, unprobeable = {}, {}
    for op in introspect.discover(include_experimental):
        probe, why = arity.probe(op) if op.has_golden else (None, "")
        if why:
            unprobeable[op.qualified_name] = why
        category = arity.category(op)
        eltwise = (
            probe
            and probe.elementwise
            and probe.real_valued
            and probe.depends_on_input
            and category
        )
        if eltwise and op.qualified_name not in EXCLUDED:
            ops[op.qualified_name] = {
                "name": op.name,
                "category": category,
                "operands": arity.operands(op),
                "signature": op.signature,
                "is_cpp": op.is_cpp,
                "is_experimental": op.is_experimental,
            }
    return _empty() | {"ops": ops, "unprobeable": unprobeable}


FIELDS = ("ops", "unprobeable", "domains", "refused", "layouts", "rejected", "probing")


def _empty() -> dict:
    """A fresh dict per field — `dict.fromkeys` would alias one across all seven."""
    return {field: {} for field in FIELDS}


def load() -> dict:
    """Defaults first, so a manifest written before a key existed still loads."""
    if not MANIFEST_FILE.exists():
        return _empty()
    manifest = _empty() | json.loads(MANIFEST_FILE.read_text())
    # Pre-arch-keyed manifests hold flat per-op dicts, all of them probed on Wormhole.
    for key in ("layouts", "rejected"):
        if any(k.startswith("ttnn.") for k in manifest[key]):
            manifest[key] = {"wh": manifest[key]}
    return manifest


def save(manifest: dict) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


MEASURED = ("name", "category", "operands")


def diff(old: dict, new: dict) -> tuple[list[str], list[str], list[str]]:
    before, after = old["ops"], new["ops"]
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    changed = sorted(
        k
        for k in before.keys() & after.keys()
        if any(before[k].get(f) != after[k].get(f) for f in MEASURED)
    )
    return added, removed, changed


def _single_axis(golden, backward: bool, x):
    from ttnn_accuracy.ops import arity

    return arity.call_golden(golden, [x], backward)


def derive_domains() -> int:
    """Per-dtype bounds for every known eltwise op; a golden that refuses is recorded."""
    from ttnn_accuracy.domain.derive import DTYPES, derive
    from ttnn_accuracy.ops import introspect

    manifest = load()
    discovered = {op.qualified_name: op for op in introspect.discover()}
    # Cleared: a stale domain beside a fresh refusal would leave the op holding both.
    manifest["domains"], manifest["refused"] = {}, {}

    for name, entry in manifest["ops"].items():
        if entry["operands"] != 1:
            continue  # binary and ternary need an operand grid, not a single axis
        op = discovered[name]
        call = partial(_single_axis, op.golden, op.is_backward)
        domains = {}
        for dtype in DTYPES:
            try:
                domains[dtype] = derive(call, dtype)
            except Exception as exc:
                manifest["refused"][name] = f"{type(exc).__name__} on {dtype}: {exc}"
        if domains:
            manifest["domains"][name] = domains

    save(manifest)
    derived, refused = len(manifest["domains"]), len(manifest["refused"])
    logger.success("domains for {} unary ops → {}", derived, MANIFEST_FILE)
    if refused:
        logger.warning("{} goldens refused the fp64 grid", refused)
    return 0 if derived else 1


def probe_layouts(device_id: int = 0) -> int:
    """Probe every op in a child process, naming it first, so one segfault costs one op."""
    import multiprocessing

    context = multiprocessing.get_context("spawn")  # a fresh interpreter, no shared device
    probed = context.SimpleQueue()  # the parent has no device, so the child names the arch
    # Every crash consumes exactly one op, so a pass per op is the bound by construction.
    for attempt in range(len(load()["ops"]) + 1):
        child = context.Process(target=_probe_pass, args=(device_id, probed))
        child.start()
        child.join()
        if child.exitcode == 0:
            return _probe_summary(probed.get())
        logger.error("probe pass {} died with {}, resuming past it", attempt + 1, child.exitcode)
    return 1


def _probe_summary(arch: str) -> int:
    manifest = load()
    layouts, rejected = manifest["layouts"][arch], manifest["rejected"][arch]
    logger.success(
        "{} layouts for {} of {} ops → {}", arch, len(layouts), len(manifest["ops"]), MANIFEST_FILE
    )
    for dtypes, count in Counter(tuple(sorted(v)) for v in layouts.values()).most_common():
        logger.info("{} ops accept {}", count, ", ".join(dtypes))
    for name, why in sorted(rejected.items()):
        logger.info("{} accepts nothing: {}", name.rsplit(".", 1)[-1], why)
    return 0 if layouts else 1


def _probe_pass(device_id: int, probed) -> None:
    """One pass over the ops still unprobed. Runs in its own process; may not return."""
    from ttnn_accuracy.measure.device import open_device
    from ttnn_accuracy.measure.schema import ARCH_OF_DEVICE
    from ttnn_accuracy.measure.sweeps import capabilities
    from ttnn_accuracy.ops import introspect
    from ttnn_accuracy.ops.overrides import OVERRIDES, bw_fn

    manifest = load()
    ops = manifest["ops"]

    with open_device(device_id) as device:
        # Keyed by arch: `softcap` is Blackhole-only, and neither probe may erase the other.
        arch = ARCH_OF_DEVICE[device.arch().name]
        layouts = manifest["layouts"].setdefault(arch, {})
        rejected = manifest["rejected"].setdefault(arch, {})
        # A name still recorded here means that op never returned: that is the finding.
        if died := manifest["probing"].pop(arch, None):
            rejected[died] = "crashed the process while probing — see the run log"
            logger.error("{} crashed the previous probe; recorded and skipped", died)

        todo = [(n, e) for n, e in ops.items() if n not in layouts and n not in rejected]
        if not todo:  # a completed pass, so start a fresh one
            layouts.clear(), rejected.clear()
            todo = list(ops.items())

        for i, (name, entry) in enumerate(todo, start=1):
            logger.debug("probing {}/{} {}", i, len(todo), name)
            try:
                op = introspect.resolve(name)
            except AttributeError:
                continue
            fn = bw_fn(op) if entry["category"].endswith("_bw") else op
            if variants := OVERRIDES.get(name):
                fn = partial(fn, **variants[0].ttnn_kwargs)
            manifest["probing"][arch] = name
            save(manifest)  # named before it runs: a segfault leaves no other trace

            found, why = capabilities(fn, entry["operands"], device)
            manifest["probing"].pop(arch, None)
            if found:
                layouts[name] = found
            elif why:
                rejected[name] = why
            save(manifest)  # every op, so a crash loses only the op that caused it
    probed.put(arch)


def discover(include_experimental: bool = False) -> int:
    existing = MANIFEST_FILE.exists()
    old = load()
    new = build(include_experimental)
    added, removed, changed = diff(old, new)
    # Derivation survives a rebuild, except where ttnn changed the op underneath it.
    stale = set(changed)
    for key in ("domains", "refused"):
        new[key] = {k: v for k, v in old[key].items() if k in new["ops"] and k not in stale}
    for key in ("layouts", "rejected"):  # arch-keyed: prune ops within each architecture
        new[key] = {
            arch: {k: v for k, v in per.items() if k in new["ops"] and k not in stale}
            for arch, per in old[key].items()
        }
    save(new)

    counts = Counter(entry["category"] for entry in new["ops"].values())

    logger.success("{} eltwise ops → {}", len(new["ops"]), MANIFEST_FILE)
    logger.info("{}", dict(sorted(counts.items())))
    if new["unprobeable"]:
        logger.info(
            "{} goldens refused the probe — each reason recorded under `unprobeable`",
            len(new["unprobeable"]),
        )
    if not existing:
        return 0
    for label, names in (("added", added), ("removed", removed), ("changed", changed)):
        if names:
            logger.warning("{} {}: {}", len(names), label, ", ".join(names))
    return 0
