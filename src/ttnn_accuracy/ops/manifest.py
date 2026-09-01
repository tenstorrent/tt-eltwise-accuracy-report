"""The committed op manifest — eltwise ops only.

`ops` holds every ttnn op that classified as real-valued elementwise mathematics, rebuilt
wholesale by `discover`, along with `unprobeable` — why each unclassifiable golden
refused, learned from the same calls. Everything else is what the later stages learned by
running goldens and calling the device: `domains` and its `refused` counterpart from
derivation, `layouts` and its `rejected` counterpart from probing, per architecture. Each
costs far more than a rebuild and does not change when ttnn's op list does, so a rebuild
keeps them for every op it did not see change.

Nothing here holds a timestamp: an unchanged ttnn must produce an unchanged file, so that
a diff means ttnn changed.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import partial

from loguru import logger

from ttnn_accuracy.domain.derive import DTYPES, derive
from ttnn_accuracy.ops import arity, introspect
from ttnn_accuracy.ops.overrides import EXCLUDED, OVERRIDES, bw_fn
from ttnn_accuracy.paths import MANIFEST_FILE


def build(include_experimental: bool = False) -> dict:
    """Eltwise facts only. `domains` is filled by derive_domains, not here.

    The scope is eltwise, so an op that is not real-valued elementwise mathematics never
    enters `ops` — matmuls, predicates, constant outputs and the excluded movers are
    dropped at the door rather than stored with a flag. `unprobeable` keeps every
    refusal: a golden that cannot be called might still be an eltwise op waiting for an
    override, so those reasons are the queue for extending coverage.
    """
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
    return {
        "ops": ops,
        "unprobeable": unprobeable,
        "domains": {},
        "refused": {},
        "layouts": {},
        "rejected": {},
        "probing": {},
    }


def load() -> dict:
    """Defaults first, so a manifest written before a key existed still loads."""
    empty = {
        "ops": {},
        "unprobeable": {},
        "domains": {},
        "refused": {},
        "layouts": {},
        "rejected": {},
        "probing": {},
    }
    if not MANIFEST_FILE.exists():
        return empty
    manifest = empty | json.loads(MANIFEST_FILE.read_text())
    # Manifests written before layouts were arch-keyed hold flat per-op dicts, and every
    # one of those was probed on Wormhole.
    for key in ("layouts", "rejected"):
        if any(k.startswith("ttnn.") for k in manifest[key]):
            manifest[key] = {"wh": manifest[key]}
    return manifest


def save(manifest: dict) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def diff(old: dict, new: dict) -> tuple[list[str], list[str], list[str]]:
    before, after = old["ops"], new["ops"]
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    return added, removed, changed


def _single_axis(golden, backward: bool, x):
    return arity.call_golden(golden, [x], backward)


def derive_domains() -> int:
    """Add per-dtype bounds to every eltwise op the manifest already knows about.

    An op whose golden rejects the fp64 grid is recorded under `refused` with the reason,
    so later stages can say why it is unmeasurable instead of guessing.
    """
    manifest = load()
    discovered = {op.qualified_name: op for op in introspect.discover()}
    # Both are recomputed for every op below, so clear them: keeping a previous domain for
    # a golden that now refuses would leave the op with bounds and a refusal at once.
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
    """Probe every op, outliving any that kills the process.

    An op can segfault the host — `bias_gelu_bw` does on some tt-metal builds — and a
    segfault cannot be caught, only outlived. Each pass therefore runs in its own
    process, having recorded the op it is about to probe; a pass that dies tells the next
    one what killed it, which is recorded as that op's result. One crash costs one op and
    a device open, not the run.
    """
    import multiprocessing

    context = multiprocessing.get_context("spawn")  # a fresh interpreter, no shared device
    # The child reports which architecture it probed: the parent has no device, and
    # guessing from the manifest would summarise the other architecture once both exist.
    probed = context.SimpleQueue()
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

    manifest = load()
    ops = manifest["ops"]

    with open_device(device_id) as device:
        # Keyed by the architecture actually probed: `softcap` exists only on Blackhole,
        # and a Wormhole probe must not overwrite what a Blackhole probe learned.
        arch = ARCH_OF_DEVICE[device.arch().name]
        layouts = manifest["layouts"].setdefault(arch, {})
        rejected = manifest["rejected"].setdefault(arch, {})
        # An op can take the whole process down — bias_gelu_bw segfaults the host on some
        # tt-metal builds — and a segfault is not catchable, so the op being probed is
        # recorded before it runs. Finding that record on the next run means it never
        # returned: that is the finding, and the remaining ops are not lost with it.
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
