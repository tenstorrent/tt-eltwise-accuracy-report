"""The result contract: what a measured row holds, and what produced it.

Imported by both `measure` and `report` so the two cannot drift. Config axes are
declared before they are varied — a column costs nothing now and cannot be
backfilled into runs that already exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

COLUMNS = (
    "x",
    "y",
    "y_ref",
    "ulp_error",
    "abs_error",
    "outcome",
    "op",
    "variant",
    "dtype",
)

ARCH_OF_DEVICE = {"WORMHOLE_B0": "wh", "BLACKHOLE": "bh"}


@dataclass(frozen=True, slots=True)
class RunMeta:
    run_id: str
    arch: str
    device_arch: str
    device_id: int
    ttnn_version: str
    tt_metal_commit: str | None
    torch_version: str
    # Reserved axes: populated once the compute-config matrix lands.
    math_fidelity: str | None = None
    fp32_dest_acc: bool | None = None
    approx_mode: bool | None = None
    ops: list[str] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)


def _tt_metal_commit(version: str) -> str | None:
    """The tt-metal SHA that ttnn was built from, e.g. 0.75.0rc10.dev817+g2adf8c65887."""
    found = re.search(r"\+g([0-9a-f]{7,40})", version)
    return found.group(1) if found else None


def describe_run(device, arch: str, ops: list[str], dtypes: list[str]) -> RunMeta:
    import importlib.metadata as metadata

    import torch

    version = metadata.version("ttnn")
    device_arch = device.arch().name
    return RunMeta(
        run_id=datetime.now().strftime("%Y%m%dT%H%M%S"),
        arch=arch,
        device_arch=device_arch,
        device_id=device.id(),
        ttnn_version=version,
        tt_metal_commit=_tt_metal_commit(version),
        torch_version=torch.__version__,
        ops=ops,
        dtypes=dtypes,
    )


def check_arch(device, arch: str) -> None:
    """Refuse to label results with an architecture the device is not."""
    actual = ARCH_OF_DEVICE.get(device.arch().name)
    if actual != arch:
        raise SystemExit(f"--arch {arch} but device reports {device.arch().name} ({actual})")
