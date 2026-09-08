"""The result contract, imported by `measure` and `report` so the two cannot drift."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

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
    "layout",
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
    ops: list[str] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)


def _commit_from_version(version: str) -> str | None:
    """The SHA ttnn's package metadata claims, e.g. 0.75.0rc10.dev817+g2adf8c65887."""
    found = re.search(r"\+g([0-9a-f]{7,40})", version)
    return found.group(1) if found else None


def _tt_metal_commit(version: str) -> str | None:
    """The SHA actually running: `build_metal.sh` rebuilds without reinstalling ttnn."""
    home = os.environ.get("TT_METAL_HOME")
    if home:
        head = subprocess.run(
            ["git", "-C", home, "rev-parse", "--short=11", "HEAD"],
            capture_output=True,
            text=True,
        )
        if head.returncode == 0:
            sha = head.stdout.strip()
            if sha not in version:
                logger.warning(
                    "ttnn metadata reports {} but {} is at {} — reinstall ttnn to agree",
                    version,
                    home,
                    sha,
                )
            return sha
    return _commit_from_version(version)


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
