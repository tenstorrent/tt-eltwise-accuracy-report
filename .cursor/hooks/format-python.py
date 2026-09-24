#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Format a just-edited Python file. Fail open: a hook crash must not block the edit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNED = (ROOT / "src", ROOT / "tests")


def _reply() -> None:
    sys.stdout.write("{}\n")


def _path(payload: dict) -> Path | None:
    raw = payload.get("file_path") or payload.get("path") or payload.get("uri") or ""
    if isinstance(raw, dict):
        raw = raw.get("path") or raw.get("fsPath") or ""
    if not raw:
        return None
    p = Path(str(raw).replace("file://", ""))
    return p if p.suffix == ".py" else None


def main() -> int:
    payload = json.load(sys.stdin)
    path = _path(payload)
    if path is None:
        _reply()
        return 0
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root) for root in OWNED):
        _reply()
        return 0
    subprocess.run(
        ["uvx", "ruff", "format", str(resolved)],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    subprocess.run(
        ["uvx", "ruff", "check", "--fix", str(resolved)],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    _reply()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
