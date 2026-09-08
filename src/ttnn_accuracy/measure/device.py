from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager

# A TT_FATAL quotes the build path; only file and line are useful, and this text is published.
_BUILD_PATH = re.compile(r"/\S+?/(?=(?:ttnn|tt_metal)/)")


@contextmanager
def open_device(device_id: int = 0) -> Iterator[object]:
    import ttnn

    device = ttnn.open_device(device_id=device_id)
    try:
        yield device
    finally:
        ttnn.close_device(device)


@contextmanager
def session(device_id: int = 0, device=None) -> Iterator[object]:
    """The caller's device or a fresh one — opening costs two minutes and `check` passes twice."""
    if device is not None:
        yield device
    else:
        with open_device(device_id) as opened:
            yield opened


def reason(exc: Exception, limit: int = 200) -> str:
    """Why the device refused, as one publishable line."""
    line = next(iter(str(exc).strip().splitlines()), "")
    if home := os.environ.get("TT_METAL_HOME"):
        line = line.replace(f"{home}/", "")
    # count=1: the path repeats `ttnn/` further along, and a second substitution eats it.
    return _BUILD_PATH.sub("", line, count=1)[:limit]
