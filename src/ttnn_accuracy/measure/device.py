from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager

# A TT_FATAL quotes the absolute path it was built from. The file and line are the useful
# part; the prefix is whoever's home directory built tt-metal, and this text is published.
_BUILD_PATH = re.compile(r"/\S+?/(?=(?:ttnn|tt_metal)/)")


@contextmanager
def open_device(device_id: int = 0) -> Iterator[object]:
    import ttnn

    device = ttnn.open_device(device_id=device_id)
    try:
        yield device
    finally:
        ttnn.close_device(device)


def reason(exc: Exception, limit: int = 200) -> str:
    """Why the device refused, as one publishable line."""
    line = next(iter(str(exc).strip().splitlines()), "")
    if home := os.environ.get("TT_METAL_HOME"):
        line = line.replace(f"{home}/", "")
    # count=1: the path repeats `ttnn/` further along, and a second substitution eats it.
    return _BUILD_PATH.sub("", line, count=1)[:limit]
