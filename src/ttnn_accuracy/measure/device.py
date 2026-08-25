from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def open_device(device_id: int = 0) -> Iterator[object]:
    import ttnn

    device = ttnn.open_device(device_id=device_id)
    try:
        yield device
    finally:
        ttnn.close_device(device)
