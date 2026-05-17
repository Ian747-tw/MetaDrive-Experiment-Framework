from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    data = {"start": time.time(), "elapsed": 0.0}
    try:
        yield data
    finally:
        data["elapsed"] = time.time() - data["start"]
