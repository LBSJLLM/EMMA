from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator


@contextmanager
def timed(bucket: Dict[str, float], key: str) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        bucket[key] = bucket.get(key, 0.0) + (time.time() - start)
