"""Find the largest GPU batch size that fits in the current VRAM budget."""
from __future__ import annotations

import logging
from typing import Callable

import torch

logger = logging.getLogger(__name__)


def find_max_batch_size(
    try_forward: Callable[[int], None],
    *,
    max_size: int = 64,
    min_size: int = 1,
) -> int:
    """Binary-search the largest batch size for which ``try_forward(n)`` succeeds.

    ``try_forward`` should run a representative forward pass of size ``n`` and
    raise ``torch.cuda.OutOfMemoryError`` (or return after succeeding). CUDA
    cache is cleared after each OOM.
    """
    max_size = max(min_size, int(max_size))
    min_size = max(1, int(min_size))

    if not torch.cuda.is_available():
        return min(8, max_size)

    lo, hi = min_size, max_size
    best = min_size
    # First try max — if it fits we are done.
    candidates = []
    n = max_size
    while n >= min_size:
        candidates.append(n)
        n //= 2
    if min_size not in candidates:
        candidates.append(min_size)

    # Ascending probe via binary search over [min_size, max_size].
    while lo <= hi:
        mid = (lo + hi) // 2
        if _fits(try_forward, mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    logger.info("auto batch size selected: %s (search max=%s)", best, max_size)
    return best


def _fits(try_forward: Callable[[int], None], batch_size: int) -> bool:
    try:
        try_forward(batch_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg and "memory" in msg:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
        raise
