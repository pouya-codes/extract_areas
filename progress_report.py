"""Mid-job progress reporting for long-running inference.

Workers set a reporter via :func:`set_progress_reporter`; models call
:func:`report_progress` from batch loops. Aimviewer polls the Redis job and
forwards fields to the browser.
"""
from __future__ import annotations

import contextvars
from typing import Any, Callable, Optional

ProgressReporter = Callable[..., None]

_reporter: contextvars.ContextVar[Optional[ProgressReporter]] = contextvars.ContextVar(
    "extract_areas_progress_reporter", default=None
)


def set_progress_reporter(fn: Optional[ProgressReporter]):
    """Bind a reporter for the current context; returns a reset token."""
    return _reporter.set(fn)


def reset_progress_reporter(token) -> None:
    _reporter.reset(token)


def report_progress(**fields: Any) -> None:
    """Best-effort progress update (no-op when no reporter is bound)."""
    fn = _reporter.get()
    if fn is None:
        return
    try:
        fn(**fields)
    except Exception:
        # Progress must never abort inference.
        pass
