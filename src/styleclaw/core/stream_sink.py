from __future__ import annotations

import contextvars
from typing import Callable

# A sink receives raw text deltas as they stream from the LLM provider.
DeltaSink = Callable[[str], None]

_current_sink: contextvars.ContextVar[DeltaSink | None] = contextvars.ContextVar(
    "styleclaw_delta_sink", default=None,
)


def set_delta_sink(sink: DeltaSink | None) -> contextvars.Token:
    """Install a delta sink for the current context. Returns a token for reset."""
    return _current_sink.set(sink)


def reset_delta_sink(token: contextvars.Token) -> None:
    _current_sink.reset(token)


def emit_delta(text: str) -> bool:
    """Send a streaming delta to the active sink if one is installed.

    Returns True if a sink consumed it, False otherwise (callers then fall
    back to their default behavior, e.g. printing to stdout).
    """
    sink = _current_sink.get()
    if sink is None:
        return False
    sink(text)
    return True
