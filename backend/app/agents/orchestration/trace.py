"""Request-local graph correlation without passing IDs through provider prompts."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_run_id: ContextVar[str | None] = ContextVar("sagematch_run_id", default=None)


@contextmanager
def graph_run_scope(run_id: str) -> Iterator[None]:
    """Bind one run ID for nested model/tool calls in the current async task."""
    token = _run_id.set(run_id)
    try:
        yield
    finally:
        _run_id.reset(token)


def current_run_id() -> str | None:
    """Return the graph run currently owning this provider/tool call."""
    return _run_id.get()
