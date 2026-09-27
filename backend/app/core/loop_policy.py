"""Uvicorn event-loop factory compatible with async Psycopg on Windows."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


def selector_loop_factory(use_subprocess: bool = False) -> Callable[[], asyncio.AbstractEventLoop]:
    """Use Selector on Windows because Psycopg rejects Proactor async I/O."""
    del use_subprocess
    return asyncio.SelectorEventLoop
