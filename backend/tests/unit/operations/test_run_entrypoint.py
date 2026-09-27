"""The executable server entry point selects a Psycopg-compatible event loop."""

import asyncio

from uvicorn import Config

from app.core.loop_policy import selector_loop_factory


def test_custom_uvicorn_loop_factory_creates_selector_loop() -> None:
    """Uvicorn must receive a factory yielding a loop, not another factory."""

    selected = selector_loop_factory()
    factory = Config("app.main:app", loop=selected).get_loop_factory()
    loop = factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()
