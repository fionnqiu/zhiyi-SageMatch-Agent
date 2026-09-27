"""Start SageMatch with an event loop compatible with async PostgreSQL saves."""

from __future__ import annotations

import argparse

import uvicorn

from app.core.loop_policy import selector_loop_factory


def main() -> None:
    """Use the Selector loop on Windows for Psycopg's async checkpointer."""
    parser = argparse.ArgumentParser(description="Run the SageMatch API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        # Uvicorn 0.53 treats a custom path as the loop factory itself. Pass
        # the SelectorEventLoop class so Runner receives a loop instance.
        loop=selector_loop_factory(),
    )


if __name__ == "__main__":
    main()
