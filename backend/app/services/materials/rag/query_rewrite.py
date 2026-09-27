"""Bounded query rewriting with a deterministic no-provider path.

Rewrites are retrieval hints only.  The original user query is retained in the
result and is always the first search query, so a provider cannot silently
replace the user's wording or turn a rewrite into answer evidence.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.integrations.llm import UsageBudgetError
from app.core.rag_config import get_rag_config


Provider = Callable[[str], Any]


@dataclass(frozen=True)
class RewriteSettings:
    """Use the shared RAG configuration for rewrite limits and enablement."""

    enabled: bool = True
    max_queries: int = 3
    max_query_chars: int = 2000

    @classmethod
    def from_env(cls) -> "RewriteSettings":
        config = get_rag_config().query_rewrite
        return cls(
            enabled=config.enabled,
            max_queries=max(1, min(3, config.max_queries)),
            max_query_chars=max(64, min(8000, config.max_query_chars)),
        )


async def rewrite_queries(
    original_query: str,
    *,
    provider: Provider | None = None,
    max_queries: int | None = None,
    max_query_chars: int | None = None,
    enabled: bool | None = None,
    timeout_seconds: float = 8.0,
) -> dict[str, Any]:
    """Return at most three bounded search queries and rewrite diagnostics.

    A provider may return a list, a ``{"queries": [...]}`` object, or a
    ``{"search_queries": [...]}`` object.  Invalid, empty, timed-out, and
    unavailable providers all use the original query as a deterministic
    fallback.  The callable is injected to keep this module network-free in
    tests and to let the graph choose its own LLM adapter.
    """

    settings = RewriteSettings.from_env()
    query = " ".join(str(original_query or "").split())
    limit_chars = max_query_chars or settings.max_query_chars
    query = query[: max(1, int(limit_chars))]
    limit = max(1, min(3, int(max_queries or settings.max_queries)))
    use_provider = settings.enabled if enabled is None else bool(enabled)
    diagnostics: dict[str, Any] = {
        "rewrite_used": False,
        "rewrite_failed": False,
        "fallback_reason": "",
        "provider_status": "unconfigured" if provider is None else "configured",
    }

    if not query:
        diagnostics.update({"fallback_reason": "empty_query", "provider_status": "skipped"})
        return {"original_query": "", "search_queries": [], "diagnostics": diagnostics}
    if not use_provider or provider is None:
        diagnostics["fallback_reason"] = "provider_unavailable" if provider is None else "disabled"
        return {"original_query": query, "search_queries": [query], "diagnostics": diagnostics}

    try:
        raw = provider(query)
        if inspect.isawaitable(raw):
            raw = await asyncio.wait_for(raw, timeout=max(0.1, timeout_seconds))
        values = _extract_values(raw)
        if not values:
            raise ValueError("rewrite provider returned no usable query")
        values = normalize_queries(values, query, max_queries=limit, max_query_chars=limit_chars)
        diagnostics["rewrite_used"] = len(values) > 1 or values[0] != query
        diagnostics["provider_status"] = "ok"
        return {"original_query": query, "search_queries": values, "diagnostics": diagnostics}
    except UsageBudgetError:
        # A configured graph budget is a policy gate, not provider unavailability.
        raise
    except Exception as exc:  # noqa: BLE001 - rewrite must never block retrieval
        diagnostics.update(
            {
                "rewrite_failed": True,
                "fallback_reason": "rewrite_failed",
                "provider_status": "error",
                "error_type": type(exc).__name__,
            }
        )
        return {"original_query": query, "search_queries": [query], "diagnostics": diagnostics}


async def rewrite_query(original_query: str, **kwargs: Any) -> dict[str, Any]:
    """Singular alias retained for callers that use the stage name directly."""

    return await rewrite_queries(original_query, **kwargs)


def normalize_queries(
    values: Any,
    original_query: str,
    *,
    max_queries: int = 3,
    max_query_chars: int = 2000,
) -> list[str]:
    """Normalize provider output while always retaining the original query."""

    original = " ".join(str(original_query or "").split())[: max(1, max_query_chars)]
    raw_values = _extract_values(values)
    cleaned: list[str] = [original] if original else []
    seen = {original.casefold()} if original else set()
    for value in raw_values:
        text = " ".join(str(value or "").split())[: max(1, max_query_chars)]
        key = text.casefold()
        if text and key not in seen:
            cleaned.append(text)
            seen.add(key)
        if len(cleaned) >= max(1, min(3, max_queries)):
            break
    return cleaned[: max(1, min(3, max_queries))] if cleaned else []


def _extract_values(raw: Any) -> list[Any]:
    if isinstance(raw, dict):
        for key in ("search_queries", "queries", "rewrites", "data"):
            if key in raw:
                return _extract_values(raw[key])
        return []
    if isinstance(raw, str):
        # Newline-separated output is a useful compatibility path for simple providers.
        return [line.strip("-• ") for line in raw.splitlines() if line.strip()]
    if isinstance(raw, (list, tuple, set)):
        return list(raw)
    return []


__all__ = ["RewriteSettings", "normalize_queries", "rewrite_queries", "rewrite_query"]
