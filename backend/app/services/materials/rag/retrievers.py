"""Independent lexical and vector coarse retrievers.

The adapters consume plain chunk mappings, which keeps the RAG quality chain
usable when no embedding or hosted provider is configured.  Every returned
candidate uses the same metadata contract before fusion.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from app.materials import knowledge


def lexical_retrieve(
    query: str,
    rows: Sequence[dict[str, Any]],
    *,
    top_k: int = 40,
    query_id: str | None = None,
    score_floor: float = 0.0,
) -> list[dict[str, Any]]:
    """Score chunks with the deterministic lexical scorer already used by ingest."""

    query_key = query_id or stable_query_id(query)
    ranked: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text") or "")
        score = float(knowledge.lexical_score(query, text))
        # A zero floor still excludes non-matches; otherwise an empty query
        # would return every chunk and exhaust the context budget.
        if score <= float(score_floor):
            continue
        ranked.append(_candidate(row, query_key, "lexical", score))
    ranked.sort(key=lambda item: (-item["retrieval_score"], str(item.get("chunk_id") or "")))
    return ranked[: max(0, int(top_k))]


def vector_retrieve(
    query_vector: Sequence[float] | None,
    rows: Sequence[dict[str, Any]],
    *,
    top_k: int = 40,
    query_id: str = "q-vector",
    score_floor: float = 0.0,
) -> list[dict[str, Any]]:
    """Score stored embeddings; an absent query vector intentionally yields an empty route."""

    if not query_vector:
        return []
    ranked: list[dict[str, Any]] = []
    query = [float(value) for value in query_vector]
    for row in rows:
        vector = row.get("embedding")
        if not isinstance(vector, (list, tuple)):
            continue
        score = float(knowledge.cosine(query, [float(value) for value in vector]))
        if score <= float(score_floor):
            continue
        ranked.append(_candidate(row, query_id, "vector", score))
    ranked.sort(key=lambda item: (-item["retrieval_score"], str(item.get("chunk_id") or "")))
    return ranked[: max(0, int(top_k))]


def dual_retrieve(
    query: str,
    rows: Sequence[dict[str, Any]],
    *,
    query_vector: Sequence[float] | None = None,
    lexical_top_k: int = 40,
    vector_top_k: int = 40,
    query_id: str | None = None,
    score_floor: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run both coarse routes locally and return ``(lexical, vector)`` results."""

    qid = query_id or stable_query_id(query)
    lexical = lexical_retrieve(
        query,
        rows,
        top_k=lexical_top_k,
        query_id=qid,
        score_floor=score_floor,
    )
    vector = vector_retrieve(
        query_vector,
        rows,
        top_k=vector_top_k,
        query_id=qid,
        score_floor=score_floor,
    )
    return lexical, vector


async def retrieve_parallel(
    query: str,
    *,
    lexical: Callable[[str], Any],
    vector: Callable[[str], Any],
) -> dict[str, Any]:
    """Execute injected provider adapters concurrently with route-level fail-open behavior."""

    async def call(adapter: Callable[[str], Any]) -> list[dict[str, Any]]:
        value = adapter(query)
        if inspect.isawaitable(value):
            value = await value
        return list(value or []) if isinstance(value, (list, tuple)) else []

    results = await asyncio.gather(call(lexical), call(vector), return_exceptions=True)
    lexical_result = results[0] if isinstance(results[0], list) else []
    vector_result = results[1] if isinstance(results[1], list) else []
    diagnostics = {
        "lexical_failed": isinstance(results[0], Exception),
        "vector_failed": isinstance(results[1], Exception),
        "retrieval_unavailable": not lexical_result and not vector_result,
    }
    return {"lexical": lexical_result, "vector": vector_result, "diagnostics": diagnostics}


def _candidate(row: dict[str, Any], query_id: str, source: str, score: float) -> dict[str, Any]:
    """Copy only retrieval metadata so ORM objects and secret fields never leak downstream."""

    item = {
        "chunk_id": str(row.get("chunk_id") or row.get("id") or ""),
        "material_id": str(row.get("material_id") or ""),
        "filename": str(row.get("filename") or ""),
        "ordinal": int(row.get("ordinal") or 0),
        "text": str(row.get("text") or ""),
        "query_id": query_id,
        "retrieval_source": source,
        "retrieval_score": round(max(0.0, min(1.0, score)), 6),
    }
    # ``score`` is retained as a compatibility alias for older ranking callers.
    item["score"] = item["retrieval_score"]
    return item


def stable_query_id(query: str) -> str:
    """Derive a repeatable query id so test runs and diagnostics are reproducible."""

    digest = hashlib.sha1(str(query).encode("utf-8")).hexdigest()[:12]
    return f"q-{digest}"


retrieve_lexical = lexical_retrieve
retrieve_vector = vector_retrieve
retrieve_two_way = dual_retrieve

__all__ = [
    "dual_retrieve",
    "lexical_retrieve",
    "retrieve_lexical",
    "retrieve_parallel",
    "retrieve_two_way",
    "retrieve_vector",
    "stable_query_id",
    "vector_retrieve",
]
