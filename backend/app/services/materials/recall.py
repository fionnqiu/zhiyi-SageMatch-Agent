"""Embedding + hybrid recall shared by chat, question generation, and the admin probe."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sqlalchemy.orm import Session

from app.materials import knowledge
from app.integrations import llm
from app.models import Material, MaterialChunk
from app.core.rag_config import get_rag_config
from app.services.operations.llm_gateway import log_call
from app.services.materials.rag.context import build_context
from app.services.materials.rag.query_rewrite import rewrite_queries
from app.services.materials.rag.ranking import fuse_candidates, govern_candidates, rerank_candidates
from app.services.materials.rag.retrievers import dual_retrieve
from app.services.materials.rag.providers import rewrite_with_provider, rerank_with_provider


def embedding_ready() -> bool:
    cfg = get_rag_config().embedding
    return bool(cfg.enabled and cfg.model.strip() and cfg.base_url.strip() and cfg.api_key.strip())


async def embed_or_empty(db: Session, texts: list[str]) -> tuple[list[list[float]], str]:
    """Embed texts, or return nothing so lexical recall can still run.

    A bad vector must not block material ingest or question generation.
    """
    cfg = get_rag_config().embedding
    if not texts or not embedding_ready():
        return [], ""
    model = cfg.model.strip()
    started = time.perf_counter()
    with llm.call_usage_scope() as measured:
        try:
            vectors = await llm.embed_texts(
                texts,
                api_key=cfg.api_key.strip(),
                base_url=cfg.base_url.strip(),
                model=model,
                batch_size=cfg.batch_size,
            )
        except llm.UsageBudgetError:
            # Budget denial is a graph policy failure, not a provider failure.
            raise
        except Exception as exc:  # noqa: BLE001 — 向量失败不挡住物料入库
            # Endpoint URLs may contain credentials; log only the adapter name.
            log_call(db, "embedding", "embedding", model, "error",
                     int((time.perf_counter() - started) * 1000), str(exc)[:240],
                     usage=llm.combine_usage(measured))
            return [], ""
        expected = cfg.dimensions
        if expected and vectors and any(len(v) != expected for v in vectors):
            log_call(
                db, "embedding", "embedding", model, "error",
                int((time.perf_counter() - started) * 1000),
                f"embedding 维度应为 {expected}，实际 {len(vectors[0])}",
                usage=llm.combine_usage(measured),
            )
            return [], ""
        # Successful embeddings belong in the denominator of provider failure
        # metrics, and retain only provider-reported token usage.
        log_call(db, "embedding", "embedding", model, "ok",
                 int((time.perf_counter() - started) * 1000), None,
                 usage=llm.combine_usage(measured))
        return vectors, model


async def recall_snippets(db: Session, query: str) -> list[dict[str, Any]]:
    """Run the structured two-route RAG chain and return legacy hit rows.

    Existing callers still receive the historical list shape, but every item now
    carries the query/source metadata produced by rewrite, dual retrieval,
    fusion, governance and optional reranking.  ``structured_recall`` exposes
    the complete contract to new graph nodes without making old APIs guess.
    """
    result = await structured_recall(db, query)
    return result["selected_chunks"]


async def structured_recall(db: Session, query: str) -> dict[str, Any]:
    """Return the full RAG result contract used by AgentState and API extras."""
    started = time.monotonic()
    cfg = get_rag_config()
    rows = (
        db.query(MaterialChunk, Material)
        .join(Material, Material.id == MaterialChunk.material_id)
        .filter(Material.status == "ready")
        .all()
    )
    packed = [
        {
            "chunk_id": chunk.id,
            "material_id": mat.id,
            "filename": mat.filename,
            "ordinal": chunk.ordinal,
            "text": chunk.text,
            "embedding": chunk.embedding,
        }
        for chunk, mat in rows
    ]
    rewrite_provider = None
    if cfg.query_rewrite.enabled:
        rewrite_provider = lambda text: rewrite_with_provider(
            text,
            model=cfg.query_rewrite.model,
            base_url=cfg.query_rewrite.base_url,
            timeout=cfg.query_rewrite.timeout_seconds,
            db=db,
        )
    rewrite = await rewrite_queries(
        query, provider=rewrite_provider, enabled=cfg.query_rewrite.enabled,
        max_queries=cfg.query_rewrite.max_queries,
        timeout_seconds=cfg.query_rewrite.timeout_seconds,
    )
    search_queries = rewrite["search_queries"] or [query]
    # Each rewritten phrase needs its own embedding; reusing the original vector
    # would silently make the vector route ignore the rewrite.
    vectors, embedding_model = await embed_or_empty(db, search_queries)
    query_vectors = vectors if len(vectors) == len(search_queries) else [None] * len(search_queries)
    lexical: list[dict[str, Any]] = []
    vector: list[dict[str, Any]] = []
    async def retrieve_one(query_id: int, search_query: str, vector: list[float] | None):
        return await asyncio.to_thread(
            dual_retrieve, search_query, packed, query_vector=vector,
            lexical_top_k=cfg.recall.lexical_top_k,
            vector_top_k=cfg.recall.vector_top_k,
            query_id=f"q{query_id}", score_floor=cfg.recall.score_floor,
        )

    for left, right in await asyncio.gather(*(
        retrieve_one(i, text, query_vectors[i - 1])
        for i, text in enumerate(search_queries, start=1)
    )):
        lexical.extend(left)
        vector.extend(right)
    fused = fuse_candidates(
        lexical,
        vector,
        strategy=cfg.recall.fusion,
        lexical_weight=cfg.recall.lexical_weight,
        candidate_k=cfg.recall.candidate_k,
    )
    governed = govern_candidates(
        fused,
        max_chunks_per_material=cfg.recall.max_chunks_per_material,
        max_materials=cfg.recall.max_materials,
        max_candidates=cfg.recall.candidate_k,
        merge_adjacent=cfg.recall.merge_adjacent,
    )
    reranker = None
    if cfg.rerank.enabled:
        reranker = lambda text, rows: rerank_with_provider(
            text, rows, model=cfg.rerank.model,
            base_url=cfg.rerank.base_url, timeout=cfg.rerank.timeout_seconds,
            db=db,
        )
    reranked = await rerank_candidates(
        query,
        governed,
        reranker=reranker,
        top_n=max(1, min(cfg.recall.top_k, cfg.rerank.top_n)),
        batch_size=cfg.rerank.batch_size,
        timeout_seconds=cfg.rerank.timeout_seconds,
        fail_open=cfg.rerank.fail_open,
    )
    context = build_context(
        reranked["candidates"],
        max_chunks=min(cfg.recall.top_k, cfg.context.max_chunks),
        max_tokens=cfg.context.max_tokens,
        max_chunk_chars=cfg.context.max_chunk_chars,
        include_source_id=cfg.context.include_source_id,
    )
    selected = list(context["selected_chunks"])
    rewrite_diagnostics = rewrite.get("diagnostics", {})
    rerank_diagnostics = reranked.get("diagnostics", {})
    fallback_reasons = [
        reason for reason in (
            rewrite_diagnostics.get("fallback_reason"),
            rerank_diagnostics.get("fallback_reason"),
            "context_truncated" if context.get("diagnostics", {}).get("context_truncated") else "",
        ) if reason and reason not in {"disabled", "provider_unavailable", "empty_candidates"}
    ]
    diagnostics = {
        **rewrite_diagnostics,
        **rerank_diagnostics,
        **context.get("diagnostics", {}),
        "rewrite_status": rewrite_diagnostics.get("provider_status"),
        "rerank_status": rerank_diagnostics.get("provider_status"),
        "fallback_reasons": fallback_reasons,
        "candidate_count": len(fused),
        "selected_count": len(selected),
        "lexical_count": len(lexical),
        "vector_count": len(vector),
        "embedding_model": embedding_model,
        "retrieval_source": "dual_route",
        "stage_latency_ms": {"total": round((time.monotonic() - started) * 1000, 2)},
    }
    return {
        "original_query": query,
        "search_queries": search_queries,
        "candidates": fused,
        "selected_chunks": selected,
        "context_text": context["context_text"],
        "citations": [
            {
                "source_id": item["source_id"],
                "chunk_id": item.get("chunk_id"),
                "material_id": item.get("material_id"),
                "filename": item.get("filename"),
                "ordinal": item.get("ordinal"),
            }
            for item in selected
        ],
        "retrieval_status": reranked["retrieval_status"] if reranked["retrieval_status"] == "failed" else ("ok" if selected else "empty"),
        "diagnostics": diagnostics,
    }
