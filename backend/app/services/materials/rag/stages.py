"""Request-local RAG stages for the knowledge graph's production callbacks."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from app.integrations.llm import UsageBudgetError
from app.models import Material, MaterialChunk
from app.core.rag_config import get_rag_config
from app.services.materials.rag.context import build_context
from app.services.materials.rag.providers import rerank_with_provider, rewrite_with_provider
from app.services.materials.rag.query_rewrite import rewrite_queries
from app.services.materials.rag.ranking import fuse_candidates, govern_candidates, rerank_candidates
from app.services.materials.rag.retrievers import lexical_retrieve, vector_retrieve
from app.services.materials.recall import embed_or_empty


class StagedRetrieval:
    """Keep raw evidence outside checkpoints while each graph node does its own work."""

    def __init__(self, db: Any, query: str, *, history: list[dict[str, str]] | None = None) -> None:
        self.db = db
        self.query = query
        # Only bounded recent turns inform rewrite; the original user query
        # remains the search contract and is never replaced by chat history.
        self.history = [
            {"role": str(item.get("role") or "")[:20], "content": str(item.get("content") or "")[:300]}
            for item in (history or [])[-4:] if isinstance(item, dict)
        ]
        self.cfg = get_rag_config()
        self.started = time.monotonic()
        self.timings: dict[str, float] = {}
        self.queries: list[str] = [query]
        self.rows: list[dict[str, Any]] = []
        self._rows_loaded = False
        self.lexical: list[dict[str, Any]] = []
        self.vector: list[dict[str, Any]] = []
        self.lexical_failed = False
        self.vector_failed = False
        self.fused: list[dict[str, Any]] = []
        self.governed: list[dict[str, Any]] = []
        self.reranked: dict[str, Any] = {}
        self.rewrite: dict[str, Any] = {}
        self.embedding_model = ""

    async def query_rewrite(self) -> None:
        started = time.monotonic()
        config = self.cfg.query_rewrite
        provider = None
        if config.enabled:
            provider = lambda text: rewrite_with_provider(
                self._rewrite_prompt(text), model=config.model, base_url=config.base_url,
                timeout=config.timeout_seconds,
                db=self.db,
            )
        self.rewrite = await rewrite_queries(
            self.query, provider=provider, enabled=config.enabled,
            max_queries=config.max_queries, timeout_seconds=config.timeout_seconds,
        )
        self.queries = self.rewrite["search_queries"] or [self.query]
        self.rewrite.setdefault("diagnostics", {})["history_used"] = bool(self.history)
        self._timed("query_rewrite", started)

    def _rewrite_prompt(self, query: str) -> str:
        """Give the provider context for ellipsis without changing the raw query."""

        if not self.history:
            return query
        turns = "\n".join(f"{item['role']}: {item['content']}" for item in self.history)
        return f"最近对话：\n{turns}\n当前问题：{query}"

    def _load_rows(self) -> None:
        if self._rows_loaded:
            return
        records = (
            self.db.query(MaterialChunk, Material)
            .join(Material, Material.id == MaterialChunk.material_id)
            .filter(Material.status == "ready")
            .all()
        )
        self.rows = [
            {"chunk_id": chunk.id, "material_id": material.id,
             "filename": material.filename, "ordinal": chunk.ordinal,
             "text": chunk.text, "embedding": chunk.embedding}
            for chunk, material in records
        ]
        self._rows_loaded = True

    async def retrieve_parallel(self) -> None:
        """Run both recall routes against one ORM-free snapshot of ready chunks.

        Loading before spawning tasks keeps the SQLAlchemy Session on its owning
        execution path; lexical workers only inspect plain dictionaries. Each
        route handles its own failure and records its own elapsed time.
        """
        started = time.monotonic()
        try:
            self._load_rows()
        except Exception:
            # A failed shared corpus read leaves neither route usable.
            self.lexical = []
            self.vector = []
            self.lexical_failed = True
            self.vector_failed = True
            self._timed("parallel_retrieve", started)
            return
        await asyncio.gather(self.lexical_retrieve(), self.vector_retrieve())
        self._timed("parallel_retrieve", started)

    async def lexical_retrieve(self) -> None:
        started = time.monotonic()
        self.lexical = []
        try:
            self._load_rows()
            for index, query in enumerate(self.queries, start=1):
                self.lexical.extend(await asyncio.to_thread(
                    lexical_retrieve, query, self.rows,
                    top_k=self.cfg.recall.lexical_top_k, query_id=f"q{index}",
                    score_floor=self.cfg.recall.score_floor,
                ))
        except Exception:
            # One failed route must not discard the other route's evidence.
            self.lexical = []
            self.lexical_failed = True
        self._timed("lexical_retrieve", started)

    async def vector_retrieve(self) -> None:
        started = time.monotonic()
        self.vector = []
        try:
            self._load_rows()
            vectors, self.embedding_model = await embed_or_empty(self.db, self.queries)
            if len(vectors) != len(self.queries):
                vectors = [None] * len(self.queries)
            for index, vector in enumerate(vectors, start=1):
                self.vector.extend(await asyncio.to_thread(
                    vector_retrieve, vector, self.rows,
                    top_k=self.cfg.recall.vector_top_k, query_id=f"q{index}",
                    score_floor=self.cfg.recall.score_floor,
                ))
        except UsageBudgetError:
            raise
        except Exception:
            self.vector = []
            self.vector_failed = True
        self._timed("vector_retrieve", started)

    def candidate_fusion(self) -> None:
        started = time.monotonic()
        config = self.cfg.recall
        self.fused = fuse_candidates(
            self.lexical, self.vector, strategy=config.fusion,
            lexical_weight=config.lexical_weight, candidate_k=config.candidate_k,
        )
        self._timed("candidate_fusion", started)

    def deduplicate_and_diversify(self) -> None:
        started = time.monotonic()
        config = self.cfg.recall
        self.governed = govern_candidates(
            self.fused, max_chunks_per_material=config.max_chunks_per_material,
            max_materials=config.max_materials, max_candidates=config.candidate_k,
            merge_adjacent=config.merge_adjacent,
        )
        self._timed("deduplicate_and_diversify", started)

    async def rerank(self) -> None:
        started = time.monotonic()
        config = self.cfg.rerank
        provider = None
        if config.enabled:
            provider = lambda query, rows: rerank_with_provider(
                query, rows, model=config.model, base_url=config.base_url,
                timeout=config.timeout_seconds,
                db=self.db,
            )
        self.reranked = await rerank_candidates(
            self.query, self.governed, reranker=provider,
            top_n=max(1, min(self.cfg.recall.top_k, config.top_n)),
            batch_size=config.batch_size, timeout_seconds=config.timeout_seconds,
            fail_open=config.fail_open,
        )
        self._timed("rerank", started)

    def context_select(self) -> dict[str, Any]:
        started = time.monotonic()
        config = self.cfg.context
        context = build_context(
            self.reranked.get("candidates", []),
            max_chunks=min(self.cfg.recall.top_k, config.max_chunks),
            max_tokens=config.max_tokens, max_chunk_chars=config.max_chunk_chars,
            include_source_id=config.include_source_id,
        )
        self._timed("context_select", started)
        selected = list(context["selected_chunks"])
        rewrite_diagnostics = self.rewrite.get("diagnostics", {})
        rerank_diagnostics = self.reranked.get("diagnostics", {})
        fallback_reasons = [
            reason for reason in (
                rewrite_diagnostics.get("fallback_reason"),
                rerank_diagnostics.get("fallback_reason"),
                "context_truncated" if context.get("diagnostics", {}).get("context_truncated") else "",
            ) if reason and reason not in {"disabled", "provider_unavailable", "empty_candidates"}
        ]
        return {
            "original_query": self.query,
            "search_queries": self.queries,
            "candidates": self.fused,
            "selected_chunks": selected,
            "context_text": context["context_text"],
            "citations": [
                {key: item.get(key) for key in ("source_id", "chunk_id", "material_id", "filename", "ordinal")}
                for item in selected
            ],
            "retrieval_status": (
                "failed" if self.lexical_failed and self.vector_failed or self.reranked.get("retrieval_status") == "failed"
                else "degraded" if selected and (self.lexical_failed or self.vector_failed)
                else "ok" if selected else "empty"
            ),
            "diagnostics": {
                **rewrite_diagnostics, **rerank_diagnostics,
                **context.get("diagnostics", {}),
                "rewrite_status": rewrite_diagnostics.get("provider_status"),
                "rerank_status": rerank_diagnostics.get("provider_status"),
                "fallback_reasons": fallback_reasons,
                "candidate_count": len(self.fused), "selected_count": len(selected),
                "lexical_count": len(self.lexical), "vector_count": len(self.vector),
                "lexical_failed": self.lexical_failed,
                "vector_failed": self.vector_failed,
                "retrieval_unavailable": self.lexical_failed and self.vector_failed,
                "embedding_model": self.embedding_model,
                "retrieval_source": "dual_route",
                "stage_latency_ms": {**self.timings, "total": round((time.monotonic() - self.started) * 1000, 2)},
            },
        }

    def _timed(self, name: str, started: float) -> None:
        self.timings[name] = round((time.monotonic() - started) * 1000, 2)
