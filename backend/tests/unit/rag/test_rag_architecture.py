"""Deterministic contracts for the stage-four RAG chain and stage-five gates."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


def test_staged_retrieval_runs_both_routes_concurrently_without_worker_db_access() -> None:
    from app.services.materials.rag.stages import StagedRetrieval

    lexical_started = threading.Event()
    vector_started = threading.Event()
    db = MagicMock()
    db.query.return_value.join.return_value.filter.return_value.all.return_value = []
    pipeline = StagedRetrieval(db, "缓存击穿")

    def lexical_route(*_args, **_kwargs):
        lexical_started.set()
        assert vector_started.wait(2), "vector route did not overlap lexical route"
        return []

    async def embedding_route(_db, _queries):
        assert await asyncio.to_thread(lexical_started.wait, 2)
        vector_started.set()
        return [], ""

    with (
        patch("app.services.materials.rag.stages.lexical_retrieve", side_effect=lexical_route),
        patch("app.services.materials.rag.stages.embed_or_empty", side_effect=embedding_route),
    ):
        asyncio.run(pipeline.retrieve_parallel())

    assert db.query.call_count == 1
    assert not pipeline.lexical_failed
    assert not pipeline.vector_failed
    assert "lexical_retrieve" in pipeline.timings
    assert "vector_retrieve" in pipeline.timings


def test_query_rewrite_without_provider_preserves_original_query() -> None:
    from app.services.materials.rag.query_rewrite import rewrite_queries

    result = asyncio.run(rewrite_queries("什么是缓存击穿？", provider=None))

    assert result["original_query"] == "什么是缓存击穿？"
    assert result["search_queries"] == ["什么是缓存击穿？"]
    assert result["diagnostics"]["rewrite_used"] is False


def test_dual_retrieval_and_fusion_attach_route_metadata() -> None:
    from app.services.materials.rag.retrievers import dual_retrieve
    from app.services.materials.rag.ranking import fuse_candidates

    rows = [
        {"chunk_id": "a", "material_id": "m1", "filename": "a.md", "ordinal": 0, "text": "缓存击穿"},
        {"chunk_id": "b", "material_id": "m2", "filename": "b.md", "ordinal": 0, "text": "缓存雪崩"},
    ]
    lexical, vector = dual_retrieve("缓存击穿", rows, query_vector=None)
    fused = fuse_candidates(lexical, vector, strategy="weighted")

    assert lexical[0]["retrieval_source"] == "lexical"
    assert fused[0]["chunk_id"] == "a"
    assert 0 <= fused[0]["retrieval_score"] <= 1
    assert fused[0]["query_id"]


def test_candidate_governance_deduplicates_merges_adjacent_and_limits_materials() -> None:
    from app.services.materials.rag.ranking import govern_candidates

    candidates = [
        {"chunk_id": "a", "material_id": "m1", "filename": "a", "ordinal": 0, "text": "一", "score": 0.9},
        {"chunk_id": "b", "material_id": "m1", "filename": "a", "ordinal": 1, "text": "二", "score": 0.8},
        {"chunk_id": "a", "material_id": "m1", "filename": "a", "ordinal": 0, "text": "一", "score": 0.7},
        {"chunk_id": "c", "material_id": "m2", "filename": "b", "ordinal": 0, "text": "三", "score": 0.6},
    ]

    selected = govern_candidates(candidates, max_chunks_per_material=2, max_materials=2, merge_adjacent=True)

    assert [item["chunk_id"] for item in selected] == ["a", "c"]
    assert selected[0]["text"] == "一\n二"


def test_context_has_stable_sources_and_citation_validation_rejects_unknown_source() -> None:
    from app.services.materials.rag.citations import validate_citations
    from app.services.materials.rag.context import build_context

    context = build_context([
        {"chunk_id": "a", "material_id": "m1", "filename": "a", "ordinal": 0, "text": "缓存击穿"}
    ])

    assert context["selected_chunks"][0]["source_id"] == "S1"
    assert "[S1]" in context["context_text"]
    valid = validate_citations("答案 [S1]", context["selected_chunks"])
    invalid = validate_citations("答案 [S9]", context["selected_chunks"])
    assert valid["valid"] is True
    assert invalid["valid"] is False


def test_retrieval_metrics_and_gate_are_deterministic() -> None:
    from app.services.operations.quality import evaluate_retrieval, quality_gate

    metrics = evaluate_retrieval(
        [{"retrieved": ["a", "b"], "relevant": ["b"]}],
        k=2,
    )

    assert metrics["recall_at_k"] == 1.0
    assert metrics["precision_at_k"] == 0.5
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg_at_k"] == 0.6309
    assert metrics["hit_rate"] == 1.0
    assert quality_gate(metrics, {"recall_at_k": 0.5, "mrr": 0.2})["passed"] is True
    assert quality_gate({"recall_at_k": float("nan")}, {"recall_at_k": 0.5})["passed"] is False
    assert quality_gate({"recall_at_k": 1.0}, {"recall_at_k": float("nan")})["passed"] is False
    incomplete = evaluate_retrieval([{"retrieved": ["a"], "relevant": ["a", "b"]}], k=2)
    assert incomplete["precision_at_k"] == 0.5
    assert incomplete["ndcg_at_k"] == 0.6131


def test_duplicate_retrieval_hits_do_not_inflate_ranking_metrics() -> None:
    from app.services.operations.quality import evaluate_retrieval

    metrics = evaluate_retrieval(
        [{"retrieved": ["a", "a"], "relevant": ["a"]}], k=2,
    )

    assert metrics["precision_at_k"] == 0.5
    assert metrics["ndcg_at_k"] == 1.0
    assert metrics["effective_candidate_rate"] == 0.5


def test_citation_metadata_and_claim_coverage_are_not_inferred_from_source_count() -> None:
    from app.services.materials.rag.citations import validate_citations

    selected = [{"source_id": "S1", "chunk_id": "c1", "material_id": "m1", "filename": "m.md"}]
    valid = validate_citations("事实 [S1]", selected)
    wrong = validate_citations("事实 [S1]", selected, citations=[{"source_id": "S1", "chunk_id": "other"}])
    malformed = validate_citations("事实 [S1]", selected, citations=[{"source_id": "prefix-S1", "chunk_id": "c1"}])
    assert valid["valid"] is True
    assert valid["citation_coverage"] is None
    assert wrong["metadata_errors"] == ["S1:chunk_id"]
    assert malformed["valid"] is False


def test_reranker_accepts_zero_score_and_restores_coarse_rank_on_failure() -> None:
    from app.services.materials.rag.ranking import rerank_candidates

    candidates = [{"chunk_id": "a", "score": 0.9}, {"chunk_id": "b", "score": 0.8}]
    async def scores(_query, _batch):
        return [{"score": 0.0}, {"score": 1.0}]
    result = asyncio.run(rerank_candidates("q", candidates, reranker=scores))
    assert [row["chunk_id"] for row in result["candidates"]] == ["b", "a"]
    assert result["candidates"][1]["rerank_score"] == 0.0


def test_fixed_synthetic_rag_gate_has_versioned_evidence() -> None:
    from app.services.interviews.eval import run_fixed_rag_eval

    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "rag_fixed_v1.json"
    report = run_fixed_rag_eval(fixture)
    assert report["dataset_version"] == "synthetic-v1"
    assert report["material_index_version"] == "synthetic-v1"
    assert report["metrics"]["case_count"] == 5
    assert report["metric_groups"]["answer_evidence"]["groundedness"] is None
    assert report["metric_groups"]["system"]["sse_completion_rate"] is None
    assert report["metric_groups"]["system"]["token_count"] is None
    assert report["metric_groups"]["citations"]["citation_validity"] is None
    assert report["metrics"]["case_count"] == 5
    assert report["gate"]["passed"] is True


def test_rewrite_timeout_returns_original_with_failure_diagnostic() -> None:
    from app.services.materials.rag.query_rewrite import rewrite_queries

    async def stalled(_query):
        await asyncio.sleep(0.2)
        return {"queries": ["wrong"]}

    result = asyncio.run(rewrite_queries("original", provider=stalled, enabled=True, timeout_seconds=0.001))
    assert result["search_queries"] == ["original"]
    assert result["diagnostics"]["rewrite_failed"] is True


def test_material_metrics_do_not_confuse_chunk_and_material_ids() -> None:
    from app.services.operations.quality import evaluate_retrieval

    row = {"retrieved": [{"chunk_id": "c1", "material_id": "m1"}], "expected_materials": ["m1"]}
    assert evaluate_retrieval([row], k=1)["recall_at_k"] == 1.0
    row = {"retrieved": [{"chunk_id": "m1", "material_id": "other"}], "expected_materials": ["m1"]}
    assert evaluate_retrieval([row], k=1)["recall_at_k"] == 0.0


def test_answer_evidence_requires_explicit_claim_review() -> None:
    from app.services.operations.quality import evaluate_answer_evidence, quality_gate

    unreviewed = evaluate_answer_evidence([{"answer": "事实 [S1]", "valid": True}])
    assert unreviewed["groundedness"] is None
    assert quality_gate(unreviewed, {"groundedness": 0.9})["passed"] is False
    reviewed = evaluate_answer_evidence([
        {"claims": [{"text": "a", "supported": True}, {"text": "b", "supported": False}]},
    ])
    assert reviewed["groundedness"] == 0.5
    assert reviewed["unsupported_claim_rate"] == 0.5
    partly_reviewed = evaluate_answer_evidence([
        {"claims": [{"text": "a", "supported": True}]},
        {"answer": "unreviewed answer"},
    ])
    assert partly_reviewed["groundedness"] is None
    assert evaluate_answer_evidence([{"answer": "unknown", "claims": []}])["groundedness"] is None
    assert quality_gate(reviewed, {"max_unsupported_claim_rate": 0.1})["passed"] is False


def test_system_quality_metrics_cover_latency_failures_cost_and_recovery() -> None:
    from app.services.operations.quality import evaluate_system_metrics

    metrics = evaluate_system_metrics([
        {
            "latency_ms": 100,
            "node_latency_ms": {"route": 10, "answer": 70},
            "provider_failed": False,
            "fallback": False,
            "sse_completed": True,
            "checkpoint_recovered": True,
            "token_count": 120,
            "cost": 0.01,
        },
        {
            "latency_ms": 300,
            "node_latency_ms": {"route": 30, "answer": 210},
            "provider_failed": True,
            "fallback": True,
            "sse_completed": False,
            "checkpoint_recovered": False,
            "token_count": 80,
            "cost": 0.02,
        },
    ])

    assert metrics["p50_latency_ms"] == 200.0
    assert metrics["p95_latency_ms"] == 290.0
    assert metrics["provider_failure_rate"] == 0.5
    assert metrics["fallback_rate"] == 0.5
    assert metrics["sse_completion_rate"] == 0.5
    assert metrics["checkpoint_recovery_rate"] == 0.5
    assert metrics["token_count"] == 200
    assert metrics["estimated_cost"] == 0.03
    assert metrics["node_latency_ms"]["answer"]["p95"] == 203.0


def test_structured_recall_uses_configured_governance_and_context_budgets() -> None:
    from app.core.rag_config import RagConfig
    from app.services.materials.recall import structured_recall

    config = RagConfig.model_validate({
        "recall": {
            "top_k": 5, "fusion": "rrf", "lexical_weight": 0.2,
            "lexical_top_k": 7, "vector_top_k": 9,
            "candidate_k": 11, "max_chunks_per_material": 2,
            "max_materials": 3, "merge_adjacent": False,
        },
        "context": {"max_chunks": 4, "max_tokens": 321, "max_chunk_chars": 87},
    })
    db = MagicMock()
    db.query.return_value.join.return_value.filter.return_value.all.return_value = []
    empty_context = {"selected_chunks": [], "context_text": "", "diagnostics": {}}
    with (
        patch("app.services.materials.recall.get_rag_config", return_value=config),
        patch("app.services.materials.recall.embed_or_empty", new=AsyncMock(return_value=([], ""))),
        patch("app.services.materials.recall.fuse_candidates", return_value=[]) as fuse,
        patch("app.services.materials.recall.govern_candidates", return_value=[]) as govern,
        patch("app.services.materials.recall.build_context", return_value=empty_context) as context,
    ):
        asyncio.run(structured_recall(db, "缓存击穿"))

    assert fuse.call_args.kwargs["strategy"] == "rrf"
    assert fuse.call_args.kwargs["lexical_weight"] == 0.2
    assert govern.call_args.kwargs["max_chunks_per_material"] == 2
    assert govern.call_args.kwargs["max_materials"] == 3
    assert govern.call_args.kwargs["merge_adjacent"] is False
    assert govern.call_args.kwargs["max_candidates"] == 11
    assert context.call_args.kwargs["max_chunks"] == 4
    assert context.call_args.kwargs["max_tokens"] == 321
    assert context.call_args.kwargs["max_chunk_chars"] == 87


def test_staged_retrieval_keeps_vector_evidence_when_lexical_route_fails() -> None:
    from app.services.materials.rag.stages import StagedRetrieval

    pipeline = StagedRetrieval(object(), "缓存击穿")
    pipeline.rows = [{
        "chunk_id": "c1", "material_id": "m1", "filename": "source.md",
        "ordinal": 0, "text": "缓存击穿", "embedding": [1.0],
    }]
    pipeline._rows_loaded = True
    with (
        patch("app.services.materials.rag.stages.lexical_retrieve", side_effect=RuntimeError("lexical unavailable")),
        patch("app.services.materials.rag.stages.embed_or_empty", new=AsyncMock(return_value=([[1.0]], "embed"))),
    ):
        asyncio.run(pipeline.lexical_retrieve())
        asyncio.run(pipeline.vector_retrieve())
    pipeline.candidate_fusion()
    pipeline.deduplicate_and_diversify()
    asyncio.run(pipeline.rerank())
    result = pipeline.context_select()

    assert result["retrieval_status"] == "degraded"
    assert result["diagnostics"]["lexical_failed"] is True
    assert result["diagnostics"]["vector_failed"] is False
    assert result["selected_chunks"][0]["chunk_id"] == "c1"

    unavailable = StagedRetrieval(object(), "缓存击穿")
    unavailable.rows = pipeline.rows
    unavailable._rows_loaded = True
    with (
        patch("app.services.materials.rag.stages.lexical_retrieve", side_effect=RuntimeError("lexical unavailable")),
        patch("app.services.materials.rag.stages.embed_or_empty", side_effect=RuntimeError("vector unavailable")),
    ):
        asyncio.run(unavailable.lexical_retrieve())
        asyncio.run(unavailable.vector_retrieve())
    unavailable.candidate_fusion()
    unavailable.deduplicate_and_diversify()
    asyncio.run(unavailable.rerank())
    empty = unavailable.context_select()
    assert empty["retrieval_status"] == "failed"
    assert empty["diagnostics"]["retrieval_unavailable"] is True


def test_rewrite_uses_bounded_history_without_replacing_original_query() -> None:
    from app.core.rag_config import RagConfig
    from app.services.materials.rag.stages import StagedRetrieval

    cfg = RagConfig.model_validate({"query_rewrite": {
        "enabled": True, "model": "rewrite-model", "base_url": "https://example.test/v1",
    }})
    seen: list[str] = []

    async def rewrite(prompt: str, **_kwargs) -> dict:
        seen.append(prompt)
        return {"queries": ["缓存击穿如何避免"]}

    with patch("app.services.materials.rag.stages.get_rag_config", return_value=cfg), patch(
        "app.services.materials.rag.stages.rewrite_with_provider", new=rewrite,
    ):
        pipeline = StagedRetrieval(object(), "那怎么避免？", history=[
            {"role": "user", "content": "缓存击穿是什么？"},
            {"role": "assistant", "content": "热点 key 过期导致大量请求。"},
        ])
        asyncio.run(pipeline.query_rewrite())

    assert pipeline.rewrite["original_query"] == "那怎么避免？"
    assert pipeline.queries == ["那怎么避免？", "缓存击穿如何避免"]
    assert "缓存击穿是什么" in seen[0]
    assert "当前问题：那怎么避免？" in seen[0]
    assert pipeline.rewrite["diagnostics"]["history_used"] is True
