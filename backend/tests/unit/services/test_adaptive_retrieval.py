"""Tests for adaptive retrieval pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.materials.rag.adaptive_retrieval import AdaptiveRetrievalPipeline
from app.services.materials.rag.query_understanding import QueryAnalysis

pytestmark = pytest.mark.anyio


class MockDB:
    """模拟数据库会话"""
    pass


def mock_chunk_dict(id: str, content: str, score: float, material_id: str) -> dict:
    """创建模拟的 chunk 字典"""
    return {
        "chunk_id": id,
        "content": content,
        "retrieval_score": score,
        "material_id": material_id,
    }


async def test_strategy_determination_simple():
    """测试简单查询的策略选择"""
    db = MockDB()
    pipeline = AdaptiveRetrievalPipeline(db)

    analysis = QueryAnalysis(
        intent="factual",
        entities=["Python"],
        temporal_scope="all",
        complexity="simple",
        requires_synthesis=False,
        confidence=0.9,
    )

    strategy = pipeline._determine_strategy(analysis)

    assert strategy["name"] == "efficient"
    assert strategy["broad_top_k"] == 30
    assert strategy["final_top_k"] == 8


async def test_strategy_determination_complex():
    """测试复杂查询的策略选择"""
    db = MockDB()
    pipeline = AdaptiveRetrievalPipeline(db)

    analysis = QueryAnalysis(
        intent="comparative",
        entities=["FastAPI", "Django"],
        temporal_scope="all",
        complexity="complex",
        requires_synthesis=False,
        confidence=0.85,
    )

    strategy = pipeline._determine_strategy(analysis)

    assert strategy["name"] == "wide_recall"
    assert strategy["broad_top_k"] == 60
    assert strategy["relevance_threshold"] == 0.5
    assert strategy["final_top_k"] == 15


async def test_strategy_determination_synthesis():
    """测试需要综合的查询策略"""
    db = MockDB()
    pipeline = AdaptiveRetrievalPipeline(db)

    analysis = QueryAnalysis(
        intent="procedural",
        entities=["微服务", "部署"],
        temporal_scope="all",
        complexity="moderate",
        requires_synthesis=True,
        confidence=0.8,
    )

    strategy = pipeline._determine_strategy(analysis)

    assert strategy["name"] == "synthesis"
    assert strategy["broad_top_k"] == 80
    assert strategy["min_required"] == 10
    assert strategy["max_per_material"] == 6


async def test_broad_recall_fusion():
    """测试宽召回的融合逻辑"""
    db = MockDB()
    pipeline = AdaptiveRetrievalPipeline(db)

    # Mock 打包的 chunks
    packed_chunks = [
        {"id": f"chunk_{i}", "content": f"content {i}", "vector": [0.1] * 768}
        for i in range(50)
    ]

    # Mock dual_retrieve 返回
    with patch("app.services.materials.rag.adaptive_retrieval.dual_retrieve") as mock_dual:
        mock_dual.return_value = (
            [mock_chunk_dict(f"lex_{i}", f"lexical {i}", 0.8, "mat_1") for i in range(10)],
            [mock_chunk_dict(f"vec_{i}", f"vector {i}", 0.75, "mat_2") for i in range(10)],
        )

        with patch("app.services.materials.rag.adaptive_retrieval.fuse_candidates") as mock_fuse:
            mock_fuse.return_value = [
                mock_chunk_dict(f"fused_{i}", f"content {i}", 0.85, "mat_1")
                for i in range(15)
            ]

            result = await pipeline._broad_recall(
                "test query",
                packed_chunks,
                query_vector=None,
                top_k=40,
            )

            assert "candidates" in result
            assert len(result["candidates"]) == 15
            assert result["lexical_count"] == 10
            assert result["vector_count"] == 10


async def test_rerank_and_filter_with_threshold():
    """测试重排和过滤"""
    db = MockDB()
    pipeline = AdaptiveRetrievalPipeline(db)

    candidates = [
        mock_chunk_dict(f"chunk_{i}", f"content {i}", 0.5 + i * 0.05, "mat_1")
        for i in range(20)
    ]

    # Mock reranker
    mock_reranker = MagicMock()

    with patch("app.services.materials.rag.ranking.rerank_candidates") as mock_rerank:
        # 模拟 reranker 返回：给每个候选加上 rerank_score
        reranked = []
        for i, c in enumerate(candidates):
            c_copy = c.copy()
            c_copy["rerank_score"] = 0.7 if i < 10 else 0.4
            reranked.append(c_copy)

        mock_rerank.return_value = {"candidates": reranked}

        result = await pipeline._rerank_and_filter(
            "test query",
            candidates,
            reranker=mock_reranker,
            threshold=0.6,
        )

        # 应该只保留 rerank_score >= 0.6 的
        assert len(result["candidates"]) == 10
        assert all(c.get("rerank_score", 0) >= 0.6 for c in result["candidates"])


async def test_supplement_recall_deduplication():
    """测试补充召回的去重逻辑"""
    db = MockDB()
    pipeline = AdaptiveRetrievalPipeline(db)

    existing_candidates = [
        mock_chunk_dict("chunk_1", "existing 1", 0.9, "mat_1"),
        mock_chunk_dict("chunk_2", "existing 2", 0.85, "mat_1"),
    ]

    packed_chunks = [{"id": f"chunk_{i}", "content": f"content {i}"} for i in range(30)]

    with patch("app.services.materials.rag.adaptive_retrieval.dual_retrieve") as mock_dual:
        # 返回包含重复的结果
        mock_dual.return_value = (
            [
                mock_chunk_dict("chunk_1", "duplicate", 0.8, "mat_1"),  # 重复
                mock_chunk_dict("chunk_3", "new 1", 0.75, "mat_2"),
            ],
            [
                mock_chunk_dict("chunk_4", "new 2", 0.7, "mat_3"),
            ],
        )

        with patch("app.services.materials.rag.adaptive_retrieval.fuse_candidates") as mock_fuse:
            mock_fuse.return_value = [
                mock_chunk_dict("chunk_1", "duplicate", 0.8, "mat_1"),
                mock_chunk_dict("chunk_3", "new 1", 0.75, "mat_2"),
                mock_chunk_dict("chunk_4", "new 2", 0.7, "mat_3"),
            ]

            result = await pipeline._supplement_recall(
                "test query",
                packed_chunks,
                existing_candidates=existing_candidates,
                entities=["Python"],
                query_vector=None,
            )

            # chunk_1 应该被去重
            chunk_ids = [c["chunk_id"] for c in result["candidates"]]
            assert "chunk_1" not in chunk_ids
            assert "chunk_3" in chunk_ids
            assert "chunk_4" in chunk_ids


async def test_build_supplement_queries():
    """测试补充查询构建"""
    db = MockDB()
    pipeline = AdaptiveRetrievalPipeline(db)

    query = "如何使用 FastAPI 实现异步接口？"
    entities = ["FastAPI", "异步接口", "Python"]

    supplements = pipeline._build_supplement_queries(query, entities)

    # 应该生成实体查询
    assert any("FastAPI" in s for s in supplements)

    # 应该生成去除疑问词的查询
    assert any("如何" not in s and "FastAPI" in s for s in supplements)


async def test_full_pipeline_integration():
    """测试完整检索流水线"""
    db = MockDB()
    pipeline = AdaptiveRetrievalPipeline(db)

    packed_chunks = [{"id": f"chunk_{i}", "content": f"content {i}"} for i in range(100)]

    # Mock 查询理解
    with patch("app.services.materials.rag.adaptive_retrieval.analyze_query") as mock_analyze:
        mock_analyze.return_value = QueryAnalysis(
            intent="factual",
            entities=["Python"],
            temporal_scope="all",
            complexity="simple",
            requires_synthesis=False,
            confidence=0.9,
        )

        # Mock 宽召回
        with patch.object(pipeline, "_broad_recall") as mock_broad:
            mock_broad.return_value = {
                "candidates": [mock_chunk_dict(f"c_{i}", f"content {i}", 0.8, "mat_1") for i in range(15)],
                "lexical_count": 10,
                "vector_count": 10,
            }

            # Mock 重排
            with patch.object(pipeline, "_rerank_and_filter") as mock_rerank:
                mock_rerank.return_value = {
                    "candidates": [mock_chunk_dict(f"c_{i}", f"content {i}", 0.8, "mat_1") for i in range(10)],
                }

                # Mock govern_candidates
                with patch("app.services.materials.rag.adaptive_retrieval.govern_candidates") as mock_govern:
                    mock_govern.return_value = [
                        mock_chunk_dict(f"c_{i}", f"content {i}", 0.8, "mat_1") for i in range(8)
                    ]

                    result = await pipeline.retrieve(
                        "什么是 Python GIL？",
                        packed_chunks,
                        query_vector=None,
                        reranker=None,
                    )

                    # 验证返回结构
                    assert "candidates" in result
                    assert "diagnostics" in result
                    assert "retrieval_status" in result

                    # 验证诊断信息
                    diagnostics = result["diagnostics"]
                    assert "query_intent" in diagnostics
                    assert "query_complexity" in diagnostics
                    assert "stages_executed" in diagnostics
                    assert "broad_recall" in diagnostics["stages_executed"]
                    assert "rerank_filter" in diagnostics["stages_executed"]

                    # 验证状态
                    assert result["retrieval_status"] == "ok"
                    assert len(result["candidates"]) == 8

