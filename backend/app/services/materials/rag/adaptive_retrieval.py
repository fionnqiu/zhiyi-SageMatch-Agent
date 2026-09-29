"""Adaptive multi-stage retrieval pipeline with dynamic strategy.

Implements a three-stage approach:
1. Broad recall (宽召回): Cast a wide net with high top_k
2. Rerank and filter (精排过滤): Apply reranker and relevance threshold
3. Supplement recall (补充召回): Query expansion if results are insufficient

This improves recall for complex queries while maintaining precision.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sqlalchemy.orm import Session

from app.core.rag_config import get_rag_config
from app.services.materials.rag.query_understanding import analyze_query
from app.services.materials.rag.retrievers import dual_retrieve
from app.services.materials.rag.ranking import fuse_candidates, govern_candidates
from app.services.materials.rag.query_rewrite import rewrite_queries


class AdaptiveRetrievalPipeline:
    """自适应多阶段检索流水线。

    根据查询理解结果动态调整检索策略：
    - 简单查询：标准召回（top_k=10）
    - 复杂查询：宽召回 + 精排（top_k=20-30）
    - 需要综合的查询：多轮补充召回
    """

    def __init__(self, db: Session):
        self.db = db
        self.cfg = get_rag_config()

    async def retrieve(
        self,
        query: str,
        packed_chunks: list[dict[str, Any]],
        *,
        query_vector: list[float] | None = None,
        reranker: Any | None = None,
    ) -> dict[str, Any]:
        """执行自适应检索流水线"""
        started = time.monotonic()

        # Stage 0: 查询理解
        analysis = analyze_query(query)
        diagnostics: dict[str, Any] = {
            "query_intent": analysis.intent,
            "query_complexity": analysis.complexity,
            "detected_entities": analysis.entities,
            "requires_synthesis": analysis.requires_synthesis,
            "stages_executed": [],
        }

        # 根据复杂度动态调整召回参数
        strategy = self._determine_strategy(analysis)
        diagnostics["retrieval_strategy"] = strategy["name"]

        # Stage 1: 宽召回
        stage1_result = await self._broad_recall(
            query,
            packed_chunks,
            query_vector=query_vector,
            top_k=strategy["broad_top_k"],
        )
        diagnostics["stages_executed"].append("broad_recall")
        diagnostics["stage1_candidates"] = len(stage1_result["candidates"])

        # Stage 2: 精排 + 相关性过滤
        stage2_result = await self._rerank_and_filter(
            query,
            stage1_result["candidates"],
            reranker=reranker,
            threshold=strategy["relevance_threshold"],
        )
        diagnostics["stages_executed"].append("rerank_filter")
        diagnostics["stage2_candidates"] = len(stage2_result["candidates"])

        # Stage 3: 补充召回（如果结果不足）
        final_candidates = stage2_result["candidates"]
        if len(final_candidates) < strategy["min_required"] and analysis.requires_synthesis:
            supplement_result = await self._supplement_recall(
                query,
                packed_chunks,
                existing_candidates=final_candidates,
                entities=analysis.entities,
                query_vector=query_vector,
            )
            final_candidates.extend(supplement_result["candidates"])
            diagnostics["stages_executed"].append("supplement_recall")
            diagnostics["stage3_supplemented"] = len(supplement_result["candidates"])

        # Stage 4: 多样性治理
        governed = govern_candidates(
            final_candidates,
            max_chunks_per_material=strategy["max_per_material"],
            max_materials=self.cfg.recall.max_materials,
            max_candidates=strategy["final_top_k"],
            merge_adjacent=self.cfg.recall.merge_adjacent,
        )

        diagnostics["final_candidates"] = len(governed)
        diagnostics["total_latency_ms"] = round((time.monotonic() - started) * 1000, 2)

        return {
            "candidates": governed,
            "diagnostics": diagnostics,
            "retrieval_status": "ok" if governed else "empty",
        }

    def _determine_strategy(self, analysis: Any) -> dict[str, Any]:
        """根据查询分析确定检索策略"""
        # 默认策略
        strategy = {
            "name": "standard",
            "broad_top_k": 40,
            "relevance_threshold": 0.0,  # 不过滤
            "min_required": 5,
            "max_per_material": 4,
            "final_top_k": 10,
        }

        # 复杂查询：增加召回范围
        if analysis.complexity == "complex":
            strategy.update({
                "name": "wide_recall",
                "broad_top_k": 60,
                "relevance_threshold": 0.5,  # 启用相关性过滤
                "min_required": 8,
                "max_per_material": 5,
                "final_top_k": 15,
            })

        # 需要综合多来源：进一步放宽
        if analysis.requires_synthesis:
            strategy.update({
                "name": "synthesis",
                "broad_top_k": 80,
                "relevance_threshold": 0.4,  # 更宽松的阈值
                "min_required": 10,
                "max_per_material": 6,
                "final_top_k": 20,
            })

        # 简单查询：标准参数即可
        if analysis.complexity == "simple" and not analysis.requires_synthesis:
            strategy.update({
                "name": "efficient",
                "broad_top_k": 30,
                "relevance_threshold": 0.0,
                "min_required": 3,
                "max_per_material": 3,
                "final_top_k": 8,
            })

        return strategy

    async def _broad_recall(
        self,
        query: str,
        packed_chunks: list[dict[str, Any]],
        *,
        query_vector: list[float] | None = None,
        top_k: int = 40,
    ) -> dict[str, Any]:
        """宽召回阶段：最大化召回率"""
        lexical, vector = await asyncio.to_thread(
            dual_retrieve,
            query,
            packed_chunks,
            query_vector=query_vector,
            lexical_top_k=top_k,
            vector_top_k=top_k,
            score_floor=self.cfg.recall.score_floor,
        )

        # 融合两路结果
        fused = fuse_candidates(
            lexical,
            vector,
            strategy=self.cfg.recall.fusion,
            lexical_weight=self.cfg.recall.lexical_weight,
            candidate_k=top_k,
        )

        return {
            "candidates": fused,
            "lexical_count": len(lexical),
            "vector_count": len(vector),
        }

    async def _rerank_and_filter(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        reranker: Any | None = None,
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        """精排 + 相关性过滤阶段"""
        if not reranker or not candidates:
            # 无 reranker 时仅过滤低分
            filtered = [c for c in candidates if c.get("retrieval_score", 0.0) >= threshold]
            return {"candidates": filtered}

        # 调用 reranker
        try:
            from app.services.materials.rag.ranking import rerank_candidates

            reranked_result = await rerank_candidates(
                query,
                candidates,
                reranker=reranker,
                top_n=len(candidates),  # 先全部重排，再过滤
                batch_size=self.cfg.rerank.batch_size,
                timeout_seconds=self.cfg.rerank.timeout_seconds,
                fail_open=True,
            )

            # 应用相关性阈值
            filtered = [
                c for c in reranked_result["candidates"]
                if c.get("rerank_score", c.get("retrieval_score", 0.0)) >= threshold
            ]

            return {"candidates": filtered}

        except Exception:  # noqa: BLE001
            # Reranker 失败时回退到原始排序
            filtered = [c for c in candidates if c.get("retrieval_score", 0.0) >= threshold]
            return {"candidates": filtered}

    async def _supplement_recall(
        self,
        query: str,
        packed_chunks: list[dict[str, Any]],
        *,
        existing_candidates: list[dict[str, Any]],
        entities: list[str],
        query_vector: list[float] | None = None,
    ) -> dict[str, Any]:
        """补充召回阶段：查询扩展 + 实体增强"""
        # 使用实体构建补充查询
        supplement_queries = self._build_supplement_queries(query, entities)

        all_supplements = []
        for sup_query in supplement_queries[:2]:  # 最多 2 个补充查询
            lexical, vector = await asyncio.to_thread(
                dual_retrieve,
                sup_query,
                packed_chunks,
                query_vector=query_vector,
                lexical_top_k=20,
                vector_top_k=20,
                score_floor=self.cfg.recall.score_floor,
            )

            fused = fuse_candidates(
                lexical,
                vector,
                strategy=self.cfg.recall.fusion,
                lexical_weight=self.cfg.recall.lexical_weight,
                candidate_k=20,
            )
            all_supplements.extend(fused)

        # 去重：排除已有的 chunk
        existing_ids = {c.get("chunk_id") for c in existing_candidates}
        new_candidates = [
            c for c in all_supplements
            if c.get("chunk_id") not in existing_ids
        ]

        # 按分数排序，取 top 5
        new_candidates.sort(key=lambda x: -x.get("retrieval_score", 0.0))

        return {"candidates": new_candidates[:5]}

    def _build_supplement_queries(self, query: str, entities: list[str]) -> list[str]:
        """构建补充查询"""
        supplements = []

        # 策略 1：仅保留实体关键词
        if entities:
            entity_query = " ".join(entities[:3])
            supplements.append(entity_query)

        # 策略 2：移除疑问词，保留核心内容
        core_query = query
        for q_word in ["如何", "怎么", "什么是", "为什么", "是什么", "？", "?"]:
            core_query = core_query.replace(q_word, "")
        core_query = core_query.strip()
        if core_query and core_query != query:
            supplements.append(core_query)

        return supplements


__all__ = ["AdaptiveRetrievalPipeline"]
