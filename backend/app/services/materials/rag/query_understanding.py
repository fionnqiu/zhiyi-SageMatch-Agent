"""Query understanding layer for adaptive retrieval strategy.

Analyzes query intent, extracts entities, and determines complexity before retrieval.
This guides the multi-stage retrieval pipeline with contextual decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session


@dataclass(frozen=True)
class QueryAnalysis:
    """Structured query understanding result."""

    # 意图类型：事实查询、流程说明、对比分析、多跳推理
    intent: str  # "factual" | "procedural" | "comparative" | "multi_hop"

    # 提取的关键实体（技术名词、框架名称等）
    entities: list[str]

    # 时间范围偏好
    temporal_scope: str  # "recent" | "all" | "historical"

    # 查询复杂度
    complexity: str  # "simple" | "moderate" | "complex"

    # 是否需要多文档综合
    requires_synthesis: bool

    # 置信度
    confidence: float


def analyze_query(query: str) -> QueryAnalysis:
    """确定性查询分析，无需调用模型。

    使用规则和模式匹配快速判断查询特征，为后续检索提供策略指导。
    这一步不消耗 token，且结果可重现用于评测。
    """
    query_lower = query.lower()

    # 意图识别
    intent = _detect_intent(query_lower)

    # 实体抽取（技术术语、框架名称等）
    entities = _extract_entities(query)

    # 时间偏好
    temporal_scope = _detect_temporal_scope(query_lower)

    # 复杂度判断
    complexity = _assess_complexity(query, entities)

    # 是否需要综合多个来源
    requires_synthesis = _needs_synthesis(query_lower, intent)

    # 置信度（基于识别规则的匹配程度）
    confidence = _calculate_confidence(query, intent, entities)

    return QueryAnalysis(
        intent=intent,
        entities=entities,
        temporal_scope=temporal_scope,
        complexity=complexity,
        requires_synthesis=requires_synthesis,
        confidence=confidence,
    )


def _detect_intent(query_lower: str) -> str:
    """检测查询意图类型"""
    # 对比分析：包含"区别"、"对比"、"vs"等关键词
    if any(kw in query_lower for kw in ["区别", "对比", "比较", "差异", "vs", "和...的不同"]):
        return "comparative"

    # 流程说明：包含"如何"、"步骤"、"流程"等
    if any(kw in query_lower for kw in ["如何", "怎么", "步骤", "流程", "过程", "方法", "实现"]):
        return "procedural"

    # 多跳推理：包含"为什么"、"原因"、"影响"等
    if any(kw in query_lower for kw in ["为什么", "原因", "影响", "导致", "解决", "优化", "改进"]):
        return "multi_hop"

    # 默认：事实查询
    return "factual"


def _extract_entities(query: str) -> list[str]:
    """提取技术实体（框架、语言、工具等）"""
    entities = []

    # 常见技术栈关键词（可扩展）
    tech_keywords = [
        # 编程语言
        r"\bPython\b", r"\bJava\b", r"\bJavaScript\b", r"\bTypeScript\b",
        r"\bGo\b", r"\bRust\b", r"\bC\+\+\b", r"\bC#\b",

        # Web 框架
        r"\bFastAPI\b", r"\bDjango\b", r"\bFlask\b", r"\bSpring\b",
        r"\bExpress\b", r"\bReact\b", r"\bVue\b", r"\bNext\.js\b",

        # 数据库
        r"\bPostgreSQL\b", r"\bMySQL\b", r"\bMongoDB\b", r"\bRedis\b",

        # AI/ML
        r"\bLangChain\b", r"\bLangGraph\b", r"\bOpenAI\b", r"\bClaude\b",
        r"\bRAG\b", r"\bembedding\b", r"\brerank\b",

        # 工具
        r"\bDocker\b", r"\bKubernetes\b", r"\bGit\b", r"\bNginx\b",
    ]

    for pattern in tech_keywords:
        matches = re.findall(pattern, query, re.IGNORECASE)
        entities.extend(matches)

    # 去重并保留原始大小写
    seen = set()
    unique_entities = []
    for entity in entities:
        if entity.lower() not in seen:
            seen.add(entity.lower())
            unique_entities.append(entity)

    return unique_entities


def _detect_temporal_scope(query_lower: str) -> str:
    """检测时间范围偏好"""
    if any(kw in query_lower for kw in ["最新", "新版", "当前", "现在", "最近"]):
        return "recent"

    if any(kw in query_lower for kw in ["历史", "过去", "旧版", "之前"]):
        return "historical"

    return "all"


def _assess_complexity(query: str, entities: list[str]) -> str:
    """评估查询复杂度"""
    # 字数 + 实体数量 + 句子结构
    word_count = len(query)
    entity_count = len(entities)
    sentence_count = query.count("？") + query.count("?") + query.count("。") + 1

    # 复杂度评分
    score = 0
    if word_count > 50:
        score += 2
    elif word_count > 20:
        score += 1

    if entity_count > 3:
        score += 2
    elif entity_count > 1:
        score += 1

    if sentence_count > 2:
        score += 1

    # 包含嵌套问题
    if "以及" in query or "同时" in query or "另外" in query:
        score += 1

    if score >= 4:
        return "complex"
    elif score >= 2:
        return "moderate"
    else:
        return "simple"


def _needs_synthesis(query_lower: str, intent: str) -> bool:
    """判断是否需要综合多个文档来源"""
    # 对比类和多跳推理类通常需要综合
    if intent in {"comparative", "multi_hop"}:
        return True

    # 包含"综合"、"整体"、"全面"等词
    if any(kw in query_lower for kw in ["综合", "整体", "全面", "各个", "多种", "所有"]):
        return True

    return False


def _calculate_confidence(query: str, intent: str, entities: list[str]) -> float:
    """计算分析置信度"""
    confidence = 0.5  # 基础置信度

    # 查询长度适中，增加置信度
    if 10 <= len(query) <= 100:
        confidence += 0.1

    # 识别到实体，增加置信度
    if entities:
        confidence += min(0.2, len(entities) * 0.05)

    # 意图识别清晰（非默认 factual），增加置信度
    if intent != "factual":
        confidence += 0.1

    # 查询包含标点和结构化内容
    if any(p in query for p in ["？", "?", "，", ","]):
        confidence += 0.1

    return min(1.0, confidence)


async def analyze_with_context(
    query: str,
    db: Session,
    recent_queries: list[str] | None = None
) -> dict[str, Any]:
    """带上下文的查询分析（可选，用于会话级优化）

    结合最近的查询历史，判断当前查询是否为追问或延续。
    """
    analysis = analyze_query(query)

    # 判断是否为追问
    is_followup = False
    if recent_queries:
        # 简单启发式：查询很短且包含指代词
        if len(query) < 20 and any(ref in query for ref in ["这个", "那个", "它", "这些", "那些"]):
            is_followup = True

    return {
        "intent": analysis.intent,
        "entities": analysis.entities,
        "temporal_scope": analysis.temporal_scope,
        "complexity": analysis.complexity,
        "requires_synthesis": analysis.requires_synthesis,
        "confidence": analysis.confidence,
        "is_followup": is_followup,
    }


__all__ = ["QueryAnalysis", "analyze_query", "analyze_with_context"]
