"""Tests for query understanding module."""

import pytest

from app.services.materials.rag.query_understanding import analyze_query


def test_intent_detection_comparative():
    """测试对比类意图识别"""
    query = "FastAPI 和 Django 的区别是什么？"
    analysis = analyze_query(query)

    assert analysis.intent == "comparative"
    assert "FastAPI" in analysis.entities
    assert "Django" in analysis.entities


def test_intent_detection_procedural():
    """测试流程类意图识别"""
    query = "如何使用 LangGraph 实现 Agent 编排？"
    analysis = analyze_query(query)

    assert analysis.intent == "procedural"
    assert "LangGraph" in analysis.entities


def test_intent_detection_multi_hop():
    """测试多跳推理类意图识别"""
    query = "为什么 PostgreSQL 的 checkpoint 会影响性能？"
    analysis = analyze_query(query)

    assert analysis.intent == "multi_hop"
    assert "PostgreSQL" in analysis.entities


def test_intent_detection_factual():
    """测试事实查询意图识别"""
    query = "Python 3.11 有哪些新特性？"
    analysis = analyze_query(query)

    assert analysis.intent == "factual"
    assert "Python" in analysis.entities


def test_entity_extraction():
    """测试实体抽取"""
    query = "用 React 和 TypeScript 开发前端，后端使用 FastAPI 和 PostgreSQL"
    analysis = analyze_query(query)

    expected_entities = {"React", "TypeScript", "FastAPI", "PostgreSQL"}
    extracted = set(analysis.entities)

    assert expected_entities.issubset(extracted)


def test_temporal_scope_recent():
    """测试时间范围：最近"""
    query = "最新的 LangChain 版本支持哪些功能？"
    analysis = analyze_query(query)

    assert analysis.temporal_scope == "recent"


def test_temporal_scope_all():
    """测试时间范围：全部"""
    query = "Python 的特性有哪些？"
    analysis = analyze_query(query)

    assert analysis.temporal_scope == "all"


def test_complexity_simple():
    """测试简单查询"""
    query = "什么是 RAG？"
    analysis = analyze_query(query)

    assert analysis.complexity == "simple"


def test_complexity_moderate():
    """测试中等复杂度查询"""
    query = "如何使用 LangChain 实现 RAG 检索增强生成？"
    analysis = analyze_query(query)

    assert analysis.complexity in {"moderate", "complex"}


def test_complexity_complex():
    """测试复杂查询"""
    query = """在多 Agent 系统中，如何设计 checkpoint 机制来保证状态一致性，
    同时支持并发执行和故障恢复？另外，如何处理不同 Agent 之间的通信和协调？"""
    analysis = analyze_query(query)

    assert analysis.complexity == "complex"


def test_requires_synthesis():
    """测试是否需要综合多来源"""
    query = "对比 FastAPI、Django、Flask 的性能和适用场景"
    analysis = analyze_query(query)

    assert analysis.requires_synthesis is True


def test_no_synthesis_needed():
    """测试不需要综合"""
    query = "FastAPI 的安装步骤"
    analysis = analyze_query(query)

    assert analysis.requires_synthesis is False


def test_confidence_calculation():
    """测试置信度计算"""
    query = "如何使用 PostgreSQL 和 Redis 构建缓存层？"
    analysis = analyze_query(query)

    # 有实体、有意图、有结构，置信度应该较高
    assert analysis.confidence > 0.6


def test_empty_query():
    """测试空查询"""
    query = ""
    analysis = analyze_query(query)

    # 空查询应该有基本的返回值
    assert analysis.intent == "factual"
    assert analysis.entities == []
    assert analysis.confidence < 0.6


def test_case_insensitive_entity_extraction():
    """测试大小写不敏感的实体抽取"""
    query = "fastapi 和 FASTAPI 以及 FastAPI"
    analysis = analyze_query(query)

    # 应该只保留一个 FastAPI（去重）
    fastapi_count = sum(1 for e in analysis.entities if e.lower() == "fastapi")
    assert fastapi_count == 1
