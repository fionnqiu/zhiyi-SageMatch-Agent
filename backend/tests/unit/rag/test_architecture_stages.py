"""Regression tests for the five-stage Agent/RAG architecture migration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import asyncio


def test_route_contract_uses_explicit_business_modes() -> None:
    from app.agents.contracts.state import RouteDecision

    decision = RouteDecision(route="knowledge_qa", confidence=0.9, source="test")

    assert decision.route == "knowledge_qa"
    assert decision.model_dump()["original_query"] == ""


def test_retrieval_result_has_structured_diagnostics() -> None:
    from app.agents.contracts.state import RetrievalResult

    result = RetrievalResult(original_query="缓存", selected_chunks=[])

    assert result.retrieval_status == "empty"
    assert result.diagnostics["selected_count"] == 0


def test_agent_result_envelope_normalizes_error_fields() -> None:
    from app.agents.contracts.state import AgentResult

    result = AgentResult.failure("agent_timeout", retryable=True, trace_id="trace-1")

    assert result.ok is False
    assert result.error_code == "agent_timeout"
    assert result.retryable is True
    assert result.trace_id == "trace-1"


def test_knowledge_answer_retrieves_even_when_legacy_flag_is_false() -> None:
    from app.services.chat.session import prepare_direct_answer

    db = SimpleNamespace()
    # prepare_direct_answer now renders long-term memory through MemoryManager,
    # which reads session.id/user_id and queries the db; the fake session carries
    # the identity fields and the manager is stubbed for this retrieval-only test.
    session = SimpleNamespace(id="session-rag", user_id="local-user", messages=[])
    hits = [{"chunk_id": "c1", "material_id": "m1", "filename": "x.md", "ordinal": 1, "text": "命中"}]

    class _NullMemoryManager:
        def __init__(self, *_args, **_kwargs):
            pass

        def render(self, *_args, **_kwargs):
            return ""

    with (
        patch("app.services.chat.session.recall_snippets", new=AsyncMock(return_value=hits)) as recall,
        patch("app.services.chat.session.MemoryManager", _NullMemoryManager),
    ):
        prompt, extra = asyncio.run(prepare_direct_answer(
            db,
            session,
            "什么是缓存？",
            {"intent": "answer", "needs_recall": False, "todos": []},
        ))

    recall.assert_awaited_once()
    assert "命中" in prompt["user"]
    assert extra["retrieval"]["diagnostics"]["selected_count"] == 1
