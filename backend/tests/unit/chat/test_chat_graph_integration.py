"""Production chat adapter contract for one graph run and one business commit."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


def test_staged_retrieval_builds_context_only_after_real_stage_work() -> None:
    """The production stage helper carries candidate evidence through every node."""
    from app.services.materials.rag.stages import StagedRetrieval

    chunk = SimpleNamespace(id="c1", material_id="m1", ordinal=0, text="缓存击穿是热点键失效。", embedding=None)
    material = SimpleNamespace(id="m1", filename="cache.md")
    query = SimpleNamespace()
    query.join = lambda *args: query
    query.filter = lambda *args: query
    query.all = lambda: [(chunk, material)]
    db = SimpleNamespace(query=lambda *args: query)

    async def run():
        pipeline = StagedRetrieval(db, "缓存击穿")
        await pipeline.query_rewrite()
        await pipeline.retrieve_parallel()
        pipeline.candidate_fusion()
        pipeline.deduplicate_and_diversify()
        await pipeline.rerank()
        return pipeline.context_select()

    with patch("app.services.materials.rag.stages.embed_or_empty", new=AsyncMock(return_value=([], ""))):
        result = asyncio.run(run())

    assert result["selected_chunks"][0]["chunk_id"] == "c1"
    assert result["citations"][0]["source_id"] == "S1"
    assert result["diagnostics"]["lexical_count"] == 1
    assert set(result["diagnostics"]["stage_latency_ms"]) >= {
        "query_rewrite", "parallel_retrieve", "lexical_retrieve", "vector_retrieve", "candidate_fusion",
        "deduplicate_and_diversify", "rerank", "context_select",
    }


def test_nonstream_chat_uses_one_route_and_commits_once() -> None:
    """The route decision is reused by begin_chat and the graph commit writes once."""
    from app.services.chat.session import send_chat_graph

    session = SimpleNamespace(id="session-1")
    turn = {
        "session": session,
        "mode": "answer",
        "run_id": "run-1",
        "route": {"route": "knowledge_qa"},
    }
    routed = {
        "run_id": "run-1",
        "mode": "stream",
        "content": "缓存击穿",
        "intent": {"intent": "answer", "needs_recall": False},
        "route": {"route": "knowledge_qa", "confidence": 1.0, "source": "test", "original_query": "缓存击穿"},
    }
    with (
        patch("app.services.chat.session.route_chat", new=AsyncMock(return_value=routed)) as route,
        patch("app.services.chat.session.begin_chat", new=AsyncMock(return_value=turn)) as begin,
        patch("app.services.chat.session.complete_chat_turn", new=AsyncMock(return_value=("说明 [S1]", {"retrieval": {}}))) as complete,
        patch("app.services.chat.session.finish_chat", new=AsyncMock(return_value=session)) as finish,
    ):
        result = asyncio.run(send_chat_graph(object(), "缓存击穿", "session-1"))

    assert result is session
    route.assert_awaited_once()
    begin.assert_awaited_once()
    complete.assert_awaited_once()
    finish.assert_awaited_once()
    assert begin.await_args.kwargs["precomputed_intent"] is routed["intent"]
    assert begin.await_args.kwargs["run_id"] == "run-1"
    assert begin.await_args.kwargs["defer_retrieval"] is True


def test_clarification_waiting_branch_commits_without_rerouting() -> None:
    """A waiting clarification still persists once through the graph commit."""
    from app.services.chat.session import send_chat_graph

    session = SimpleNamespace(id="session-2")
    routed = {
        "run_id": "run-2", "mode": "blocked", "content": "岗位不明确",
        "intent": {"intent": "clarify", "needs_recall": False},
        "route": {"route": "clarification", "confidence": 1.0, "source": "test", "original_query": "岗位不明确"},
    }
    turn = {"session": session, "mode": "clarify", "run_id": "run-2", "route": routed["route"]}
    with (
        patch("app.services.chat.session.route_chat", new=AsyncMock(return_value=routed)) as route,
        patch("app.services.chat.session.begin_chat", new=AsyncMock(return_value=turn)) as begin,
        patch("app.services.chat.session.complete_chat_turn", new=AsyncMock(return_value=("请明确岗位", {}))),
        patch("app.services.chat.session.finish_chat", new=AsyncMock(return_value=session)) as finish,
    ):
        result = asyncio.run(send_chat_graph(object(), "岗位不明确", "session-2"))

    assert result is session
    route.assert_awaited_once()
    begin.assert_awaited_once()
    finish.assert_awaited_once()


def test_nonstream_knowledge_answer_repairs_citations_before_commit() -> None:
    """Graph citation stages run once before the assistant fact is written."""
    from app.services.chat.session import send_chat_graph

    session = SimpleNamespace(id="session-3")
    routed = {
        "run_id": "run-3", "mode": "stream", "content": "缓存",
        "intent": {"intent": "answer", "needs_recall": False},
        "route": {"route": "knowledge_qa", "confidence": 1.0, "source": "test", "original_query": "缓存"},
    }
    turn = {"session": session, "mode": "answer", "run_id": "run-3"}
    selected = [{"source_id": "S1", "chunk_id": "c1", "material_id": "m1", "filename": "a.md", "text": "缓存定义"}]
    extra = {"retrieval": {"selected_chunks": selected}}
    with (
        patch("app.services.chat.session.route_chat", new=AsyncMock(return_value=routed)),
        patch("app.services.chat.session.begin_chat", new=AsyncMock(return_value=turn)),
        patch("app.services.chat.session.complete_chat_turn", new=AsyncMock(return_value=("错误引用 [S9]", extra))) as complete_turn,
        patch("app.services.chat.session.complete", new=AsyncMock(return_value="修正引用 [S1]")) as repair,
        patch("app.services.chat.session.finish_chat", new=AsyncMock(return_value=session)) as finish,
    ):
        result = asyncio.run(send_chat_graph(object(), "缓存", "session-3"))

    assert result is session
    complete_turn.assert_awaited_once()
    repair.assert_awaited_once()
    finish.assert_awaited_once()
    assert finish.await_args.args[2] == "修正引用 [S1]"
    assert finish.await_args.args[3]["citation_validation"]["valid"] is True
    assert finish.await_args.kwargs["citation_checked"] is True


def test_stream_followup_repairs_citations_from_selected_evidence() -> None:
    """A follow-up answer uses the same one-attempt citation rule as a first answer."""
    from app.services.chat.session import repair_stream_answer, validate_stream_answer

    selected = [{"source_id": "S1", "chunk_id": "c1", "material_id": "m1", "filename": "a.md", "text": "缓存定义"}]
    extra = {"retrieval": {"selected_chunks": selected}}
    turn = {"mode": "followup"}
    validation = validate_stream_answer(turn, "错误引用 [S9]", extra)
    assert validation is not None and validation["valid"] is False
    with patch("app.services.chat.session.complete", new=AsyncMock(return_value="修正引用 [S1]")) as repair:
        reply, repaired_extra = asyncio.run(repair_stream_answer(object(), turn, "错误引用 [S9]", extra, validation))
    assert reply == "修正引用 [S1]"
    assert repaired_extra["citation_validation"]["valid"] is True
    repair.assert_awaited_once()
