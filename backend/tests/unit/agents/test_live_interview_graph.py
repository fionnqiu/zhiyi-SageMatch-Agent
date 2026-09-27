"""The live answer graph commits the answer before asking for a follow-up."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.orchestration.checkpoint import MemoryCheckpointer
from app.agents.workflows.live_interview import run_live_answer_graph
from app.agents.orchestration.workflows import GraphExecutionError, run_business_graph
from app.services.interviews.interview import AnswerProgress


def test_live_answer_subgraph_orders_durable_answer_before_followup() -> None:
    calls: list[str] = []
    interview = SimpleNamespace(id="iv-1", status="live", current_question_index=1)
    progress = AnswerProgress(interview, None, None, "q-2", "fallback", True)
    db = SimpleNamespace(get=lambda model, _id: interview if model.__name__ == "Interview" else None,
                         rollback=lambda: None)
    def persist(*_args, **_kwargs):
        calls.append("persist")
        return progress

    async def propose(*_args, **_kwargs):
        calls.append("followup")
        return "next question"

    def finalize(*_args, **_kwargs):
        calls.append("finalize")
        return interview

    with patch("app.services.interviews.interview.persist_interview_answer", side_effect=persist), patch(
        "app.services.interviews.interview.propose_interview_followup", new=AsyncMock(side_effect=propose)
    ), patch("app.services.interviews.interview.finalize_interview_answer", side_effect=finalize):
        result = asyncio.run(run_live_answer_graph(
            db, "iv-1", "candidate answer", "text", "answer-run",
        ))

    assert result is interview
    assert calls == ["persist", "followup", "finalize"]


def test_answer_first_commit_precedes_followup_graph() -> None:
    """The answer writer runs in commit_node, before provider preparation."""
    calls: list[str] = []
    interview = SimpleNamespace(id="iv-1", status="live", current_question_index=0)
    progress = AnswerProgress(interview, None, None, "q-1", "fallback", False)
    db = SimpleNamespace(get=lambda model, _id: interview if model.__name__ == "Interview" else None,
                         rollback=lambda: None)

    def persist(*_args, **_kwargs):
        calls.append("persist")
        return progress

    async def propose(*_args, **_kwargs):
        calls.append("followup")
        return "next question"

    def finalize(*_args, **_kwargs):
        calls.append("finalize")
        return interview

    async def one_graph(*args, **kwargs):
        calls.append("graph")
        assert kwargs["run_id"] == "answer-run"
        assert kwargs["first_commit_after"] == "persist_answer"
        return await run_business_graph(*args, **kwargs)

    with patch("app.agents.workflows.live_interview.run_business_graph", side_effect=one_graph), patch(
        "app.services.interviews.interview.persist_interview_answer", side_effect=persist
    ), patch(
        "app.services.interviews.interview.propose_interview_followup", new=AsyncMock(side_effect=propose)
    ), patch("app.services.interviews.interview.finalize_interview_answer", side_effect=finalize):
        result = asyncio.run(run_live_answer_graph(
            db, "iv-1", "candidate answer", "text", "answer-run",
        ))

    assert result is interview
    assert calls == ["graph", "persist", "followup", "finalize"]


def test_failed_followup_checkpoint_restarts_without_second_answer() -> None:
    """A terminal failed checkpoint can re-enter through the durable answer ID."""
    interview = SimpleNamespace(id="iv-1", status="live", current_question_index=0)
    progress = AnswerProgress(interview, None, None, "q-1", "fallback", False)
    pending = None
    writes = 0

    def get(model, _id):
        return interview if model.__name__ == "Interview" else pending

    db = SimpleNamespace(get=get, commit=lambda: None, rollback=lambda: None)

    def persist(*_args, **_kwargs):
        nonlocal pending, writes
        if pending is None:
            pending = SimpleNamespace(interview_id="iv-1", role="user",
                                      content="answer", answer_mode="text", cite="answer_pending")
            writes += 1
        return progress

    def finalize(*_args, **_kwargs):
        pending.cite = None
        return interview

    async def failed(*_args, **_kwargs):
        raise RuntimeError("provider failed")

    async def recovered(*_args, **_kwargs):
        return "follow-up"

    saver = MemoryCheckpointer()

    async def run():
        return await run_live_answer_graph(
            db, "iv-1", "answer", "text", "answer-run", checkpointer=saver,
        )

    with patch("app.agents.orchestration.workflows.authorize_checkpoint"), patch(
        "app.services.interviews.interview.persist_interview_answer", side_effect=persist
    ), patch("app.services.interviews.interview.finalize_interview_answer", side_effect=finalize), patch(
        "app.services.interviews.interview.propose_interview_followup", new=AsyncMock(side_effect=failed)
    ):
        with pytest.raises(GraphExecutionError):
            asyncio.run(run())

    with patch("app.agents.orchestration.workflows.authorize_checkpoint"), patch(
        "app.services.interviews.interview.persist_interview_answer", side_effect=persist
    ), patch("app.services.interviews.interview.finalize_interview_answer", side_effect=finalize), patch(
        "app.services.interviews.interview.propose_interview_followup", new=AsyncMock(side_effect=recovered)
    ):
        assert asyncio.run(run()) is interview
    assert writes == 1
    assert pending.cite is None
