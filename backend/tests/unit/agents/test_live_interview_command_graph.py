"""Command traces reflect only work performed before the controlled commit."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.agents.workflows.live_interview import LiveInterviewCommandStages
from app.agents.orchestration.graph import AgentState, build_application_graph
from app.agents.orchestration.workflows import run_business_graph


@pytest.mark.parametrize("command,status", [
    ("create", "question_set_ready"),
    ("resume", "ready"),
    ("end", "live"),
    ("abandon", "live"),
])
def test_command_graph_reads_and_decides_before_commit(command: str, status: str) -> None:
    order: list[str] = []
    interview = SimpleNamespace(id="iv-1", status=status)

    def get(*_args):
        order.append("read")
        return interview

    db = SimpleNamespace(get=get)
    stages = LiveInterviewCommandStages(
        db, command, interview_id="iv-1" if command != "create" else None,
        question_set_id="qset-1" if command == "create" else None,
    )
    original_decide = stages.decide_command

    async def decide(state):
        order.append("decide")
        return await original_decide(state)

    stages.decide_command = decide

    async def commit():
        order.append("commit")
        return interview

    async def run():
        return await run_business_graph(
            db, mode="live_interview", interview_id="iv-1", run_id=f"{command}-run",
            stages=stages.callbacks(), stage_sequences=stages.sequences(), action=commit,
        )

    with patch("app.services.interviews.interview.resolve_question_set", side_effect=lambda *_: get()), patch(
        "app.services.interviews.interview.get_interview", side_effect=lambda *_: get()
    ):
        assert asyncio.run(run()) is interview
    assert order == ["read", "decide", "commit"]


def test_command_graph_emits_only_executed_stages() -> None:
    from app.agents.orchestration.graph import AgentState, build_application_graph

    db = SimpleNamespace(get=lambda *_: SimpleNamespace(status="live"))
    stages = LiveInterviewCommandStages(db, "end", interview_id="iv-1")

    async def commit(_state):
        return {"status": "committed", "result": {"valid": True}}

    graph = build_application_graph(
        stages=stages.callbacks(), stage_sequences=stages.sequences(), commit=commit,
    )
    state = AgentState(
        request_id="command-trace", run_id="command-trace", thread_id="interview:iv-1",
        user_id="anonymous", tenant_id="anonymous", requested_mode="live_interview",
    )
    with patch("app.services.interviews.interview.get_interview", return_value=SimpleNamespace(status="live")):
        outcome = asyncio.run(graph.ainvoke(state))
    stage_nodes = [event["node"] for event in outcome["events"]
                   if event["event_type"] == "node_started" and
                   event["node"].startswith("live_interview.")]
    assert stage_nodes == [f"live_interview.{name}" for name in stages.STAGE_NAMES]
    assert not any("persist_answer" in name or "interviewer_followup" in name
                   for name in outcome["completed_nodes"])


@pytest.mark.parametrize("command,status", [
    ("create", "missing"),
    ("resume", "ended"),
    ("end", "abandoned"),
    ("abandon", "missing"),
])
def test_invalid_command_never_reaches_business_commit(command: str, status: str) -> None:
    """A failed read-only precheck must stop before the write callback."""

    calls: list[str] = []
    stages = LiveInterviewCommandStages(
        SimpleNamespace(), command, interview_id="iv-1", question_set_id="qset-1",
    )

    async def commit(_state: AgentState) -> dict:
        calls.append("write")
        return {"status": "committed"}

    graph = build_application_graph(
        stages=stages.callbacks(), stage_sequences=stages.sequences(), commit=commit,
    )
    found = None if status == "missing" else SimpleNamespace(status=status)
    with patch("app.services.interviews.interview.resolve_question_set", return_value=found), patch(
        "app.services.interviews.interview.get_interview", return_value=found,
    ):
        output = asyncio.run(graph.ainvoke(AgentState(
            requested_mode="live_interview", max_retries=0,
        )))
    assert output["status"] == "failed"
    assert output["error"]["error_code"] == "interview_command_invalid"
    assert output["retry_count"] == 0
    assert calls == []
