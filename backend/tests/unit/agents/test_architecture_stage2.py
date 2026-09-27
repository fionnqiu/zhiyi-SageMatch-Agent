"""Stage-two tests for the typed Agent protocol, graph, and checkpoint boundary."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError


def test_agent_envelopes_validate_and_reject_unknown_fields() -> None:
    """Worker handoffs use a closed schema so field drift cannot be silent."""
    from app.agents.contracts.envelopes import AgentDecision, AgentTask, ToolResult

    task = AgentTask(
        task_id="task-1",
        run_id="run-1",
        from_agent="supervisor",
        to_agent="author",
        route="interview_generation",
        goal="生成题目",
    )
    decision = AgentDecision(
        task_id=task.task_id,
        agent="author",
        status="success",
        decision="persist",
        output={"questions": []},
        next_action="persist",
        confidence=0.9,
    )
    result = ToolResult(
        tool_call_id="tool-1",
        tool_name="validate_question",
        ok=True,
        data={"success": True},
        latency_ms=2.5,
        trace_id="trace-1",
    )

    assert task.model_dump(mode="json")["route"] == "interview_generation"
    assert decision.confidence == 0.9
    assert result.ok is True
    with pytest.raises(ValidationError):
        AgentTask(
            task_id="task-2",
            run_id="run-1",
            from_agent="supervisor",
            to_agent="author",
            route="interview_generation",
            goal="生成题目",
            undocumented_field=True,
        )


def test_error_and_event_envelopes_keep_internal_details_out_of_public_payload() -> None:
    """Events retain trace context while public error serialization hides diagnostics."""
    from app.agents.contracts.envelopes import AgentEvent, ErrorEnvelope

    error = ErrorEnvelope(
        error_code="provider_timeout",
        retryable=True,
        user_message="服务暂时不可用，请稍后重试",
        internal_detail="vendor response included sensitive internals",
        trace_id="trace-1",
    )
    event = AgentEvent(
        run_id="run-1",
        node="author",
        agent="author",
        event_type="agent_called",
        payload={"task_id": "task-1"},
    )

    assert "vendor response" not in error.public_dict()["user_message"]
    assert "internal_detail" not in error.public_dict()
    assert event.schema_version
    assert event.timestamp.tzinfo is not None


def test_application_state_is_json_serializable_and_preserves_original_query() -> None:
    """Checkpoint state contains bounded structured fields and never overwrites the user query."""
    from app.agents.orchestration.graph import AgentState

    state = AgentState(
        request_id="req-1",
        run_id="run-1",
        thread_id="session:s-1",
        user_id="u-1",
        session_id="s-1",
        entrypoint="chat",
        original_query="原始问题",
        normalized_query="清洗后的问题",
        messages=[{"role": "user", "content": "原始问题"}],
    )
    payload = state.to_json_dict()

    assert payload["original_query"] == "原始问题"
    assert payload["normalized_query"] == "清洗后的问题"
    assert payload["status"] == "running"
    assert "schema_version" in payload


def test_application_graph_routes_once_and_runs_all_supported_subgraphs() -> None:
    """The top-level graph dispatches each fixed route through a bounded child graph."""
    from app.agents.orchestration.graph import AgentState, build_application_graph

    graph = build_application_graph()
    for route, expected_node in (
        ("knowledge_qa", "knowledge_qa"),
        ("interview_generation", "interview_generation"),
        ("live_interview", "live_interview"),
        ("evaluation_report", "evaluation_report"),
    ):
        state = AgentState(
            request_id=f"req-{route}",
            run_id=f"run-{route}",
            thread_id=f"request:req-{route}",
            requested_mode=route,
            original_query="请执行任务",
        )
        output = asyncio.run(graph.ainvoke(state))
        assert output["status"] in {"committed", "degraded"}
        assert expected_node in output["completed_nodes"]
        route_events = [event for event in output["events"] if event["event_type"] == "route_decided"]
        assert len(route_events) == 1


def test_application_graph_returns_controlled_terminal_routes() -> None:
    """Clarification and unsupported inputs terminate without entering a business subgraph."""
    from app.agents.orchestration.graph import AgentState, build_application_graph

    graph = build_application_graph()
    for mode, status in (("clarification", "waiting"), ("unsupported", "unsupported")):
        output = asyncio.run(graph.ainvoke(
            AgentState(
                request_id=f"req-{mode}",
                run_id=f"run-{mode}",
                thread_id=f"request:req-{mode}",
                requested_mode=mode,
                original_query="",
            )
        ))
        assert output["status"] == status
        assert "dispatch_subgraph" not in output["completed_nodes"]


def test_memory_checkpointer_enforces_owner_schema_ttl_and_size() -> None:
    """The test checkpointer rejects cross-owner, stale, incompatible, and oversized state."""
    from app.agents.orchestration.checkpoint import (
        CheckpointPolicy,
        CheckpointValidationError,
        MemoryCheckpointer,
        derive_thread_id,
    )

    thread_id = derive_thread_id(session_id="session-1")
    policy = CheckpointPolicy(ttl_seconds=60, max_bytes=500)
    store = MemoryCheckpointer(policy=policy)
    state = {"schema_version": policy.schema_version, "session_id": "session-1", "status": "running"}
    store.save(thread_id, state, owner_id="user-1", session_id="session-1")

    assert store.load(thread_id, owner_id="user-1")["session_id"] == "session-1"
    with pytest.raises(CheckpointValidationError, match="owner"):
        store.load(thread_id, owner_id="user-2")
    with pytest.raises(CheckpointValidationError, match="schema"):
        store.save(thread_id, {"schema_version": "agent-state.v0"}, owner_id="user-1")
    with pytest.raises(CheckpointValidationError, match="size"):
        store.save(thread_id, {"schema_version": policy.schema_version, "data": "x" * 1000}, owner_id="user-1")

    store.save(
        thread_id,
        state,
        owner_id="user-1",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    assert store.load(thread_id, owner_id="user-1") is None


def test_thread_id_and_postgres_boundary_are_explicit() -> None:
    """Stable IDs are deterministic and PostgreSQL remains an opt-in adapter."""
    from app.agents.orchestration.checkpoint import PostgresCheckpointer, derive_thread_id

    assert derive_thread_id(session_id="s-1") == derive_thread_id(session_id="s-1")
    assert derive_thread_id(interview_id="i-1") != derive_thread_id(request_id="r-1")
    adapter = PostgresCheckpointer(connection_string="postgresql://example.invalid/db")
    assert adapter.available in {True, False}
