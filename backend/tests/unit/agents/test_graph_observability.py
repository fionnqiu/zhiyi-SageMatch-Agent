"""Trace persistence and provider correlation without sensitive prompt data."""

from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agents.orchestration.observability import persist_graph_trace
from app.agents.orchestration.trace import graph_run_scope
from app.core.db import Base
from app.models.platform.runtime import AgentEvent, GraphRun, NodeRun, ToolRun
from app.services.operations.llm_gateway import log_call


def test_graph_trace_stores_compact_events_and_node_runs() -> None:
    """The trace tables contain IDs and event kinds, never raw question text."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[GraphRun.__table__, NodeRun.__table__, AgentEvent.__table__, ToolRun.__table__])
    outcome = {
        "run_id": "run-1", "request_id": "req-1", "thread_id": "session:s1", "user_id": "u1",
        "status": "committed", "current_node": "finish_node", "loop_count": 4,
        "schema_version": "agent-state.v1", "diagnostics": {"candidate_count": 2, "checkpoint_recovered": True},
        "events": [{
            "event_id": "event-1", "node": "dispatch_subgraph", "event_type": "node_started",
            "payload": {"route": "knowledge_qa", "original_query": "private question"},
        }],
    }
    with Session(engine) as db:
        persist_graph_trace(db, outcome)
        assert db.get(GraphRun, "run-1").status == "committed"
        assert db.get(GraphRun, "run-1").diagnostics["checkpoint_recovered"] is True
        # Legacy events without execution timestamps remain auditable events,
        # but cannot truthfully produce a timed node run.
        assert db.query(NodeRun).count() == 0
        event = db.get(AgentEvent, "event-1")
        assert event.payload == {"route": "knowledge_qa"}
        persist_graph_trace(db, outcome)
        assert db.query(AgentEvent).count() == 1
        assert db.query(NodeRun).count() == 0


def test_graph_trace_pairs_actual_attempts_and_keeps_failure_duration() -> None:
    """Retry attempts have independent timings and only matching finishes close them."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[GraphRun.__table__, NodeRun.__table__, AgentEvent.__table__, ToolRun.__table__])
    outcome = {
        "run_id": "run-3", "status": "committed", "current_node": "finish_node",
        "events": [
            {"event_id": "start-1", "node": "knowledge_qa.answer", "event_type": "node_started",
             "payload": {"attempt": 1, "timestamp": "2026-09-26T10:00:00+00:00"}},
            {"event_id": "finish-1", "node": "knowledge_qa.answer", "event_type": "node_finished",
             "payload": {"attempt": 1, "status": "failed", "error_code": "provider_failed",
                         "timestamp": "2026-09-26T10:00:01+00:00", "duration_ms": 750.5}},
            {"event_id": "start-2", "node": "knowledge_qa.answer", "event_type": "node_started",
             "payload": {"attempt": 2, "timestamp": "2026-09-26T10:00:02+00:00"}},
            {"event_id": "finish-2", "node": "knowledge_qa.answer", "event_type": "node_finished",
             "payload": {"attempt": 2, "status": "success",
                         "timestamp": "2026-09-26T10:00:03+00:00", "duration_ms": 900.0}},
            {"event_id": "start-3", "node": "knowledge_qa.answer", "event_type": "node_started",
             "payload": {"attempt": 3, "timestamp": "2026-09-26T10:00:04+00:00"}},
        ],
    }
    with Session(engine) as db:
        persist_graph_trace(db, outcome)
        first, second, third = db.query(NodeRun).order_by(NodeRun.attempt).all()
        assert (first.status, first.duration_ms, first.error) == (
            "failed", 750.5, {"error_code": "provider_failed"})
        assert (second.status, second.duration_ms) == ("completed", 900.0)
        assert (third.status, third.completed_at, third.duration_ms) == ("running", None, None)
        assert first.started_at != first.completed_at
        persist_graph_trace(db, outcome)
        assert db.query(NodeRun).count() == 3
        assert db.query(AgentEvent).count() == 5


def test_graph_run_uses_lifecycle_timestamps_instead_of_trace_write_time() -> None:
    """A delayed trace sink preserves the actual graph wall duration."""

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[GraphRun.__table__, NodeRun.__table__, AgentEvent.__table__, ToolRun.__table__])
    outcome = {
        "run_id": "run-timed", "status": "committed", "current_node": "finish_node",
        "events": [
            {"event_id": "run-start", "node": "load_context", "event_type": "run_started",
             "payload": {"timestamp": "2026-09-26T10:00:00+00:00"}},
            {"event_id": "run-finish", "node": "finish_node", "event_type": "run_finished",
             "payload": {"status": "committed", "timestamp": "2026-09-26T10:00:00.250000+00:00"}},
        ],
    }
    with Session(engine) as db:
        persist_graph_trace(db, outcome)
        run = db.get(GraphRun, "run-timed")
        assert (run.completed_at - run.created_at).total_seconds() == 0.25


def test_supervisor_event_persists_handoff_identity_without_task_body() -> None:
    """The trace retains correlation IDs while excluding worker prompt text."""

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[GraphRun.__table__, NodeRun.__table__, AgentEvent.__table__, ToolRun.__table__])
    outcome = {
        "run_id": "handoff-run", "status": "committed",
        "events": [{"event_id": "handoff-event", "node": "supervisor_handoff", "event_type": "agent_called",
                    "payload": {"task_id": "task-1", "agent": "knowledge_qa", "attempt": 1,
                                "goal": "private question body"}}],
    }
    with Session(engine) as db:
        persist_graph_trace(db, outcome)
        payload = db.get(AgentEvent, "handoff-event").payload
        assert payload == {"task_id": "task-1", "agent": "knowledge_qa", "attempt": 1}


def test_supervisor_decision_persists_role_without_model_reason() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[
        GraphRun.__table__, NodeRun.__table__, AgentEvent.__table__, ToolRun.__table__,
    ])
    outcome = {"run_id": "role-run", "status": "committed", "events": [{
        "event_id": "role-choice", "node": "supervisor_decide", "event_type": "supervisor_decision",
        "payload": {"next_role": "critic", "reason": "private candidate text"},
    }]}
    with Session(engine) as db:
        persist_graph_trace(db, outcome)
        assert db.get(AgentEvent, "role-choice").payload == {"next_role": "critic"}


def test_tool_events_derive_from_actual_tool_run_without_arguments() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[
        GraphRun.__table__, NodeRun.__table__, AgentEvent.__table__, ToolRun.__table__,
    ])
    with Session(engine) as db:
        db.add(ToolRun(
            id="tool-row", run_id="graph-tool", tool_call_id="tool-call-1",
            tool_name="search", status="success", cached=False, latency_ms=12.5,
        ))
        outcome = {"run_id": "graph-tool", "status": "committed", "events": [
            {"event_id": "graph-finished", "node": "finish_node", "event_type": "run_finished",
             "payload": {"status": "committed", "timestamp": "2026-09-26T10:00:01+00:00"}},
        ]}
        persist_graph_trace(db, outcome)
        events = db.query(AgentEvent).order_by(AgentEvent.sequence).all()
        assert [event.event_type for event in events] == ["tool_called", "tool_finished", "run_finished"]
        assert events[1].payload["duration_ms"] == 12.5
        assert "arguments" not in str([event.payload for event in events])
        persist_graph_trace(db, outcome)
        assert db.query(AgentEvent).count() == 3
        assert [event.sequence for event in db.query(AgentEvent).order_by(AgentEvent.sequence)] == [1, 2, 3]


def test_model_call_log_inherits_run_id_from_async_context() -> None:
    """A provider call uses the enclosing graph ID without prompt propagation."""
    added = []
    db = SimpleNamespace(add=added.append)
    with graph_run_scope("run-2"):
        log_call(db, "analyst", "provider", "model", "ok", 12, None)
    assert added[0].run_id == "run-2"
