"""Metrics must come from recorded graph, provider, and stream evidence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.db import Base
from app.models.platform.audit import LlmCallLog
from app.models.platform.runtime import GraphRun, NodeRun
from app.models.platform.stream import StreamEvent, StreamRun
from app.services.interviews.eval import collect_system_run_metrics
from app.services.operations.overview import admin_overview
from app.services.operations.quality import evaluate_system_metrics


def test_collect_system_run_metrics_uses_observed_rows_only() -> None:
    """A completed graph alone does not prove SSE or checkpoint recovery."""

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[
        GraphRun.__table__, NodeRun.__table__, LlmCallLog.__table__, StreamRun.__table__, StreamEvent.__table__,
    ])
    start = datetime(2026, 9, 26, 10, 0, tzinfo=timezone.utc)
    with Session(engine) as db:
        db.add(GraphRun(
            id="run-1", status="committed", created_at=start,
            completed_at=start + timedelta(milliseconds=200), diagnostics={},
        ))
        db.add(NodeRun(
            id="node-1", run_id="run-1", node="knowledge_qa.answer",
            status="completed", attempt=1, duration_ms=120.0,
        ))
        db.add(LlmCallLog(
            id="call-1", run_id="run-1", role="analyst", status="error", latency_ms=120,
        ))
        db.add(StreamRun(id="run-1", owner_id="local-user", kind="chat", status="failed"))
        db.commit()

        metrics = collect_system_run_metrics(db, ["run-1", "run-1", "missing"])

    assert metrics["run_count"] == 1
    assert metrics["p50_latency_ms"] == 200.0
    assert metrics["node_latency_ms"]["knowledge_qa.answer"]["p50"] == 120.0
    assert metrics["provider_failure_rate"] == 1.0
    assert metrics["sse_completion_rate"] == 0.0
    assert metrics["checkpoint_recovery_rate"] is None
    assert metrics["token_count"] is None


def test_provider_metrics_use_call_denominator_and_recorded_usage() -> None:
    """One failed call in a busy run must not count as a failed run rate."""

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[
        GraphRun.__table__, NodeRun.__table__, LlmCallLog.__table__, StreamRun.__table__, StreamEvent.__table__,
    ])
    with Session(engine) as db:
        db.add(GraphRun(id="run-many", status="committed", diagnostics={"token_count": 999, "cost": 9.0}))
        for index in range(10):
            db.add(LlmCallLog(
                id=f"call-{index}", run_id="run-many", role="analyst",
                status="error" if index == 0 else "ok", latency_ms=10,
                total_tokens=20, estimated_cost=0.001,
            ))
        db.commit()
        metrics = collect_system_run_metrics(db, ["run-many"])

    assert metrics["provider_failure_rate"] == 0.1
    assert metrics["provider_failure_rate_basis"] == "logged_model_calls"
    assert metrics["logged_model_call_count"] == 10
    assert metrics["logged_model_token_count"] == 200
    assert metrics["logged_model_estimated_cost"] == 0.01
    assert metrics["token_count"] == 999
    assert metrics["estimated_cost"] == 9.0


def test_partial_provider_usage_remains_unobserved() -> None:
    """Missing usage in one call makes the complete run total unknown."""

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[
        GraphRun.__table__, NodeRun.__table__, LlmCallLog.__table__, StreamRun.__table__, StreamEvent.__table__,
    ])
    with Session(engine) as db:
        db.add(GraphRun(id="run-partial", status="failed", diagnostics={}))
        db.add_all([
            LlmCallLog(id="known", run_id="run-partial", status="ok", total_tokens=10, estimated_cost=0.1),
            LlmCallLog(id="unknown", run_id="run-partial", status="error"),
        ])
        db.commit()
        metrics = collect_system_run_metrics(db, ["run-partial"])

    assert metrics["provider_failure_rate"] == 0.5
    assert metrics["token_count"] is None
    assert metrics["estimated_cost"] is None
    assert metrics["logged_model_token_count"] is None
    assert metrics["logged_model_estimated_cost"] is None


def test_logged_calls_do_not_mix_with_case_failure_observations() -> None:
    """A case-only row cannot dilute a measured model-call failure rate."""
    metrics = evaluate_system_metrics([
        {"provider_calls": [True], "logged_model_token_count": 20, "logged_model_cost": 0.01},
        {"provider_failed": False},
    ])
    assert metrics["provider_failure_rate"] == 1.0
    assert metrics["provider_failure_rate_basis"] == "logged_model_calls"
    assert metrics["logged_model_call_count"] == 1
    assert metrics["logged_model_token_count"] == 20
    assert metrics["logged_model_estimated_cost"] == 0.01


def test_sse_completion_uses_terminal_event_instead_of_closed_run_status() -> None:
    """An error frame closes the stream but cannot count as a successful SSE."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[
        GraphRun.__table__, NodeRun.__table__, LlmCallLog.__table__, StreamRun.__table__, StreamEvent.__table__,
    ])
    with Session(engine) as db:
        for run_id in ("done-run", "error-run", "unknown-run"):
            recovery = {"checkpoint_recovered": run_id == "done-run"} if run_id != "unknown-run" else {}
            db.add(GraphRun(id=run_id, status="committed", diagnostics=recovery))
            db.add(StreamRun(id=run_id, owner_id="local-user", kind="chat", status="completed"))
        db.flush()
        db.add_all([
            StreamEvent(event_id="done-run:1", run_id="done-run", seq=1,
                        event_type="done", payload={"type": "done"}),
            StreamEvent(event_id="error-run:1", run_id="error-run", seq=1,
                        event_type="error", payload={"type": "error"}),
        ])
        db.commit()
        metrics = collect_system_run_metrics(db, ["done-run", "error-run", "unknown-run"])

    assert metrics["sse_completion_rate"] == 0.5
    assert metrics["checkpoint_recovery_rate"] == 0.5


def test_admin_overview_reports_observed_sse_completion_only() -> None:
    """The operator view reads durable terminal frames and shows no invented rate."""
    db = MagicMock()
    streams = [SimpleNamespace(id="done"), SimpleNamespace(id="error"), SimpleNamespace(id="open")]
    graphs = []

    def query(model):
        result = MagicMock()
        result.count.return_value = 0
        result.all.return_value = []
        result.filter.return_value.all.return_value = (
            streams if model is StreamRun else graphs if model is GraphRun else []
        )
        return result

    db.query.side_effect = query
    with patch("app.services.operations.overview.seed_providers"), patch(
        "app.services.operations.overview.embedding_ready", return_value=False
    ), patch("app.services.operations.overview.stream_completion_outcomes", return_value={}) as outcomes:
        assert admin_overview(db)["sse_completion_rate_24h"] is None
        assert admin_overview(db)["checkpoint_recovery_rate_24h"] is None
        outcomes.return_value = {"done": True, "error": False}
        graphs.extend([
            SimpleNamespace(diagnostics={"checkpoint_recovered": True}),
            SimpleNamespace(diagnostics={"checkpoint_recovered": False}),
            SimpleNamespace(diagnostics={}),
        ])
        overview = admin_overview(db)
        assert overview["sse_completion_rate_24h"] == 0.5
        assert overview["checkpoint_recovery_rate_24h"] == 0.5
