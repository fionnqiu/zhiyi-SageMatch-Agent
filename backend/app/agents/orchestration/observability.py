"""Persist compact graph lifecycle records without prompt or secret payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from sqlalchemy.orm import Session

from app.models.platform.runtime import AgentEvent as AgentEventRow
from app.models.platform.runtime import GraphRun, NodeRun, ToolRun


_EVENT_KEYS = frozenset({"route", "node", "next_role", "result_keys", "status", "error_code", "attempt", "thread_id", "timestamp", "duration_ms", "task_id", "agent", "candidate_count", "selected_count", "handoff_count"})
_DIAGNOSTIC_KEYS = frozenset({"candidate_count", "selected_count", "handoff_count", "retry_count", "fallback_reason", "token_count", "cost", "checkpoint_recovered"})


def persist_graph_trace(db: Session, outcome: dict[str, Any]) -> None:
    """Write one graph summary and append-only node/event records.

    Events are trimmed to operational fields because graph state can contain
    private interview answers, attachment text, and provider reasoning.
    """
    run_id = str(outcome["run_id"])
    row = db.get(GraphRun, run_id)
    if row is None:
        row = GraphRun(id=run_id)
        db.add(row)
    row.request_id = str(outcome.get("request_id") or run_id)
    row.thread_id = str(outcome.get("thread_id") or "")
    row.owner_id = str(outcome.get("user_id") or "")
    row.status = str(outcome.get("status") or "failed")[:20]
    row.current_node = str(outcome.get("current_node") or "")[:120]
    row.loop_count = int(outcome.get("loop_count") or 0)
    row.schema_version = str(outcome.get("schema_version") or "agent-state.v1")[:40]
    error = outcome.get("error")
    row.error = {key: error[key] for key in ("error_code", "retryable", "trace_id") if key in error} if isinstance(error, dict) else None
    diagnostics = outcome.get("diagnostics")
    row.diagnostics = {key: diagnostics[key] for key in _DIAGNOSTIC_KEYS if key in diagnostics} if isinstance(diagnostics, dict) else {}
    lifecycle = [event for event in outcome.get("events") or [] if isinstance(event, dict)]
    started = next((
        _event_time((event.get("payload") or {}).get("timestamp"))
        for event in lifecycle if event.get("event_type") == "run_started"
    ), None)
    finished = next((
        _event_time((event.get("payload") or {}).get("timestamp"))
        for event in reversed(lifecycle) if event.get("event_type") == "run_finished"
    ), None)
    if started is not None:
        row.created_at = started
    row.completed_at = finished if row.status in {"committed", "failed", "degraded", "unsupported", "waiting"} else None
    db.flush()

    existing = {item[0] for item in db.query(AgentEventRow.id).filter(AgentEventRow.run_id == run_id).all()}
    node_runs = {(item.node, item.attempt): item for item in db.query(NodeRun).filter(NodeRun.run_id == run_id).all()}
    for sequence, event in enumerate(outcome.get("events") or [], start=1):
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("event_id") or uuid4().hex)[:36]
        node = str(event.get("node") or "")[:120]
        event_type = str(event.get("event_type") or "")[:80]
        raw_payload = event.get("payload")
        safe_payload = {key: raw_payload[key] for key in _EVENT_KEYS if key in raw_payload} if isinstance(raw_payload, dict) else {}
        if event_id not in existing:
            db.add(AgentEventRow(
                id=event_id,
                run_id=run_id,
                node=node,
                agent=str(event.get("agent") or "")[:80] or None,
                event_type=event_type,
                sequence=sequence,
                payload=safe_payload,
                schema_version=str(event.get("schema_version") or "agent-event.v1")[:40],
            ))
            existing.add(event_id)
        if event_type not in {"node_started", "node_finished"} or not node:
            continue
        try:
            attempt = max(1, int(safe_payload.get("attempt") or 1))
        except (TypeError, ValueError):
            attempt = 1
        key = (node, attempt)
        node_run = node_runs.get(key)
        if event_type == "node_started" and node_run is None:
            started_at = _event_time(safe_payload.get("timestamp"))
            if started_at is None:
                continue
            node_run = NodeRun(
                id=uuid4().hex, run_id=run_id, node=node, attempt=attempt,
                status="running", started_at=started_at,
            )
            db.add(node_run)
            node_runs[key] = node_run
        elif event_type == "node_finished" and node_run is not None and node_run.completed_at is None:
            finished_at = _event_time(safe_payload.get("timestamp"))
            if finished_at is None:
                continue
            status = safe_payload.get("status")
            node_run.status = "completed" if status == "success" else (
                status if status in {"completed", "failed", "degraded", "waiting"} else "failed"
            )
            node_run.completed_at = finished_at
            duration = safe_payload.get("duration_ms")
            # The graph measures elapsed time with a monotonic clock. Older traces
            # can still derive a wall-clock approximation from paired timestamps.
            if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 0:
                node_run.duration_ms = float(duration)
            else:
                started_at = node_run.started_at.replace(tzinfo=node_run.started_at.tzinfo or timezone.utc)
                node_run.duration_ms = max(0.0, (finished_at - started_at).total_seconds() * 1000)
            if safe_payload.get("error_code"):
                node_run.error = {"error_code": str(safe_payload["error_code"])[:120]}
    persisted = db.query(AgentEventRow).filter(AgentEventRow.run_id == run_id).all()
    finish_event = next((item for item in persisted if item.event_type == "run_finished"), None)
    persisted_by_id = {item.id: item for item in persisted}
    sequence = (len(outcome.get("events") or []) - 1) if finish_event else max((item.sequence for item in persisted), default=0)
    tool_rows = db.query(ToolRun).filter(ToolRun.run_id == run_id).order_by(ToolRun.started_at, ToolRun.id).all()
    for tool in tool_rows:
        # ToolRun is written by the actual ToolNode invocation. Derived events
        # use stable IDs so trace persistence can safely retry after a crash.
        for event_type, timestamp, payload in (
            ("tool_called", tool.started_at, {"tool_call_id": tool.tool_call_id, "tool_name": tool.tool_name}),
            ("tool_finished", tool.completed_at, {
                "tool_call_id": tool.tool_call_id, "tool_name": tool.tool_name,
                "status": tool.status, "error_code": tool.error_code,
                "cached": tool.cached, "duration_ms": tool.latency_ms,
            }),
        ):
            event_id = str(uuid5(NAMESPACE_URL, f"{run_id}:{tool.id}:{event_type}"))
            sequence += 1
            if event_id in existing:
                persisted_by_id[event_id].sequence = sequence
                continue
            db.add(AgentEventRow(
                id=event_id, run_id=run_id, node=tool.tool_name,
                event_type=event_type, sequence=sequence,
                payload={**{key: value for key, value in payload.items() if value is not None},
                         "timestamp": timestamp.isoformat() if timestamp else None},
                schema_version="agent-event.v1",
            ))
            existing.add(event_id)
    if finish_event is not None:
        # Tool calls execute before the graph's terminal event even though
        # their rows are joined into the trace after the business callback.
        finish_event.sequence = sequence + 1
    db.commit()


def _event_time(value: Any) -> datetime | None:
    """Accept only timezone-aware lifecycle timestamps supplied by the graph."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None


__all__ = ["persist_graph_trace"]
