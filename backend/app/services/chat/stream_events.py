"""Commit public SSE frames before delivery and replay by owner/run/sequence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.models.platform.stream import StreamEvent, StreamRun


PUBLIC_TYPES = frozenset({"meta", "delta", "reset", "done", "error", "blocked", "redirect"})
PRIVATE_KEYS = frozenset({"thinking", "reasoning", "reasoning_content", "thought", "thoughts"})


def _public(value):
    """Strip provider reasoning from nested session or interview snapshots."""
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if key.lower() not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


def create_stream_run(
    db: Session, kind: str, owner_id: str, *, run_id: str | None = None,
    business_id: str | None = None, commit: bool = True,
) -> StreamRun:
    """Bind a stable run ID to its owner before any frame can be appended."""
    if not owner_id or kind not in {"chat", "interview_generation"}:
        raise ValueError("invalid stream scope")
    run_id = run_id or str(uuid4())
    row = db.get(StreamRun, run_id)
    if row is not None:
        if (row.owner_id, row.kind, row.business_id) != (owner_id, kind, business_id):
            raise PermissionError("stream scope mismatch")
        return row
    row = StreamRun(id=run_id, kind=kind, owner_id=owner_id, business_id=business_id, next_seq=1)
    db.add(row)
    if commit:
        db.commit()
    return row


def set_stream_recovery(db: Session, run_id: str, owner_id: str, recovery: dict, *, worker_id: str | None = None) -> None:
    """Store the route once the user turn has committed, before generation starts."""
    row = db.get(StreamRun, run_id)
    if row is None or row.owner_id != owner_id or row.status != "running":
        raise PermissionError("stream scope mismatch")
    row.recovery = recovery
    if worker_id is not None:
        row.lease_owner = worker_id
        row.lease_until = datetime.now(timezone.utc) + timedelta(seconds=30)
    db.commit()


def claim_stream_run(db: Session, run_id: str, worker_id: str, *, lease_seconds: int = 30) -> bool:
    """Claim only an unowned or expired run; PostgreSQL serializes contenders."""
    current = datetime.now(timezone.utc)
    row = db.query(StreamRun).filter(StreamRun.id == run_id).with_for_update().one_or_none()
    if row is None or row.status != "running":
        db.rollback()
        return False
    if row.recovery is None:
        created_at = row.created_at
        if created_at is None:
            db.rollback()
            return False
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        # An old deployment or crash before staging left no replayable input.
        # Give active route handlers time to finish staging before closing it.
        if current - created_at < timedelta(seconds=30):
            db.rollback()
            return False
        row.recovery = {"recoverable": False}
    lease_until = row.lease_until
    if lease_until is not None and lease_until.tzinfo is None:
        lease_until = lease_until.replace(tzinfo=timezone.utc)
    if lease_until is not None and lease_until > current:
        db.rollback()
        return False
    row.lease_owner = worker_id
    row.lease_until = current + timedelta(seconds=lease_seconds)
    db.commit()
    return True


def renew_stream_lease(db: Session, run_id: str, worker_id: str, *, lease_seconds: int = 30) -> bool:
    """Only the current worker may extend its lease."""
    result = db.execute(
        update(StreamRun)
        .where(StreamRun.id == run_id, StreamRun.status == "running",
               StreamRun.lease_owner == worker_id, StreamRun.lease_until > datetime.now(timezone.utc))
        .values(lease_until=datetime.now(timezone.utc) + timedelta(seconds=lease_seconds))
    )
    db.commit()
    return result.rowcount == 1


def expire_stream_lease(db: Session, run_id: str, worker_id: str) -> None:
    """Release a conflicting recovery attempt without admitting its stale writer."""
    db.execute(
        update(StreamRun)
        .where(StreamRun.id == run_id, StreamRun.status == "running",
               StreamRun.lease_owner == worker_id)
        .values(lease_until=datetime.now(timezone.utc) - timedelta(microseconds=1))
    )
    db.commit()


def append_stream_event(
    db: Session, run_id: str, owner_id: str, payload: dict, *, terminal: bool = True,
    worker_id: str | None = None,
) -> StreamEvent:
    """Atomically reserve a sequence and commit its event before it is sent."""
    event_type = payload.get("type")
    if event_type not in PUBLIC_TYPES:
        raise ValueError("event type is not replayable")
    # UPDATE ... RETURNING serializes allocation on the run row in PostgreSQL.
    # A reclaimed run must reject frames from its previous worker, including
    # terminal frames that would otherwise close the new worker's stream.
    current = datetime.now(timezone.utc)
    lease_guard = (
        StreamRun.lease_owner.is_(None) if worker_id is None else
        (StreamRun.lease_owner.is_(None) |
         ((StreamRun.lease_owner == worker_id) & (StreamRun.lease_until > current)))
    )
    seq = db.execute(
        update(StreamRun)
        .where(StreamRun.id == run_id, StreamRun.owner_id == owner_id,
               StreamRun.status == "running", lease_guard)
        .values(next_seq=StreamRun.next_seq + 1)
        .returning(StreamRun.next_seq)
    ).scalar_one_or_none()
    if seq is None:
        db.rollback()
        raise PermissionError("stream is missing, closed, or owned by another user")
    seq -= 1
    clean = _public(payload)
    clean.update(run_id=run_id, seq=seq)
    row = StreamEvent(event_id=f"{run_id}:{seq}", run_id=run_id, seq=seq, event_type=event_type, payload=clean)
    db.add(row)
    if terminal and event_type in {"done", "error", "blocked", "redirect"}:
        db.execute(
            update(StreamRun).where(StreamRun.id == run_id).values(status="completed", completed_at=datetime.now(timezone.utc))
        )
    db.commit()
    return row


def replay_stream_events(db: Session, run_id: str, owner_id: str, after_event_id: str | None = None) -> list[StreamEvent]:
    """Return strictly later frames after verifying the owner, including closed runs."""
    row = db.get(StreamRun, run_id)
    if row is None or row.owner_id != owner_id:
        raise PermissionError("stream is missing or owned by another user")
    after_seq = 0
    if after_event_id:
        prefix = f"{run_id}:"
        if not after_event_id.startswith(prefix):
            raise ValueError("event ID does not belong to this run")
        suffix = after_event_id[len(prefix):]
        if not suffix.isdecimal():
            raise ValueError("invalid event sequence")
        after_seq = int(suffix)
    return (
        db.query(StreamEvent)
        .filter(StreamEvent.run_id == run_id, StreamEvent.seq > after_seq)
        .order_by(StreamEvent.seq)
        .all()
    )


def event_frame(event: StreamEvent) -> str:
    """Encode the persisted payload with an SSE ID for Last-Event-ID clients."""
    return f"id: {event.event_id}\ndata: {json.dumps(event.payload, ensure_ascii=False)}\n\n"


def stream_completion_outcomes(db: Session, runs: Sequence[StreamRun]) -> dict[str, bool]:
    """Classify observed terminal frames without counting open streams as failures."""
    ids = [run.id for run in runs]
    if not ids:
        return {}
    events = db.query(StreamEvent).filter(
        StreamEvent.run_id.in_(ids),
        StreamEvent.event_type.in_(["done", "error", "blocked", "redirect"]),
    ).order_by(StreamEvent.seq.desc()).all()
    # The first row per run is its latest persisted terminal decision.
    observed = {}
    for event in events:
        observed.setdefault(event.run_id, event.event_type != "error")
    for run in runs:
        if run.id not in observed and run.status == "failed":
            observed[run.id] = False
    return observed
