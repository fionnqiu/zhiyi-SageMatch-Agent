"""Stage-three contracts for durable jobs, recovery, and readiness checks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.db import Base
from app.models.platform.runtime import DurableJob, RuntimeSchemaVersion, WorkerHeartbeat
from app.services.operations.jobs import (
    claim_job,
    complete_job,
    create_job,
    fail_job,
    heartbeat,
    readiness_check,
    reclaim_expired_jobs,
    renew_lease,
)


def _db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    # Existing business tables use PostgreSQL JSONB; the job contracts are
    # independently SQLite-testable without pretending SQLite is production.
    Base.metadata.create_all(
        engine,
        tables=[
            DurableJob.__table__,
            RuntimeSchemaVersion.__table__,
            WorkerHeartbeat.__table__,
        ],
    )
    return Session(engine)


def test_create_job_is_idempotent_by_key() -> None:
    db = _db()
    first = create_job(db, "material_ingest", "material-1", {"x": 1}, idempotency_key="idem-1")
    second = create_job(db, "material_ingest", "material-1", {"x": 2}, idempotency_key="idem-1")
    db.commit()

    assert first.id == second.id
    assert second.payload == {"x": 1}
    assert db.query(DurableJob).count() == 1


def test_claim_renew_and_complete_are_worker_scoped() -> None:
    db = _db()
    row = create_job(db, "report", "interview-1", idempotency_key="report-1")
    claimed = claim_job(db, "worker-a", now_at=datetime.now(timezone.utc), lease_seconds=30)
    assert claimed is not None and claimed.id == row.id
    assert renew_lease(db, row.id, "worker-b", lease_seconds=30) is False
    assert renew_lease(db, row.id, "worker-a", lease_seconds=30) is True

    assert complete_job(db, row.id, worker_id="worker-b", result={"ok": True}) is None
    completed = complete_job(db, row.id, worker_id="worker-a", result={"ok": True})
    db.commit()
    assert completed is not None and completed.status == "succeeded"
    assert complete_job(db, row.id, worker_id="worker-a", result={"ok": False}).status == "succeeded"


def test_expired_lease_is_reclaimed_then_dead_lettered_at_attempt_limit() -> None:
    db = _db()
    created = create_job(db, "report", "interview-2", max_attempts=2, idempotency_key="report-2")
    old = datetime.now(timezone.utc) - timedelta(minutes=2)
    claimed = claim_job(db, "worker-a", now_at=old, lease_seconds=1)
    assert claimed is not None
    assert reclaim_expired_jobs(db, now_at=datetime.now(timezone.utc)) == 1
    assert db.get(DurableJob, created.id).status == "retry_wait"

    second = claim_job(db, "worker-b", now_at=datetime.now(timezone.utc), lease_seconds=1)
    assert second is not None and second.attempts == 2
    failed = fail_job(db, created.id, "provider_timeout", retryable=True, worker_id="worker-b")
    db.commit()
    assert failed is not None and failed.status == "dead_letter"
    assert failed.last_error_code == "provider_timeout"


def test_generation_reclaim_leaves_unrelated_expired_jobs_running() -> None:
    db = _db()
    old = datetime.now(timezone.utc) - timedelta(minutes=2)
    generation = create_job(db, "interview_generation", "generation-1")
    report = create_job(db, "report_generation", "interview-1")
    material = create_job(db, "material_ingest", "material-1")
    for kind in ("interview_generation", "report_generation", "material_ingest"):
        assert claim_job(db, f"{kind}-worker", kinds=[kind], now_at=old, lease_seconds=1) is not None
    db.commit()

    # The generation poller must not alter another worker's recovery state.
    assert reclaim_expired_jobs(db, kinds=["interview_generation"]) == 1
    assert db.get(DurableJob, generation.id).status == "retry_wait"
    assert db.get(DurableJob, report.id).status == "running"
    assert db.get(DurableJob, material.id).status == "running"


def test_stale_worker_cannot_renew_or_complete_an_expired_lease() -> None:
    db = _db()
    row = create_job(db, "report", "interview-stale", idempotency_key="report-stale")
    old = datetime.now(timezone.utc) - timedelta(minutes=2)
    assert claim_job(db, "worker-old", now_at=old, lease_seconds=1) is not None
    db.commit()

    assert renew_lease(db, row.id, "worker-old") is False
    assert complete_job(db, row.id, worker_id="worker-old") is None
    assert reclaim_expired_jobs(db) == 1
    assert db.get(DurableJob, row.id).status == "retry_wait"


def test_deadline_is_a_non_retryable_boundary() -> None:
    db = _db()
    deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
    row = create_job(db, "report", "interview-3", deadline_at=deadline, idempotency_key="report-3")
    assert claim_job(db, "worker-a", now_at=datetime.now(timezone.utc)) is None
    assert db.get(DurableJob, row.id).status == "dead_letter"
    assert db.get(DurableJob, row.id).last_error_code == "deadline_exceeded"


def test_readiness_requires_database_migration_worker_and_checkpointer() -> None:
    db = _db()
    db.add(RuntimeSchemaVersion(name="runtime", version=1))
    heartbeat(db, "worker-a", status="alive")
    db.commit()
    old = readiness_check(db, worker_id="worker-a", checkpointer=lambda: True)
    assert old["checks"]["migrations"]["error_code"] == "migration_pending"
    db.get(RuntimeSchemaVersion, "runtime").version = 2
    db.commit()
    result = readiness_check(db, worker_id="worker-a", checkpointer=lambda: True)
    assert result["status"] == "ready"
    assert all(item["status"] == "ok" for item in result["checks"].values())
