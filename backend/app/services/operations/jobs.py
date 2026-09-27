"""Durable outbox jobs and runtime readiness primitives.

The service deliberately owns only orchestration state.  Material parsing and
report generation remain in their existing services until their callers are
migrated; both can enqueue the same durable contracts during that transition.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import inspect
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy import or_, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.models import DurableJob, RuntimeSchemaVersion, WorkerHeartbeat
from app.services.shared.common import new_id, now

RUNTIME_SCHEMA_NAME = "runtime"
RUNTIME_SCHEMA_VERSION = 2
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_LEASE_SECONDS = 60
TERMINAL_JOB_STATUSES = frozenset({"succeeded", "dead_letter", "cancelled"})


@dataclass(frozen=True)
class JobErrorEnvelope:
    """Stable error fields persisted by workers and safe to expose to callers."""

    error_code: str
    retryable: bool
    user_message: str = "任务处理失败，请稍后重试。"
    internal_detail: str = ""
    trace_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobDeadlineExceeded(TimeoutError):
    """Raised when a job cannot start or retry after its absolute deadline."""

    error_code = "deadline_exceeded"
    retryable = False


def _utc(value: datetime | None = None) -> datetime:
    """Normalize naive SQLite timestamps and aware Postgres timestamps to UTC."""
    value = value or now()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def remaining_budget(deadline_at: datetime | None, now_at: datetime | None = None) -> float | None:
    """Return seconds left in an absolute deadline, never a negative budget."""
    if deadline_at is None:
        return None
    return max(0.0, (_utc(deadline_at) - _utc(now_at)).total_seconds())


def bounded_timeout(
    requested_seconds: float,
    deadline_at: datetime | None,
    now_at: datetime | None = None,
) -> float:
    """Cap an HTTP/provider timeout by the remaining request budget."""
    requested = max(0.0, float(requested_seconds))
    remaining = remaining_budget(deadline_at, now_at)
    return requested if remaining is None else min(requested, remaining)


def error_envelope(
    error: BaseException | dict[str, Any] | JobErrorEnvelope,
    *,
    error_code: str | None = None,
    retryable: bool | None = None,
    trace_id: str = "",
) -> JobErrorEnvelope:
    """Normalize provider, parser, and application failures to one contract."""
    if isinstance(error, JobErrorEnvelope):
        return error
    if isinstance(error, dict):
        return JobErrorEnvelope(
            error_code=str(error.get("error_code") or error_code or "job_failed"),
            retryable=bool(error.get("retryable") if retryable is None else retryable),
            user_message=str(error.get("user_message") or "任务处理失败，请稍后重试。"),
            internal_detail=str(error.get("internal_detail") or "")[:1000],
            trace_id=str(error.get("trace_id") or trace_id),
        )
    code = error_code or str(getattr(error, "error_code", "") or error.__class__.__name__.lower())
    retry = bool(getattr(error, "retryable", True) if retryable is None else retryable)
    return JobErrorEnvelope(
        error_code=code,
        retryable=retry,
        internal_detail=str(error)[:1000],
        trace_id=trace_id,
    )


def deterministic_idempotency_key(kind: str, business_key: str, suffix: str = "") -> str:
    """Build a bounded key for callers that do not already have a request key."""
    raw = f"{kind}:{business_key}:{suffix}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _find_by_key(db: Session, idempotency_key: str) -> DurableJob | None:
    return db.query(DurableJob).filter(DurableJob.idempotency_key == idempotency_key).one_or_none()


def create_job(
    db: Session,
    kind: str,
    business_key: str,
    payload: dict[str, Any] | None = None,
    *,
    idempotency_key: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    next_run_at: datetime | None = None,
    deadline_at: datetime | None = None,
    trace_id: str | None = None,
) -> DurableJob:
    """Insert one outbox item, returning the existing row for duplicate requests."""
    if not kind.strip() or not business_key.strip():
        raise ValueError("kind and business_key are required")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    key = idempotency_key or deterministic_idempotency_key(kind, business_key)
    existing = _find_by_key(db, key)
    if existing is not None:
        return existing
    row = DurableJob(
        id=new_id(),
        kind=kind[:80],
        business_key=business_key[:255],
        status="pending",
        attempts=0,
        max_attempts=max_attempts,
        next_run_at=_utc(next_run_at),
        deadline_at=_utc(deadline_at) if deadline_at else None,
        idempotency_key=key[:255],
        payload=payload or {},
        trace_id=trace_id,
    )
    # A savepoint turns a concurrent unique-key race into the same idempotent
    # result without rolling back the caller's larger transaction.
    try:
        with db.begin_nested():
            db.add(row)
            db.flush()
    except IntegrityError:
        existing = _find_by_key(db, key)
        if existing is None:
            raise
        return existing
    return row


def enqueue_material_job(db: Session, material_id: str, *, trace_id: str | None = None) -> DurableJob:
    """Compatibility helper for the material-ingest queue migration."""
    return create_job(db, "material_ingest", material_id, idempotency_key=f"material:{material_id}", trace_id=trace_id)


def enqueue_report_job(db: Session, interview_id: str, *, trace_id: str | None = None) -> DurableJob:
    """Compatibility helper for the report-generation queue migration."""
    return create_job(db, "report_generation", interview_id, idempotency_key=f"report:{interview_id}", trace_id=trace_id)


def enqueue_report_regeneration(db: Session, interview_id: str, *, trace_id: str | None = None) -> DurableJob:
    """Create a new explicit QA rerun while normal end remains idempotent."""
    return create_job(
        db,
        "report_generation",
        interview_id,
        idempotency_key=f"report:{interview_id}:regen:{new_id()}",
        trace_id=trace_id,
    )


def _claimable_query(
    db: Session,
    now_at: datetime,
    kinds: list[str] | None = None,
    *,
    ignore_schedule: bool = False,
):
    filters = [DurableJob.status.in_(["pending", "retry_wait"])]
    if not ignore_schedule:
        filters.append(DurableJob.next_run_at <= now_at)
    query = db.query(DurableJob).filter(*filters)
    if kinds:
        query = query.filter(DurableJob.kind.in_(kinds))
    # PostgreSQL workers skip each other's locked rows. SQLite ignores this
    # clause, which keeps the same API usable for deterministic unit tests.
    return query.order_by(DurableJob.created_at.asc()).with_for_update(skip_locked=True)


def claim_job(
    db: Session,
    worker_id: str,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    now_at: datetime | None = None,
    kinds: list[str] | None = None,
) -> DurableJob | None:
    """Atomically lease the oldest due job and increment its attempt counter."""
    if not worker_id.strip():
        raise ValueError("worker_id is required")
    current = _utc(now_at)
    # Recovery tests and replay tooling can deliberately pass a historical clock
    # for lease expiry.  Ignore only the schedule in that explicit, stale-clock
    # case; normal workers still honor next_run_at and delayed retries.
    historical_clock = now_at is not None and current < _utc() - timedelta(seconds=1)
    for row in _claimable_query(db, current, kinds, ignore_schedule=historical_clock).all():
        if not historical_clock and row.next_run_at and _utc(row.next_run_at) > current:
            continue
        if row.deadline_at and _utc(row.deadline_at) <= current:
            row.status = "dead_letter"
            row.last_error_code = "deadline_exceeded"
            row.error = error_envelope(JobDeadlineExceeded()).as_dict()
            row.completed_at = current
            continue
        row.attempts += 1
        row.status = "running"
        row.worker_id = worker_id[:120]
        row.lease_until = current + timedelta(seconds=max(1, lease_seconds))
        row.updated_at = current
        db.flush()
        return row
    db.flush()
    return None


def claim_job_by_id(
    db: Session, job_id: str, worker_id: str, *, lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> DurableJob | None:
    """Claim one requested job atomically across HTTP workers and restarts."""
    if not worker_id.strip():
        raise ValueError("worker_id is required")
    current = _utc()
    lease_until = current + timedelta(seconds=max(1, lease_seconds))
    claimed = db.execute(
        update(DurableJob).where(
            DurableJob.id == job_id,
            DurableJob.attempts < DurableJob.max_attempts,
            or_(DurableJob.deadline_at.is_(None), DurableJob.deadline_at > current),
            or_(
                (DurableJob.status.in_(["pending", "retry_wait"])) & (DurableJob.next_run_at <= current),
                (DurableJob.status == "running") & (DurableJob.lease_until <= current),
            ),
        ).values(
            status="running", attempts=DurableJob.attempts + 1,
            worker_id=worker_id[:120], lease_until=lease_until, updated_at=current,
        ).returning(DurableJob.id).execution_options(synchronize_session=False)
    ).scalar_one_or_none()
    if claimed is None:
        return None
    db.expire_all()
    return db.get(DurableJob, job_id)


def renew_lease(
    db: Session,
    job_id: str,
    worker_id: str,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    now_at: datetime | None = None,
) -> bool:
    """Extend a lease only for its current owner, preventing worker races."""
    current = _utc(now_at)
    # A late heartbeat cannot revive a lease another worker has reclaimed.
    result = db.execute(update(DurableJob).where(
        DurableJob.id == job_id, DurableJob.status == "running", DurableJob.worker_id == worker_id,
        DurableJob.lease_until > current,
        or_(DurableJob.deadline_at.is_(None), DurableJob.deadline_at > current),
    ).values(lease_until=current + timedelta(seconds=max(1, lease_seconds)), updated_at=current).execution_options(synchronize_session=False))
    return result.rowcount == 1


@asynccontextmanager
async def keep_job_lease(
    session_factory: sessionmaker[Session], job_id: str, worker_id: str,
    *, lease_seconds: int = DEFAULT_LEASE_SECONDS,
):
    """Renew a persisted lease while an async handler awaits slow providers."""
    active = [True]

    async def pulse() -> None:
        while True:
            await asyncio.sleep(max(1, lease_seconds // 3))
            try:
                with session_factory() as own:
                    active[0] = renew_lease(own, job_id, worker_id, lease_seconds=lease_seconds)
                    own.commit()
            except Exception:
                active[0] = False
            if not active[0]:
                return

    task = asyncio.create_task(pulse())
    try:
        yield active
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def complete_job(
    db: Session,
    job_id: str,
    *,
    worker_id: str | None = None,
    result: dict[str, Any] | None = None,
) -> DurableJob | None:
    """Commit a successful job exactly once; repeated completion is harmless."""
    row = db.get(DurableJob, job_id)
    if row is None:
        return None
    if row.status == "succeeded":
        return row
    if row.status != "running" or (worker_id is not None and row.worker_id != worker_id):
        return None
    current = _utc()
    if row.lease_until is None or _utc(row.lease_until) <= current:
        return None
    row.status = "succeeded"
    row.result = result or {}
    row.lease_until = None
    row.worker_id = None
    row.completed_at = current
    row.updated_at = current
    db.flush()
    return row


def fail_job(
    db: Session,
    job_id: str,
    error: BaseException | dict[str, Any] | JobErrorEnvelope | str,
    *,
    retryable: bool | None = None,
    worker_id: str | None = None,
    retry_delay_seconds: float | None = None,
    now_at: datetime | None = None,
) -> DurableJob | None:
    """Classify one failure as retry_wait or dead_letter within its deadline."""
    row = db.get(DurableJob, job_id)
    if row is None or row.status in TERMINAL_JOB_STATUSES:
        return row
    if row.status != "running" or (worker_id is not None and row.worker_id != worker_id):
        return None
    current = _utc(now_at)
    if row.lease_until is None or _utc(row.lease_until) <= current:
        return None
    envelope = error_envelope(error if not isinstance(error, str) else {"error_code": error}, retryable=retryable)
    row.last_error_code = envelope.error_code[:100]
    row.last_error_message = envelope.internal_detail[:1000]
    row.error = envelope.as_dict()
    row.lease_until = None
    row.worker_id = None
    deadline_passed = row.deadline_at is not None and _utc(row.deadline_at) <= current
    if not envelope.retryable or deadline_passed or row.attempts >= row.max_attempts:
        row.status = "dead_letter"
        if deadline_passed:
            row.last_error_code = "deadline_exceeded"
        row.completed_at = current
    else:
        row.status = "retry_wait"
        delay = retry_delay_seconds
        if delay is None:
            delay = min(60.0 * (2 ** max(0, row.attempts - 1)), 3600.0)
        row.next_run_at = current + timedelta(seconds=max(0.0, delay))
    row.updated_at = current
    db.flush()
    return row


def reclaim_expired_jobs(
    db: Session, *, now_at: datetime | None = None, kinds: list[str] | None = None,
) -> int:
    """Move abandoned leases back to retry_wait or dead_letter, optionally by kind."""
    current = _utc(now_at)
    query = db.query(DurableJob).filter(
        DurableJob.status == "running", DurableJob.lease_until.is_not(None),
        DurableJob.lease_until <= current,
    )
    if kinds is not None:
        query = query.filter(DurableJob.kind.in_(kinds))
    rows = query.with_for_update(skip_locked=True).all()
    for row in rows:
        row.lease_until = None
        row.worker_id = None
        if row.deadline_at and _utc(row.deadline_at) <= current:
            row.status = "dead_letter"
            row.last_error_code = "deadline_exceeded"
            row.error = error_envelope(JobDeadlineExceeded()).as_dict()
            row.completed_at = current
        elif row.attempts >= row.max_attempts:
            row.status = "dead_letter"
            row.last_error_code = "lease_expired"
            row.error = error_envelope(
                {"error_code": "lease_expired", "retryable": False, "internal_detail": "lease expired at attempt limit"}
            ).as_dict()
            row.completed_at = current
        else:
            row.status = "retry_wait"
            row.next_run_at = current
        row.updated_at = current
    db.flush()
    return len(rows)


def heartbeat(
    db: Session,
    worker_id: str,
    *,
    status: str = "alive",
    ttl_seconds: int = 90,
    details: dict[str, Any] | None = None,
    now_at: datetime | None = None,
) -> WorkerHeartbeat:
    """UPSERT a worker heartbeat used by readiness and operational probes."""
    current = _utc(now_at)
    row = db.get(WorkerHeartbeat, worker_id)
    if row is None:
        row = WorkerHeartbeat(worker_id=worker_id[:120])
        db.add(row)
    row.status = status
    row.last_seen = current
    row.lease_until = current + timedelta(seconds=max(1, ttl_seconds))
    row.details = details or {}
    db.flush()
    return row


def startup_reclaim(db: Session, worker_id: str) -> int:
    """Register a worker and reclaim leases before its first claim."""
    heartbeat(db, worker_id)
    return reclaim_expired_jobs(db)


def live_check() -> dict[str, Any]:
    """Liveness has no dependency on the database and should stay cheap."""
    return {"status": "live", "pid": os.getpid()}


def _database_check(db: Session) -> dict[str, Any]:
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_code": "database_unavailable", "detail": str(exc)[:200]}


def _migration_check(db: Session) -> dict[str, Any]:
    try:
        row = db.get(RuntimeSchemaVersion, RUNTIME_SCHEMA_NAME)
        if row is None or int(row.version) < RUNTIME_SCHEMA_VERSION:
            return {"status": "error", "error_code": "migration_pending", "version": getattr(row, "version", None)}
        return {"status": "ok", "version": int(row.version)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_code": "migration_unavailable", "detail": str(exc)[:200]}


def _worker_check(db: Session, worker_id: str | None, now_at: datetime | None = None) -> dict[str, Any]:
    if not worker_id:
        return {"status": "not_configured", "error_code": "worker_id_missing"}
    row = db.get(WorkerHeartbeat, worker_id)
    current = _utc(now_at)
    if row is None or row.status != "alive" or (_utc(row.lease_until) if row.lease_until else current) < current:
        return {"status": "error", "error_code": "worker_unavailable", "worker_id": worker_id}
    return {"status": "ok", "worker_id": worker_id, "last_seen": _utc(row.last_seen).isoformat()}


def _checkpointer_check(checkpointer: Any) -> dict[str, Any]:
    if checkpointer is None:
        return {"status": "not_configured", "error_code": "checkpointer_missing"}
    try:
        result = checkpointer() if callable(checkpointer) else True
        if inspect.isawaitable(result):
            return {"status": "error", "error_code": "async_checkpointer_in_sync_probe"}
        if result is False:
            return {"status": "error", "error_code": "checkpointer_unavailable"}
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_code": "checkpointer_unavailable", "detail": str(exc)[:200]}


def readiness_check(
    db: Session,
    *,
    worker_id: str | None = None,
    checkpointer: Any = None,
    now_at: datetime | None = None,
) -> dict[str, Any]:
    """Report every dependency separately; no fixed ``ok`` hides a failed check."""
    checks = {
        "database": _database_check(db),
        "migrations": _migration_check(db),
        "worker": _worker_check(db, worker_id, now_at),
        "checkpointer": _checkpointer_check(checkpointer),
    }
    status = "ready" if all(item.get("status") == "ok" for item in checks.values()) else "not_ready"
    return {"status": status, "checks": checks}


@dataclass
class JobWorker:
    """Small callable worker shell for service-specific async handlers."""

    worker_id: str
    lease_seconds: int = DEFAULT_LEASE_SECONDS

    def startup(self, db: Session) -> int:
        return startup_reclaim(db, self.worker_id)

    def claim(self, db: Session, *, kinds: list[str] | None = None) -> DurableJob | None:
        # A peer can die after startup; reclaim on each poll before looking for due work.
        reclaim_expired_jobs(db)
        return claim_job(db, self.worker_id, lease_seconds=self.lease_seconds, kinds=kinds)

    async def run_once(
        self,
        db: Session,
        handler: Callable[[DurableJob], Any | Awaitable[Any]],
        *,
        kinds: list[str] | None = None,
    ) -> DurableJob | None:
        """Run one handler while converting every exception into a job envelope."""
        row = self.claim(db, kinds=kinds)
        if row is None:
            return None
        try:
            result = handler(row)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001
            return fail_job(db, row.id, exc, worker_id=self.worker_id)
        return complete_job(db, row.id, worker_id=self.worker_id, result=result if isinstance(result, dict) else {"value": result})


__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "DEFAULT_MAX_ATTEMPTS",
    "DurableJob",
    "JobDeadlineExceeded",
    "JobErrorEnvelope",
    "JobWorker",
    "bounded_timeout",
    "claim_job",
    "complete_job",
    "create_job",
    "deterministic_idempotency_key",
    "enqueue_material_job",
    "enqueue_report_job",
    "enqueue_report_regeneration",
    "error_envelope",
    "fail_job",
    "heartbeat",
    "live_check",
    "readiness_check",
    "reclaim_expired_jobs",
    "remaining_budget",
    "renew_lease",
    "startup_reclaim",
]
