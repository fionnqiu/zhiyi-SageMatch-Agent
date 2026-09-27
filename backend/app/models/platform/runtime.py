"""Runtime tables for governance, memory, durable jobs, and graph runs."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

# JSONB is authoritative in PostgreSQL, while the generic JSON variant keeps
# runtime-only SQLite tests and recovery tooling executable.
RuntimeJSON = JSON().with_variant(JSONB, "postgresql")


class ProviderHealth(Base):
    """Persisted breaker and latency for one vendor. Routing reads this, not RAM."""

    __tablename__ = "provider_health"

    provider_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    provider_name: Mapped[str] = mapped_column(String(200), default="")
    state: Mapped[str] = mapped_column(String(20), default="closed")  # closed | open | half_open
    consecutive_fails: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[int] = mapped_column(Integer, default=0)
    total_ms: Mapped[int] = mapped_column(Integer, default=0)
    opened_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    probe_lease_until: Mapped[float | None] = mapped_column(Float, nullable=True)
    probe_owner: Mapped[str | None] = mapped_column(String(36), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ToolCacheEntry(Base):
    """Short-lived tool result. Same arguments skip a second retrieval."""

    __tablename__ = "tool_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    tool_name: Mapped[str] = mapped_column(String(60))
    scope: Mapped[str] = mapped_column(String(120), default="legacy")
    index_version: Mapped[str] = mapped_column(String(120), default="")
    payload: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    expires_at: Mapped[float] = mapped_column(Float)


class EpisodeBrief(Base):
    """Structured brief for one session or one interview. Not a transcript copy."""

    __tablename__ = "episode_briefs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scope: Mapped[str] = mapped_column(String(20))  # session | interview
    scope_id: Mapped[str] = mapped_column(String(36), index=True)
    owner_id: Mapped[str] = mapped_column(String(120), default="local-user", index=True)
    source: Mapped[str] = mapped_column(String(120), default="legacy")
    provenance: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    summary: Mapped[str] = mapped_column(Text, default="")
    slots: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class UserProfileMemory(Base):
    """Durable facts for the anonymous local user: job title, focus, preferences."""

    __tablename__ = "user_profile_memory"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(120), default="local-user", index=True)
    scope: Mapped[str] = mapped_column(String(20), default="profile")
    source: Mapped[str] = mapped_column(String(120), default="legacy")
    provenance: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    profile: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DurableJob(Base):
    """A database-backed outbox item that survives process restarts.

    ``idempotency_key`` is unique so two HTTP retries cannot create two logical
    jobs.  ``lease_until`` and ``worker_id`` are deliberately persisted: a new
    worker can reclaim an abandoned lease without relying on process memory.
    """

    __tablename__ = "durable_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(80), index=True)
    business_key: Mapped[str] = mapped_column(String(255), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    next_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    last_error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    payload: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    result: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    trace_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class InterviewCreateReceipt(Base):
    """Committed create result for a caller key, independent of interview status."""

    __tablename__ = "interview_create_receipts"

    owner_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    key_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    question_set_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    interview_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WorkerHeartbeat(Base):
    """A short-lived liveness record used by the readiness probe."""

    __tablename__ = "worker_heartbeats"

    worker_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default="alive")
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    details: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)


class RuntimeSchemaVersion(Base):
    """Small migration marker so readiness can distinguish an old database."""

    __tablename__ = "runtime_schema_versions"

    name: Mapped[str] = mapped_column(String(80), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class GraphRun(Base):
    """Durable summary of one top-level graph execution."""

    __tablename__ = "graph_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    thread_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    owner_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="running", index=True)
    current_node: Mapped[str | None] = mapped_column(String(120), nullable=True)
    loop_count: Mapped[int] = mapped_column(Integer, default=0)
    schema_version: Mapped[str] = mapped_column(String(40), default="agent-state-v1")
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    diagnostics: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class NodeRun(Base):
    """Attempt-level trace for a graph node, retained independently of payloads."""

    __tablename__ = "node_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("graph_runs.id"), index=True)
    node: Mapped[str] = mapped_column(String(120), index=True)
    status: Mapped[str] = mapped_column(String(20), default="running")
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    error: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    output_refs: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ToolRun(Base):
    """One executed role tool, correlated with a graph without storing its payload."""

    __tablename__ = "tool_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # Tool calls may flush before the graph summary is committed, so this is
    # an indexed correlation key rather than an immediate foreign key.
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(120), index=True)
    status: Mapped[str] = mapped_column(String(20))
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    cached: Mapped[bool] = mapped_column(default=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentEvent(Base):
    """Append-only event envelope for route, node, tool, and commit traces."""

    __tablename__ = "agent_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("graph_runs.id"), index=True)
    node: Mapped[str | None] = mapped_column(String(120), nullable=True)
    agent: Mapped[str | None] = mapped_column(String(80), nullable=True)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    sequence: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    schema_version: Mapped[str] = mapped_column(String(40), default="event-v1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class GraphCheckpointOwner(Base):
    """Durable ownership gate checked before a LangGraph thread is resumed.

    The saver stores node snapshots; this row binds the thread to the business
    owner and scene so a guessed thread_id cannot cross that boundary.
    """

    __tablename__ = "graph_checkpoint_owners"

    thread_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(120), index=True)
    tenant_id: Mapped[str] = mapped_column(String(120), default="local")
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    interview_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    schema_version: Mapped[str] = mapped_column(String(40), default="agent-state.v1")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
