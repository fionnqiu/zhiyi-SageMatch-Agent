"""Checkpoint contracts and adapters for the application graph.

LangGraph's checkpointer API is intentionally kept at this module boundary.  The
application can use the in-memory implementation in tests, while deployments can
inject the optional PostgreSQL saver without making PostgreSQL a test dependency.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Mapping

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.platform.runtime import GraphCheckpointOwner


CHECKPOINT_SCHEMA_VERSION = "agent-state.v1"


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint cannot safely be read or written."""


class CheckpointRunActive(CheckpointValidationError):
    """A valid thread is temporarily held by another graph invocation."""


class CheckpointerBackendUnavailable(RuntimeError):
    """Raised when an optional production backend is not installed/configured."""


class CheckpointPolicy(BaseModel):
    """Bounds applied before state enters a checkpoint store."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    ttl_seconds: int | None = Field(default=24 * 60 * 60, ge=1)
    # Small values are useful in deterministic tests; production callers should
    # use the 256 KiB default or a larger deployment-specific limit.
    max_bytes: int = Field(default=256 * 1024, ge=128)


class CheckpointRecord(BaseModel):
    """Portable state record used by the explicit test/store API."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    owner_id: str
    tenant_id: str | None = None
    session_id: str | None = None
    interview_id: str | None = None
    schema_version: str
    saved_at: datetime
    expires_at: datetime | None = None
    state: dict[str, Any]


class CheckpointStore(ABC):
    """Small owner-aware store contract independent from LangGraph internals."""

    @abstractmethod
    def save(
        self,
        thread_id: str,
        state: Mapping[str, Any] | Any,
        *,
        owner_id: str,
        tenant_id: str | None = None,
        session_id: str | None = None,
        interview_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> CheckpointRecord:
        """Validate and save a state snapshot."""

    @abstractmethod
    def load(
        self,
        thread_id: str,
        *,
        owner_id: str,
        tenant_id: str | None = None,
        session_id: str | None = None,
        interview_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Load a state snapshot only after owner/scope validation."""


class MemoryCheckpointer(InMemorySaver, CheckpointStore):
    """LangGraph-compatible in-memory saver with explicit validation metadata.

    This implementation is deliberately limited to tests and local development.  It
    still subclasses LangGraph's saver so graph invocation exercises the same
    ``thread_id`` contract as a PostgreSQL deployment.
    """

    def __init__(self, *, policy: CheckpointPolicy | None = None) -> None:
        super().__init__()
        self.policy = policy or CheckpointPolicy()
        self.records: dict[str, CheckpointRecord] = {}
        self._thread_owners: dict[str, tuple[str, str | None]] = {}

    def save(
        self,
        thread_id: str,
        state: Mapping[str, Any] | Any,
        *,
        owner_id: str,
        tenant_id: str | None = None,
        session_id: str | None = None,
        interview_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> CheckpointRecord:
        """Validate a portable state snapshot before retaining it in memory."""
        normalized = _state_dict(state)
        _validate_state(
            normalized,
            policy=self.policy,
            owner_id=owner_id,
            session_id=session_id,
            interview_id=interview_id,
        )
        _validate_thread_owner(self._thread_owners, thread_id, owner_id, tenant_id)
        now = datetime.now(timezone.utc)
        record = CheckpointRecord(
            thread_id=thread_id,
            owner_id=owner_id,
            tenant_id=tenant_id,
            session_id=session_id or _optional_text(normalized.get("session_id")),
            interview_id=interview_id or _optional_text(normalized.get("interview_id")),
            schema_version=str(normalized.get("schema_version") or self.policy.schema_version),
            saved_at=now,
            expires_at=expires_at
            if expires_at is not None
            else (now + timedelta(seconds=self.policy.ttl_seconds) if self.policy.ttl_seconds else None),
            state=normalized,
        )
        self.records[thread_id] = record
        self._thread_owners[thread_id] = (owner_id, tenant_id)
        return record

    def load(
        self,
        thread_id: str,
        *,
        owner_id: str,
        tenant_id: str | None = None,
        session_id: str | None = None,
        interview_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the latest portable state, or ``None`` for a missing/expired record."""
        record = self.records.get(thread_id)
        if record is None:
            return None
        _validate_thread_owner(self._thread_owners, thread_id, owner_id, tenant_id)
        if record.expires_at and record.expires_at <= datetime.now(timezone.utc):
            self.records.pop(thread_id, None)
            return None
        if session_id is not None and record.session_id != session_id:
            raise CheckpointValidationError("session owner mismatch")
        if interview_id is not None and record.interview_id != interview_id:
            raise CheckpointValidationError("interview owner mismatch")
        _validate_state(
            record.state,
            policy=self.policy,
            owner_id=record.owner_id,
            session_id=record.session_id,
            interview_id=record.interview_id,
        )
        return dict(record.state)

    # These aliases make the explicit store API readable in adapters that call
    # checkpoint operations asynchronously without changing LangGraph's methods.
    save_state = save
    load_state = load

    def put(self, config: dict[str, Any], checkpoint: Any, metadata: Mapping[str, Any], new_versions: Mapping[str, Any]) -> dict[str, Any]:
        """Guard LangGraph writes with the owner binding when one is provided."""
        configurable = config.setdefault("configurable", {})
        thread_id = str(configurable.get("thread_id") or "")
        if not thread_id:
            raise CheckpointValidationError("thread_id is required")
        owner_id = _optional_text(configurable.get("owner_id"))
        tenant_id = _optional_text(configurable.get("tenant_id"))
        if owner_id:
            _validate_thread_owner(self._thread_owners, thread_id, owner_id, tenant_id)
            # LangGraph writes through put/aput, not the explicit save method.
            # Bind the first writer here too or a later owner could reuse the
            # same thread before any save() call populated the owner map.
            self._thread_owners.setdefault(thread_id, (owner_id, tenant_id))
        payload = checkpoint.get("channel_values", {}).get("payload") if isinstance(checkpoint, Mapping) else None
        if isinstance(payload, Mapping):
            _validate_state(
                payload,
                policy=self.policy,
                owner_id=owner_id or str(payload.get("user_id") or ""),
                session_id=_optional_text(payload.get("session_id")),
                interview_id=_optional_text(payload.get("interview_id")),
            )
        # LangGraph stores its own compact channel snapshot.  Metadata validation is
        # performed only when callers explicitly provide a schema version; raw graph
        # checkpoints do not contain the application's AgentState envelope.
        schema_version = _optional_text(configurable.get("schema_version")) or _optional_text(metadata.get("schema_version"))
        if schema_version and schema_version != self.policy.schema_version:
            raise CheckpointValidationError("checkpoint schema version mismatch")
        return super().put(config, checkpoint, metadata, new_versions)

    async def aput(self, config: dict[str, Any], checkpoint: Any, metadata: Mapping[str, Any], new_versions: Mapping[str, Any]) -> dict[str, Any]:
        """Async LangGraph hook with the same validation as the sync implementation."""
        return self.put(config, checkpoint, metadata, new_versions)

    def get_tuple(self, config: dict[str, Any]) -> Any:
        """Keep development reads subject to the same owner binding as writes."""
        configurable = config.get("configurable") or {}
        thread_id = _optional_text(configurable.get("thread_id"))
        owner_id = _optional_text(configurable.get("owner_id"))
        if not thread_id or not owner_id:
            raise CheckpointValidationError("thread_id and owner_id are required")
        _validate_thread_owner(self._thread_owners, thread_id, owner_id, _optional_text(configurable.get("tenant_id")))
        return super().get_tuple(config)


class PostgresCheckpointer:
    """Own the async PostgreSQL saver connection for the FastAPI lifespan.

    LangGraph's ``from_conn_string`` returns a context manager, not a saver.
    The owner must keep that context open for every graph invocation and close it
    on shutdown; a saver returned from a temporary context would hold a dead DB
    connection after startup.
    """

    def __init__(self, *, connection_string: str, policy: CheckpointPolicy | None = None, backend: Any | None = None) -> None:
        self.connection_string = connection_string
        self.policy = policy or CheckpointPolicy()
        self._backend = backend
        if backend is not None:
            self.available = True
            return
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # type: ignore

            self._backend_factory = AsyncPostgresSaver
            self.available = True
        except ImportError:
            self._backend_factory = None
            self.available = False

    def build(self) -> Any:
        """Return a managed async context for the real saver."""
        if self._backend is not None:
            return self._backend
        if self._backend_factory is None:
            raise CheckpointerBackendUnavailable(
                "langgraph-checkpoint-postgres is not installed; use MemoryCheckpointer for tests"
            )
        return self._backend_factory.from_conn_string(self.connection_string)

    @asynccontextmanager
    async def open(self) -> AsyncIterator[Any]:
        """Open, migrate and yield a saver for the whole application lifespan."""
        if self._backend is not None:
            yield self._backend
            return
        async with self.build() as saver:
            await saver.setup()
            # Guard queries use a separate connection. Interleaving them on
            # the saver connection would conflict with its pipeline writes.
            import psycopg
            from psycopg.rows import dict_row

            async with await psycopg.AsyncConnection.connect(
                self.connection_string, autocommit=True, row_factory=dict_row,
            ) as guard_conn:
                yield GuardedPostgresSaver(saver, guard_conn=guard_conn, policy=self.policy,
                                           connection_string=self.connection_string)

    @property
    def backend(self) -> Any:
        """Return the configured adapter; use ``open`` for a live connection."""
        return self.build()


class GuardedPostgresSaver(BaseCheckpointSaver):
    """Enforce durable ownership and bounds on every LangGraph saver access."""

    def __init__(self, saver: Any, *, guard_conn: Any, policy: CheckpointPolicy | None = None,
                 connection_string: str | None = None) -> None:
        super().__init__(serde=saver.serde)
        self.saver = saver
        self.guard_conn = guard_conn
        self.policy = policy or CheckpointPolicy()
        self.connection_string = connection_string

    @asynccontextmanager
    async def exclusive_run(self, thread_id: str) -> AsyncIterator[None]:
        """Hold a transaction-scoped lock across checkpoint read and graph commit.

        A dedicated connection keeps the lock independent of business commits and
        PostgreSQL releases it if the process dies mid-run.
        """
        if not self.connection_string:
            raise CheckpointerBackendUnavailable("checkpoint run lock needs a PostgreSQL connection")
        import psycopg

        async with await psycopg.AsyncConnection.connect(self.connection_string) as conn:
            async with conn.transaction():
                cursor = await conn.execute(
                    "SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0))", (thread_id,),
                )
                if not (await cursor.fetchone())[0]:
                    raise CheckpointRunActive("checkpoint run already active")
                yield

    @property
    def config_specs(self) -> list[Any]:
        return self.saver.config_specs

    def get_next_version(self, current: Any, channel: Any) -> Any:
        return self.saver.get_next_version(current, channel)

    def __getattr__(self, name: str) -> Any:
        # LangGraph also asks for version allocation and serializer metadata;
        # those are delegated, while data reads and writes below stay guarded.
        return getattr(self.saver, name)

    async def ping(self) -> bool:
        """Check the actual saver connection without fabricating a thread."""
        await self.guard_conn.execute("SELECT 1")
        return True

    async def _authorize(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        values = config.get("configurable") or {}
        thread_id = _optional_text(values.get("thread_id"))
        owner_id = _optional_text(values.get("owner_id"))
        tenant_id = _optional_text(values.get("tenant_id")) or "local"
        if not thread_id or not owner_id:
            raise CheckpointValidationError("thread_id and owner_id are required")
        cursor = await self.guard_conn.execute(
            "SELECT owner_id, tenant_id, session_id, interview_id, schema_version, expires_at "
            "FROM graph_checkpoint_owners WHERE thread_id = %s",
            (thread_id,),
        )
        row = await cursor.fetchone()
        if row is None or row["owner_id"] != owner_id or row["tenant_id"] != tenant_id:
            raise CheckpointValidationError("checkpoint owner mismatch")
        if row["schema_version"] != self.policy.schema_version:
            raise CheckpointValidationError("checkpoint schema version mismatch")
        expires = row["expires_at"]
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires <= datetime.now(timezone.utc):
            raise CheckpointValidationError("checkpoint expired")
        return row

    async def aget_tuple(self, config: Mapping[str, Any]) -> Any:
        await self._authorize(config)
        return await self.saver.aget_tuple(config)

    async def aput(self, config: Mapping[str, Any], checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        binding = await self._authorize(config)
        payload = checkpoint.get("channel_values", {}).get("payload") if isinstance(checkpoint, Mapping) else None
        if isinstance(payload, Mapping):
            self._validate_bound_payload(payload, binding)
        return await self.saver.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config: Mapping[str, Any], writes: Any, task_id: str, task_path: str = "") -> None:
        binding = await self._authorize(config)
        if len(json.dumps(writes, ensure_ascii=False, default=str).encode("utf-8")) > self.policy.max_bytes:
            raise CheckpointValidationError("checkpoint write size exceeds limit")
        # LangGraph can persist a node's pending payload before a full
        # checkpoint; guard that write with the same durable thread binding.
        for channel, value in writes:
            if channel == "payload" and isinstance(value, Mapping):
                self._validate_bound_payload(value, binding)
        await self.saver.aput_writes(config, writes, task_id, task_path)

    def _validate_bound_payload(self, payload: Mapping[str, Any], binding: Mapping[str, Any]) -> None:
        """Reject state that does not belong to the authorized thread."""
        if _optional_text(payload.get("user_id")) != binding["owner_id"]:
            raise CheckpointValidationError("checkpoint owner mismatch")
        if _optional_text(payload.get("tenant_id")) != binding["tenant_id"]:
            raise CheckpointValidationError("checkpoint tenant mismatch")
        if _optional_text(payload.get("session_id")) != binding["session_id"]:
            raise CheckpointValidationError("session owner mismatch")
        if _optional_text(payload.get("interview_id")) != binding["interview_id"]:
            raise CheckpointValidationError("interview owner mismatch")
        _validate_state(
            payload, policy=self.policy, owner_id=binding["owner_id"],
            session_id=binding["session_id"], interview_id=binding["interview_id"],
        )

    async def alist(self, config: Mapping[str, Any] | None, **kwargs: Any) -> AsyncIterator[Any]:
        if config is None:
            raise CheckpointValidationError("thread scope is required")
        await self._authorize(config)
        async for item in self.saver.alist(config, **kwargs):
            yield item


def derive_thread_id(*, session_id: str | None = None, interview_id: str | None = None, request_id: str | None = None) -> str:
    """Derive a stable, namespaced graph thread ID from the business owner."""
    for prefix, value in (("session", session_id), ("interview", interview_id), ("request", request_id)):
        clean = _optional_text(value)
        if clean:
            return f"{prefix}:{clean}"
    raise ValueError("session_id, interview_id, or request_id is required for thread_id")


thread_id_for = derive_thread_id


def checkpoint_config(*, thread_id: str, owner_id: str | None = None, tenant_id: str | None = None, schema_version: str = CHECKPOINT_SCHEMA_VERSION) -> dict[str, dict[str, str]]:
    """Build the LangGraph config with the metadata needed for owner checks."""
    configurable: dict[str, str] = {"thread_id": thread_id, "schema_version": schema_version}
    if owner_id:
        configurable["owner_id"] = owner_id
    if tenant_id:
        configurable["tenant_id"] = tenant_id
    return {"configurable": configurable}


def authorize_checkpoint(
    db: Session,
    thread_id: str,
    state: Mapping[str, Any] | Any,
    *,
    owner_id: str,
    tenant_id: str = "local",
    session_id: str | None = None,
    interview_id: str | None = None,
    policy: CheckpointPolicy | None = None,
) -> GraphCheckpointOwner:
    """Bind or verify a thread before the saver may read or write its state."""
    normalized = _state_dict(state)
    current_policy = policy or CheckpointPolicy()
    if normalized.get("user_id") not in (None, "", owner_id):
        raise CheckpointValidationError("checkpoint owner mismatch")
    if normalized.get("tenant_id") not in (None, "", tenant_id):
        raise CheckpointValidationError("checkpoint tenant mismatch")
    _validate_state(
        normalized,
        policy=current_policy,
        owner_id=owner_id,
        session_id=session_id,
        interview_id=interview_id,
    )
    if not thread_id:
        raise CheckpointValidationError("thread_id is required")
    now = datetime.now(timezone.utc)
    row = db.get(GraphCheckpointOwner, thread_id)
    if row is None:
        row = GraphCheckpointOwner(
            thread_id=thread_id,
            owner_id=owner_id,
            tenant_id=tenant_id,
            session_id=session_id,
            interview_id=interview_id,
            schema_version=current_policy.schema_version,
            expires_at=now + timedelta(seconds=current_policy.ttl_seconds or 86400),
        )
        try:
            with db.begin_nested():
                db.add(row)
                db.flush()
        except IntegrityError:
            # A concurrent request won the unique thread key. Re-read its
            # ownership before allowing this request to touch the saver.
            row = db.get(GraphCheckpointOwner, thread_id)
            if row is None:
                raise
    if (row.owner_id, row.tenant_id, row.session_id, row.interview_id) != (
        owner_id, tenant_id, session_id, interview_id,
    ):
        raise CheckpointValidationError("checkpoint owner mismatch")
    if row.schema_version != current_policy.schema_version:
        raise CheckpointValidationError("checkpoint schema version mismatch")
    expires = row.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires <= now:
        raise CheckpointValidationError("checkpoint expired")
    # Active conversations renew their recovery window; idle checkpoints still
    # expire and cannot be silently resumed after their TTL.
    if current_policy.ttl_seconds:
        row.expires_at = now + timedelta(seconds=current_policy.ttl_seconds)
    db.flush()
    return row


def _state_dict(state: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(state, Mapping):
        return dict(state)
    if hasattr(state, "to_json_dict"):
        return dict(state.to_json_dict())
    if hasattr(state, "model_dump"):
        return dict(state.model_dump(mode="json"))
    raise CheckpointValidationError("checkpoint state must be a mapping or Pydantic model")


def _validate_state(
    state: Mapping[str, Any],
    *,
    policy: CheckpointPolicy,
    owner_id: str,
    session_id: str | None,
    interview_id: str | None,
) -> None:
    if not _optional_text(owner_id):
        raise CheckpointValidationError("owner is required")
    schema_version = str(state.get("schema_version") or "")
    if schema_version != policy.schema_version:
        raise CheckpointValidationError("checkpoint schema version mismatch")
    actual_session = _optional_text(state.get("session_id"))
    actual_interview = _optional_text(state.get("interview_id"))
    if session_id is not None and actual_session not in {None, session_id}:
        raise CheckpointValidationError("session owner mismatch")
    if interview_id is not None and actual_interview not in {None, interview_id}:
        raise CheckpointValidationError("interview owner mismatch")
    encoded = json.dumps(state, ensure_ascii=False, default=str, separators=(",", ":")).encode("utf-8")
    if len(encoded) > policy.max_bytes:
        raise CheckpointValidationError("checkpoint size exceeds limit")


def _validate_thread_owner(owners: dict[str, tuple[str, str | None]], thread_id: str, owner_id: str, tenant_id: str | None) -> None:
    if not _optional_text(owner_id):
        raise CheckpointValidationError("owner is required")
    previous = owners.get(thread_id)
    if previous and previous != (owner_id, tenant_id):
        raise CheckpointValidationError("checkpoint owner mismatch")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointPolicy",
    "CheckpointRecord",
    "CheckpointStore",
    "CheckpointValidationError",
    "CheckpointerBackendUnavailable",
    "MemoryCheckpointer",
    "GuardedPostgresSaver",
    "PostgresCheckpointer",
    "checkpoint_config",
    "authorize_checkpoint",
    "derive_thread_id",
    "thread_id_for",
]
