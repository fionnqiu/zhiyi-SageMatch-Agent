"""Durable thread ownership checks before LangGraph resumes a checkpoint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agents.orchestration.checkpoint import CheckpointPolicy, CheckpointRunActive, CheckpointValidationError, authorize_checkpoint
from app.agents.orchestration.checkpoint import GuardedPostgresSaver, MemoryCheckpointer, checkpoint_config
from app.agents.orchestration.workflows import run_business_graph
from app.agents.orchestration.graph import AgentState as GraphAgentState, build_application_graph
from app.agents.contracts.state import AgentState
from app.core.db import Base
from app.models.platform.runtime import GraphCheckpointOwner


def test_checkpoint_http_conflict_distinguishes_active_run_from_invalid_resume() -> None:
    """An occupied run is retryable without exposing its thread or owner."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.main import checkpoint_validation_error

    test_app = FastAPI()
    test_app.add_exception_handler(CheckpointValidationError, checkpoint_validation_error)

    @test_app.get("/checkpoint/{kind}")
    def conflict(kind: str) -> None:
        if kind == "active":
            raise CheckpointRunActive("secret thread:123")
        raise CheckpointValidationError("secret owner:456")

    with TestClient(test_app) as client:
        active = client.get("/checkpoint/active")
        invalid = client.get("/checkpoint/invalid")
    active_body = active.json()
    invalid_body = invalid.json()
    assert active.status_code == invalid.status_code == 409
    assert active_body["error_code"] == "checkpoint_run_active"
    assert active_body["retryable"] is True
    assert invalid_body["error_code"] == "checkpoint_conflict"
    assert invalid_body["retryable"] is False
    assert "secret" not in active.text + invalid.text


def test_checkpoint_owner_is_bound_to_session_and_schema() -> None:
    """A guessed thread id cannot be resumed by another owner or scene."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[GraphCheckpointOwner.__table__])
    with Session(engine) as db:
        state = AgentState(thread_id="session:s1", session_id="s1", user_id="user-1")
        row = authorize_checkpoint(db, "session:s1", state, owner_id="user-1", session_id="s1")
        db.commit()
        assert row.owner_id == "user-1"
        assert authorize_checkpoint(db, "session:s1", state, owner_id="user-1", session_id="s1") is row
        with pytest.raises(CheckpointValidationError, match="owner"):
            authorize_checkpoint(db, "session:s1", state, owner_id="user-2", session_id="s1")
        with pytest.raises(CheckpointValidationError, match="schema"):
            authorize_checkpoint(
                db, "session:s1", {**state.as_checkpoint(), "schema_version": "old"},
                owner_id="user-1", session_id="s1",
            )


def test_expired_checkpoint_cannot_resume() -> None:
    """TTL is enforced against the persisted row before the saver is read."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[GraphCheckpointOwner.__table__])
    with Session(engine) as db:
        state = AgentState(thread_id="request:r1", user_id="user-1")
        row = authorize_checkpoint(db, "request:r1", state, owner_id="user-1")
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
        with pytest.raises(CheckpointValidationError, match="expired"):
            authorize_checkpoint(db, "request:r1", state, owner_id="user-1")


def test_checkpoint_size_limit_is_checked_before_binding() -> None:
    """Oversized state is rejected without creating an ownership row."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[GraphCheckpointOwner.__table__])
    with Session(engine) as db:
        state = AgentState(original_query="x" * 1000, user_id="user-1")
        with pytest.raises(CheckpointValidationError, match="size"):
            authorize_checkpoint(
                db, "request:r2", state, owner_id="user-1",
                policy=CheckpointPolicy(max_bytes=500),
            )
        assert db.get(GraphCheckpointOwner, "request:r2") is None


def test_langgraph_memory_saver_binds_thread_owner() -> None:
    """The saver used by graph.compile enforces owner isolation on real writes."""
    saver = MemoryCheckpointer()
    graph = build_application_graph(checkpointer=saver)
    state = GraphAgentState(
        request_id="r1", run_id="r1", thread_id="request:r1",
        requested_mode="unsupported", original_query="unknown",
    )
    first = asyncio.run(graph.ainvoke(
        state, config=checkpoint_config(thread_id="request:r1", owner_id="user-1"),
    ))
    assert first["status"] == "unsupported"
    with pytest.raises(CheckpointValidationError, match="owner"):
        asyncio.run(graph.ainvoke(
            state, config=checkpoint_config(thread_id="request:r1", owner_id="user-2"),
        ))


def test_business_graph_resumes_only_the_same_unfinished_run() -> None:
    """A process loss at commit resumes from the saved node without rerouting."""
    class ProcessLost(BaseException):
        pass

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[GraphCheckpointOwner.__table__])
    saver = MemoryCheckpointer()
    calls = 0
    recovered_traces = []

    async def action() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProcessLost()
        return "persisted-result"

    async def scenario() -> None:
        with Session(engine) as db:
            with pytest.raises(ProcessLost):
                await run_business_graph(db, mode="live_interview", action=action,
                                         interview_id="i1", checkpointer=saver, run_id="r1")
            snapshot = await build_application_graph(checkpointer=saver).compiled.aget_state(
                checkpoint_config(thread_id="interview:i1", owner_id="local-user")
            )
            assert snapshot.next == ("commit_node",)
            with pytest.raises(CheckpointValidationError, match="unfinished"):
                await run_business_graph(db, mode="live_interview", action=action,
                                         interview_id="i1", checkpointer=saver, run_id="r2")
            with pytest.raises(CheckpointValidationError, match="request mismatch"):
                await run_business_graph(db, mode="live_interview", action=action,
                                         original_query="changed answer", interview_id="i1",
                                         checkpointer=saver, run_id="r1")
            assert calls == 1
            with patch("app.agents.orchestration.workflows.persist_graph_trace",
                       side_effect=lambda _db, outcome: recovered_traces.append(outcome)):
                assert await run_business_graph(db, mode="live_interview", action=action,
                                                interview_id="i1", checkpointer=saver, run_id="r1") == "persisted-result"
            assert calls == 2
            assert recovered_traces[-1]["diagnostics"]["checkpoint_recovered"] is True
            with pytest.raises(CheckpointValidationError, match="already completed"):
                await run_business_graph(db, mode="live_interview", action=action,
                                         interview_id="i1", checkpointer=saver, run_id="r1")
            assert calls == 2

    asyncio.run(scenario())


def test_exclusive_checkpoint_run_derives_thread_when_run_id_is_none() -> None:
    """The lock and graph must use the same generated request thread."""
    class LockingMemorySaver(MemoryCheckpointer):
        locked_threads: list[str] = []

        @asynccontextmanager
        async def exclusive_run(self, thread_id: str):
            self.locked_threads.append(thread_id)
            yield

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[GraphCheckpointOwner.__table__])
    saver = LockingMemorySaver()

    async def action() -> str:
        return "committed"

    async def scenario() -> None:
        with Session(engine) as db:
            assert await run_business_graph(db, mode="live_interview", action=action,
                                            checkpointer=saver, run_id=None) == "committed"
            assert len(saver.locked_threads) == 1
            thread_id = saver.locked_threads[0]
            assert thread_id.startswith("request:")
            assert db.get(GraphCheckpointOwner, thread_id) is not None
            snapshot = await build_application_graph(checkpointer=saver).compiled.aget_state(
                checkpoint_config(thread_id=thread_id, owner_id="local-user")
            )
            assert snapshot.values["payload"]["run_id"] == thread_id.removeprefix("request:")

    asyncio.run(scenario())


def test_memory_checkpoint_read_checks_owner() -> None:
    """A guessed thread ID cannot disclose snapshot values in development."""
    saver = MemoryCheckpointer()
    graph = build_application_graph(checkpointer=saver)

    async def scenario() -> None:
        await graph.ainvoke(GraphAgentState(request_id="r1", run_id="r1", thread_id="request:r1",
                                           requested_mode="unsupported"),
                            config=checkpoint_config(thread_id="request:r1", owner_id="user-1"))
        with pytest.raises(CheckpointValidationError, match="owner"):
            await graph.compiled.aget_state(checkpoint_config(thread_id="request:r1", owner_id="user-2"))

    asyncio.run(scenario())


def test_postgres_saver_rejects_payload_outside_thread_binding() -> None:
    """A valid thread config cannot write another user's or scene's state."""
    class Cursor:
        async def fetchone(self):
            return {
                "owner_id": "user-1", "tenant_id": "local", "session_id": "s1",
                "interview_id": None, "schema_version": "agent-state.v1",
                "expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
            }

    class GuardConnection:
        async def execute(self, *_args):
            return Cursor()

    class Backend:
        serde = None
        config_specs = []
        writes = 0
        pending_writes = 0

        async def aput(self, *_args):
            self.writes += 1

        async def aput_writes(self, *_args):
            self.pending_writes += 1

    async def scenario() -> None:
        backend = Backend()
        saver = GuardedPostgresSaver(backend, guard_conn=GuardConnection())
        config = checkpoint_config(thread_id="session:s1", owner_id="user-1")
        state = AgentState(user_id="user-1", session_id="s1").as_checkpoint()
        for changes, message in (
            ({"user_id": "user-2"}, "owner"),
            ({"tenant_id": "other"}, "tenant"),
            ({"session_id": "s2"}, "session"),
            ({"interview_id": "i2"}, "interview"),
        ):
            with pytest.raises(CheckpointValidationError, match=message):
                await saver.aput(config, {"channel_values": {"payload": {**state, **changes}}}, {}, {})
        assert backend.writes == 0
        await saver.aput(config, {"channel_values": {"payload": state}}, {}, {})
        assert backend.writes == 1
        with pytest.raises(CheckpointValidationError, match="owner"):
            await saver.aput_writes(config, [("payload", {**state, "user_id": "user-2"})], "task-1")
        with pytest.raises(CheckpointValidationError, match="session"):
            await saver.aput_writes(config, [("payload", {**state, "session_id": "s2"})], "task-1")
        assert backend.pending_writes == 0
        await saver.aput_writes(config, [("__error__", "failed")], "task-1")
        await saver.aput_writes(config, [("payload", state)], "task-1")
        assert backend.pending_writes == 2

    asyncio.run(scenario())
