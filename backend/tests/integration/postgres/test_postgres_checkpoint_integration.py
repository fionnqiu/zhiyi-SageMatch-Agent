"""Opt-in live PostgreSQL recovery check using an isolated temporary database."""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
import psycopg
from psycopg import sql
from sqlalchemy import create_engine, make_url
from sqlalchemy.orm import Session

from app.agents.orchestration.checkpoint import CheckpointValidationError, PostgresCheckpointer, authorize_checkpoint, checkpoint_config
from app.agents.orchestration.graph import AgentState, build_application_graph
from app.agents.orchestration.workflows import run_business_graph
from app.core.config import get_settings
from app.core.db import Base
from app.models.platform.runtime import GraphCheckpointOwner


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_postgres_checkpoint_recovers_after_saver_reconnect() -> None:
    """A new saver process can read a prior graph run; ownership remains durable."""
    settings = get_settings()
    name = f"sagematch_checkpoint_test_{uuid.uuid4().hex[:10]}"
    base_url = make_url(settings.database_url)
    admin_url = base_url.set(database="postgres", drivername="postgresql")
    test_url = base_url.set(database=name)
    admin_dsn = admin_url.render_as_string(hide_password=False)
    saver_dsn = test_url.set(drivername="postgresql").render_as_string(hide_password=False)
    created = False
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        engine = create_engine(test_url)
        try:
            Base.metadata.create_all(engine, tables=[GraphCheckpointOwner.__table__])
            state = AgentState(
                request_id="checkpoint-r1", run_id="checkpoint-r1", thread_id="request:checkpoint-r1",
                requested_mode="unsupported", original_query="unrouted", user_id="owner-1",
            )
            with Session(engine) as db:
                authorize_checkpoint(db, state.thread_id, state, owner_id="owner-1")
                db.commit()

            async def roundtrip() -> None:
                adapter = PostgresCheckpointer(connection_string=saver_dsn)
                config = checkpoint_config(thread_id=state.thread_id, owner_id="owner-1")
                class ProcessLost(BaseException):
                    pass

                commit_calls = 0

                async def interrupted_action() -> str:
                    nonlocal commit_calls
                    commit_calls += 1
                    if commit_calls == 1:
                        raise ProcessLost()
                    return "resumed"

                async with adapter.open() as saver:
                    graph = build_application_graph(checkpointer=saver)
                    result = await graph.ainvoke(state, config=config)
                    assert result["status"] == "unsupported"
                    with Session(engine) as db:
                        with pytest.raises(ProcessLost):
                            await run_business_graph(
                                db, mode="live_interview", action=interrupted_action,
                                interview_id="i1", checkpointer=saver, run_id="interrupted-r1",
                            )
                # A fresh connection proves recovery is not process-memory state.
                async with adapter.open() as saver:
                    graph = build_application_graph(checkpointer=saver)
                    snapshot = await graph.compiled.aget_state(config)
                    assert snapshot.values["payload"]["run_id"] == state.run_id
                    with Session(engine) as db, patch("app.agents.orchestration.workflows.persist_graph_trace"):
                        recovered = await run_business_graph(
                            db, mode="live_interview", action=interrupted_action,
                            interview_id="i1", checkpointer=saver, run_id="interrupted-r1",
                        )
                        assert recovered == "resumed"
                        assert commit_calls == 2
                        with pytest.raises(CheckpointValidationError, match="already completed"):
                            await run_business_graph(
                                db, mode="live_interview", action=interrupted_action,
                                interview_id="i1", checkpointer=saver, run_id="interrupted-r1",
                            )
                        assert commit_calls == 2
                    with pytest.raises(CheckpointValidationError, match="owner"):
                        await saver.aget_tuple(checkpoint_config(thread_id=state.thread_id, owner_id="owner-2"))
                    # The authorized config cannot persist state from another
                    # owner or business scene through LangGraph's write hook.
                    for state_changes, message in (({"user_id": "owner-2"}, "owner"),
                                                   ({"session_id": "other-session"}, "session")):
                        with pytest.raises(CheckpointValidationError, match=message):
                            await saver.aput(config, {
                                "channel_values": {"payload": {**state.to_json_dict(), **state_changes}},
                            }, {}, {})
                    # Pending node writes also reach the PostgreSQL saver; a
                    # rejected payload must leave no checkpoint_writes row.
                    for state_changes, message in (({"user_id": "owner-2"}, "owner"),
                                                   ({"session_id": "other-session"}, "session")):
                        task_id = f"foreign-{message}-{uuid.uuid4().hex}"
                        with pytest.raises(CheckpointValidationError, match=message):
                            await saver.aput_writes(
                                config, [("payload", {**state.to_json_dict(), **state_changes})], task_id,
                            )
                        cursor = await saver.guard_conn.execute(
                            "SELECT COUNT(*) AS count FROM checkpoint_writes WHERE task_id = %s", (task_id,),
                        )
                        assert (await cursor.fetchone())["count"] == 0
                    with Session(engine) as db:
                        row = db.get(GraphCheckpointOwner, state.thread_id)
                        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
                        db.commit()
                    with pytest.raises(CheckpointValidationError, match="expired"):
                        await saver.aget_tuple(config)

            # Windows async Psycopg requires the same Selector loop configured
            # for Uvicorn; this also tests the documented runtime setup.
            from app.core.loop_policy import selector_loop_factory

            with asyncio.Runner(loop_factory=selector_loop_factory()) as runner:
                runner.run(roundtrip())
            with Session(engine) as db:
                with pytest.raises(ValueError, match="owner"):
                    authorize_checkpoint(db, state.thread_id, state, owner_id="owner-2")
        finally:
            engine.dispose()
    finally:
        if created:
            admin.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
            )
        admin.close()
