"""Opt-in isolated PostgreSQL proof for interrupted question generation."""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import create_engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from app import schemas
from app.api.interviews.interview import _publish_generation_event, generate_interview
from app.agents.orchestration.checkpoint import PostgresCheckpointer, checkpoint_config
from app.agents.orchestration.workflows import run_business_graph
from app.core.config import get_settings
from app.core.db import Base
from app.core.loop_policy import selector_loop_factory
from app.models.platform.runtime import GraphCheckpointOwner
from app.models import ChatMessage, ChatSession, DurableJob, Interview, QuestionSet, StreamEvent, StreamRun
from app.services.operations.jobs import claim_job_by_id, complete_job, create_job, fail_job
from app.services.chat.stream_events import create_stream_run


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_generation_events_follow_durable_job_fence_and_terminal_result() -> None:
    """A reclaimed worker cannot publish frames or end a retrying stream."""
    name = f"sagematch_generation_fence_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    test_url = base_url.set(database=name)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        engine = create_engine(test_url)
        try:
            Base.metadata.create_all(engine)
            with Session(engine) as db:
                run = create_stream_run(db, "interview_generation", "local-user")
                job = create_job(db, "interview_generation", run.id,
                                 idempotency_key=f"generation:{run.id}")
                db.commit()
                run_id, job_id = run.id, job.id
                assert claim_job_by_id(db, job_id, "old-worker") is not None
                db.commit()
                assert _publish_generation_event(db, job_id, run_id, {"type": "meta"},
                                                 worker_id="old-worker") is not None
                job = db.get(DurableJob, job_id)
                job.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
                db.commit()
                assert claim_job_by_id(db, job_id, "new-worker") is not None
                db.commit()
                assert _publish_generation_event(db, job_id, run_id, {"type": "meta"},
                                                 worker_id="old-worker") is None
                assert fail_job(db, job_id, "transient", retryable=True,
                                worker_id="new-worker", retry_delay_seconds=0) is not None
                db.commit()
                assert _publish_generation_event(db, job_id, run_id,
                                                 {"type": "error", "message": "failed"}) is None
                assert db.get(StreamRun, run_id).status == "running"
                assert claim_job_by_id(db, job_id, "final-worker") is not None
                db.commit()
                assert complete_job(db, job_id, worker_id="final-worker",
                                    result={"interview_id": "result-1"}) is not None
                db.commit()
                # The committed result, rather than the former worker, owns the final frame.
                with patch("app.api.interviews.interview._generation_result_interview",
                           return_value={"id": "result-1", "title": "岗位", "status": "ready"}):
                    assert _publish_generation_event(db, job_id, run_id, {"type": "done"}) is not None
                assert db.get(StreamRun, run_id).status == "completed"
                assert [row.event_type for row in db.query(StreamEvent).all()] == ["meta", "done"]
                failed_run = create_stream_run(db, "interview_generation", "local-user")
                failed_job = create_job(db, "interview_generation", failed_run.id,
                                        idempotency_key=f"generation:{failed_run.id}")
                db.commit()
                assert claim_job_by_id(db, failed_job.id, "failed-worker") is not None
                db.commit()
                assert fail_job(db, failed_job.id, "invalid_input", retryable=False,
                                worker_id="failed-worker") is not None
                db.commit()
                assert _publish_generation_event(db, failed_job.id, failed_run.id,
                                                 {"type": "error"}) is not None
                assert db.get(StreamRun, failed_run.id).status == "completed"
        finally:
            engine.dispose()
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_postgres_generation_restarts_without_raw_job_in_checkpoint() -> None:
    name = f"sagematch_generation_test_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    test_url = base_url.set(database=name)
    saver_dsn = test_url.set(drivername="postgresql").render_as_string(hide_password=False)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        engine = create_engine(test_url)
        try:
            Base.metadata.create_all(engine, tables=[GraphCheckpointOwner.__table__])
            raw_job = "后端开发岗位负责缓存一致性与线上故障处理"
            digest = hashlib.sha256(raw_job.encode("utf-8")).hexdigest()
            calls = 0

            class ProcessLost(BaseException):
                pass

            async def author(_state):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise ProcessLost()
                return {"result": {"valid": True}}

            async def persist():
                return "same-run-committed"

            async def scenario():
                adapter = PostgresCheckpointer(connection_string=saver_dsn)
                args = dict(mode="interview_generation", run_id="generation-restart",
                            original_query=digest, restart_incomplete=True,
                            stages={"interview_generation.author": author}, action=persist)
                async with adapter.open() as saver:
                    with Session(engine) as db, patch("app.agents.orchestration.workflows.persist_graph_trace"):
                        with pytest.raises(ProcessLost):
                            await run_business_graph(db, checkpointer=saver, **args)
                async with adapter.open() as saver:
                    config = checkpoint_config(thread_id="request:generation-restart", owner_id="local-user")
                    snapshot = await saver.aget_tuple(config)
                    assert snapshot is not None
                    assert raw_job not in str(snapshot.checkpoint)
                    with Session(engine) as db, patch("app.agents.orchestration.workflows.persist_graph_trace"):
                        assert await run_business_graph(db, checkpointer=saver, **args) == "same-run-committed"
                assert calls == 2

            with asyncio.Runner(loop_factory=selector_loop_factory()) as runner:
                runner.run(scenario())
        finally:
            engine.dispose()
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_postgres_generation_http_commit_and_same_key_replay() -> None:
    name = f"sagematch_generation_http_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    test_url = base_url.set(database=name)
    saver_dsn = test_url.set(drivername="postgresql").render_as_string(hide_password=False)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        engine = create_engine(test_url)
        try:
            Base.metadata.create_all(engine)
            SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

            async def scenario():
                adapter = PostgresCheckpointer(connection_string=saver_dsn)
                async with adapter.open() as saver:
                    request = SimpleNamespace(
                        headers={"Idempotency-Key": "same-generation"},
                        app=SimpleNamespace(state=SimpleNamespace(checkpointer=saver)),
                    )
                    payload = schemas.InterviewGenerateIn(content="后端开发岗位负责缓存一致性和线上故障排查")
                    with patch("app.api.interviews.interview.SessionLocal", SessionLocal), patch(
                        "app.agents.workflows.generation.recall_snippets", new=AsyncMock(return_value=[])
                    ), patch("app.agents.workflows.generation.llm.llm_available", return_value=False):
                        with SessionLocal() as db:
                            first = await generate_interview(payload, request, db)
                        first_meta = await asyncio.wait_for(anext(first.body_iterator), timeout=3)
                        assert '"type": "meta"' in first_meta
                        with SessionLocal() as duplicate_db:
                            follower = await generate_interview(payload, request, duplicate_db)
                            jobs = duplicate_db.query(DurableJob).all()
                            assert len(jobs) == 1 and jobs[0].attempts == 1
                            staged = duplicate_db.get(ChatSession, first.headers["X-Run-ID"])
                            assert staged is not None and staged.origin == "interview"
                            assert duplicate_db.query(ChatMessage).filter(
                                ChatMessage.session_id == staged.id, ChatMessage.role == "user",
                            ).one().content == payload.content
                        first_followed = await asyncio.wait_for(anext(follower.body_iterator), timeout=3)
                        assert first_followed == first_meta
                        frames = [first_meta, *[frame async for frame in first.body_iterator]]
                        assert any('"type": "done"' in frame for frame in frames)
                        followed = [first_followed, *[frame async for frame in follower.body_iterator]]
                        assert any('"type": "done"' in frame for frame in followed)
                        with SessionLocal() as db:
                            replay = await generate_interview(payload, request, db)
                            repeated = [frame async for frame in replay.body_iterator]
                        assert repeated == frames
                    with SessionLocal() as db:
                        assert db.query(QuestionSet).count() == 1
                        assert db.query(Interview).count() == 1
                        assert db.query(StreamRun).one().status == "completed"

            with asyncio.Runner(loop_factory=selector_loop_factory()) as runner:
                runner.run(scenario())
        finally:
            engine.dispose()
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_postgres_running_queue_recovers_expired_generation_and_missing_done_frame() -> None:
    name = f"sagematch_generation_boot_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    test_url = base_url.set(database=name)
    saver_dsn = test_url.set(drivername="postgresql").render_as_string(hide_password=False)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        engine = create_engine(test_url)
        try:
            Base.metadata.create_all(engine)
            SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

            async def wait_done(run_id: str) -> None:
                for _ in range(80):
                    with SessionLocal() as db:
                        if db.get(StreamRun, run_id).status == "completed":
                            return
                    await asyncio.sleep(0.1)
                raise AssertionError("generation recovery did not complete")

            async def scenario():
                from app.api.interviews import interview as interview_api

                adapter = PostgresCheckpointer(connection_string=saver_dsn)
                async with adapter.open() as saver:
                    request = SimpleNamespace(
                        headers={"Idempotency-Key": "accepted-before-crash"},
                        app=SimpleNamespace(state=SimpleNamespace(checkpointer=saver)),
                    )
                    payload = schemas.InterviewGenerateIn(content="后端开发岗位负责缓存一致性和线上故障排查")
                    with patch("app.api.interviews.interview.SessionLocal", SessionLocal), patch(
                        "app.agents.workflows.generation.recall_snippets", new=AsyncMock(return_value=[])
                    ), patch("app.agents.workflows.generation.llm.llm_available", return_value=False):
                        with SessionLocal() as db:
                            accepted = await generate_interview(payload, request, db)
                            run_id = accepted.headers["X-Run-ID"]
                        # The recovery poller is already running when this
                        # claimed request loses its worker and its lease expires.
                        interview_api.start_generation_worker(saver)
                        try:
                            await asyncio.sleep(0.1)
                            with SessionLocal() as db:
                                job = db.query(DurableJob).filter(DurableJob.business_key == run_id).one()
                                job.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
                                db.commit()
                            await wait_done(run_id)
                        finally:
                            await asyncio.wait_for(interview_api.stop_generation_worker(), timeout=5)
                        with SessionLocal() as db:
                            assert db.query(Interview).count() == 1
                            assert db.query(QuestionSet).count() == 1
                            assert [row.event_type for row in db.query(StreamEvent).filter(
                                StreamEvent.run_id == run_id,
                            ).all()].count("done") == 1
                            interview_id = db.query(Interview).one().id
                            gap = create_stream_run(db, "interview_generation", "local-user",
                                                    run_id="generation-gap")
                            gap_job = create_job(db, "interview_generation", gap.id,
                                                 idempotency_key="generation:generation-gap")
                            gap_job.status = "succeeded"
                            gap_job.result = {"interview_id": interview_id}
                            db.commit()
                        interview_api.start_generation_worker(saver)
                        try:
                            await wait_done("generation-gap")
                        finally:
                            await asyncio.wait_for(interview_api.stop_generation_worker(), timeout=5)
                        with SessionLocal() as db:
                            assert db.query(StreamEvent).filter(
                                StreamEvent.run_id == "generation-gap", StreamEvent.event_type == "done",
                            ).count() == 1

            with asyncio.Runner(loop_factory=selector_loop_factory()) as runner:
                runner.run(scenario())
        finally:
            engine.dispose()
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()
