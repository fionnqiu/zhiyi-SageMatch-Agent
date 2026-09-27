"""Exercise report leases across independent processes against disposable PostgreSQL."""

from __future__ import annotations

import multiprocessing as mp
import os
import uuid
import asyncio
from datetime import datetime, timedelta, timezone

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import create_engine, make_url
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import Base
from app.models import DurableJob, Interview, JobProfile, Report
from app.services.interviews.interview import _store_report
from app.services.operations.jobs import claim_job, complete_job, create_job, reclaim_expired_jobs


def _resume_worker(test_url: str, gate: mp.synchronize.Event, entered: mp.Queue, release: mp.synchronize.Event, result: mp.Queue) -> None:
    """Resume one durable graph from an interpreter with its own saver connection."""
    from app.agents.orchestration.checkpoint import PostgresCheckpointer
    from app.agents.orchestration.workflows import run_business_graph, CheckpointValidationError
    from app.core.loop_policy import selector_loop_factory

    engine = create_engine(test_url, connect_args={"connect_timeout": 5})
    saver_dsn = make_url(test_url).set(drivername="postgresql").render_as_string(hide_password=False)

    async def run() -> None:
        async with PostgresCheckpointer(connection_string=saver_dsn).open() as saver:
            assert gate.wait(10)
            with Session(engine) as db:
                async def commit() -> str:
                    entered.put(os.getpid())
                    assert release.wait(10)
                    db.add(Report(id=uuid.uuid4().hex, interview_id="interview-1", review="resumed", score=80))
                    db.commit()
                    return "committed"

                try:
                    value = await run_business_graph(db, mode="live_interview", action=commit,
                                                     interview_id="interview-1", run_id="same-run",
                                                     checkpointer=saver)
                    result.put(("ok", value))
                except CheckpointValidationError as exc:
                    result.put(("conflict", str(exc)))

    try:
        with asyncio.Runner(loop_factory=selector_loop_factory()) as runner:
            runner.run(run())
    finally:
        engine.dispose()


def _worker(test_url: str, action: str, start: mp.synchronize.Event | None, result: mp.Queue) -> None:
    """Use a fresh engine and connection, as a separate server instance would."""
    engine = create_engine(test_url, connect_args={"connect_timeout": 5})
    try:
        if start is not None:
            assert start.wait(10), "workers did not start together"
        with Session(engine) as db:
            job = claim_job(db, action, lease_seconds=30, kinds=["report_generation"])
            db.commit()
            if job is None:
                result.put((action, "idle"))
                return
            if action == "lost":
                # A clean process exit after the claim models abrupt host loss:
                # no Python finally block releases the persisted lease.
                return
            current = (
                db.query(DurableJob)
                .filter(DurableJob.id == job.id, DurableJob.status == "running",
                        DurableJob.worker_id == action, DurableJob.lease_until > datetime.now(timezone.utc))
                .with_for_update()
                .one_or_none()
            )
            assert current is not None
            _store_report(db, job.business_key, {"score": 80, "review": action, "issues": []}, commit=False)
            assert complete_job(db, job.id, worker_id=action, result={"interview_id": job.business_key})
            db.commit()
            result.put((action, "committed"))
    finally:
        engine.dispose()


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_separate_workers_recover_lost_report_lease_and_commit_once() -> None:
    """A dead owner cannot commit, and competing replacement instances commit once."""
    name = f"sagematch_multi_test_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    test_url = base_url.set(database=name)
    worker_url = test_url.render_as_string(hide_password=False)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        engine = create_engine(test_url)
        children: list[mp.Process] = []
        try:
            Base.metadata.create_all(engine)
            with Session(engine) as db:
                db.add(JobProfile(id="profile-1", user_id=get_settings().anonymous_user_id,
                                  raw_text="Backend", job_title="Backend"))
                db.flush()
                db.add(Interview(id="interview-1", profile_id="profile-1", title="Interview", status="ended"))
                job = create_job(db, "report_generation", "interview-1", idempotency_key="report:interview-1")
                db.commit()
                job_id = job.id

            # Spawn guarantees distinct interpreter processes and database connections.
            ctx = mp.get_context("spawn")
            output = ctx.Queue()
            lost = ctx.Process(target=_worker, args=(worker_url, "lost", None, output))
            children.append(lost)
            lost.start()
            lost.join(15)
            assert lost.exitcode == 0
            with Session(engine) as db:
                row = db.get(DurableJob, job_id)
                assert row.status == "running" and row.worker_id == "lost" and row.attempts == 1
                assert db.query(Report).count() == 0
                # Advance only this disposable row's lease to model elapsed time.
                row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
                db.commit()
                assert reclaim_expired_jobs(db) == 1
                db.commit()
                assert complete_job(db, job_id, worker_id="lost") is None
                db.rollback()

            gate = ctx.Event()
            peers = [ctx.Process(target=_worker, args=(worker_url, f"peer-{i}", gate, output))
                     for i in range(2)]
            children.extend(peers)
            for peer in peers:
                peer.start()
            gate.set()
            for peer in peers:
                peer.join(20)
                assert peer.exitcode == 0
            outcomes = [output.get(timeout=5) for _ in peers]
            assert sorted(status for _, status in outcomes) == ["committed", "idle"]
            with Session(engine) as db:
                row = db.get(DurableJob, job_id)
                reports = db.query(Report).filter(Report.interview_id == "interview-1").all()
                assert row.status == "succeeded" and row.attempts == 2
                assert len(reports) == 1
                assert reports[0].review in {"peer-0", "peer-1"}
        finally:
            # A failed assertion must not leave test-owned worker processes
            # holding connections to the disposable database.
            for child in children:
                if child.is_alive():
                    child.terminate()
                    child.join(5)
            engine.dispose()
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_two_processes_cannot_resume_same_checkpoint_into_duplicate_commit() -> None:
    """Only one instance can enter a resumed graph's business commit node."""
    from app.agents.orchestration.checkpoint import PostgresCheckpointer
    from app.agents.orchestration.workflows import run_business_graph
    from app.core.loop_policy import selector_loop_factory

    name = f"sagematch_resume_test_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    test_url = base_url.set(database=name)
    worker_url = test_url.render_as_string(hide_password=False)
    saver_dsn = test_url.set(drivername="postgresql").render_as_string(hide_password=False)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    children: list[mp.Process] = []
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        engine = create_engine(test_url)
        try:
            Base.metadata.create_all(engine)
            with Session(engine) as db:
                db.add(JobProfile(id="profile-1", user_id=get_settings().anonymous_user_id,
                                  raw_text="Backend", job_title="Backend"))
                db.flush()
                db.add(Interview(id="interview-1", profile_id="profile-1", title="Interview", status="ended"))
                db.commit()

            class ProcessLost(BaseException):
                pass

            async def interrupted() -> None:
                raise ProcessLost()

            async def seed() -> None:
                async with PostgresCheckpointer(connection_string=saver_dsn).open() as saver:
                    with Session(engine) as db:
                        with pytest.raises(ProcessLost):
                            await run_business_graph(db, mode="live_interview", action=interrupted,
                                                     interview_id="interview-1", run_id="same-run", checkpointer=saver)

            with asyncio.Runner(loop_factory=selector_loop_factory()) as runner:
                runner.run(seed())

            ctx = mp.get_context("spawn")
            gate, release, entered, results = ctx.Event(), ctx.Event(), ctx.Queue(), ctx.Queue()
            children = [ctx.Process(target=_resume_worker, args=(worker_url, gate, entered, release, results))
                        for _ in range(2)]
            for child in children:
                child.start()
            gate.set()
            entered.get(timeout=20)
            # The rival must lose while the first commit is still blocked;
            # otherwise a later completed-checkpoint rejection masks the race.
            loser = results.get(timeout=20)
            assert loser == ("conflict", "checkpoint run already active")
            release.set()
            for child in children:
                child.join(20)
                assert child.exitcode == 0
            assert results.get(timeout=5) == ("ok", "committed")
            assert entered.empty()
            with Session(engine) as db:
                assert db.query(Report).filter_by(interview_id="interview-1").count() == 1
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
                    child.join(5)
            engine.dispose()
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()
