"""Exit and report completion serialize on the same PostgreSQL interview row."""

from __future__ import annotations

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event
from time import sleep

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import create_engine, make_url
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import ensure_schema
from app.models import DurableJob, Interview, InterviewTurn, JobProfile
from app.services.interviews.interview import _lock_interview, abandon_interview, finish_interview


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_finish_and_abandon_obey_the_locked_commit_order() -> None:
    """End first preserves recap; exit first prevents a later report command."""

    name = f"sagematch_exit_race_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        engine = create_engine(base_url.set(database=name))
        try:
            ensure_schema(engine)
            with Session(engine) as db:
                db.add(JobProfile(id="profile-exit", user_id=get_settings().anonymous_user_id,
                                  raw_text="Backend", job_title="Backend"))
                db.flush()
                for interview_id in ("finish-first", "exit-first"):
                    db.add(Interview(id=interview_id, profile_id="profile-exit", title="Interview",
                                     status="live", started_at=datetime.now(timezone.utc)))
                    db.add(InterviewTurn(id=f"turn-{interview_id}", interview_id=interview_id,
                                         role="interviewer", content="请回答"))
                db.commit()

            def race(interview_id: str, first_command: str) -> tuple[str, str]:
                locked = Event()
                release = Event()

                def first() -> str:
                    with Session(engine) as db:
                        assert _lock_interview(db, interview_id) is not None
                        locked.set()
                        assert release.wait(10)
                        if first_command == "end":
                            return asyncio.run(finish_interview(db, interview_id, enqueue_report=True)).status
                        return abandon_interview(db, interview_id).status

                def second() -> str:
                    with Session(engine) as db:
                        try:
                            if first_command == "end":
                                return abandon_interview(db, interview_id).status
                            return asyncio.run(finish_interview(db, interview_id, enqueue_report=True)).status
                        except ValueError:
                            db.rollback()
                            return "rejected"

                with ThreadPoolExecutor(max_workers=2) as pool:
                    leading = pool.submit(first)
                    assert locked.wait(5)
                    trailing = pool.submit(second)
                    try:
                        sleep(0.1)
                        assert not trailing.done(), "second command bypassed the interview row lock"
                    finally:
                        release.set()
                    return leading.result(timeout=10), trailing.result(timeout=10)

            assert race("finish-first", "end") == ("ended", "ended")
            assert race("exit-first", "abandon") == ("ready", "rejected")
            with Session(engine) as db:
                ended = db.get(Interview, "finish-first")
                ready = db.get(Interview, "exit-first")
                assert (ended.status, ready.status) == ("ended", "ready")
                assert db.query(InterviewTurn).filter(InterviewTurn.interview_id == "finish-first").count() == 1
                assert db.query(InterviewTurn).filter(InterviewTurn.interview_id == "exit-first").count() == 0
                assert db.query(DurableJob).filter(DurableJob.business_key == "finish-first").count() == 1
                assert db.query(DurableJob).filter(DurableJob.business_key == "exit-first").count() == 0
        finally:
            engine.dispose()
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()
