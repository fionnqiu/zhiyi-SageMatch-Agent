"""Opt-in FastAPI startup/readiness smoke test in a disposable PostgreSQL DB."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import make_url

from app.core.config import get_settings


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_startup_opens_saver_and_readiness_is_live() -> None:
    """The real startup path reaches ready with migrations and a live saver."""
    name = f"sagematch_startup_test_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        env = os.environ.copy()
        env["POSTGRES_DB"] = name
        # TestClient runs startup and shutdown in one isolated interpreter so
        # app.core.db's module-level engine uses the temporary database from import.
        code = """
import asyncio
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from app.agents.orchestration.checkpoint import authorize_checkpoint
from app.agents.orchestration.graph import AgentState
from app.core.db import SessionLocal
from app.main import app
from app.models import DurableJob, Interview, JobProfile, Report
from app.models.platform.runtime import GraphCheckpointOwner, GraphRun
from app.services.interviews import interview as report_service
from app.services.shared.common import ANON
from app.services.operations.jobs import create_job

def safe_hook(error_type, _error, trace):
    # Keep DSNs and provider diagnostics out of the parent test output.
    frame = traceback.extract_tb(trace)[-1]
    print(f'REPORT_SMOKE:{error_type.__name__}:{frame.name}:{frame.lineno}', file=sys.stderr)

sys.excepthook = safe_hook

selector = getattr(asyncio, 'WindowsSelectorEventLoopPolicy', None)
if selector:
    asyncio.set_event_loop_policy(selector())
with TestClient(app) as client:
    assert report_service._report_checkpointer is app.state.checkpointer
    assert report_service._report_worker is not None
    response = client.get('/api/health/ready')
    assert response.status_code == 200, response.json()
    assert response.json()['status'] == 'ready'
    with SessionLocal() as db:
        db.add(JobProfile(id='report-startup-profile', user_id=ANON,
                          raw_text='测试岗位', job_title='测试岗位'))
        db.flush()
        db.add(Interview(id='report-startup-i1', profile_id='report-startup-profile', title='空场次', status='ended'))
        db.add(Interview(id='report-startup-i2', profile_id='report-startup-profile', title='回收场次', status='ended'))
        job = create_job(db, 'report_generation', 'report-startup-i1', idempotency_key='report:startup')
        job_id = job.id
        reclaimed = create_job(db, 'report_generation', 'report-startup-i2', idempotency_key='report:reclaimed')
        reclaimed_id = reclaimed.id
        reclaimed.status = 'running'
        reclaimed.attempts = 1
        reclaimed.worker_id = 'dead-worker'
        reclaimed.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        stale_thread = f'report:{reclaimed_id}:attempt:1'
        authorize_checkpoint(db, stale_thread,
                             AgentState(thread_id=stale_thread, interview_id='report-startup-i2',
                                        requested_mode='evaluation_report'),
                             owner_id=report_service.ANON, interview_id='report-startup-i2')
        db.commit()
    for _ in range(50):
        time.sleep(0.2)
        with SessionLocal() as db:
            current = db.get(DurableJob, job_id)
            retried = db.get(DurableJob, reclaimed_id)
            first_trace = db.query(GraphRun).filter(GraphRun.thread_id == f'report:{job_id}:attempt:1').count()
            reclaimed_trace = db.query(GraphRun).filter(
                GraphRun.thread_id == f'report:{reclaimed_id}:attempt:2').count()
            # The business job commits before its secondary trace transaction.
            if current.status == retried.status == 'succeeded' and first_trace == reclaimed_trace == 1:
                assert db.query(Report).filter(Report.interview_id == 'report-startup-i1').count() == 1
                assert db.query(Report).filter(Report.interview_id == 'report-startup-i2').count() == 1
                assert db.get(GraphCheckpointOwner, f'report:{job_id}:attempt:1') is not None
                assert retried.attempts == 2
                assert db.get(GraphCheckpointOwner, stale_thread) is not None
                assert db.get(GraphCheckpointOwner, f'report:{reclaimed_id}:attempt:2') is not None
                assert first_trace == 1
                assert db.query(GraphRun).filter(GraphRun.thread_id == stale_thread).count() == 0
                assert reclaimed_trace == 1
                break
            assert current.status != 'dead_letter'
            assert retried.status != 'dead_letter'
    else:
        raise AssertionError('report worker did not commit through checkpointed graph')
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        # Do not echo subprocess stderr: connection diagnostics can contain a DSN.
        if completed.returncode != 0:
            marker = next((line for line in completed.stderr.splitlines() if line.startswith("REPORT_SMOKE:")), "")
            pytest.fail(f"isolated FastAPI startup/readiness failed {marker}", pytrace=False)
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_failed_saver_startup_leaves_pending_jobs_unclaimed() -> None:
    """A broken saver prevents every worker from consuming durable work."""
    name = f"sagematch_startup_test_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        env = os.environ.copy()
        env["POSTGRES_DB"] = name
        code = """
from contextlib import asynccontextmanager
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from app import main
from app.core.db import SessionLocal, engine, ensure_schema
from app.models import DurableJob
from app.services.operations.jobs import create_job

ensure_schema(engine)
with SessionLocal() as db:
    job = create_job(db, 'material_ingest', 'pending-material', idempotency_key='saver-failure-job')
    db.commit()
    job_id = job.id

@asynccontextmanager
async def broken_saver():
    raise RuntimeError('simulated saver failure')
    yield

with patch.object(main.PostgresCheckpointer, 'open', return_value=broken_saver()):
    with pytest.raises(RuntimeError, match='PostgreSQL graph checkpointer unavailable'):
        with TestClient(main.app):
            pass

assert not main.material_worker_running()
assert not main.report_worker_running()
assert not main.interview.generation_worker_running()
assert not main.session.chat_worker_running()
assert getattr(main.app.state, 'heartbeat_task', None) is None
with SessionLocal() as db:
    stored = db.get(DurableJob, job_id)
    assert stored.status == 'pending'
    assert stored.attempts == 0
    assert stored.worker_id is None
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        # Failure details can include connection settings; report only the exit code.
        assert completed.returncode == 0, f"isolated failed-saver startup exited {completed.returncode}"
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_failure_after_saver_entry_closes_resources_and_preserves_jobs() -> None:
    """Startup faults after saver entry must unwind partially started workers."""
    name = f"sagematch_startup_test_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        env = os.environ.copy()
        env["POSTGRES_DB"] = name
        code = '''
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from app import main
from app.core.db import SessionLocal, engine, ensure_schema
from app.models import DurableJob
from app.services.operations.jobs import create_job

selector = getattr(asyncio, 'WindowsSelectorEventLoopPolicy', None)
if selector:
    asyncio.set_event_loop_policy(selector())
ensure_schema(engine)
with SessionLocal() as db:
    job = create_job(db, 'report_generation', 'pending-report', idempotency_key='startup-fault-job')
    db.commit()
    job_id = job.id

real_open = main.PostgresCheckpointer.open
for fault in ('seed', 'report_worker'):
    state = {'entered': False, 'closed': False}
    @asynccontextmanager
    async def tracked_saver(checkpointer):
        async with real_open(checkpointer) as saver:
            state['entered'] = True
            try:
                yield saver
            finally:
                state['closed'] = True

    def tracked_open(checkpointer):
        return tracked_saver(checkpointer)

    def fail(*_args, **_kwargs):
        raise RuntimeError('injected startup fault')

    target = 'seed_providers' if fault == 'seed' else 'start_report_worker'
    with patch.object(main.PostgresCheckpointer, 'open', tracked_open), patch.object(main.services, target, fail):
        with pytest.raises(RuntimeError, match='injected startup fault'):
            with TestClient(main.app):
                pass
    assert state == {'entered': True, 'closed': True}, fault
    assert not main.material_worker_running(), fault
    assert not main.report_worker_running(), fault
    assert not main.interview.generation_worker_running(), fault
    assert not main.session.chat_worker_running(), fault
    assert getattr(main.app.state, 'heartbeat_task', None) is None, fault
    assert getattr(main.app.state, 'checkpointer', None) is None, fault
    assert getattr(main.app.state, 'checkpointer_context', None) is None, fault
    with SessionLocal() as db:
        stored = db.get(DurableJob, job_id)
        assert stored.status == 'pending', fault
        assert stored.attempts == 0, fault
        assert stored.worker_id is None, fault
'''
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            env=env, capture_output=True, text=True, timeout=30, check=False,
        )
        # Tracebacks can contain connection details; retain only a categorical failure.
        assert completed.returncode == 0, f"isolated post-saver startup fault exited {completed.returncode}"
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()
