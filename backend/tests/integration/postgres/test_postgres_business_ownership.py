"""Exercise business-object ownership against isolated PostgreSQL rows."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import uuid

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import make_url

from app.core.config import get_settings


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_foreign_business_objects_are_not_read_or_mutated() -> None:
    """A seeded foreign owner must not become this installation's anonymous user."""
    name = f"sagematch_owner_test_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        env = os.environ.copy()
        env["POSTGRES_DB"] = name
        code = r'''
import asyncio
import sys
from app.core.db import Base, SessionLocal, engine, ensure_schema
from app.models import ChatSession, Interview, JobProfile, QuestionSet
from app.services.chat import session as sessions
from app.services.interviews import interview as interviews
from app.services.chat.stream_events import create_stream_run

# Match run.py: psycopg's async saver needs the Selector loop on Windows.
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

Base.metadata.create_all(bind=engine)
ensure_schema(engine)
with SessionLocal() as db:
    db.add_all([
        ChatSession(id='local-session', user_id='local-user'),
        ChatSession(id='foreign-session', user_id='other-user'),
        JobProfile(id='local-profile', session_id='local-session', user_id='local-user', raw_text='local'),
        JobProfile(id='foreign-profile', session_id='foreign-session', user_id='other-user', raw_text='foreign'),
        QuestionSet(id='local-set', profile_id='local-profile', status='ready'),
        QuestionSet(id='foreign-set', profile_id='foreign-profile', status='ready'),
        Interview(id='local-interview', profile_id='local-profile', question_set_id='local-set', title='local', status='ready'),
        Interview(id='foreign-interview', profile_id='foreign-profile', question_set_id='foreign-set', title='foreign', status='ready'),
        Interview(id='unattributed-interview', title='legacy', status='ready'),
    ])
    db.commit()
    assert sessions.get_session(db, 'local-session') is not None
    assert sessions.get_session(db, 'foreign-session') is None
    assert [row.id for row in sessions.list_sessions(db)] == []  # Empty drafts are excluded.
    assert sessions.latest_question_set_for_session(db, 'foreign-session') is None
    assert interviews.get_interview(db, 'local-interview') is not None
    assert interviews.get_interview(db, 'foreign-interview') is None
    assert interviews.get_interview(db, 'unattributed-interview') is None
    assert [row.id for row in interviews.list_interviews(db)] == ['local-interview']
    assert interviews.resolve_question_set(db, None, 'foreign-set') is None
    for command in (
        lambda: sessions.begin_chat(db, '出题', 'foreign-session'),
        lambda: sessions.route_chat(db, '出题', 'foreign-session'),
    ):
        try:
            asyncio.run(command())
        except ValueError:
            db.rollback()
        else:
            raise AssertionError('foreign session accepted as a new local chat')
    for operation in (
        lambda: sessions.clear_session(db, 'foreign-session'),
        lambda: sessions.delete_session(db, 'foreign-session'),
        lambda: interviews.delete_interview(db, 'foreign-interview'),
        lambda: interviews.delete_interview(db, 'unattributed-interview'),
        lambda: interviews.resume_or_start(db, 'foreign-interview'),
    ):
        try:
            operation()
        except ValueError:
            db.rollback()
        else:
            raise AssertionError('foreign mutation was allowed')
    assert db.get(ChatSession, 'foreign-session') is not None
    assert db.query(ChatSession).count() == 2
    assert db.get(Interview, 'foreign-interview').status == 'ready'
    create_stream_run(db, 'chat', 'local-user', run_id='foreign-stream', business_id='foreign-session')
    db.commit()

# HTTP uses the same fixed identity and must not turn a foreign object into a
# successful response, including replay through a matching stream owner label.
from fastapi.testclient import TestClient
from app.main import app
with TestClient(app) as client:
    assert client.get('/api/sessions/foreign-session').status_code == 404
    assert client.post('/api/chat', json={'content': '出题', 'session_id': 'foreign-session'}).status_code == 404
    assert client.post('/api/chat/stream', json={'content': '出题', 'session_id': 'foreign-session'}).status_code == 404
    assert client.get('/api/interviews/foreign-interview').status_code == 404
    assert client.get('/api/interviews/unattributed-interview').status_code == 404
    assert client.post('/api/interviews', json={'question_set_id': 'foreign-set'}).status_code == 400
    assert client.get('/api/streams/foreign-stream/events').status_code == 404
'''
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            env=env, capture_output=True, text=True, timeout=60, check=False,
        )
        # PostgreSQL diagnostics may include connection details; keep them out of test output.
        if completed.returncode != 0:
            frames = re.findall(r'File "<string>", line (\d+)', completed.stderr)
            errors = re.findall(r'^([A-Za-z_][\w.]*(?:Error|Exception))(?::|$)', completed.stderr, re.MULTILINE)
            location = frames[-1] if frames else "unknown"
            kind = errors[-1] if errors else "unknown"
            saver = re.findall(r'PostgreSQL graph checkpointer unavailable: ([A-Za-z_]+)', completed.stderr)
            cause = f", saver={saver[-1]}" if saver else ""
            pytest.fail(f"isolated business ownership test failed at child line {location} ({kind}{cause})", pytrace=False)
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()
