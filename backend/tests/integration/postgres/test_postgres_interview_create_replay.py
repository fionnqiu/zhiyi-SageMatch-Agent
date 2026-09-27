"""Durable HTTP idempotency for starting an interview."""

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
def test_create_replay_survives_status_change_and_concurrent_requests() -> None:
    """A committed key identifies one business fact after the graph callback."""
    name = f"sagematch_create_replay_{uuid.uuid4().hex[:10]}"
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
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from app.core.db import SessionLocal, ensure_schema, engine
from app.main import app
from app.models import Interview, InterviewCreateReceipt, InterviewTurn, JobProfile, Question, QuestionSet

# Match run.py when TestClient opens the async PostgreSQL saver on Windows.
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ensure_schema(engine)
with SessionLocal() as db:
    db.add(JobProfile(id='create-profile', raw_text='backend', job_title='Backend', analysis={}, user_id='local-user'))
    for suffix in ('one', 'two', 'three'):
        db.add(QuestionSet(id='set-' + suffix, profile_id='create-profile', status='ready'))
        db.add(Question(id='question-' + suffix, question_set_id='set-' + suffix,
                        ordinal=0, stem='如何保证幂等？'))
    db.commit()

with TestClient(app) as client:
    first = client.post('/api/interviews', json={'question_set_id': 'set-one'},
                        headers={'Idempotency-Key': 'create-once'})
    assert first.status_code == 200, first.status_code
    interview_id = first.json()['id']
    with SessionLocal() as db:
        db.get(Interview, interview_id).status = 'ended'
        db.commit()
    replay = client.post('/api/interviews', json={'question_set_id': 'set-one'},
                         headers={'Idempotency-Key': 'create-once'})
    assert replay.status_code == 200 and replay.json()['id'] == interview_id
    assert replay.json()['status'] == 'ended'
    changed = client.post('/api/interviews', json={'question_set_id': 'set-two'},
                          headers={'Idempotency-Key': 'create-once'})
    assert changed.status_code == 400

    def create_same_key(_):
        response = client.post('/api/interviews', json={'question_set_id': 'set-two'},
                               headers={'Idempotency-Key': 'concurrent-create'})
        return response.status_code, response.json().get('id')

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(create_same_key, range(2)))
    assert outcomes[0][0] == outcomes[1][0] == 200, outcomes
    assert outcomes[0][1] == outcomes[1][1]
    no_key = client.post('/api/interviews', json={'question_set_id': 'set-three'})
    assert no_key.status_code == 200
    reused = client.post('/api/interviews', json={'question_set_id': 'set-three'})
    assert reused.status_code == 200 and reused.json()['id'] == no_key.json()['id']
    deleted = client.delete('/api/interviews/' + interview_id)
    assert deleted.status_code == 200
    deleted_replay = client.post('/api/interviews', json={'question_set_id': 'set-one'},
                                 headers={'Idempotency-Key': 'create-once'})
    assert deleted_replay.status_code == 400

with SessionLocal() as db:
    assert db.query(Interview).filter(Interview.question_set_id == 'set-one').count() == 0
    assert db.query(Interview).filter(Interview.question_set_id == 'set-two').count() == 1
    assert db.query(InterviewTurn).filter(InterviewTurn.interview_id == outcomes[0][1]).count() == 1
    assert db.query(InterviewCreateReceipt).count() == 2
'''
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            env=env, capture_output=True, text=True, timeout=90, check=False,
        )
        # Child diagnostics can contain credentials; do not print them.
        if completed.returncode != 0:
            frames = re.findall(r'File "<string>", line (\d+)', completed.stderr)
            errors = re.findall(r'^([A-Za-z_][\w.]*(?:Error|Exception))(?::|$)', completed.stderr, re.MULTILINE)
            location = frames[-1] if frames else "unknown"
            kind = errors[-1] if errors else "unknown"
            saver = re.findall(r'PostgreSQL graph checkpointer unavailable: ([A-Za-z_]+)', completed.stderr)
            cause = f", saver={saver[-1]}" if saver else ""
            pytest.fail(f"isolated PostgreSQL interview create replay check failed at child line {location} ({kind}{cause})", pytrace=False)
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()
