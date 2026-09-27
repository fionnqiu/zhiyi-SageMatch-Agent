"""Opt-in PostgreSQL concurrency checks for interview state transitions."""

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
def test_concurrent_answer_and_report_regeneration_are_serialized() -> None:
    """Row locks admit one answer and one regeneration command per state."""

    name = f"sagematch_interview_test_{uuid.uuid4().hex[:10]}"
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
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from app.core.db import SessionLocal, ensure_schema, engine
from app.models import ChatSession, DurableJob, Interview, InterviewTurn, JobProfile, Question, QuestionSet, Report
from app.services.interviews.interview import answer_interview, finish_interview, prepare_interview, regenerate_report, resume_or_start, start_interview
from app.services.interviews import interview as interview_service
from app.services.chat.session import stub_pack

ensure_schema(engine)
with SessionLocal() as db:
    profile = JobProfile(id='profile-1', raw_text='backend', job_title='Backend', analysis={})
    qset = QuestionSet(id='qset-1', profile_id=profile.id, status='ready')
    question = Question(id='question-1', question_set_id=qset.id, ordinal=0, stem='如何处理缓存击穿？')
    fresh_qset = QuestionSet(id='qset-2', profile_id=profile.id, status='ready')
    fresh_question = Question(id='question-2', question_set_id=fresh_qset.id, ordinal=0, stem='如何保证幂等？')
    interview = Interview(
        id='iv-live', profile_id=profile.id, question_set_id=qset.id,
        title='Backend interview', status='live', current_question_index=0,
    )
    ready = Interview(
        id='iv-ready', profile_id=profile.id, question_set_id=qset.id,
        title='Ready interview', status='ready', current_question_index=0,
    )
    ended = Interview(id='iv-ended', profile_id=profile.id, title='Ended interview', status='ended')
    db.add_all([profile, qset, question, fresh_qset, fresh_question, interview, ready, ended])
    db.flush()
    db.add(InterviewTurn(id='opening', interview_id=interview.id, role='interviewer', content='请回答', question_id=question.id))
    db.add(Report(id='report-old', interview_id=ended.id, score=80, review='old', issues=[]))
    db.commit()

async def slow_followup(*_args, **_kwargs):
    await asyncio.sleep(0.25)
    return '追问'

def submit_answer(content):
    with SessionLocal() as db:
        try:
            with patch('app.services.interviews.interview.followup_line', side_effect=slow_followup):
                asyncio.run(answer_interview(db, 'iv-live', content, 'text'))
            return 'accepted'
        except ValueError:
            db.rollback()
            return 'rejected'

barrier = threading.Barrier(2)
arrival = threading.local()
original_lock = interview_service._lock_interview
def synchronized_lock(db, interview_id):
    if not getattr(arrival, 'seen', False):
        arrival.seen = True
        barrier.wait(timeout=5)
    return original_lock(db, interview_id)
with patch('app.services.interviews.interview._lock_interview', side_effect=synchronized_lock):
    with ThreadPoolExecutor(max_workers=2) as pool:
        answer_results = list(pool.map(submit_answer, ['answer-a', 'answer-b']))
assert sorted(answer_results) == ['accepted', 'rejected']
with SessionLocal() as db:
    user_turns = db.query(InterviewTurn).filter(InterviewTurn.interview_id == 'iv-live', InterviewTurn.role == 'user').all()
    assert len(user_turns) == 1
    assert user_turns[0].cite is None

# The answer ID survives a replay after the first commit, and a pending turn
# left by a crashed worker resumes without advancing the question twice.
retry_id = '00000000-0000-4000-8000-000000000001'
with SessionLocal() as db:
    asyncio.run(answer_interview(db, 'iv-live', 'stable answer', 'text', run_id=retry_id))
with SessionLocal() as db:
    before = db.query(InterviewTurn).filter(InterviewTurn.interview_id == 'iv-live').count()
    state = (db.get(Interview, 'iv-live').current_question_index, db.get(Interview, 'iv-live').followups_on_question)
    asyncio.run(answer_interview(db, 'iv-live', 'stable answer', 'text', run_id=retry_id))
    assert db.query(InterviewTurn).filter(InterviewTurn.interview_id == 'iv-live').count() == before
    assert (db.get(Interview, 'iv-live').current_question_index, db.get(Interview, 'iv-live').followups_on_question) == state
    assert db.get(InterviewTurn, retry_id).cite is None

pending_id = '00000000-0000-4000-8000-000000000002'
with SessionLocal() as db:
    interview = db.get(Interview, 'iv-live')
    interview.followups_on_question = 1
    db.add(InterviewTurn(id=pending_id, interview_id='iv-live', role='user', content='crash answer', answer_mode='text', cite='answer_pending', question_id='question-1'))
    db.commit()
with SessionLocal() as db:
    before = db.query(InterviewTurn).filter(InterviewTurn.interview_id == 'iv-live').count()
    asyncio.run(answer_interview(db, 'iv-live', 'crash answer', 'text', run_id=pending_id))
    assert db.query(InterviewTurn).filter(InterviewTurn.interview_id == 'iv-live').count() == before + 1
    assert db.get(InterviewTurn, pending_id).cite is None
    assert db.get(Interview, 'iv-live').followups_on_question == 1

# A refreshed client no longer has the original Idempotency-Key. The HTTP
# route must recover the pending turn's run ID without advancing twice.
from fastapi import HTTPException
from starlette.requests import Request
from types import SimpleNamespace
from app.api.interviews.interview import answer_interview as answer_route
from app.schemas.business.interview import InterviewAnswerIn
from app.services.interviews.interview import pending_answer_run_id

lost_key_id = '00000000-0000-4000-8000-000000000003'
reload_request = Request({'type': 'http', 'headers': [],
                          'app': SimpleNamespace(state=SimpleNamespace(checkpointer=None))})
with SessionLocal() as db:
    db.add(InterviewTurn(id=lost_key_id, interview_id='iv-live', role='user',
                         content='answer after reload', answer_mode='text',
                         cite='answer_pending', question_id='question-1'))
    db.commit()
with SessionLocal() as db:
    assert pending_answer_run_id(db, 'iv-live', 'answer after reload', 'text') == lost_key_id
    try:
        asyncio.run(answer_route('iv-live', InterviewAnswerIn(content='different answer'), db,
                                 reload_request))
        raise AssertionError('different answer accepted while pending')
    except HTTPException as exc:
        assert exc.status_code == 400
    before = db.query(InterviewTurn).filter(InterviewTurn.interview_id == 'iv-live').count()
    state = (db.get(Interview, 'iv-live').current_question_index,
             db.get(Interview, 'iv-live').followups_on_question)
    result = asyncio.run(answer_route('iv-live', InterviewAnswerIn(content='answer after reload'), db,
                                      reload_request))
    assert result.id == 'iv-live'
    assert db.get(InterviewTurn, lost_key_id).cite is None
    assert db.query(InterviewTurn).filter(InterviewTurn.interview_id == 'iv-live').count() == before + 1
    assert (db.get(Interview, 'iv-live').current_question_index,
            db.get(Interview, 'iv-live').followups_on_question) == state

def start_ready(_index):
    with SessionLocal() as db:
        resume_or_start(db, 'iv-ready')

with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(start_ready, range(2)))
with SessionLocal() as db:
    openings = db.query(InterviewTurn).filter(InterviewTurn.interview_id == 'iv-ready').all()
    assert len(openings) == 1
    assert db.get(Interview, 'iv-ready').status == 'live'

def create_from_question_set(_index):
    with SessionLocal() as db:
        return start_interview(db, None, 'qset-2').id

with ThreadPoolExecutor(max_workers=2) as pool:
    fresh_ids = list(pool.map(create_from_question_set, range(2)))
assert fresh_ids[0] == fresh_ids[1]
with SessionLocal() as db:
    fresh = db.query(Interview).filter(Interview.question_set_id == 'qset-2').all()
    assert len(fresh) == 1
    assert db.query(InterviewTurn).filter(InterviewTurn.interview_id == fresh[0].id).count() == 1

def finish_live(_index):
    with SessionLocal() as db:
        asyncio.run(finish_interview(db, 'iv-ready', enqueue_report=True))

with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(finish_live, range(2)))
with SessionLocal() as db:
    finish_jobs = db.query(DurableJob).filter(DurableJob.business_key == 'iv-ready').all()
    assert len(finish_jobs) == 1
    assert db.get(Interview, 'iv-ready').status == 'ended'

def regenerate(_index):
    with SessionLocal() as db:
        regenerate_report(db, 'iv-ended')

with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(regenerate, range(2)))
with SessionLocal() as db:
    jobs = db.query(DurableJob).filter(
        DurableJob.kind == 'report_generation',
        DurableJob.business_key == 'iv-ended',
    ).all()
    assert len(jobs) == 1
    assert jobs[0].idempotency_key == 'report:iv-ended:regen:report-old'
    assert db.query(Report).filter(Report.interview_id == 'iv-ended').count() == 0

from unittest.mock import AsyncMock
with patch('app.services.chat.session.recall_snippets', new=AsyncMock(return_value=[])), patch(
    'app.services.chat.session.generate_question_pack', new=AsyncMock(return_value=stub_pack('后端开发岗位')),
):
    with SessionLocal() as db:
        first = asyncio.run(prepare_interview(db, '后端开发岗位职责与要求', run_id='generation-run-1'))
    with SessionLocal() as db:
        replay = asyncio.run(prepare_interview(db, '后端开发岗位职责与要求', run_id='generation-run-1'))
        assert replay['id'] == first['id']
        assert db.query(ChatSession).filter(ChatSession.id == 'generation-run-1').count() == 1
        assert db.query(Interview).join(JobProfile, JobProfile.id == Interview.profile_id).filter(
            JobProfile.session_id == 'generation-run-1').count() == 1
        try:
            asyncio.run(prepare_interview(db, '完全不同的岗位描述内容', run_id='generation-run-1'))
            raise AssertionError('reused run id accepted different input')
        except ValueError:
            pass
'''
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), env=env,
            capture_output=True, text=True, timeout=90, check=False,
        )
        # Database diagnostics can contain credentials, so keep subprocess
        # output private while still producing a deterministic test failure.
        if completed.returncode != 0:
            pytest.fail("isolated PostgreSQL interview concurrency check failed", pytrace=False)
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()
