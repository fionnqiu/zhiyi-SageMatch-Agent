"""A fresh process reclaims interrupted text SSE runs from isolated PostgreSQL."""

from __future__ import annotations

import os
import asyncio
import subprocess
import sys
import uuid

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import make_url

from app.core.config import get_settings


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_chat_stream_recovers_across_process_restart() -> None:
    name = f"sagematch_chat_recovery_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        env = os.environ.copy()
        env["POSTGRES_DB"] = name
        prepare = '''
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from app.core.db import Base, SessionLocal, engine, ensure_schema
from app.models.business.session import ChatMessage, ChatSession
from app.models.platform.stream import StreamRun
from app.api.chat.session import _save_chat_recovery
from app.agents.orchestration.checkpoint import PostgresCheckpointer
from app.agents.orchestration.workflows import run_business_graph
from app.core.config import get_settings
from app.core.loop_policy import selector_loop_factory
from app.services.chat.session import begin_chat, route_chat
from app.services.chat.stream_events import append_stream_event, create_stream_run, set_stream_recovery
from sqlalchemy import make_url

Base.metadata.create_all(engine)
ensure_schema(engine)
async def classify(*args, **kwargs):
    return {'intent': 'answer', 'needs_recall': False, 'todos': [], 'source': 'test'}
async def stage():
    with SessionLocal() as db, patch('app.services.chat.session.resolve_intent', new=classify):
        routed = await route_chat(db, '缓存击穿是什么', None)
        create_stream_run(db, 'chat', 'local-user', run_id=routed['run_id'])
        turn = await begin_chat(db, '缓存击穿是什么', None, precomputed_intent=routed['intent'],
                                run_id=routed['run_id'], defer_retrieval=True)
        _save_chat_recovery(db, turn, [])
        worker_id = db.get(StreamRun, routed['run_id']).lease_owner
        append_stream_event(db, routed['run_id'], 'local-user', {'type': 'meta', 'session_id': turn['session'].id}, worker_id=worker_id)
        append_stream_event(db, routed['run_id'], 'local-user', {'type': 'delta', 'text': '旧候选'}, worker_id=worker_id)
        row = db.get(StreamRun, routed['run_id'])
        row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        create_stream_run(db, 'chat', 'local-user', run_id='attachment-run')
        file_turn = dict(turn, run_id='attachment-run', content='private extracted attachment text')
        _save_chat_recovery(db, file_turn, [{'name': 'private.txt', 'text': file_turn['content']}])
        db.get(StreamRun, 'attachment-run').lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        create_stream_run(db, 'chat', 'local-user', run_id='legacy-run')
        db.get(StreamRun, 'legacy-run').created_at = datetime.now(timezone.utc) - timedelta(minutes=2)
        db.add(ChatSession(id='conflict-session', title='Conflict', user_id='local-user'))
        db.flush()
        db.add(ChatMessage(id='original-user', session_id='conflict-session', role='user', content='old',
                           created_at=datetime.now(timezone.utc) - timedelta(seconds=2)))
        db.add(ChatMessage(id='newer-user', session_id='conflict-session', role='user', content='new',
                           created_at=datetime.now(timezone.utc) - timedelta(seconds=1)))
        create_stream_run(db, 'chat', 'local-user', run_id='conflict-run')
        set_stream_recovery(db, 'conflict-run', 'local-user', {
            'recoverable': True, 'session_id': 'conflict-session', 'user_message_id': 'original-user',
            'content': 'old', 'intent': {'intent': 'answer'}, 'mode': 'answer',
        }, worker_id='lost-worker')
        db.get(StreamRun, 'conflict-run').lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
        class ProcessLost(BaseException):
            pass
        async def noop(_state):
            return {}
        async def interrupted_answer(_state):
            raise ProcessLost()
        stages = {f'knowledge_qa.{name}': noop for name in (
            'query_rewrite', 'parallel_retrieve', 'candidate_fusion',
            'deduplicate_and_diversify', 'rerank', 'context_select',
        )}
        stages['knowledge_qa.answer'] = interrupted_answer
        async def commit():
            raise AssertionError('interrupted graph committed')
        dsn = make_url(get_settings().database_url).set(drivername='postgresql').render_as_string(hide_password=False)
        async with PostgresCheckpointer(connection_string=dsn).open() as saver:
            try:
                await run_business_graph(db, mode='knowledge_qa', action=commit,
                    original_query='缓存击穿是什么', session_id=turn['session'].id,
                    run_id=routed['run_id'], checkpointer=saver, stages=stages)
            except ProcessLost:
                pass
            else:
                raise AssertionError('graph did not interrupt at answer stage')
        return routed['run_id']
with asyncio.Runner(loop_factory=selector_loop_factory()) as runner:
    print(runner.run(stage()))
'''
        result = subprocess.run([sys.executable, "-c", prepare], cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                                env=env, capture_output=True, text=True, timeout=60, check=False)
        if result.returncode != 0:
            pytest.fail("isolated PostgreSQL chat recovery setup failed", pytrace=False)
        run_id = result.stdout.strip().splitlines()[-1]
        env["SAGEMATCH_RECOVERY_RUN_ID"] = run_id
        resume = '''
import asyncio
from unittest.mock import AsyncMock, patch
from app.api.chat.session import recover_pending_chat_streams
from app.agents.orchestration.checkpoint import PostgresCheckpointer
from app.core.config import get_settings
from app.core.db import SessionLocal
from app.core.loop_policy import selector_loop_factory
from app.models.business.session import ChatMessage
from app.models.platform.stream import StreamRun
from app.services.chat.stream_events import replay_stream_events
from sqlalchemy import make_url

run_id = __import__('os').environ['SAGEMATCH_RECOVERY_RUN_ID']
async def forbidden_classify(*args, **kwargs):
    raise AssertionError('route classifier ran again after restart')
async def answer(*args, **kwargs):
    yield 'content', '新的完整回答'
async def noop(_state):
    return {}
def retrieval_stages(_db, _turn):
    return {f'knowledge_qa.{name}': noop for name in (
        'query_rewrite', 'parallel_retrieve', 'candidate_fusion',
        'deduplicate_and_diversify', 'rerank', 'context_select',
    )}
async def resume():
    dsn = make_url(get_settings().database_url).set(drivername='postgresql').render_as_string(hide_password=False)
    async with PostgresCheckpointer(connection_string=dsn).open() as saver:
        await recover_pending_chat_streams(saver)
with patch('app.services.chat.session.resolve_intent', new=forbidden_classify), patch(
    'app.api.chat.session.services.iter_turn_parts', new=answer
), patch('app.api.chat.session.services.staged_chat_retrieval_callbacks', new=retrieval_stages), patch(
    'app.services.chat.session.session_title', new=AsyncMock(return_value='缓存击穿')
):
    with asyncio.Runner(loop_factory=selector_loop_factory()) as runner:
        runner.run(resume())
with SessionLocal() as db:
    frames = replay_stream_events(db, run_id, 'local-user')
    assert [frame.event_type for frame in frames] == ['meta', 'delta', 'reset', 'meta', 'delta', 'done'], [frame.event_type for frame in frames]
    assert frames[-1].payload['session']['messages'][-1]['content'] == '新的完整回答', 'answer mismatch'
    session_id = frames[-1].payload['session']['id']
    assert db.query(ChatMessage).filter(ChatMessage.session_id == session_id, ChatMessage.role == 'user').count() == 1, 'user duplicate'
    assert db.query(ChatMessage).filter(ChatMessage.session_id == session_id, ChatMessage.role == 'assistant').count() == 1, 'assistant duplicate'
    assert db.get(StreamRun, run_id).status == 'completed', 'text stream still running'
    file_run = db.get(StreamRun, 'attachment-run')
    assert file_run.status == 'completed', 'attachment stream still running'
    assert file_run.recovery == {'recoverable': False}, 'attachment recovery leaked data'
    assert replay_stream_events(db, 'attachment-run', 'local-user')[-1].payload['error_code'] == 'stream_resume_unavailable', 'attachment error missing'
    legacy_run = db.get(StreamRun, 'legacy-run')
    assert legacy_run.status == 'completed', 'unstaged legacy stream still running'
    assert replay_stream_events(db, 'legacy-run', 'local-user')[-1].payload['error_code'] == 'stream_resume_unavailable', 'legacy recovery error missing'
    assert db.get(StreamRun, 'conflict-run').status == 'completed', 'conflicted stream still running'
    assert replay_stream_events(db, 'conflict-run', 'local-user')[-1].payload['error_code'] == 'stream_resume_conflict', 'conflict error missing'
print('recovered')
'''
        result = subprocess.run([sys.executable, "-c", resume], cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                                env=env, capture_output=True, text=True, timeout=90, check=False)
        if result.returncode != 0 or "recovered" not in result.stdout:
            # Driver diagnostics can contain credentials; do not echo subprocess stderr.
            pytest.fail("isolated PostgreSQL chat stream restart recovery failed", pytrace=False)
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_reclaimed_stream_fences_assistant_commit() -> None:
    """Only the current stream worker may commit its assistant fact once."""
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.core.db import ensure_schema
    from app.models.business.session import ChatMessage, ChatSession
    from app.models.platform.stream import StreamRun
    from app.services.chat.session import finish_chat
    from app.services.chat.stream_events import claim_stream_run, create_stream_run, set_stream_recovery

    name = f"sagematch_chat_fence_{uuid.uuid4().hex[:10]}"
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
                db.add(ChatSession(id="fenced-session", title="Question", user_id=get_settings().anonymous_user_id))
                db.commit()
                create_stream_run(db, "chat", get_settings().anonymous_user_id, run_id="fenced-run")
                set_stream_recovery(db, "fenced-run", get_settings().anonymous_user_id,
                                    {"recoverable": True}, worker_id="worker-a")
                stream = db.get(StreamRun, "fenced-run")
                stream.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
                db.commit()

            with Session(engine) as db:
                assert claim_stream_run(db, "fenced-run", "worker-b")

            def turn(db):
                return {"session": db.get(ChatSession, "fenced-session"), "run_id": "fenced-run",
                        "mode": "answer", "first_turn": False, "opening": "", "route": {}}

            with Session(engine) as stale:
                with pytest.raises(PermissionError):
                    asyncio.run(finish_chat(
                        stale, turn(stale), "stale answer", {}, citation_checked=True,
                        worker_id="worker-a",
                    ))
                stale.rollback()

            with Session(engine) as current:
                asyncio.run(finish_chat(
                    current, turn(current), "current answer", {}, citation_checked=True,
                    worker_id="worker-b",
                ))
                asyncio.run(finish_chat(
                    current, turn(current), "current answer", {}, citation_checked=True,
                    worker_id="worker-b",
                ))
            with Session(engine) as check:
                answers = check.query(ChatMessage).filter_by(session_id="fenced-session", role="assistant").all()
                assert len(answers) == 1
                assert answers[0].content == "current answer"

            # Redirect cleanup is also a business write: a stale worker must
            # not delete the staged user message after another worker claims it.
            with Session(engine) as db:
                db.add(ChatMessage(id="redirect-user", session_id="fenced-session",
                                   role="user", content="Start interview"))
                db.commit()
                create_stream_run(db, "chat", get_settings().anonymous_user_id, run_id="redirect-run")
                set_stream_recovery(db, "redirect-run", get_settings().anonymous_user_id,
                                    {"recoverable": True}, worker_id="worker-a")
                db.get(StreamRun, "redirect-run").lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
                db.commit()
            with Session(engine) as db:
                assert claim_stream_run(db, "redirect-run", "worker-b")

            def redirect_turn(db):
                return {"session": db.get(ChatSession, "fenced-session"), "run_id": "redirect-run",
                        "user_message": db.get(ChatMessage, "redirect-user"), "mode": "redirect",
                        "first_turn": False, "opening": "", "route": {}}

            with Session(engine) as stale:
                with pytest.raises(PermissionError):
                    asyncio.run(finish_chat(stale, redirect_turn(stale), "Open interview", {},
                                            worker_id="worker-a"))
                stale.rollback()
            with Session(engine) as check:
                assert check.get(ChatMessage, "redirect-user") is not None
            with Session(engine) as current:
                asyncio.run(finish_chat(current, redirect_turn(current), "Open interview", {},
                                        worker_id="worker-b"))
            with Session(engine) as check:
                assert check.get(ChatMessage, "redirect-user") is None
        finally:
            engine.dispose()
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()
