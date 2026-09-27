"""Opt-in PostgreSQL tests for SSE sequence, reconnect, and owner boundaries."""

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
def test_stream_concurrent_append_replay_and_owner_scope() -> None:
    """Real PostgreSQL row locks serialize frame IDs and replay respects the owner."""
    name = f"sagematch_stream_test_{uuid.uuid4().hex[:10]}"
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
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, patch
getattr(asyncio, 'WindowsSelectorEventLoopPolicy', None) and asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from fastapi.testclient import TestClient
from app.main import app
from app.core.db import SessionLocal
from app.services.chat.stream_events import append_stream_event, create_stream_run, replay_stream_events
from app.services.chat.stream_events import claim_stream_run, renew_stream_lease, set_stream_recovery
from app.models.platform.stream import StreamRun, StreamEvent
from app.services.shared.common import ANON
from app.models.business.knowledge import Material, MaterialChunk
from datetime import datetime, timedelta, timezone
with TestClient(app) as client:
    with SessionLocal() as db:
        create_stream_run(db, 'chat', 'owner-a', run_id='leased-stream')
        set_stream_recovery(db, 'leased-stream', 'owner-a', {'recoverable': True}, worker_id='old-worker')
        append_stream_event(db, 'leased-stream', 'owner-a', {'type': 'meta'}, worker_id='old-worker')
        db.get(StreamRun, 'leased-stream').lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
        assert not renew_stream_lease(db, 'leased-stream', 'old-worker')
    with SessionLocal() as db:
        assert claim_stream_run(db, 'leased-stream', 'new-worker')
        assert not renew_stream_lease(db, 'leased-stream', 'old-worker')
        for stale_owner in ('old-worker', None):
            try:
                append_stream_event(db, 'leased-stream', 'owner-a', {'type': 'done'}, worker_id=stale_owner)
                raise AssertionError('stale worker closed a reclaimed stream')
            except PermissionError:
                pass
        assert append_stream_event(db, 'leased-stream', 'owner-a', {'type': 'reset'},
                                   terminal=False, worker_id='new-worker').seq == 2
        assert append_stream_event(db, 'leased-stream', 'owner-a', {'type': 'done'},
                                   worker_id='new-worker').seq == 3
        assert [event.event_type for event in replay_stream_events(db, 'leased-stream', 'owner-a')] == [
            'meta', 'reset', 'done',
        ]
    with SessionLocal() as db:
        create_stream_run(db, 'chat', 'owner-a', run_id='stream-1')
    def append(index):
        with SessionLocal() as db:
            return append_stream_event(db, 'stream-1', 'owner-a', {'type': 'delta', 'text': str(index), 'reasoning': 'private'}).event_id
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(append, range(24)))
    assert sorted(int(value.split(':')[1]) for value in ids) == list(range(1, 25))
    with SessionLocal() as db:
        append_stream_event(db, 'stream-1', 'owner-a', {'type': 'done'})
        frames = replay_stream_events(db, 'stream-1', 'owner-a', 'stream-1:12')
        assert [frame.seq for frame in frames] == list(range(13, 26))
        assert all('reasoning' not in frame.payload for frame in frames)
        try:
            replay_stream_events(db, 'stream-1', 'owner-b')
            raise AssertionError('cross-owner replay succeeded')
        except PermissionError:
            pass
        try:
            append_stream_event(db, 'stream-1', 'owner-b', {'type': 'delta'})
            raise AssertionError('cross-owner append succeeded')
        except PermissionError:
            pass
        assert db.get(StreamRun, 'stream-1').status == 'completed'
        assert db.query(StreamEvent).filter(StreamEvent.run_id == 'stream-1').count() == 25
    response = client.get('/api/streams/stream-1/events', headers={'Last-Event-ID': 'stream-1:24'})
    assert response.status_code == 404  # HTTP principal is anonymous, not owner-a.
    with SessionLocal() as db:
        create_stream_run(db, 'chat', ANON, run_id='anonymous-stream')
        append_stream_event(db, 'anonymous-stream', ANON, {'type': 'meta', 'reasoning_content': 'private'})
        append_stream_event(db, 'anonymous-stream', ANON, {'type': 'done'})
    response = client.get('/api/streams/anonymous-stream/events', headers={'Last-Event-ID': 'anonymous-stream:1'})
    assert response.status_code == 200
    assert 'id: anonymous-stream:2' in response.text
    assert 'id: anonymous-stream:1' not in response.text
    assert 'reasoning_content' not in response.text
    assert client.get('/api/streams/anonymous-stream/events', headers={'Last-Event-ID': 'wrong:1'}).status_code == 400

    # Run the HTTP stream against real PostgreSQL rows and SSE replay. Only
    # the provider outputs are deterministic replacements in this scenario.
    with SessionLocal() as db:
        db.add(Material(id='material-1', filename='cache.md', status='ready', chunk_count=1))
        db.add(MaterialChunk(id='chunk-1', material_id='material-1', ordinal=0,
                             text='缓存击穿是热点键失效后大量请求同时回源。'))
        db.commit()
    calls = []
    async def classify(*args, **kwargs):
        calls.append('route')
        return {'intent': 'answer', 'needs_recall': False, 'todos': [], 'source': 'test'}
    async def stream_answer(*args, **kwargs):
        yield 'content', '缓存击穿会同时回源 [S9]'
    with patch('app.services.chat.session.resolve_intent', new=classify), patch(
        'app.services.chat.session.stream_parts', new=stream_answer
    ), patch('app.services.chat.session.session_title', new=AsyncMock(return_value='缓存击穿')), patch(
        'app.services.chat.session.complete', new=AsyncMock(return_value='缓存击穿会同时回源 [S1]')
    ) as repair, patch('app.services.materials.rag.stages.embed_or_empty', new=AsyncMock(return_value=([], ''))):
        streamed = client.post('/api/chat/stream', json={'content': '缓存击穿'})
    assert streamed.status_code == 200
    assert calls == ['route']
    assert repair.await_count == 1
    import json
    frames = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith('data: ')]
    assert [frame['type'] for frame in frames] == ['meta', 'delta', 'done']
    done = frames[-1]
    answer = done['session']['messages'][-1]
    assert answer['content'] == '缓存击穿会同时回源 [S1]'
    assert answer['extra']['retrieval']['selected_chunks'][0]['chunk_id'] == 'chunk-1'
    assert answer['extra']['citation_validation']['valid'] is True
    run_id = frames[0]['run_id']
    replayed = client.get('/api/streams/' + run_id + '/events',
                          headers={'Last-Event-ID': run_id + ':1'})
    assert replayed.status_code == 200
    replay_frames = [json.loads(line[6:]) for line in replayed.text.splitlines() if line.startswith('data: ')]
    assert [frame['type'] for frame in replay_frames] == ['delta', 'done']
    assert replay_frames[-1]['session']['messages'][-1]['content'] == '缓存击穿会同时回源 [S1]'
    assert '"type": "reasoning"' not in replayed.text
'''
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), env=env,
            capture_output=True, text=True, timeout=90, check=False,
        )
        # Never echo database diagnostics; connection errors may contain credentials.
        if completed.returncode != 0:
            safe = [line.strip() for line in completed.stderr.splitlines()
                    if line.strip().startswith(("AssertionError:", "KeyError:", "TypeError:", "AttributeError:", "NameError:"))]
            suffix = f": {safe[-1]}" if safe else ""
            pytest.fail(f"isolated PostgreSQL stream integration failed{suffix}", pytrace=False)
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()

