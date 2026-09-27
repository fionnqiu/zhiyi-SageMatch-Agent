"""Opt-in HTTP chat graph integration in a disposable PostgreSQL database."""

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
def test_http_chat_creates_one_graph_run_and_committed_answer() -> None:
    """One API command shares a run id across graph, checkpoint, and answer."""
    name = f"sagematch_chat_test_{uuid.uuid4().hex[:10]}"
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
import asyncio
from unittest.mock import AsyncMock, patch
getattr(asyncio, 'WindowsSelectorEventLoopPolicy', None) and asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from fastapi.testclient import TestClient
from app.main import app
from app.core.db import SessionLocal
from app.models import ChatMessage, GraphCheckpointOwner, GraphRun
routed = {
    'run_id': 'chat-run-1', 'mode': 'stream', 'content': '缓存击穿',
    'intent': {'intent': 'answer', 'route': 'knowledge_qa', 'confidence': 0.9, 'needs_recall': False},
    'route': {'route': 'knowledge_qa', 'confidence': 0.9, 'source': 'test', 'original_query': '缓存击穿'},
}
prompt = {'system': 'answer', 'user': 'question', 'fallback': 'unavailable'}
extra = {'kind': 'answer', 'retrieval': {'selected_chunks': []}}
with patch('app.services.chat.session.route_chat', new=AsyncMock(return_value=routed)), patch('app.services.chat.session.prepare_direct_answer', new=AsyncMock(return_value=(prompt, extra))), patch('app.services.chat.session.complete', new=AsyncMock(return_value='测试回答')), patch('app.services.chat.session.session_title', new=AsyncMock(return_value='测试会话')):
    with TestClient(app) as client:
        response = client.post('/api/chat', json={'content': '缓存击穿'})
        assert response.status_code == 200, response.status_code
        with SessionLocal() as db:
            assert db.query(GraphRun).count() == 1
            run = db.query(GraphRun).one()
            assert run.id == 'chat-run-1' and run.status == 'committed'
            assert db.query(GraphCheckpointOwner).count() == 1
            answer = db.query(ChatMessage).filter(ChatMessage.role == 'assistant').one()
            assert answer.extra['run_id'] == run.id
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), env=env,
            capture_output=True, text=True, timeout=60, check=False,
        )
        # Avoid echoing subprocess stderr because DB diagnostics can include a DSN.
        if completed.returncode != 0:
            pytest.fail("isolated HTTP chat graph integration failed", pytrace=False)
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()
