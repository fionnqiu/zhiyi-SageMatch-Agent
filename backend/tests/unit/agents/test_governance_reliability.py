"""Regression checks for cross-user cache isolation and breaker probe ownership."""

import asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import psycopg
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from psycopg import sql

from app.core.db import Base
from app.agents.providers.governance import ProviderGovernor, cache_get, cache_put
from app.models.platform.runtime import ProviderHealth, ToolCacheEntry, ToolRun
from app.models import Material, MaterialChunk, ProviderConfig, RoleBinding
from app.agents.providers.router import choose_provider
from app.agents.roles.loop import _langchain_tool
from app.agents.roles.authoring import author_question_candidate
from app.agents.tools.registry import ToolSpec
from app.services.operations.llm_gateway import complete_with, stream_parts
from app.core.config import get_settings
from app.services.materials.rag.live_eval import index_version


def _db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[ProviderHealth.__table__, ToolCacheEntry.__table__, ToolRun.__table__])
    return Session(engine)


def test_cache_requires_scope_and_index_version() -> None:
    db = _db()
    cache_put(db, "hybrid_search", {"query": "python"}, {"hits": [1]}, scope="user-a", index_version="v1")
    assert cache_get(db, "hybrid_search", {"query": "python"}, scope="user-a", index_version="v1") == {"hits": [1]}
    cache_put(db, "hybrid_search", {"query": "python"}, {"hits": [2]}, scope="user-a", index_version="v1")
    db.commit()

    assert cache_get(db, "hybrid_search", {"query": "python"}, scope="user-a", index_version="v1") == {"hits": [2]}
    assert cache_get(db, "hybrid_search", {"query": "python"}, scope="user-b", index_version="v1") is None
    assert cache_get(db, "hybrid_search", {"query": "python"}, scope="user-a", index_version="v2") is None
    assert cache_get(db, "hybrid_search", {"query": "python"}) is None


def test_role_tool_cache_binds_owner_session_and_index() -> None:
    """The actual role adapter must hit the cache only for the same index scope."""
    db = _db()
    calls = 0

    async def handler(_db: Session, _args: dict) -> dict:
        nonlocal calls
        calls += 1
        return {"success": True, "hits": [calls]}

    spec = ToolSpec("hybrid_search", "search", ("query",), 120.0, handler,
                    {"query": {"type": "string"}})

    async def run(seed: dict[str, str]) -> str:
        tool = _langchain_tool(spec, db, seed)
        return await tool.coroutine(tool_call_id=uuid.uuid4().hex, query="python")

    seed = {"cache_owner_id": "user-a", "cache_session_id": "session-a", "cache_index_version": "v1"}
    assert '"hits": [1]' in asyncio.run(run(seed))
    assert '"cached": true' in asyncio.run(run(seed))
    assert calls == 1
    assert '"hits": [2]' in asyncio.run(run({**seed, "cache_session_id": "session-b"}))
    assert '"hits": [3]' in asyncio.run(run({**seed, "cache_index_version": "v2"}))
    assert '"hits": [4]' in asyncio.run(run({"cache_owner_id": "user-a"}))


def test_author_without_tool_call_does_not_scan_ready_index() -> None:
    """Candidate generation should not fingerprint the corpus before a tool call."""
    async def handoff(_db, _task, **_kwargs):
        return {"output": {"questions": []}}

    with patch("app.agents.roles.authoring.run_handoff", side_effect=handoff), patch(
        "app.services.materials.rag.live_eval.index_version", side_effect=AssertionError("eager index scan")
    ):
        result = asyncio.run(author_question_candidate(
            SimpleNamespace(), "backend role", context="", hits=[], run_id="run-1", attempt=1,
        ))
    assert result["output"]["questions"] == []


def test_role_tool_lazily_fingerprints_index_and_invalidates_changed_corpus() -> None:
    """Actual retrieval keeps the exact ready-index boundary without eager work."""
    db = _db()
    calls = 0
    versions = iter(("v1", "v1", "v2"))

    async def handler(_db: Session, _args: dict) -> dict:
        nonlocal calls
        calls += 1
        return {"success": True, "hits": [calls]}

    spec = ToolSpec("hybrid_search", "search", ("query",), 120.0, handler,
                    {"query": {"type": "string"}})
    seed = {"cache_owner_id": "user-a", "cache_session_id": "session-a"}

    async def run() -> str:
        tool = _langchain_tool(spec, db, seed)
        return await tool.coroutine(tool_call_id=uuid.uuid4().hex, query="python")

    with patch("app.services.materials.rag.live_eval.index_version", side_effect=lambda _db: (next(versions), set(), set())):
        assert '"hits": [1]' in asyncio.run(run())
        assert '"cached": true' in asyncio.run(run())
        assert '"hits": [2]' in asyncio.run(run())
    assert calls == 2


def test_half_open_probe_is_leased_to_one_governor() -> None:
    db = _db()
    db.add(ProviderHealth(provider_id="provider-a", provider_name="A", state="open", opened_at=0,
                          consecutive_fails=3, total=3, success=0, total_ms=0))
    db.commit()

    first = ProviderGovernor(db, "provider-a", "A")
    second = ProviderGovernor(db, "provider-a", "A")
    assert first.allow() is True
    assert second.allow() is False
    second.record_success(1)
    assert db.get(ProviderHealth, "provider-a").state == "half_open"
    first.record_failure(1)
    assert db.get(ProviderHealth, "provider-a").state == "open"


def test_legacy_half_open_row_without_lease_is_eligible_for_probe() -> None:
    """An upgraded breaker row without lease columns must not stay excluded."""
    db = _db()
    db.add(ProviderHealth(provider_id="legacy-provider", provider_name="Legacy", state="half_open",
                          probe_lease_until=None, probe_owner=None,
                          consecutive_fails=3, total=3, success=0, total_ms=0))
    db.commit()

    governor = ProviderGovernor(db, "legacy-provider", "Legacy")
    assert governor.eligible() is True
    assert governor.allow() is True


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_routing_does_not_consume_the_only_recovery_probe() -> None:
    """A read-only route must leave the half-open claim for the actual model call."""
    name = f"sagematch_governance_test_{uuid.uuid4().hex[:10]}"
    base = make_url(get_settings().database_url)
    admin_url = base.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            engine = create_engine(base.set(database=name))
            try:
                Base.metadata.create_all(engine)
                with Session(engine) as db:
                    provider = ProviderConfig(id="provider-a", name="A", protocol="chat_completions", capability="llm", base_url="http://example.invalid", api_key="test")
                    db.add(provider)
                    db.add(RoleBinding(id="binding-a", role="author", label="Author", provider_id=provider.id, model="test-model"))
                    db.add(ProviderHealth(provider_id=provider.id, provider_name="A", state="open", opened_at=0,
                                          consecutive_fails=3, total=3, success=0, total_ms=0))
                    db.commit()

                    assert choose_provider(db, "author").provider.id == provider.id
                    db.refresh(db.get(ProviderHealth, provider.id))
                    assert db.get(ProviderHealth, provider.id).state == "open"
                    with patch("app.integrations.llm.complete_text", new_callable=AsyncMock, return_value="probe succeeded") as model:
                        result = asyncio.run(complete_with(db, "author", "system", "prompt"))
                    db.commit()
                    assert result == "probe succeeded"
                    model.assert_awaited_once()
                    assert db.get(ProviderHealth, provider.id).state == "closed"

                    health = db.get(ProviderHealth, provider.id)
                    health.state = "open"
                    health.opened_at = 0
                    db.commit()

                    async def fake_stream(*_args, **_kwargs):
                        yield "answer", "streamed probe"

                    async def consume() -> list[tuple[str, str]]:
                        return [part async for part in stream_parts(db, "author", "system", "prompt")]

                    with patch("app.integrations.llm.stream_chat_parts", side_effect=fake_stream) as stream:
                        assert asyncio.run(consume()) == [("answer", "streamed probe")]
                    db.commit()
                    stream.assert_called_once()
                    assert db.get(ProviderHealth, provider.id).state == "closed"

                    health.state = "open"
                    health.opened_at = 0
                    db.commit()
                    first = ProviderGovernor(db, provider.id, provider.name)
                    assert first.allow() is True
                    db.commit()
                    with Session(engine) as contender:
                        assert ProviderGovernor(contender, provider.id, provider.name).allow() is False
                        contender.rollback()

                    db.add(Material(id="material-a", filename="notes.txt", status="ready"))
                    db.add(MaterialChunk(id="chunk-a", material_id="material-a", ordinal=0,
                                         text="first content", embedding_model="test"))
                    db.commit()
                    calls = 0

                    async def handler(_db: Session, _args: dict) -> dict:
                        nonlocal calls
                        calls += 1
                        return {"success": True, "hits": [calls]}

                    spec = ToolSpec("hybrid_search", "search", ("query",), 120.0, handler,
                                    {"query": {"type": "string"}})

                    async def search(version: str) -> str:
                        tool = _langchain_tool(spec, db, {"cache_owner_id": "local-user",
                                                         "cache_session_id": "session-a",
                                                         "cache_index_version": version})
                        return await tool.coroutine(tool_call_id=uuid.uuid4().hex, query="python")

                    first_version, _, _ = index_version(db)
                    assert '"hits": [1]' in asyncio.run(search(first_version))
                    assert '"cached": true' in asyncio.run(search(first_version))
                    db.get(MaterialChunk, "chunk-a").text = "changed content"
                    db.commit()
                    second_version, _, _ = index_version(db)
                    assert second_version != first_version
                    assert '"hits": [2]' in asyncio.run(search(second_version))
                    assert calls == 2
            finally:
                engine.dispose()
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
