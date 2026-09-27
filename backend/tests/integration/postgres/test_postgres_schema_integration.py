"""Opt-in PostgreSQL migration smoke test in a disposable database."""

from __future__ import annotations

import os
import uuid

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import create_engine, inspect, make_url, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import ensure_schema
from app.models import RuntimeSchemaVersion


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_runtime_schema_builds_on_postgres() -> None:
    """All runtime contracts must exist before startup advertises readiness."""
    name = f"sagematch_schema_test_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    test_url = base_url.set(database=name)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        engine = create_engine(test_url)
        try:
            ensure_schema(engine)
            with Session(engine) as db:
                assert db.get(RuntimeSchemaVersion, "runtime").version == 2
            inspector = inspect(engine)
            required = {
                "durable_jobs", "graph_runs", "node_runs", "tool_runs", "agent_events",
                "graph_checkpoint_owners", "stream_runs", "stream_events",
            }
            assert required <= set(inspector.get_table_names())
            assert {"probe_owner", "probe_lease_until"} <= {
                column["name"] for column in inspector.get_columns("provider_health")
            }
            assert {"scope", "index_version"} <= {
                column["name"] for column in inspector.get_columns("tool_cache")
            }
            assert "run_id" in {column["name"] for column in inspector.get_columns("llm_call_logs")}
            assert "duration_ms" in {column["name"] for column in inspector.get_columns("node_runs")}
            # Exercise the upgrade path, not only fresh create_all metadata.
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE provider_health DROP COLUMN probe_owner"))
                conn.execute(text("ALTER TABLE tool_cache DROP COLUMN index_version"))
                conn.execute(text("ALTER TABLE user_profile_memory DROP COLUMN provenance"))
                conn.execute(text("ALTER TABLE llm_call_logs DROP COLUMN run_id"))
                conn.execute(text("ALTER TABLE node_runs DROP COLUMN duration_ms"))
            ensure_schema(engine)
            inspector = inspect(engine)
            assert "probe_owner" in {column["name"] for column in inspector.get_columns("provider_health")}
            assert "index_version" in {column["name"] for column in inspector.get_columns("tool_cache")}
            assert "provenance" in {column["name"] for column in inspector.get_columns("user_profile_memory")}
            assert "run_id" in {column["name"] for column in inspector.get_columns("llm_call_logs")}
            assert "duration_ms" in {column["name"] for column in inspector.get_columns("node_runs")}
        finally:
            engine.dispose()
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()
