"""Rehearse a runtime migration against a disposable copy of the live schema."""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import create_engine, inspect, make_url
from sqlalchemy.orm import Session

from app import models  # noqa: F401 - register every ORM table for the schema comparison
from app.core.config import get_settings
from app.core.db import Base, ensure_schema
from app.models import RuntimeSchemaVersion


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_runtime_schema_upgrades_copy_of_business_schema() -> None:
    """Catch upgrade failures that a fresh, empty ORM-created database misses."""
    pg_dump = shutil.which("pg_dump")
    psql = shutil.which("psql")
    if not pg_dump or not psql:
        pytest.skip("pg_dump and psql are required for the schema rehearsal")

    source = make_url(get_settings().database_url)
    clone_name = f"sagematch_rehearsal_{uuid.uuid4().hex[:12]}"
    clone = source.set(database=clone_name)
    admin_dsn = source.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    connection_env = os.environ.copy()
    connection_env["PGPASSWORD"] = source.password or ""
    connection_args = ["-h", source.host or "127.0.0.1", "-p", str(source.port or 5432), "-U", source.username or "postgres"]

    # Only DDL is copied; no rows or local dump artifact leave the source DB.
    with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(clone_name)))
        engine = None
        try:
            dumped = subprocess.run(
                [pg_dump, *connection_args, "--schema-only", "--no-owner", "--no-acl", "-d", source.database],
                env=connection_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            assert dumped.returncode == 0, "schema-only pg_dump failed"
            restored = subprocess.run(
                [psql, *connection_args, "-X", "-v", "ON_ERROR_STOP=1", "-d", clone_name],
                env=connection_env, input=dumped.stdout, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, check=False,
            )
            assert restored.returncode == 0, "schema-only restore failed"

            engine = create_engine(clone)
            ensure_schema(engine)
            ensure_schema(engine)  # A later application boot must be idempotent.
            inspector = inspect(engine)
            missing_tables = set(Base.metadata.tables) - set(inspector.get_table_names())
            assert not missing_tables, f"missing tables after migration: {sorted(missing_tables)}"
            for table in Base.metadata.tables.values():
                actual = {column["name"] for column in inspector.get_columns(table.name)}
                expected = {column.name for column in table.columns}
                assert expected <= actual, f"missing {table.name} columns: {sorted(expected - actual)}"
            with Session(engine) as db:
                assert db.get(RuntimeSchemaVersion, "runtime").version == 2
        finally:
            if engine is not None:
                engine.dispose()
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(clone_name)))
