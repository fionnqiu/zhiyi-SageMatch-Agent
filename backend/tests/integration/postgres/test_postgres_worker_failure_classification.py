"""Run durable worker failure classification against disposable PostgreSQL."""

from __future__ import annotations

import asyncio
import os
import uuid
from unittest.mock import patch

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import create_engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.agents.orchestration.workflows import GraphExecutionError
from app.core.config import get_settings
from app.core.db import Base
from app.models.platform.runtime import DurableJob
from app.services.interviews import interview
from app.services.materials import knowledge
from app.services.operations.jobs import create_job


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
@pytest.mark.parametrize("service, kind", [("report", "report_generation"), ("material", "material_ingest")])
def test_postgres_worker_preserves_terminal_and_transient_errors(service: str, kind: str) -> None:
    name = f"sagematch_worker_error_test_{uuid.uuid4().hex[:10]}"
    base_url = make_url(get_settings().database_url)
    admin_dsn = base_url.set(database="postgres", drivername="postgresql").render_as_string(hide_password=False)
    admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    created = False
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        engine = create_engine(base_url.set(database=name))
        try:
            Base.metadata.create_all(engine, tables=[DurableJob.__table__])
            factory = sessionmaker(engine)
            module = interview if service == "report" else knowledge
            queue = interview._run_report_queue if service == "report" else knowledge.run_material_queue

            async def stop_polling(_delay: float) -> None:
                raise asyncio.CancelledError

            for error, expected in (
                (GraphExecutionError("model_budget_exceeded", retryable=False), "dead_letter"),
                (RuntimeError("temporary provider outage"), "retry_wait"),
            ):
                with factory() as db:
                    job_id = create_job(db, kind, f"{service}-{expected}").id
                    db.commit()

                async def fail_material(_db: Session, _id: str) -> None:
                    raise error

                def fail_report(_db: Session, _id: str):
                    raise error

                failure = patch.object(module, "get_interview", fail_report) if service == "report" else patch.object(module, "process_material", fail_material)
                # A cancelled idle poll stops the infinite worker loop after
                # the single claimed job has committed its classified failure.
                with patch.object(module, "SessionLocal", factory), patch.object(module.asyncio, "sleep", stop_polling), failure:
                    with pytest.raises(asyncio.CancelledError):
                        asyncio.run(queue())

                with factory() as db:
                    row = db.get(DurableJob, job_id)
                    assert row is not None
                    assert row.status == expected
                    assert row.attempts == 1
                    assert row.error["retryable"] is (expected == "retry_wait")
        finally:
            engine.dispose()
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()
