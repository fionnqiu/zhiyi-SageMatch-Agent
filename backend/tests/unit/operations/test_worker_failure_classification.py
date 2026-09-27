"""Worker failures preserve the retryability declared by their error type."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.db import Base
from app.models.platform.runtime import DurableJob
from app.services.interviews import interview
from app.services.materials import knowledge
from app.services.operations.jobs import create_job
from app.agents.orchestration.workflows import GraphExecutionError


@pytest.mark.parametrize(
    ("service", "kind", "error", "expected_status"),
    [
        ("report", "report_generation", GraphExecutionError("model_budget_exceeded", retryable=False), "dead_letter"),
        ("report", "report_generation", RuntimeError("temporary provider outage"), "retry_wait"),
        ("material", "material_ingest", GraphExecutionError("model_budget_exceeded", retryable=False), "dead_letter"),
        ("material", "material_ingest", RuntimeError("temporary provider outage"), "retry_wait"),
    ],
)
def test_worker_keeps_failure_retryability(service: str, kind: str, error: Exception, expected_status: str) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[DurableJob.__table__])
    factory = sessionmaker(engine)
    with factory() as db:
        job_id = create_job(db, kind, f"{service}-item").id
        db.commit()

    async def stop_polling(_delay: float) -> None:
        raise asyncio.CancelledError

    async def fail_material(_db: Session, _id: str) -> None:
        raise error

    def fail_report(_db: Session, _id: str):
        raise error

    module = interview if service == "report" else knowledge
    queue = interview._run_report_queue if service == "report" else knowledge.run_material_queue
    failure = patch.object(module, "get_interview", fail_report) if service == "report" else patch.object(module, "process_material", fail_material)
    with patch.object(module, "SessionLocal", factory), patch.object(module.asyncio, "sleep", stop_polling), failure:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(queue())

    with factory() as db:
        row = db.get(DurableJob, job_id)
        assert row is not None
        assert row.status == expected_status
        assert row.attempts == 1
    engine.dispose()
