"""Opt-in PostgreSQL lease and report uniqueness check in a disposable DB."""

from __future__ import annotations

import os
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import create_engine, make_url
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import ensure_schema
from app.models import DurableJob, Interview, JobProfile, Material, MaterialChunk, Report
from app.services.interviews.interview import _store_report
from app.services.materials.knowledge import commit_material_job, process_material
from app.services.operations.jobs import claim_job, claim_job_by_id, complete_job, create_job, reclaim_expired_jobs, renew_lease


@pytest.mark.skipif(os.getenv("SAGEMATCH_TEST_POSTGRES") != "1", reason="requires isolated local PostgreSQL")
def test_expired_worker_cannot_complete_and_report_is_unique(tmp_path, monkeypatch) -> None:
    """PostgreSQL enforces lease ownership and one report per interview."""
    name = f"sagematch_jobs_test_{uuid.uuid4().hex[:10]}"
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
                job = create_job(db, "report_generation", "interview-1", idempotency_key="report:i1")
                db.add(JobProfile(id="profile-1", user_id=get_settings().anonymous_user_id,
                                  raw_text="测试岗位", job_title="测试岗位"))
                db.flush()
                db.add(Interview(id="interview-1", profile_id="profile-1", title="测试面试", status="ended"))
                db.commit()
                job_id = job.id

            with Session(engine) as first:
                assert claim_job(first, "worker-a", lease_seconds=60).id == job_id
                first.commit()
            with Session(engine) as second:
                assert claim_job(second, "worker-b") is None
                row = second.get(DurableJob, job_id)
                row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
                second.commit()
                assert reclaim_expired_jobs(second) == 1
                second.commit()
                assert claim_job(second, "worker-b").id == job_id
                second.commit()

            with Session(engine) as first:
                assert complete_job(first, job_id, worker_id="worker-a") is None
                first.rollback()
            with Session(engine) as second:
                assert complete_job(second, job_id, worker_id="worker-b") is not None
                second.commit()

            recap = {"score": 0, "review": "报告", "summary": "报告", "issues": []}
            with Session(engine) as first:
                _store_report(first, "interview-1", recap)
            with Session(engine) as second:
                _store_report(second, "interview-1", {**recap, "review": "重复报告"})
                assert second.query(Report).filter(Report.interview_id == "interview-1").count() == 1
                assert second.query(Report).one().review == "报告"

            with Session(engine) as db:
                db.add(Interview(id="interview-2", profile_id="profile-1", title="并发复盘", status="ended"))
                db.commit()
            barrier = Barrier(2)

            def write_competing_report(label: str) -> None:
                with Session(engine) as db:
                    barrier.wait(timeout=5)
                    _store_report(db, "interview-2", {**recap, "review": label})

            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(write_competing_report, ("worker-a", "worker-b")))
            with Session(engine) as db:
                rows = db.query(Report).filter(Report.interview_id == "interview-2").all()
                assert len(rows) == 1
                assert rows[0].review in {"worker-a", "worker-b"}

            # A parser's uncommitted chunks and missing-file status both disappear
            # if another process takes the lease before the publish transaction.
            from app.services.materials import knowledge as ingest

            async def fake_embed(_db, parts):
                return [None] * len(parts), ""

            monkeypatch.setattr(ingest, "embed_or_empty", fake_embed)
            monkeypatch.setattr(ingest.knowledge, "parse_bytes", lambda _name, _data: "staged text")
            monkeypatch.setattr(ingest.knowledge, "split_chunks", lambda text: [text])
            monkeypatch.setattr(ingest, "upload_dir", lambda: tmp_path)

            for case_name, exists in (("staged", True), ("missing", False)):
                material_id = f"material-{case_name}"
                if exists:
                    ingest.knowledge.upload_path(tmp_path, material_id, "source.txt").write_bytes(b"source")
                with Session(engine) as db:
                    db.add(Material(id=material_id, filename="source.txt", status="pending"))
                    job = create_job(db, "material_ingest", material_id,
                                     idempotency_key=f"material:{material_id}")
                    db.commit()
                    job_id = job.id
                with Session(engine) as owner:
                    assert claim_job(owner, "parser-a", kinds=["material_ingest"]) is not None
                    owner.commit()
                    asyncio.run(process_material(owner, material_id))
                    with Session(engine) as replacement:
                        leased = replacement.get(DurableJob, job_id)
                        leased.worker_id = "parser-b"
                        replacement.commit()
                    assert commit_material_job(owner, job_id, material_id, "parser-a") is False
                with Session(engine) as check:
                    assert check.get(Material, material_id).status == "pending"
                    assert check.query(MaterialChunk).filter_by(material_id=material_id).count() == 0

            # The parser session caches the original short lease while the
            # heartbeat extends the same row through an independent session.
            material_id = "material-renewed"
            ingest.knowledge.upload_path(tmp_path, material_id, "source.txt").write_bytes(b"source")
            with Session(engine) as db:
                db.add(Material(id=material_id, filename="source.txt", status="pending"))
                job = create_job(db, "material_ingest", material_id,
                                 idempotency_key=f"material:{material_id}")
                db.commit()
                job_id = job.id
            with Session(engine) as owner:
                assert claim_job_by_id(owner, job_id, "parser-renewed", lease_seconds=1) is not None
                owner.commit()
                cached_job = owner.get(DurableJob, job_id)
                original_lease = cached_job.lease_until
                with Session(engine) as heartbeat:
                    assert renew_lease(heartbeat, job_id, "parser-renewed", lease_seconds=60)
                    heartbeat.commit()
                time.sleep(1.1)
                assert original_lease < datetime.now(timezone.utc)
                assert cached_job.lease_until == original_lease
                asyncio.run(process_material(owner, material_id))
                assert commit_material_job(owner, job_id, material_id, "parser-renewed") is True
            with Session(engine) as check:
                assert check.get(Material, material_id).status == "ready"
                assert check.query(MaterialChunk).filter_by(material_id=material_id).count() == 1
                assert check.get(DurableJob, job_id).status == "succeeded"
        finally:
            engine.dispose()
    finally:
        if created:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()
