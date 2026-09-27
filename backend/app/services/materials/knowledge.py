"""Knowledge ingest: accept a file, queue parsing, and serve admin recall."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session, selectinload

from app.materials import knowledge
from app.core.db import SessionLocal
from app.models import DurableJob, Material, MaterialChunk
from app.services.shared.common import audit, new_id, upload_dir
from app.services.operations.jobs import JobWorker, complete_job, create_job, enqueue_material_job, fail_job, keep_job_lease
from app.services.materials.recall import embed_or_empty, recall_snippets

# The task handle only keeps one poller per process; work itself lives in the
# durable_jobs table so a restart or a second worker can reclaim it.
_material_worker: asyncio.Task[None] | None = None
_material_worker_id = f"material-{os.getpid()}"


def list_materials(db: Session) -> list[Material]:
    return db.query(Material).order_by(Material.created_at.desc()).all()


def get_material(db: Session, material_id: str) -> Material | None:
    return (
        db.query(Material)
        .options(selectinload(Material.chunks))
        .filter(Material.id == material_id)
        .one_or_none()
    )


def accept_material(db: Session, filename: str, data: bytes, mime: str = "text/plain") -> Material:
    """Persist the file and return immediately. Parsing runs on the background queue."""
    mid = new_id()
    row = Material(
        id=mid,
        filename=filename,
        mime=mime,
        size_bytes=len(data),
        status="pending",
        source="upload",
    )
    db.add(row)
    db.flush()
    knowledge.write_upload(upload_dir(), mid, filename, data)
    audit(db, "material.upload", filename, {"status": "pending"})
    db.commit()
    db.refresh(row)
    return row


async def finish_material(db: Session, row: Material, data: bytes) -> None:
    try:
        text = knowledge.parse_bytes(row.filename, data)
        parts = knowledge.split_chunks(text)
        if not parts:
            raise ValueError("文件没有可切分的文本")
        vectors, model_name = await embed_or_empty(db, parts)
        for i, part in enumerate(parts):
            db.add(
                MaterialChunk(
                    id=new_id(),
                    material_id=row.id,
                    ordinal=i,
                    text=part,
                    token_estimate=knowledge.token_estimate(part),
                    embedding=vectors[i] if i < len(vectors) else None,
                    embedding_model=model_name if i < len(vectors) else "",
                )
            )
        row.chunk_count = len(parts)
        row.status = "ready"
        row.error = None
    except Exception as exc:  # noqa: BLE001
        row.status = "failed"
        row.error = str(exc)[:400]
    # The caller publishes this result in the same transaction as job completion.


def pending_material_ids(db: Session) -> list[str]:
    rows = db.query(Material.id).filter(Material.status == "pending").all()
    return [row[0] for row in rows]


def start_material_worker() -> None:
    """Start one process-local poller over durable material-ingest jobs."""
    global _material_worker
    if _material_worker and not _material_worker.done():
        return
    _material_worker = asyncio.create_task(run_material_queue())


def material_worker_running() -> bool:
    """Report the actual poller state, including unexpected task completion."""
    return _material_worker is not None and not _material_worker.done()


async def stop_material_worker() -> None:
    """Stop the process-local poller during application shutdown."""
    global _material_worker
    task = _material_worker
    _material_worker = None
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def enqueue_material_id(material_id: str) -> None:
    """Create an idempotent outbox row before waking the local poller."""
    db = SessionLocal()
    try:
        enqueue_material_job(db, material_id)
        db.commit()
    finally:
        db.close()
    start_material_worker()


def commit_material_job(db: Session, job_id: str, material_id: str, worker_id: str) -> bool:
    """Publish a staged parse only while this worker still owns the SQL lease."""
    # The row lock fences a concurrent reclaim through the business commit.
    current_job = (
        db.query(DurableJob)
        .filter(
            DurableJob.id == job_id,
            DurableJob.status == "running",
            DurableJob.worker_id == worker_id,
            DurableJob.lease_until > datetime.now(timezone.utc),
        )
        .with_for_update()
        .one_or_none()
    )
    if current_job is None:
        db.rollback()
        return False
    # The heartbeat renews this row in another session. Refresh the cached ORM
    # instance under our row lock before complete_job checks its lease deadline.
    db.refresh(current_job)
    material = db.get(Material, material_id)
    if material is not None and material.status == "failed":
        finished = fail_job(db, job_id, "material_processing_failed", retryable=False, worker_id=worker_id)
    else:
        finished = complete_job(db, job_id, worker_id=worker_id, result={"material_id": material_id})
    if finished is None:
        db.rollback()
        return False
    db.commit()
    return True


async def run_material_queue() -> None:
    """Poll persisted jobs; no request-owned asyncio queue is used for work."""
    worker = JobWorker(_material_worker_id)
    while True:
        db = SessionLocal()
        row = None
        try:
            row = worker.claim(db, kinds=["material_ingest"])
            if row is None:
                db.commit()
                await asyncio.sleep(0.5)
                continue
            # Commit the lease before parsing or embedding. Recovery can then
            # see exactly which job was interrupted by a process restart.
            db.commit()
            material_id = row.business_key
            async with keep_job_lease(SessionLocal, row.id, worker.worker_id, lease_seconds=worker.lease_seconds) as lease:
                await process_material(db, material_id)
                if not lease[0]:
                    db.rollback()
                    continue
                commit_material_job(db, row.id, material_id, worker.worker_id)
        except asyncio.CancelledError:
            db.rollback()
            raise
        except Exception as exc:  # noqa: BLE001 - worker continues after one bad job
            db.rollback()
            # A claimed row may be retried or dead-lettered according to its
            # persisted attempt budget; the process itself must stay alive.
            if row is not None:
                # Preserve parser/provider error classification across the durable queue.
                fail_job(db, row.id, exc, worker_id=worker.worker_id)
                db.commit()
        finally:
            db.close()


async def process_material(db: Session, material_id: str) -> None:
    """Stage material changes in the worker transaction until its lease is fenced."""
    row = db.get(Material, material_id)
    if not row or row.status != "pending":
        return
    path = knowledge.upload_path(upload_dir(), row.id, row.filename)
    if not path.exists():
        row.status = "failed"
        row.error = "原件丢失，无法继续入库"
        return
    await finish_material(db, row, path.read_bytes())


def delete_material(db: Session, material_id: str) -> None:
    row = db.get(Material, material_id)
    if not row:
        raise ValueError("物料不存在")
    name = row.filename
    status = row.status
    db.delete(row)
    audit(db, "material.delete", name, {"status": status})
    db.commit()


async def recall(db: Session, query: str) -> dict[str, Any]:
    hits = await recall_snippets(db, query)
    c_count = db.query(MaterialChunk).count()
    return {
        "query": query,
        "hits": hits,
        "index_status": "就绪" if c_count else "空索引",
    }
