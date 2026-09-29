"""Interview routes: hub cards, live turns, and the downloadable recap."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from sqlalchemy.orm import Session

from app import schemas, services
from app.api.shared.deps import interview_detail, interview_out, report_text
from app.core.db import SessionLocal, get_db
from app.models import ChatMessage, ChatSession, DurableJob, Interview, InterviewTurn, JobProfile, StreamRun
from app.agents.orchestration.workflows import run_business_graph
from app.agents.workflows.generation import InterviewGenerationStages
from app.agents.workflows.evaluation import ReportRegenerationCommandStages
from app.agents.workflows.live_interview import LiveInterviewCommandStages, run_live_answer_graph
from app.services.shared.common import ANON
from app.services.chat.stream_events import append_stream_event, create_stream_run, event_frame, replay_stream_events
from app.services.operations.jobs import claim_job_by_id, complete_job, create_job, fail_job, keep_job_lease, reclaim_expired_jobs, renew_lease

router = APIRouter()
logger = logging.getLogger(__name__)
_generation_worker: asyncio.Task[None] | None = None
_generation_checkpointer = None
_active_generation_tasks: set[asyncio.Task] = set()


def _stream_headers() -> dict[str, str]:
    # 反向代理看到这些头就不要把阶段事件攒成一整包。
    return {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}


def _request_checkpointer(request: Request | None):
    """Read the lifespan saver while preserving direct service-route callers."""
    state = getattr(getattr(request, "app", None), "state", None)
    return getattr(state, "checkpointer", None)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _surface_task_error(task: asyncio.Task) -> None:
    """生成任务如果在推送前就失败，异常要被取走，不能在流还开着时打进事件循环。"""
    if task.cancelled():
        return
    try:
        task.exception()
    except Exception:
        return


def _generation_result_interview(db: Session, interview_id: str) -> dict | None:
    """Resolve the interview named by the committed job result."""
    ready = db.get(Interview, interview_id)
    return {"id": ready.id, "title": ready.title, "status": ready.status} if ready else None


def _publish_generation_event(
    db: Session, job_id: str, run_id: str, payload: dict, *, worker_id: str | None = None,
) -> str | None:
    """Fence a generation frame to the leased job or its committed terminal fact.

    The job row lock remains held until append_stream_event commits its frame,
    so a concurrent claimant cannot replace the worker between the check and
    publication. A successful job may be published by a recovery process.
    """
    job = db.query(DurableJob).filter(DurableJob.id == job_id).with_for_update().one_or_none()
    if job is None or job.kind != "interview_generation" or job.business_key != run_id:
        db.rollback()
        return None
    kind = payload.get("type")
    if kind == "meta":
        until = job.lease_until
        if until is not None and until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        if (job.status != "running" or not worker_id or job.worker_id != worker_id or
                until is None or until <= datetime.now(timezone.utc)):
            db.rollback()
            return None
    elif kind == "done":
        interview_id = str((job.result or {}).get("interview_id") or "")
        ready = _generation_result_interview(db, interview_id) if job.status == "succeeded" else None
        if ready is None:
            db.rollback()
            return None
        payload = {"type": "done", "interview": ready}
    elif kind == "error":
        # A retry_wait job has no terminal result. Keep its stream open for the
        # next claimant, including clients already following the SSE sequence.
        if job.status != "dead_letter":
            db.rollback()
            return None
        payload = {"type": "error", "message": "题目没有生成"}
    else:
        db.rollback()
        raise ValueError("invalid generation event")
    try:
        return event_frame(append_stream_event(db, run_id, ANON, payload))
    except PermissionError:
        db.rollback()
        if kind == "done":
            events = replay_stream_events(db, run_id, ANON)
            if events and events[-1].event_type == "done":
                return event_frame(events[-1])
        return None


async def _generate_events(
    db: Session, content: str, checkpointer=None, run_id: str | None = None,
    job_id: str | None = None, worker_id: str | None = None,
) -> AsyncIterator[str]:
    """出题在后台跑。思考原文不推给页面，完成或失败才发一条事件。"""
    # 请求上的会话只用来做前置校验。生成可能很久，不能占着这条连接的事务。
    db.close()
    done: asyncio.Queue[None] = asyncio.Queue()

    async def _drop_thought(_text: str) -> None:
        # 模型思考里有题干和工具结构。创建页只需要知道还在生成。
        return None

    async def _run() -> tuple[dict | None, str | None]:
        own = SessionLocal()
        try:
            try:
                generation = InterviewGenerationStages(own, content, run_id, _drop_thought) if run_id and (job_id or checkpointer is not None) else None
                # A process may have committed the pack before losing its
                # checkpoint or SSE terminal frame. Reuse that durable fact.
                ready = (
                    own.query(Interview).join(JobProfile, JobProfile.id == Interview.profile_id)
                    .filter(JobProfile.session_id == run_id).first()
                    if run_id and (job_id or checkpointer is not None) else None
                )
                interview = {"id": ready.id, "title": ready.title, "status": ready.status} if ready else None
                if interview is None:
                    async def persist_question_set():
                        # The lease row stays locked through the business commit.
                        if job_id and worker_id and not renew_lease(own, job_id, worker_id):
                            raise RuntimeError("generation lease lost")
                        return await services.prepare_interview(
                            own, content, on_thought=_drop_thought, run_id=run_id,
                            **({"prepared_hits": generation.hits, "prepared_payload": generation.payload,
                                "generation_trace": generation.handoffs} if generation else {}),
                        )

                    async def execute_graph():
                        return await run_business_graph(
                            own, mode="interview_generation",
                            # Raw job text is held by this worker, never by the checkpoint.
                            original_query=hashlib.sha256(content.strip().encode("utf-8")).hexdigest(),
                            checkpointer=checkpointer, run_id=run_id,
                            stages=generation.callbacks() if generation else None,
                            supervised_route=generation.supervised_route() if generation else None,
                            restart_incomplete=bool(generation), action=persist_question_set,
                            restart_completed_failed=bool(generation),
                        )

                    if job_id and worker_id:
                        async with keep_job_lease(SessionLocal, job_id, worker_id) as lease_active:
                            interview = await execute_graph()
                            if not lease_active[0]:
                                raise RuntimeError("generation lease lost")
                    else:
                        interview = await execute_graph()
                if job_id and worker_id:
                    if not renew_lease(own, job_id, worker_id) or complete_job(
                        own, job_id, worker_id=worker_id, result={"interview_id": interview["id"]},
                    ) is None:
                        raise RuntimeError("generation lease lost")
                    own.commit()
                payload = {"type": "done", "interview": interview} if interview and interview.get("id") else {"type": "error", "message": "题目没有生成"}
            except ValueError as exc:
                payload = {"type": "error", "message": str(exc)}
                interview = None
            except Exception:
                logger.exception("interview generation failed")
                payload = {"type": "error", "message": "题目没有生成"}
                interview = None
            # The worker commits the terminal event itself, so a dropped HTTP
            # connection cannot erase a successfully generated interview.
            if job_id and worker_id and payload["type"] == "error":
                own.rollback()
                if renew_lease(own, job_id, worker_id):
                    fail_job(own, job_id, "generation_failed", retryable=True,
                             worker_id=worker_id, retry_delay_seconds=0)
                    own.commit()
                frame = _publish_generation_event(own, job_id, run_id, payload)
            else:
                if run_id:
                    frame = (_publish_generation_event(own, job_id, run_id, payload)
                             if job_id else event_frame(append_stream_event(own, run_id, ANON, payload)))
                else:
                    frame = _sse(payload)
            return interview, frame
        finally:
            own.close()
            await done.put(None)

    meta_frame = None
    if run_id:
        event_db = SessionLocal()
        try:
            meta_frame = (_publish_generation_event(event_db, job_id, run_id, {"type": "meta"},
                                                    worker_id=worker_id)
                          if job_id else event_frame(append_stream_event(event_db, run_id, ANON,
                                                                          {"type": "meta"})))
        finally:
            event_db.close()
    task = asyncio.create_task(_run())
    _active_generation_tasks.add(task)
    task.add_done_callback(_active_generation_tasks.discard)
    if meta_frame:
        yield meta_frame
    # 任务自己的异常不能留到流结束才看。否则连接还开着时，失败会先打进事件循环。
    task.add_done_callback(_surface_task_error)
    while True:
        if task.done() and done.empty():
            break
        try:
            piece = await asyncio.wait_for(done.get(), timeout=15)
        except asyncio.TimeoutError:
            # 长时间没有完成事件时发一个空心跳，避免代理把这条连接当成空闲掐掉。
            yield ":\n\n"
            continue
        if piece is None:
            break
    _interview, frame = task.result()
    if frame:
        yield frame
    elif run_id and job_id:
        async for recovered in _follow_generation_events(run_id, meta_frame.split("\n", 1)[0][4:] if meta_frame else None):
            yield recovered


async def _follow_generation_events(run_id: str, after_event_id: str | None) -> AsyncIterator[str]:
    """A duplicate request watches the owning worker's committed SSE frames."""
    cursor = after_event_id
    while True:
        own = SessionLocal()
        try:
            events = replay_stream_events(own, run_id, ANON, cursor)
            for event in events:
                cursor = event.event_id
                yield event_frame(event)
                if event.event_type in {"done", "error"} and own.get(StreamRun, run_id).status == "completed":
                    return
            job = own.query(DurableJob).filter(
                DurableJob.kind == "interview_generation", DurableJob.business_key == run_id,
            ).one_or_none()
            if job is None:
                return
            if job.status == "dead_letter":
                frame = _publish_generation_event(own, job.id, run_id, {"type": "error"})
                if frame:
                    yield frame
                return
            if job.status == "succeeded":
                interview_id = str((job.result or {}).get("interview_id") or "")
                ready = own.get(Interview, interview_id) if interview_id else None
                if ready is not None:
                    _publish_generation_event(own, job.id, run_id, {"type": "done"})
        finally:
            own.close()
        await asyncio.sleep(1)
        yield ":\n\n"


def _publish_committed_generation(db: Session, job: DurableJob) -> None:
    """Close the SSE gap after a committed terminal job state."""
    run = db.get(StreamRun, job.business_key)
    if run is None or run.status != "running":
        return
    _publish_generation_event(db, job.id, run.id,
                              {"type": "done" if job.status == "succeeded" else "error"})


async def _run_generation_queue() -> None:
    """Recover accepted generation jobs from their interview-origin business row."""
    while True:
        try:
            with SessionLocal() as db:
                # A worker can die after this process starts; reclaim its expired
                # lease on every poll so its committed input can be regenerated.
                reclaim_expired_jobs(db, kinds=["interview_generation"])
                db.commit()
                completed = [row.id for row in db.query(DurableJob.id).join(
                    StreamRun, StreamRun.id == DurableJob.business_key,
                ).filter(
                    DurableJob.kind == "interview_generation",
                    DurableJob.status.in_(["succeeded", "dead_letter"]),
                    StreamRun.status == "running",
                ).limit(8).all()]
                due = [row.id for row in db.query(DurableJob.id).filter(
                    DurableJob.kind == "interview_generation",
                    DurableJob.status.in_(["pending", "retry_wait"]),
                    DurableJob.next_run_at <= datetime.now(timezone.utc),
                ).order_by(DurableJob.created_at).limit(8).all()]
            for job_id in completed:
                with SessionLocal() as db:
                    job = db.get(DurableJob, job_id)
                    if job is not None:
                        _publish_committed_generation(db, job)
            for job_id in due:
                worker_id = uuid.uuid4().hex
                with SessionLocal() as db:
                    job = claim_job_by_id(db, job_id, worker_id)
                    db.commit()
                    if job is None:
                        continue
                    staged = db.get(ChatSession, job.business_key)
                    message = next((row for row in staged.messages if row.role == "user"), None) if staged else None
                    run = db.get(StreamRun, job.business_key)
                    content = message.content if message else ""
                    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    if not content or run is None or run.business_id != digest or (job.payload or {}).get("input_digest") != digest:
                        fail_job(db, job.id, "generation_input_missing", retryable=False, worker_id=worker_id)
                        db.commit()
                        if run is not None and run.status == "running":
                            _publish_generation_event(db, job.id, run.id, {"type": "error"})
                        continue
                    async for _ in _generate_events(db, content, _generation_checkpointer,
                                                    run.id, job.id, worker_id):
                        pass
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("interview generation recovery worker failed")
        await asyncio.sleep(2)


def start_generation_worker(checkpointer=None) -> None:
    """Start job recovery after the graph saver has entered its lifespan."""
    global _generation_worker, _generation_checkpointer
    _generation_checkpointer = checkpointer
    if _generation_worker is None or _generation_worker.done():
        _generation_worker = asyncio.create_task(_run_generation_queue())


def generation_worker_running() -> bool:
    """Expose recovery-poller liveness without inspecting active request tasks."""
    return _generation_worker is not None and not _generation_worker.done()


async def stop_generation_worker() -> None:
    """Drain active generation commits before closing the graph saver."""
    global _generation_worker, _generation_checkpointer
    worker = _generation_worker
    _generation_worker = None
    if worker is not None:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
    if _active_generation_tasks:
        await asyncio.gather(*tuple(_active_generation_tasks), return_exceptions=True)
    _generation_checkpointer = None


@router.get("/api/interviews", response_model=list[schemas.InterviewOut])
def list_interviews(db: Session = Depends(get_db)) -> list[schemas.InterviewOut]:
    return [interview_out(item) for item in services.list_interviews(db)]


@router.post("/api/interviews/generate/stream")
async def generate_interview(payload: schemas.InterviewGenerateIn, request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    """面试页自己的生成流。思考原文只在当次连接里推，不写入面试记录。"""
    if len(payload.content.strip()) < 8:
        raise HTTPException(400, "请先写下岗位描述")
    key = request.headers.get("Idempotency-Key")
    if key is not None and (not key.strip() or len(key) > 200):
        raise HTTPException(400, "无效的请求标识")
    digest = hashlib.sha256(payload.content.strip().encode("utf-8")).hexdigest()
    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"interview-generation:{ANON}:{key}")) if key else None
    try:
        run = create_stream_run(db, "interview_generation", ANON, run_id=run_id,
                                business_id=digest, commit=False)
    except PermissionError as exc:
        raise HTTPException(409, "请求标识与岗位描述不匹配") from exc
    staged = db.get(ChatSession, run.id)
    if staged is None:
        # The original job text is a business fact, committed with the job.
        staged = ChatSession(id=run.id, title="新会话", user_id=ANON, origin="interview")
        db.add(staged)
        db.add(ChatMessage(id=uuid.uuid4().hex, session_id=run.id,
                           role="user", content=payload.content.strip()))
    elif staged.origin != "interview" or not any(
        message.role == "user" and message.content == payload.content.strip()
        for message in staged.messages
    ):
        raise HTTPException(409, "请求标识与岗位描述不匹配")
    if run.status == "completed":
        # A retry after a lost final frame reads the exact committed sequence.
        frames = (event_frame(event) for event in replay_stream_events(db, run.id, ANON))
        return StreamingResponse(iter(frames), media_type="text/event-stream",
                                 headers={**_stream_headers(), "X-Run-ID": run.id})
    job = create_job(
        db, "interview_generation", run.id, payload={"input_digest": digest},
        idempotency_key=f"generation:{run.id}", max_attempts=3,
        deadline_at=datetime.now(timezone.utc) + timedelta(hours=1), trace_id=run.id,
    )
    db.commit()
    worker_id = uuid.uuid4().hex
    claimed = claim_job_by_id(db, job.id, worker_id)
    db.commit()
    if claimed is None:
        db.refresh(run)
        db.refresh(job)
        if run.status == "running" and job.status in {"succeeded", "dead_letter"}:
            _publish_committed_generation(db, job)
            db.refresh(run)
        events = replay_stream_events(db, run.id, ANON)
        if run.status == "completed":
            frames = (event_frame(event) for event in events)
        else:
            cursor = events[-1].event_id if events else None
            async def replay_then_follow():
                # A same-key retry starts with committed frames, then follows
                # subsequent events without duplicating their sequence IDs.
                for event in events:
                    yield event_frame(event)
                async for frame in _follow_generation_events(run.id, cursor):
                    yield frame
            frames = replay_then_follow()
        return StreamingResponse(frames, media_type="text/event-stream",
                                 headers={**_stream_headers(), "X-Run-ID": run.id})
    return StreamingResponse(
        _generate_events(db, payload.content, _request_checkpointer(request), run.id, job.id, worker_id),
        media_type="text/event-stream", headers={**_stream_headers(), "X-Run-ID": run.id},
    )


@router.post("/api/interviews", response_model=schemas.InterviewDetail)
async def create_interview(payload: schemas.InterviewCreateIn, request: Request, db: Session = Depends(get_db)) -> schemas.InterviewDetail:
    key = request.headers.get("Idempotency-Key")
    if key is not None:
        try:
            replay = services.find_interview_create_replay(db, key, payload.session_id, payload.question_set_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if replay is not None:
            return interview_detail(replay)
    stages = LiveInterviewCommandStages(
        db, "create", session_id=payload.session_id, question_set_id=payload.question_set_id,
    )
    try:
        async def start():
            return services.start_interview(db, payload.session_id, payload.question_set_id,
                                            idempotency_key=key)

        interview = await run_business_graph(
            db, mode="live_interview", action=start,
            session_id=payload.session_id,
            checkpointer=_request_checkpointer(request),
            stages=stages.callbacks(), stage_sequences=stages.sequences(),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return interview_detail(interview)


@router.get("/api/interviews/{interview_id}", response_model=schemas.InterviewDetail)
def get_interview(interview_id: str, db: Session = Depends(get_db)) -> schemas.InterviewDetail:
    interview = services.get_interview(db, interview_id)
    if not interview:
        raise HTTPException(404, "interview not found")
    return interview_detail(interview)


@router.post("/api/interviews/{interview_id}/start", response_model=schemas.InterviewDetail)
async def start_existing_interview(interview_id: str, request: Request, db: Session = Depends(get_db)) -> schemas.InterviewDetail:
    stages = LiveInterviewCommandStages(db, "resume", interview_id=interview_id)
    try:
        async def start():
            return services.resume_or_start(db, interview_id)

        interview = await run_business_graph(
            db, mode="live_interview", action=start, interview_id=interview_id,
            checkpointer=_request_checkpointer(request),
            stages=stages.callbacks(), stage_sequences=stages.sequences(),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return interview_detail(interview)


@router.post("/api/interviews/{interview_id}/answer", response_model=schemas.InterviewDetail)
async def answer_interview(
    interview_id: str, payload: schemas.InterviewAnswerIn,
    db: Session = Depends(get_db), request: Request = None,
) -> schemas.InterviewDetail:
    # Bind the graph run and durable answer turn to the same client request.
    # UUID5 keeps the database key within InterviewTurn.id's 36-character limit.
    key = request.headers.get("Idempotency-Key") if request is not None else None
    if key is not None and (not key.strip() or len(key) > 200):
        raise HTTPException(400, "无效的请求标识")
    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"interview-answer:{interview_id}:{key}")) if key else str(uuid.uuid4())
    if key:
        turn = db.get(InterviewTurn, run_id)
        if turn is not None:
            if turn.interview_id != interview_id or turn.role != "user" or turn.content != payload.content or turn.answer_mode != payload.answer_mode:
                raise HTTPException(400, "同一请求标识对应另一条回答")
            if turn.cite != "answer_pending":
                # The business commit can succeed before the graph checkpoint is
                # written; the saved answer is authoritative on a replay.
                interview = services.get_interview(db, interview_id)
                if interview is not None:
                    return interview_detail(interview)
        db.rollback()
    try:
        # A reload can lose the original request key while the answer is
        # already durable. Resume that exact graph run and question progress.
        pending_run_id = services.pending_answer_run_id(db, interview_id, payload.content, payload.answer_mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if pending_run_id is not None:
        run_id = pending_run_id
    try:
        interview = await run_live_answer_graph(
            db, interview_id, payload.content, payload.answer_mode, run_id,
            checkpointer=_request_checkpointer(request),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        # Answer persistence and response serialization share one request. Roll back
        # any failed transaction so the pooled connection remains usable, then expose
        # a retryable status instead of leaking an unhandled 500 to the candidate.
        db.rollback()
        logger.exception("failed to submit interview answer: interview_id=%s", interview_id)
        raise HTTPException(503, "回答暂时未提交，请稍后重试") from exc
    return interview_detail(interview)


@router.post("/api/interviews/{interview_id}/end", response_model=schemas.InterviewDetail)
async def end_interview(interview_id: str, request: Request, db: Session = Depends(get_db)) -> schemas.InterviewDetail:
    """先停表返回。复盘用另一条数据库会话在后台写，不占着这次请求。"""
    stages = LiveInterviewCommandStages(db, "end", interview_id=interview_id)
    try:
        async def finish_and_schedule():
            interview = await services.finish_interview(db, interview_id, enqueue_report=True)
            if interview.report is None:
                # finish_interview committed the outbox with the status change;
                # this process-local action only reduces worker pickup latency.
                services.start_report_worker()
            return interview

        interview = await run_business_graph(
            db, mode="live_interview", interview_id=interview_id,
            checkpointer=_request_checkpointer(request),
            stages=stages.callbacks(), stage_sequences=stages.sequences(),
            action=finish_and_schedule,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return interview_detail(interview)


@router.post("/api/interviews/{interview_id}/abandon", response_model=schemas.InterviewDetail)
async def abandon_interview(interview_id: str, request: Request, db: Session = Depends(get_db)) -> schemas.InterviewDetail:
    """Direct exit returns the attempt to ready; no recap job is queued."""
    stages = LiveInterviewCommandStages(db, "abandon", interview_id=interview_id)
    try:
        async def abandon():
            return services.abandon_interview(db, interview_id)

        interview = await run_business_graph(
            db, mode="live_interview", interview_id=interview_id,
            checkpointer=_request_checkpointer(request), action=abandon,
            stages=stages.callbacks(), stage_sequences=stages.sequences(),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return interview_detail(interview)


@router.post("/api/interviews/{interview_id}/report/regenerate", response_model=schemas.InterviewDetail)
async def regenerate_interview_report(interview_id: str, request: Request, db: Session = Depends(get_db)) -> schemas.InterviewDetail:
    """Temporary QA action: replace the current report and enqueue a fresh run."""
    stages = ReportRegenerationCommandStages(db, interview_id)
    try:
        async def regenerate():
            interview = services.regenerate_report(db, interview_id)
            # regenerate_report commits deletion and its outbox row together;
            # this call only wakes the process-local poller after that boundary.
            services.start_report_worker()
            return interview

        interview = await run_business_graph(
            db, mode="evaluation_report", interview_id=interview_id,
            checkpointer=_request_checkpointer(request), action=regenerate,
            stages=stages.callbacks(), stage_sequences=stages.sequences(),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return interview_detail(interview)


@router.delete("/api/interviews/{interview_id}")
def delete_interview(interview_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        services.delete_interview(db, interview_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": "deleted"}


@router.get("/api/interviews/{interview_id}/report.txt")
def interview_report_text(interview_id: str, db: Session = Depends(get_db)) -> PlainTextResponse:
    interview = services.get_interview(db, interview_id)
    if not interview or not interview.report:
        raise HTTPException(404, "report not found")
    body = report_text(interview)
    filename = f"interview-{interview_id[:8]}.txt"
    return PlainTextResponse(
        body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
