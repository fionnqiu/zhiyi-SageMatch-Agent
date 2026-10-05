"""Session cockpit routes: history, chat, and JD file ingest."""

from __future__ import annotations

import json
import asyncio
import logging
import time
from uuid import uuid4
from collections.abc import AsyncIterator
from contextlib import nullcontext
from types import SimpleNamespace

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app import schemas, services
from app.integrations import llm
from app.core.config import get_settings
from app.api.shared.deps import session_detail
from app.core.db import SessionLocal, get_db
from app.models.platform.stream import StreamEvent, StreamRun
from app.services.shared.common import ANON
from app.services.chat.stream_events import (
    append_stream_event, claim_stream_run, create_stream_run, event_frame,
    expire_stream_lease, renew_stream_lease, replay_stream_events, set_stream_recovery,
)
from app.agents.orchestration.checkpoint import CheckpointRunActive
from app.agents.orchestration.workflows import run_business_graph

router = APIRouter()
logger = logging.getLogger(__name__)
_chat_recovery_task: asyncio.Task | None = None


@router.get("/api/sessions", response_model=list[schemas.ChatSessionOut])
def list_sessions(db: Session = Depends(get_db)) -> list[schemas.ChatSessionOut]:
    return [schemas.ChatSessionOut.model_validate(s, from_attributes=True) for s in services.list_sessions(db)]


@router.post("/api/sessions", response_model=schemas.ChatSessionDetail)
def create_session(db: Session = Depends(get_db)) -> schemas.ChatSessionDetail:
    session = services.create_session(db)
    return session_detail(session)


@router.get("/api/sessions/{session_id}", response_model=schemas.ChatSessionDetail)
def get_session(session_id: str, db: Session = Depends(get_db)) -> schemas.ChatSessionDetail:
    session = services.get_session(db, session_id)
    if not session:
        raise HTTPException(404, "session not found")
    return session_detail(session)


@router.post("/api/sessions/{session_id}/clear", response_model=schemas.ChatSessionDetail)
def clear_session(session_id: str, request: Request, db: Session = Depends(get_db)) -> schemas.ChatSessionDetail:
    try:
        session = services.clear_session(db, session_id,
                                         checkpointer=getattr(request.app.state, "checkpointer", None))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return session_detail(session)


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, request: Request, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        services.delete_session(db, session_id,
                                checkpointer=getattr(request.app.state, "checkpointer", None))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": "deleted"}


@router.post("/api/chat", response_model=schemas.ChatSessionDetail)
async def chat(payload: schemas.ChatSendIn, request: Request, db: Session = Depends(get_db)) -> schemas.ChatSessionDetail:
    answers = [item.model_dump() for item in payload.answers]
    attachments = [item.model_dump() for item in payload.attachments]
    if not payload.content.strip() and not answers and not attachments:
        raise HTTPException(400, "请输入内容")
    if len(payload.content) > 4000:
        # The graph bounds visible input at 4000 characters; reject before
        # begin_chat commits the user's text so the stored fact matches it.
        raise HTTPException(400, "输入内容不能超过 4000 字")
    try:
        session = await services.send_chat_graph(
            db, payload.content, payload.session_id, answers or None, attachments or None,
            checkpointer=getattr(request.app.state, "checkpointer", None),
        )
    except ValueError as exc:
        if str(exc) == "session not found":
            raise HTTPException(404, "session not found") from exc
        raise
    return session_detail(session)


@router.post("/api/chat/stream")
async def chat_stream(payload: schemas.ChatSendIn, request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    """知识回答和追问走这条流。澄清整段返回，出题请求只回 redirect。"""
    answers = [item.model_dump() for item in payload.answers]
    attachments = [item.model_dump() for item in payload.attachments]
    if not payload.content.strip() and not answers and not attachments:
        raise HTTPException(400, "请输入内容")
    if len(payload.content) > 4000:
        raise HTTPException(400, "输入内容不能超过 4000 字")
    deadline_at = time.monotonic() + 300.0
    settings = get_settings()
    with llm.usage_budget_scope(
        token_budget=settings.graph_token_budget, cost_budget=settings.graph_cost_budget,
    ) as usage_ledger, llm.deadline_scope(deadline_at):
        try:
            routed = await services.route_chat(
                db, payload.content, payload.session_id, answers or None, attachments or None
            )
        except ValueError as exc:
            if str(exc) == "session not found":
                raise HTTPException(404, "session not found") from exc
            raise
        mode = routed["mode"]
        create_stream_run(db, "chat", ANON, run_id=routed["run_id"], business_id=payload.session_id)
        if mode == "redirect":
            # Navigation requests do not create a conversation turn.
            return StreamingResponse(_redirect_events(routed["run_id"], db), media_type="text/event-stream", headers=_stream_headers(routed["run_id"]))
        if mode != "stream":
            return StreamingResponse(_blocked_events(routed["run_id"], db), media_type="text/event-stream", headers=_stream_headers(routed["run_id"]))
        turn = await services.begin_chat(
            db, payload.content, payload.session_id, answers or None, attachments or None,
            precomputed_intent=routed["intent"], run_id=routed["run_id"],
            defer_retrieval=True,
        )
    if turn["mode"] == "redirect":
        # Intent can change after the request was staged; remove that transient user row.
        services.discard_redirect_turn(db, turn)
        return StreamingResponse(_redirect_events(turn["run_id"], db), media_type="text/event-stream", headers=_stream_headers(turn["run_id"]))
    if turn["mode"] not in {"answer", "followup"}:
        # 澄清流不支持增量正文，沿用整段收尾。
        worker_id = _save_chat_recovery(db, turn, attachments)
        return StreamingResponse(_detached_turn_events(turn, fallback=True, checkpointer=getattr(request.app.state, "checkpointer", None), deadline_at=deadline_at, usage_ledger=usage_ledger, claimed_owner=worker_id), media_type="text/event-stream", headers=_stream_headers(turn["run_id"]))
    worker_id = _save_chat_recovery(db, turn, attachments)
    return StreamingResponse(_detached_turn_events(turn, checkpointer=getattr(request.app.state, "checkpointer", None), deadline_at=deadline_at, usage_ledger=usage_ledger, claimed_owner=worker_id), media_type="text/event-stream", headers=_stream_headers(turn["run_id"]))


def _save_chat_recovery(db: Session, turn: dict, attachments: list[dict]) -> str:
    """Persist only text turn inputs and the chosen route; never persist file text."""
    recovery = {"recoverable": False} if attachments else {
        "recoverable": True,
        "session_id": turn["session"].id,
        "user_message_id": turn["user_message"].id if turn.get("user_message") else None,
        "content": turn["content"],
        "intent": {key: turn["intent"].get(key) for key in ("intent", "needs_recall", "todos", "source")},
        "route": turn.get("route") or {},
        "mode": turn["mode"],
        "first_turn": turn["first_turn"],
        "opening": turn["opening"],
    }
    worker_id = f"chat-{uuid4().hex}"
    set_stream_recovery(db, turn["run_id"], ANON, recovery, worker_id=worker_id)
    return worker_id


@router.get("/api/streams/{run_id}/events")
async def replay_events(run_id: str, request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    """Replay committed frames, then follow a still-running stream."""
    after_id = request.headers.get("Last-Event-ID") or request.query_params.get("after_event_id")
    try:
        replay_stream_events(db, run_id, ANON, after_id)
        run = db.get(StreamRun, run_id)
        if run is not None and run.kind == "chat" and run.business_id and services.get_session(db, run.business_id) is None:
            raise PermissionError("stream business owner mismatch")
    except PermissionError as exc:
        raise HTTPException(404, "stream not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        # The dependency session was only needed for authorization.  Do not
        # hold its read transaction across the long-lived response iterator.
        db.rollback()

    async def follow() -> AsyncIterator[str]:
        cursor = after_id
        while True:
            with SessionLocal() as poll_db:
                events = replay_stream_events(poll_db, run_id, ANON, cursor)
                row = poll_db.get(StreamRun, run_id)
                running = row is not None and row.status == "running"
                frames = [(event.event_id, event_frame(event)) for event in events]
            for event_id, frame in frames:
                cursor = event_id
                yield frame
            if not running or await request.is_disconnected():
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(follow(), media_type="text/event-stream", headers=_stream_headers(run_id))


async def recover_pending_chat_streams(checkpointer=None) -> None:
    """Claim interrupted streams, preserving route decisions and committed frames."""
    with SessionLocal() as db:
        run_ids = [row.id for row in db.query(StreamRun).filter(
            StreamRun.kind == "chat", StreamRun.status == "running",
        ).all()]
    for run_id in run_ids:
        worker_id = f"chat-recovery-{uuid4().hex}"
        with SessionLocal() as db:
            if not claim_stream_run(db, run_id, worker_id):
                continue
            run = db.get(StreamRun, run_id)
            recovery = dict(run.recovery or {}) if run is not None else {}
            if not recovery.get("recoverable"):
                append_stream_event(db, run_id, ANON, {
                    "type": "error", "error_code": "stream_resume_unavailable",
                    "message": "本次回答已中断，请重新发送消息。", "retryable": True,
                }, worker_id=worker_id)
                continue
            # A business commit can precede its terminal SSE frame. Replaying
            # that committed fact remains valid even if the user has since sent
            # another message; it must not run generation a second time.
            saved_session_id = recovery.get("session_id")
            saved_session = services.get_session(db, saved_session_id) if isinstance(saved_session_id, str) else None
            if saved_session is not None and saved_session.user_id == ANON and any(
                item.role == "assistant" and isinstance(item.extra, dict) and item.extra.get("run_id") == run_id
                for item in saved_session.messages
            ):
                append_stream_event(db, run_id, ANON, {
                    "type": "done", "session": session_detail(saved_session).model_dump(mode="json"),
                }, worker_id=worker_id)
                continue
            turn = services.recover_chat_turn(db, recovery, run_id)
            if turn is None:
                append_stream_event(db, run_id, ANON, {
                    "type": "error", "error_code": "stream_resume_conflict",
                    "message": "会话状态已变化，请重新发送消息。", "retryable": True,
                }, worker_id=worker_id)
                continue
            had_delta = db.query(StreamEvent).filter(
                StreamEvent.run_id == run_id, StreamEvent.event_type == "delta",
            ).first() is not None
            if had_delta:
                append_stream_event(db, run_id, ANON, {"type": "reset"}, terminal=False, worker_id=worker_id)
            # The worker reloads the session in its own transaction; retain
            # only its stable ID after this setup transaction closes.
            turn["session"] = SimpleNamespace(id=turn["session"].id)
        try:
            async for _frame in _detached_turn_events(
                turn, fallback=turn["mode"] == "clarify", checkpointer=checkpointer,
                deadline_at=time.monotonic() + 300, claimed_owner=worker_id,
            ):
                pass
        except CheckpointRunActive:
            # The original graph still owns its checkpoint. Retry after it
            # finishes instead of publishing a false terminal SSE failure.
            with SessionLocal() as db:
                expire_stream_lease(db, run_id, worker_id)
            continue
        with SessionLocal() as db:
            row = db.get(StreamRun, run_id)
            if row is not None and row.status == "running":
                append_stream_event(db, run_id, ANON, {
                    "type": "error", "error_code": "stream_resume_incomplete",
                    "message": "回答恢复未完成，请重新发送消息。", "retryable": True,
                }, worker_id=worker_id)


def start_chat_recovery_worker(checkpointer=None) -> None:
    """Poll leases so recovery also works after a worker dies between boots."""
    global _chat_recovery_task
    if _chat_recovery_task is not None and not _chat_recovery_task.done():
        return

    async def poll() -> None:
        while True:
            try:
                await recover_pending_chat_streams(checkpointer)
            except Exception:
                # A transient database error must not permanently disable recovery.
                logger.exception("chat stream recovery poll failed")
            await asyncio.sleep(5)

    _chat_recovery_task = asyncio.create_task(poll())


def chat_worker_running() -> bool:
    """Expose the poller's actual process state to readiness checks."""
    return _chat_recovery_task is not None and not _chat_recovery_task.done()


async def stop_chat_recovery_worker() -> None:
    """Stop lease polling during graceful shutdown."""
    global _chat_recovery_task
    task = _chat_recovery_task
    _chat_recovery_task = None
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _blocked_events(run_id: str = "", db: Session | None = None) -> AsyncIterator[str]:
    yield event_frame(append_stream_event(db, run_id, ANON, {"type": "blocked"})) if db else _sse({"type": "blocked", "run_id": run_id}, event_id=f"{run_id}:1" if run_id else None)


async def _redirect_events(run_id: str = "", db: Session | None = None) -> AsyncIterator[str]:
    yield event_frame(append_stream_event(db, run_id, ANON, {"type": "redirect"})) if db else _sse({"type": "redirect", "run_id": run_id}, event_id=f"{run_id}:1" if run_id else None)


async def _fallback_events(db: Session, turn: dict, worker_id: str) -> AsyncIterator[str]:
    """预判能流、真正开始时却变成澄清或改去面试页。沿用已写入的用户消息收尾。"""
    reply, extra = await services.complete_chat_turn(db, turn)
    session = await services.finish_chat(db, turn, reply, extra, worker_id=worker_id)
    yield event_frame(append_stream_event(db, turn["run_id"], ANON, {"type": "done", "session": session_detail(session).model_dump(mode="json")}, worker_id=worker_id))


async def _detached_turn_events(turn: dict, *, fallback: bool = False, checkpointer=None, deadline_at: float | None = None, usage_ledger=None, claimed_owner: str | None = None) -> AsyncIterator[str]:
    """Let generation finish after a client disconnects, using its own DB session."""
    queue: asyncio.Queue[str | CheckpointRunActive | None] = asyncio.Queue()
    connected = True
    worker_id = claimed_owner or f"chat-{uuid4().hex}"

    async def _produce() -> None:
        own = SessionLocal()
        lease_task: asyncio.Task | None = None
        producer_task = asyncio.current_task()
        try:
            if claimed_owner is None:
                row = own.get(StreamRun, turn["run_id"])
                if row is not None and row.recovery is not None and not claim_stream_run(own, turn["run_id"], worker_id):
                    return
            async def keep_lease() -> None:
                # A separate transaction keeps a slow provider call claimable
                # only after this process has actually stopped.
                while True:
                    await asyncio.sleep(10)
                    with SessionLocal() as heartbeat_db:
                        if not renew_stream_lease(heartbeat_db, turn["run_id"], worker_id):
                            # A successor owns the stream now; interrupt this
                            # worker before it can commit more business work.
                            if producer_task is not None:
                                producer_task.cancel()
                            return
            lease_task = asyncio.create_task(keep_lease())
            # begin_chat committed the user turn. Reload the ORM session into
            # this worker's transaction before later messages are written.
            worker_turn = {**turn, "session": services.get_session(own, turn["session"].id)}
            async def emit(source: AsyncIterator[str]) -> None:
                async for frame in source:
                    if connected:
                        queue.put_nowait(frame)

            remaining = max(0.1, (deadline_at or time.monotonic() + 300.0) - time.monotonic())
            if fallback:
                # A route changed into clarification before the stream began;
                # this waiting branch has no knowledge subgraph to commit.
                with llm.deadline_scope(deadline_at):
                    await asyncio.wait_for(emit(_fallback_events(own, worker_turn, worker_id)), timeout=remaining)
            else:
                # Retrieval and citation work now occupy the same graph stages
                # used by whole-response chat. Only the final commit writes the
                # assistant fact and terminal frame.
                prepared: dict = {}
                stages = services.staged_chat_retrieval_callbacks(own, worker_turn)

                async def answer(_state) -> dict:
                    await emit(_answer_events(own, worker_turn, prepared, worker_id))
                    return {"result": {"valid": True, "answer_ready": True}}

                async def citation_validate(_state) -> dict:
                    prepared["citation_validation"] = services.validate_stream_answer(
                        worker_turn, prepared["reply"], prepared["extra"],
                    )
                    return {"diagnostics": {"citation_valid": (prepared["citation_validation"] or {}).get("valid")}}

                async def repair_once(_state) -> dict:
                    streamed_reply = prepared["reply"]
                    prepared["reply"], prepared["extra"] = await services.repair_stream_answer(
                        own, worker_turn, prepared["reply"], prepared["extra"],
                        prepared.get("citation_validation"),
                    )
                    if prepared["reply"] != streamed_reply:
                        # Citation repair runs after provider streaming. Replay the
                        # corrected answer through the same transient channel so
                        # the visible bubble and the durable `done` payload agree.
                        async def correction() -> AsyncIterator[str]:
                            yield _sse({"type": "reset", "run_id": worker_turn["run_id"]})
                            yield _sse({"type": "delta", "text": prepared["reply"], "run_id": worker_turn["run_id"]})
                        await emit(correction())
                    return {"result": {"valid": True, "answer_ready": True}}

                async def commit() -> bool:
                    session = await services.finish_chat(
                        own, worker_turn, prepared["reply"], prepared["extra"],
                        citation_checked=True, worker_id=worker_id,
                    )
                    await emit(_done_event(own, worker_turn["run_id"], session, worker_id))
                    return True

                await asyncio.wait_for(
                    run_business_graph(
                        own,
                        mode="knowledge_qa",
                        action=commit,
                        original_query=turn.get("content") or "",
                        session_id=turn["session"].id,
                        checkpointer=checkpointer,
                        run_id=turn["run_id"],
                        deadline_at=deadline_at,
                        restart_incomplete=bool(claimed_owner and claimed_owner.startswith("chat-recovery-")),
                        stages={
                            **stages,
                            "knowledge_qa.answer": answer,
                            "knowledge_qa.citation_validate": citation_validate,
                            "knowledge_qa.repair_once": repair_once,
                        },
                    ),
                    timeout=remaining,
                )
        except CheckpointRunActive as exc:
            own.rollback()
            if connected:
                queue.put_nowait(exc)
        except Exception:
            own.rollback()
            logger.exception("chat stream worker failed: run_id=%s", turn["run_id"])
            # The terminal frame is committed by the worker even if the HTTP
            # consumer has vanished; reconnects can see an honest failure.
            try:
                event = append_stream_event(own, turn["run_id"], ANON, {"type": "error", "message": "回答生成失败，请重试"}, worker_id=worker_id)
                if connected:
                    queue.put_nowait(event_frame(event))
            except Exception:
                own.rollback()
        finally:
            if lease_task is not None:
                lease_task.cancel()
                try:
                    await lease_task
                except asyncio.CancelledError:
                    pass
            own.close()
            if connected:
                queue.put_nowait(None)

    async def produce() -> None:
        # StreamingResponse begins after the route handler exits, so explicitly
        # rebind its measured classifier ledger to the detached graph worker.
        with llm.bind_usage_ledger(usage_ledger) if usage_ledger is not None else nullcontext():
            await _produce()

    task = asyncio.create_task(produce())
    try:
        while True:
            frame = await queue.get()
            if frame is None:
                break
            if isinstance(frame, CheckpointRunActive):
                raise frame
            yield frame
    finally:
        connected = False
        # The worker owns its deadline and transaction. Retain its task across
        # cancellation of the response iterator so the run reaches a terminal event.
        task.add_done_callback(lambda completed: completed.exception() if not completed.cancelled() else None)


async def _done_event(db: Session, run_id: str, session, worker_id: str) -> AsyncIterator[str]:
    """Publish the committed session only after graph citation stages finish."""
    yield event_frame(append_stream_event(
        db, run_id, ANON, {"type": "done", "session": session_detail(session).model_dump(mode="json")}, worker_id=worker_id,
    ))


async def _answer_events(db: Session, turn: dict, prepared: dict, worker_id: str) -> AsyncIterator[str]:
    """Stream a candidate while retaining its complete text for graph validation."""
    extra = {**(turn.get("extra") or {}), "intent": turn["intent"]["intent"]}
    run_id = turn["run_id"]
    def frame(payload: dict) -> str:
        if payload["type"] in {"thinking", "reasoning"}:
            # Provider reasoning is ephemeral and must never enter replay storage.
            return _sse({**payload, "run_id": run_id})
        return event_frame(append_stream_event(db, run_id, ANON, payload, worker_id=worker_id))

    yield frame({"type": "meta", "session_id": turn["session"].id, "extra": extra})
    answer: list[str] = []
    async for kind, delta in services.iter_turn_parts(db, turn):
        if kind == "thinking":
            yield frame({"type": "thinking", "text": delta})
            continue
        if kind == "reasoning":
            yield frame({"type": "reasoning", "text": delta})
            continue
        answer.append(delta)
        yield frame({"type": "delta", "text": delta})
    # Private provider traces are transient; replay only the public answer.
    prepared.update(reply="".join(answer).strip(), extra=extra)


def _stream_headers(run_id: str | None = None) -> dict[str, str]:
    # Keep SSE unbuffered; the header lets clients replay before the first frame.
    headers = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}
    if run_id:
        headers["X-Run-ID"] = run_id
    return headers


def _sse(payload: dict, *, event_id: str | None = None) -> str:
    # 中文按原文写进 data。前端按行解析，不依赖浏览器的 EventSource 自动重连。
    prefix = f"id: {event_id}\n" if event_id else ""
    return f"{prefix}data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/api/chat/prepare")
async def chat_prepare(file: UploadFile = File(...)) -> dict[str, str | int]:
    """只抽出文本，不建消息。文件要留在输入框里，等用户和文字一起发送。"""
    data = await file.read()
    try:
        text = services.prepare_chat_file(file.filename or "upload.txt", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"name": file.filename or "upload.txt", "size": len(data), "text": text}
