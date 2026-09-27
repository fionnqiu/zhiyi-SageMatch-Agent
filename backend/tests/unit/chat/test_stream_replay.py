"""Durable SSE identity, owner isolation, and privacy contracts."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.db import Base
from app.models.platform.stream import StreamEvent, StreamRun
from app.services.chat.stream_events import (
    append_stream_event, claim_stream_run, create_stream_run, event_frame,
    replay_stream_events, set_stream_recovery,
)


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[StreamRun.__table__, StreamEvent.__table__])
    return Session(engine)


def test_replay_returns_only_later_committed_frames_after_reconnect():
    db = _db()
    create_stream_run(db, "chat", "alice", run_id="run-1", business_id="session-1")
    first = append_stream_event(db, "run-1", "alice", {"type": "meta", "session_id": "session-1"})
    append_stream_event(db, "run-1", "alice", {"type": "delta", "text": "你好"})
    append_stream_event(db, "run-1", "alice", {"type": "done", "session": {"extra": {"reasoning": "secret"}}})
    db.expire_all()  # Replay must load committed rows, not the caller's identity map.
    replayed = replay_stream_events(db, "run-1", "alice", first.event_id)
    assert [event.seq for event in replayed] == [2, 3]
    assert "id: run-1:2\n" in event_frame(replayed[0])
    assert "你好" in event_frame(replayed[0])
    assert "secret" not in event_frame(replayed[1])


def test_owner_and_event_id_must_match_run():
    db = _db()
    create_stream_run(db, "interview_generation", "alice", run_id="run-2")
    append_stream_event(db, "run-2", "alice", {"type": "done", "interview": {"id": "interview-1"}})
    for action in (
        lambda: replay_stream_events(db, "run-2", "bob"),
        lambda: replay_stream_events(db, "run-2", "alice", "other-run:1"),
        lambda: append_stream_event(db, "run-2", "bob", {"type": "delta", "text": "x"}),
    ):
        try:
            action()
        except (PermissionError, ValueError):
            pass
        else:
            raise AssertionError("stream scope was not enforced")


def test_reasoning_events_cannot_be_persisted():
    db = _db()
    create_stream_run(db, "chat", "alice", run_id="run-3")
    try:
        append_stream_event(db, "run-3", "alice", {"type": "reasoning", "text": "private"})
    except ValueError:
        pass
    else:
        raise AssertionError("private event was accepted")
    assert replay_stream_events(db, "run-3", "alice") == []


def test_recovery_lease_prevents_a_second_worker_until_expired():
    from datetime import datetime, timedelta, timezone

    db = _db()
    create_stream_run(db, "chat", "alice", run_id="run-lease")
    set_stream_recovery(db, "run-lease", "alice", {"recoverable": True}, worker_id="worker-1")
    assert not claim_stream_run(db, "run-lease", "worker-2")
    row = db.get(StreamRun, "run-lease")
    row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
    db.commit()
    assert claim_stream_run(db, "run-lease", "worker-2")
    assert db.get(StreamRun, "run-lease").lease_owner == "worker-2"


def test_active_checkpoint_does_not_terminalize_reclaimed_stream():
    """A graph lock conflict means another worker may still commit this run."""
    from app.agents.orchestration.checkpoint import CheckpointRunActive
    from app.api.chat.session import recover_pending_chat_streams
    from datetime import datetime, timedelta, timezone

    db = _db()
    create_stream_run(db, "chat", "local-user", run_id="active-graph")
    set_stream_recovery(db, "active-graph", "local-user", {
        "recoverable": True, "session_id": "session-1", "mode": "answer",
    }, worker_id="old-worker")
    db.get(StreamRun, "active-graph").lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
    db.commit()

    async def conflicting_worker(*_args, **_kwargs):
        raise CheckpointRunActive("checkpoint run already active")
        yield ""  # Keep the fake an async generator like the real worker.

    saved_session = SimpleNamespace(id="session-1", user_id="local-user", messages=[])
    turn = {"run_id": "active-graph", "session": saved_session, "mode": "answer"}
    with patch("app.api.chat.session.SessionLocal", return_value=db), patch(
        "app.api.chat.session.services.get_session", return_value=saved_session,
    ), patch("app.api.chat.session.services.recover_chat_turn", return_value=turn), patch(
        "app.api.chat.session._detached_turn_events", new=conflicting_worker,
    ):
        asyncio.run(recover_pending_chat_streams())

    row = db.get(StreamRun, "active-graph")
    assert row.status == "running"
    assert row.lease_until.replace(tzinfo=timezone.utc) <= datetime.now(timezone.utc)
    assert replay_stream_events(db, "active-graph", "local-user") == []


def test_attachment_stream_recovery_never_stores_extracted_text():
    from app.api.chat.session import _save_chat_recovery

    db = _db()
    create_stream_run(db, "chat", "local-user", run_id="run-file")
    private_text = "confidential extracted file content"
    _save_chat_recovery(db, {
        "run_id": "run-file", "session": SimpleNamespace(id="session-1"),
        "content": private_text, "intent": {"intent": "answer"},
        "user_message": SimpleNamespace(id="message-1"), "mode": "answer",
        "first_turn": True, "opening": "upload.txt",
    }, [{"name": "upload.txt", "text": private_text}])
    row = db.get(StreamRun, "run-file")
    assert row.recovery == {"recoverable": False}
    assert private_text not in str(row.__dict__)


def test_recover_chat_turn_rejects_a_newer_user_message():
    from app.services.chat.session import recover_chat_turn

    session = SimpleNamespace(
        id="session-1", user_id="local-user",
        messages=[
            SimpleNamespace(id="original", role="user"),
            SimpleNamespace(id="newer", role="user"),
        ],
    )
    with patch("app.services.chat.session.get_session", return_value=session):
        assert recover_chat_turn(object(), {
            "session_id": "session-1", "user_message_id": "original",
            "intent": {"intent": "answer"}, "mode": "answer", "content": "old question",
        }, "old-run") is None


def test_chat_endpoints_reject_oversized_visible_text_before_writing():
    from fastapi import HTTPException
    from app.api.chat.session import chat, chat_stream
    from app.schemas.business.session import ChatSendIn

    async def check():
        for endpoint in (chat, chat_stream):
            try:
                await endpoint(ChatSendIn(content="x" * 4001), SimpleNamespace(), object())
            except HTTPException as exc:
                assert exc.status_code == 400
            else:
                raise AssertionError("oversized request reached chat persistence")

    asyncio.run(check())


def test_chat_stream_exposes_run_id_before_first_event():
    """A dropped response can replay even if no SSE frame reached the browser."""
    from app.api.chat.session import chat_stream
    from app.schemas.business.session import ChatSendIn

    db = _db()
    with patch("app.api.chat.session.services.route_chat", return_value={
        "mode": "redirect", "run_id": "accepted-chat-run",
    }):
        response = asyncio.run(chat_stream(
            ChatSendIn(content="生成面试"), SimpleNamespace(), db,
        ))
    assert response.headers["x-run-id"] == "accepted-chat-run"


def test_chat_worker_commits_terminal_event_after_response_disconnect():
    """Closing the response iterator must not cancel its generation worker."""
    from app.api.chat.session import _detached_turn_events

    db = _db()
    create_stream_run(db, "chat", "local-user", run_id="run-4")

    async def fake_answer(worker_db, turn, _worker_id):
        yield event_frame(append_stream_event(worker_db, turn["run_id"], "local-user", {"type": "meta"}))
        await asyncio.sleep(0.02)
        yield event_frame(append_stream_event(worker_db, turn["run_id"], "local-user", {"type": "done"}))

    async def check():
        with patch("app.api.chat.session.SessionLocal", return_value=db), patch(
            "app.api.chat.session.services.get_session", return_value=SimpleNamespace(id="session-1")
        ), patch("app.api.chat.session._fallback_events", new=fake_answer):
            stream = _detached_turn_events({"run_id": "run-4", "session": SimpleNamespace(id="session-1")}, fallback=True)
            assert "id: run-4:1" in await stream.__anext__()
            await stream.aclose()
            await asyncio.sleep(0.05)

    asyncio.run(check())
    assert [event.event_type for event in replay_stream_events(db, "run-4", "local-user")] == ["meta", "done"]


def test_chat_graph_worker_finishes_validation_after_response_disconnect():
    """Closing an SSE reader does not interrupt the staged answer or its commit."""
    from app.api.chat.session import _detached_turn_events

    db = _db()
    create_stream_run(db, "chat", "local-user", run_id="run-graph")
    session = SimpleNamespace(id="session-graph")
    observed = []

    async def fake_answer(worker_db, turn, prepared, _worker_id):
        yield event_frame(append_stream_event(worker_db, turn["run_id"], "local-user", {"type": "meta"}))
        await asyncio.sleep(0.02)
        prepared.update(reply="answer [S1]", extra={"retrieval": {"selected_chunks": []}})

    async def fake_graph(_db, *, action, stages, **_kwargs):
        for name in ("knowledge_qa.answer", "knowledge_qa.citation_validate", "knowledge_qa.repair_once"):
            observed.append(name)
            await stages[name](None)
        return await action()

    async def repair(_db, _turn, reply, extra, _initial):
        return reply, extra

    async def check():
        with patch("app.api.chat.session.SessionLocal", return_value=db), patch(
            "app.api.chat.session.services.get_session", return_value=session
        ), patch("app.api.chat.session.services.staged_chat_retrieval_callbacks", return_value={}), patch(
            "app.api.chat.session.services.validate_stream_answer", return_value={"valid": True}
        ), patch("app.api.chat.session.services.repair_stream_answer", new=repair), patch(
            "app.api.chat.session.services.finish_chat", new=AsyncMock(return_value=session)
        ), patch("app.api.chat.session.session_detail", return_value=SimpleNamespace(model_dump=lambda **_kwargs: {"id": session.id})), patch(
            "app.api.chat.session.run_business_graph", new=fake_graph
        ), patch("app.api.chat.session._answer_events", new=fake_answer):
            stream = _detached_turn_events({"run_id": "run-graph", "session": session})
            assert "id: run-graph:1" in await stream.__anext__()
            await stream.aclose()
            await asyncio.sleep(0.05)

    from unittest.mock import AsyncMock
    asyncio.run(check())
    assert observed == ["knowledge_qa.answer", "knowledge_qa.citation_validate", "knowledge_qa.repair_once"]
    assert [event.event_type for event in replay_stream_events(db, "run-graph", "local-user")] == ["meta", "done"]


def test_interview_worker_commits_done_after_response_disconnect():
    """The interview task starts before yielding meta and survives iterator close."""
    from app.api.interviews.interview import _generate_events

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[StreamRun.__table__, StreamEvent.__table__])
    setup = Session(engine)
    create_stream_run(setup, "interview_generation", "local-user", run_id="run-5")

    async def graph(_db, *, action, **_kwargs):
        return await action()

    async def prepare(_db, _content, *, on_thought, run_id=None):
        assert run_id == "run-5"
        await asyncio.sleep(0.02)
        return {"id": "interview-1"}

    async def check():
        with patch("app.api.interviews.interview.SessionLocal", side_effect=lambda: Session(engine)), patch(
            "app.api.interviews.interview.run_business_graph", new=graph
        ), patch("app.api.interviews.interview.services.prepare_interview", new=prepare):
            stream = _generate_events(Session(engine), "后端工程师，负责缓存与一致性", run_id="run-5")
            assert "id: run-5:1" in await stream.__anext__()
            await stream.aclose()
            await asyncio.sleep(0.05)

    asyncio.run(check())
    assert [event.event_type for event in replay_stream_events(setup, "run-5", "local-user")] == ["meta", "done"]
