"""The production generation callbacks execute inside the application graph."""

from __future__ import annotations

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agents.orchestration.checkpoint import MemoryCheckpointer
from app.agents.workflows.generation import InterviewGenerationStages
from app.agents.orchestration.graph import AgentState, build_application_graph
from app.agents.orchestration.workflows import run_business_graph
from app.models.platform.runtime import DurableJob, GraphCheckpointOwner
from app.services.operations.jobs import claim_job_by_id, create_job, fail_job
from app.services.chat.stream_events import append_stream_event, create_stream_run, replay_stream_events
from app.models import ChatSession, QuestionSet, StreamEvent, StreamRun
from app.services.chat.session import generate_and_store, stub_pack


def test_generation_stages_revise_once_then_commit_validated_pack() -> None:
    calls: list[str] = []
    candidate = stub_pack("后端开发岗位职责与要求")
    candidate["questions"][0]["stem"] = "RAW-SECRET 岗位题干"

    async def author(_db, _text, **kwargs):
        calls.append(f"author:{kwargs['attempt']}")
        return {"ok": True, "output": candidate, "handoff": {
            "task": {"task_id": "author-task", "goal": "RAW-SECRET 原岗位描述"},
            "decision": {"agent": "author", "status": "success", "output": candidate,
                         "evidence_refs": ["S1", "RAW-SECRET 证据"],
                         "observations": [{"tool_call_id": "call-1", "tool": "search",
                                           "ok": True, "data": "RAW-SECRET 工具结果"}]},
        }}

    async def critic(_db, _candidate, **_kwargs):
        calls.append("critic")
        return {"verdict": {"pass": len([item for item in calls if item == "critic"]) == 2,
                            "reason": "职责覆盖不足"}, "handoff": {
            "task": {"task_id": "critic-task", "goal": "RAW-SECRET 题干"},
            "decision": {"agent": "critic", "status": "success", "output": {"pass": True}},
        }}

    pipeline = InterviewGenerationStages(object(), "后端开发岗位职责与要求", "generation-run")

    async def commit(_state):
        calls.append("persist")
        assert pipeline.payload == candidate
        return {"status": "committed", "result": {"valid": True}}

    with patch("app.agents.workflows.generation.recall_snippets", new=AsyncMock(return_value=[])), patch(
        "app.agents.workflows.generation.MemoryManager"
    ) as memory, patch("app.agents.workflows.generation.llm.llm_available", return_value=True), patch(
        "app.agents.workflows.generation.author_question_candidate", new=author
    ), patch("app.agents.workflows.generation.critique_question_candidate", new=critic):
        memory.return_value.render.return_value = ""
        graph = build_application_graph(stages=pipeline.callbacks(), commit=commit)
        result = asyncio.run(graph.ainvoke(AgentState(
            requested_mode="interview_generation", run_id="generation-run",
        )))

    assert result["status"] == "committed"
    assert calls == ["author:1", "critic", "author:2", "critic", "persist"]
    compact = result["diagnostics"]["generation_handoffs"]
    assert [item["agent"] for item in compact] == ["author", "critic", "author", "critic"]
    assert compact[0]["evidence_refs"][0] == "S1"
    assert compact[0]["observations"][0]["tool_call_id"] == "call-1"
    assert "RAW-SECRET" not in json.dumps(compact, ensure_ascii=False)
    finished = [event["node"] for event in result["events"] if event["event_type"] in {"node_finished", "business_committed"}]
    assert finished.index("interview_generation.question_set_validate") < finished.index("commit_node")


def test_invalid_revised_pack_uses_deterministic_fallback() -> None:
    bad = {"questions": [{"kind": "choice", "stem": "坏题"}]}
    pipeline = InterviewGenerationStages(object(), "后端开发岗位职责与要求", "generation-run")

    with patch("app.agents.workflows.generation.recall_snippets", new=AsyncMock(return_value=[])), patch(
        "app.agents.workflows.generation.MemoryManager"
    ) as memory, patch("app.agents.workflows.generation.llm.llm_available", return_value=True), patch(
        "app.agents.workflows.generation.author_question_candidate",
        new=AsyncMock(return_value={"ok": True, "output": bad, "handoff": {"task": {"task_id": "author-task"}}}),
    ) as author, patch("app.agents.workflows.generation.critique_question_candidate") as critic:
        memory.return_value.render.return_value = ""
        result = asyncio.run(build_application_graph(
            stages=pipeline.callbacks(), commit=lambda _: {"status": "committed"},
        ).ainvoke(AgentState(requested_mode="interview_generation", run_id="generation-run")))

    assert result["status"] == "committed"
    assert author.await_count == 2
    critic.assert_not_called()
    assert len(pipeline.payload["questions"]) == 8


def test_interrupted_generation_restarts_with_same_run_id_and_compact_checkpoint() -> None:
    """A new worker can rebuild its candidate without checkpointing the JD."""
    class ProcessLost(BaseException):
        pass

    engine = create_engine("sqlite://")
    GraphCheckpointOwner.__table__.create(engine)
    saver = MemoryCheckpointer()
    author_calls = 0
    raw_job = "后端开发岗位，负责缓存一致性和线上故障排查"

    async def stage(_state):
        nonlocal author_calls
        author_calls += 1
        if author_calls == 1:
            raise ProcessLost()
        return {"result": {"valid": True}}

    async def action():
        return "persisted"

    async def scenario(db):
        args = dict(mode="interview_generation", run_id="generation-restart",
                    original_query=hashlib.sha256(raw_job.encode("utf-8")).hexdigest(), checkpointer=saver,
                    restart_incomplete=True, stages={"interview_generation.author": stage}, action=action)
        with patch("app.agents.orchestration.workflows.persist_graph_trace"):
            try:
                await run_business_graph(db, **args)
                assert False, "the first worker must be interrupted"
            except ProcessLost:
                pass
            return await run_business_graph(db, **args)

    try:
        with Session(engine) as db:
            assert asyncio.run(scenario(db)) == "persisted"
            assert author_calls == 2
            payloads = [checkpoint.get("channel_values", {}).get("payload", {})
                        for checkpoint in saver.storage.get("request:generation-restart", {}).values()]
            assert payloads and all(raw_job not in str(payload) for payload in payloads)
    finally:
        engine.dispose()


def test_generation_job_allows_only_one_cross_session_owner(tmp_path) -> None:
    """Two HTTP sessions cannot both enter model stages for one run."""
    engine = create_engine(f"sqlite:///{tmp_path / 'generation-lease.sqlite'}")
    DurableJob.__table__.create(engine)
    try:
        with Session(engine) as first:
            job = create_job(first, "interview_generation", "run-1", idempotency_key="generation:run-1")
            first.commit()
            job_id = job.id
        with Session(engine) as first, Session(engine) as second:
            assert claim_job_by_id(first, job_id, "worker-a") is not None
            first.commit()
            assert claim_job_by_id(second, job_id, "worker-b") is None
            second.commit()
            assert second.get(DurableJob, job_id).worker_id == "worker-a"
    finally:
        engine.dispose()


def test_question_set_snapshot_retains_only_compact_handoff_refs() -> None:
    """The role trace is written with the pack, without prompt or tool body."""
    class PendingSession:
        def __init__(self):
            self.added = []

        def add(self, row):
            self.added.append(row)

        def flush(self):
            return None

    db = PendingSession()
    session = ChatSession(id="run-1", title="新会话", origin="interview")
    trace = [{"task_id": "task-1", "agent": "author", "output_sha256": "digest-1",
              "evidence_refs": ["S1"], "observations": [{"tool_call_id": "call-1", "ok": True}]}]
    with patch("app.services.chat.session.MemoryManager"):
        asyncio.run(generate_and_store(
            db, session, "后端开发岗位职责与要求", prepared_hits=[],
            prepared_payload=stub_pack("后端开发岗位职责与要求"), generation_trace=trace,
        ))
    saved = next(row for row in db.added if isinstance(row, QuestionSet))
    assert saved.snapshot["generation_handoffs"] == trace
    assert "后端开发岗位职责与要求" not in json.dumps(saved.snapshot, ensure_ascii=False)


def test_failed_generation_retry_keeps_same_stream_and_run_id(tmp_path) -> None:
    """A retryable error is visible, while a later attempt may finish the run."""
    engine = create_engine(f"sqlite:///{tmp_path / 'generation-retry.sqlite'}")
    for table in (StreamRun.__table__, StreamEvent.__table__, DurableJob.__table__):
        table.create(engine)
    try:
        with Session(engine) as db:
            run = create_stream_run(db, "interview_generation", "local-user", run_id="generation-1")
            job = create_job(db, "interview_generation", run.id, idempotency_key="generation:generation-1")
            db.commit()
            job_id = job.id
            assert claim_job_by_id(db, job_id, "worker-a") is not None
            db.commit()
            failed = fail_job(db, job_id, "provider_failed", retryable=True,
                              worker_id="worker-a", retry_delay_seconds=0)
            assert failed.status == "retry_wait"
            db.commit()
            append_stream_event(db, run.id, "local-user", {"type": "error", "message": "稍后重试"}, terminal=False)
            assert db.get(StreamRun, run.id).status == "running"
        with Session(engine) as retry:
            assert claim_job_by_id(retry, job_id, "worker-b") is not None
            retry.commit()
            append_stream_event(retry, run.id, "local-user", {"type": "done", "interview": {"id": "iv-1"}})
            assert [event.event_type for event in replay_stream_events(retry, run.id, "local-user")] == ["error", "done"]
    finally:
        engine.dispose()
