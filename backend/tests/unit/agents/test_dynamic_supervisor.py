"""The application graph must own bounded, validated role handoffs."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agents.orchestration.checkpoint import MemoryCheckpointer
from app.agents.orchestration.graph import AgentState, SupervisedRoute, build_application_graph
from app.agents.orchestration.supervisor import choose_generation_revision
from app.agents.orchestration.workflows import GraphExecutionError, run_business_graph
from app.agents.workflows.generation import InterviewGenerationStages
from app.agents.roles.authoring import author_questions
from app.models.platform.runtime import GraphCheckpointOwner
from app.services.chat.session import generate_question_pack, stub_pack


def test_supervisor_selects_workers_and_commits_after_validation() -> None:
    calls: list[str] = []

    async def choose(_state: AgentState, allowed: tuple[str, ...]) -> str:
        calls.append("choose")
        return "author" if calls.count("choose") == 1 else (
            "critic" if calls.count("choose") == 2 else "finish"
        )

    async def worker(role: str, _state: AgentState) -> dict:
        calls.append(role)
        return {"diagnostics": {f"{role}_ran": True}}

    async def finalize(_state: AgentState) -> dict:
        calls.append("validate")
        return {"result": {"valid": True}}

    async def commit(_state: AgentState) -> dict:
        calls.append("commit")
        return {"status": "committed", "result": {"valid": True}}

    route = SupervisedRoute(
        mode="interview_generation",
        allowed_roles=lambda _state: ("author", "critic", "finish"),
        select_role=choose,
        workers={"author": lambda state: worker("author", state),
                 "critic": lambda state: worker("critic", state)},
        finalize=finalize,
        max_handoffs=4,
    )
    output = asyncio.run(build_application_graph(supervised_route=route, commit=commit).ainvoke(
        AgentState(requested_mode="interview_generation")
    ))

    assert output["status"] == "committed"
    assert calls == ["choose", "author", "choose", "critic", "choose", "validate", "commit"]
    assert [item["task"]["to_agent"] for item in output["handoffs"][-2:]] == ["author", "critic"]
    assert [item["decision"]["status"] for item in output["handoffs"][-2:]] == ["success", "success"]


def test_supervisor_rejects_forbidden_role_before_worker_or_commit() -> None:
    calls: list[str] = []

    async def choose(_state: AgentState, _allowed: tuple[str, ...]) -> str:
        return "scorer"

    async def author(_state: AgentState) -> dict:
        calls.append("author")
        return {"result": {"valid": True}}

    async def commit(_state: AgentState) -> dict:
        calls.append("commit")
        return {"status": "committed"}

    route = SupervisedRoute(
        mode="interview_generation", allowed_roles=lambda _state: ("author",),
        select_role=choose, workers={"author": author},
        finalize=author, max_handoffs=2,
    )
    output = asyncio.run(build_application_graph(supervised_route=route, commit=commit).ainvoke(
        AgentState(requested_mode="interview_generation")
    ))

    assert output["status"] == "failed"
    assert output["error"]["error_code"] == "supervisor_role_forbidden"
    assert calls == []


def test_supervised_route_requires_explicit_valid_result_before_commit() -> None:
    writes: list[str] = []
    route = SupervisedRoute(
        mode="interview_generation", allowed_roles=lambda _state: ("finish",),
        select_role=lambda _state, _allowed: "finish", workers={},
        finalize=lambda _state: {}, max_handoffs=1,
    )

    async def commit(_state: AgentState) -> dict:
        writes.append("commit")
        return {"status": "committed"}

    output = asyncio.run(build_application_graph(
        supervised_route=route, commit=commit,
    ).ainvoke(AgentState(requested_mode="interview_generation", max_retries=0)))
    assert output["status"] == "failed"
    assert output["error"]["error_code"] == "graph_validation_missing"
    assert writes == []


def test_supervisor_handoff_budget_stops_repeated_role_calls() -> None:
    calls: list[str] = []

    async def choose(_state: AgentState, _allowed: tuple[str, ...]) -> str:
        return "author"

    async def author(_state: AgentState) -> dict:
        calls.append("author")
        return {}

    async def commit(_state: AgentState) -> dict:
        calls.append("commit")
        return {"status": "committed"}

    route = SupervisedRoute(
        mode="interview_generation", allowed_roles=lambda _state: ("author",),
        select_role=choose, workers={"author": author},
        finalize=author, max_handoffs=2,
    )
    output = asyncio.run(build_application_graph(supervised_route=route, commit=commit).ainvoke(
        AgentState(requested_mode="interview_generation")
    ))

    assert output["status"] == "failed"
    assert output["error"]["error_code"] == "supervisor_handoff_budget_exceeded"
    assert calls == ["author", "author"]


def test_supervised_stage_retry_resets_adapter_state() -> None:
    attempts = 0
    adapter_attempts = 0

    async def worker(_state: AgentState) -> dict:
        nonlocal attempts, adapter_attempts
        attempts += 1
        adapter_attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary worker failure")
        assert adapter_attempts == 1
        return {"result": {"valid": True}, "diagnostics": {"worked": True}}

    def reset() -> None:
        nonlocal adapter_attempts
        adapter_attempts = 0

    route = SupervisedRoute(
        mode="interview_generation",
        allowed_roles=lambda state: ("finish",) if state.diagnostics.get("worked") else ("author",),
        select_role=lambda _state, allowed: allowed[0], workers={"author": worker},
        finalize=lambda _state: {"result": {"valid": True}},
        reset=reset, max_handoffs=2,
    )
    output = asyncio.run(build_application_graph(
        supervised_route=route,
        stages={"interview_generation.author": worker},
        commit=lambda _state: {"status": "committed", "result": {"valid": True}},
    ).ainvoke(AgentState(requested_mode="interview_generation", max_retries=1)))
    assert output["status"] == "committed"
    assert attempts == 2
    assert output["retry_count"] == 1


def test_generation_supervisor_can_choose_fallback_after_critic_veto() -> None:
    calls: list[str] = []
    candidate = stub_pack("后端开发岗位职责与要求")
    pipeline = InterviewGenerationStages(object(), "后端开发岗位职责与要求", "supervised-run")

    async def author(_db, _text, **kwargs):
        calls.append(f"author:{kwargs['attempt']}")
        return {"ok": True, "output": candidate}

    async def critic(_db, _candidate, **_kwargs):
        calls.append("critic")
        return {"verdict": {"pass": False, "reason": "覆盖不足"}}

    async def commit(_state):
        calls.append("commit")
        return {"status": "committed", "result": {"valid": True}}

    with patch("app.agents.workflows.generation.recall_snippets", new=AsyncMock(return_value=[])), patch(
        "app.agents.workflows.generation.MemoryManager"
    ) as memory, patch("app.agents.workflows.generation.provider_available", return_value=True), patch(
        "app.agents.workflows.generation.author_question_candidate", new=author
    ), patch("app.agents.workflows.generation.critique_question_candidate", new=critic), patch(
        "app.services.operations.llm_gateway.complete_with",
        new=AsyncMock(return_value={"next_role": "finish"}),
    ) as supervisor:
        memory.return_value.render.return_value = ""
        output = asyncio.run(build_application_graph(
            supervised_route=pipeline.supervised_route(), commit=commit,
        ).ainvoke(AgentState(requested_mode="interview_generation", run_id="supervised-run")))

    assert output["status"] == "committed"
    assert calls == ["author:1", "critic", "commit"]
    assert len(pipeline.payload["questions"]) == 8
    assert output["diagnostics"]["generation_fallback"] == "覆盖不足"
    assert [item["decision"]["output"] for item in output["handoffs"][-2:]] == [
        {"ok": True, "question_count": 8}, {"passed": False},
    ]
    assert supervisor.await_count == 1


def test_generation_supervisor_can_select_one_author_revision() -> None:
    calls: list[str] = []
    candidate = stub_pack("后端开发岗位职责与要求")
    pipeline = InterviewGenerationStages(object(), "后端开发岗位职责与要求", "supervised-run")

    async def author(_db, _text, **kwargs):
        calls.append(f"author:{kwargs['attempt']}")
        return {"ok": True, "output": candidate}

    async def critic(_db, _candidate, **_kwargs):
        calls.append("critic")
        return {"verdict": {"pass": calls.count("critic") == 2, "reason": "覆盖不足"}}

    with patch("app.agents.workflows.generation.recall_snippets", new=AsyncMock(return_value=[])), patch(
        "app.agents.workflows.generation.MemoryManager"
    ) as memory, patch("app.agents.workflows.generation.provider_available", return_value=True), patch(
        "app.agents.workflows.generation.author_question_candidate", new=author
    ), patch("app.agents.workflows.generation.critique_question_candidate", new=critic), patch(
        "app.services.operations.llm_gateway.complete_with",
        new=AsyncMock(return_value={"next_role": "author"}),
    ) as supervisor:
        memory.return_value.render.return_value = ""
        output = asyncio.run(build_application_graph(
            supervised_route=pipeline.supervised_route(),
            commit=lambda _: {"status": "committed", "result": {"valid": True}},
        ).ainvoke(AgentState(requested_mode="interview_generation", run_id="supervised-run")))

    assert output["status"] == "committed"
    assert calls == ["author:1", "critic", "author:2", "critic"]
    assert pipeline.payload == candidate
    assert supervisor.await_count == 1
    author_starts = [event["payload"]["attempt"] for event in output["events"]
                     if event["node"] == "interview_generation.author"
                     and event["event_type"] == "node_started"]
    assert author_starts == [1, 2]
    assert [event["payload"]["next_role"] for event in output["events"]
            if event["event_type"] == "supervisor_decision"] == [
                "author", "critic", "author", "critic", "finish",
            ]


def test_supervisor_rejects_unexpected_model_choice_and_uses_author_fallback() -> None:
    with patch("app.services.operations.llm_gateway.complete_with",
               new=AsyncMock(return_value={"next_role": "scorer"})):
        role, fallback = asyncio.run(choose_generation_revision(object(), "题干重复"))
    assert (role, fallback) == ("author", "invalid_decision")


def test_supervisor_accepts_bounded_finish_choice() -> None:
    with patch("app.services.operations.llm_gateway.complete_with",
               new=AsyncMock(return_value={"next_role": "finish"})) as model:
        role, fallback = asyncio.run(choose_generation_revision(object(), "职责覆盖不足"))
    assert (role, fallback) == ("finish", "")
    assert model.call_args.args[1] == "analyst"


def test_chat_authoring_honors_supervisor_finish_after_veto() -> None:
    calls: list[str] = []
    candidate = stub_pack("后端开发岗位职责与要求")

    async def worker(_db, profile, **_kwargs):
        calls.append(profile.role)
        if profile.role == "critic":
            return {"ok": True, "output": {"pass": False, "reason": "题干重复"}}
        return {"ok": True, "output": candidate}

    with patch("app.agents.roles.loop.run_agent", new=worker), patch(
        "app.agents.roles.authoring.MemoryManager"
    ) as memory, patch("app.services.operations.llm_gateway.complete_with",
                       new=AsyncMock(return_value={"next_role": "finish"})):
        memory.return_value.render.return_value = ""
        result = asyncio.run(author_questions(object(), "后端开发岗位职责与要求"))

    assert calls == ["author", "critic"]
    assert result["_agent"]["verdict"] == "rejected"


def test_chat_pack_replaces_rejected_candidate_with_deterministic_fallback() -> None:
    rejected = {**stub_pack("后端开发岗位职责与要求"),
                "_agent": {"verdict": "rejected"}}
    rejected["questions"][0]["stem"] = "REJECTED"
    with patch("app.services.chat.session.provider_available", return_value=True), patch(
        "app.services.chat.session.author_questions", new=AsyncMock(return_value=rejected)
    ):
        output = asyncio.run(generate_question_pack(object(), "后端开发岗位职责与要求", []))
    assert output["questions"][0]["stem"] != "REJECTED"


def test_failed_supervised_run_restarts_on_same_checkpoint_thread() -> None:
    engine = create_engine("sqlite://")
    GraphCheckpointOwner.__table__.create(engine)
    saver = MemoryCheckpointer()
    attempts = 0
    commits = 0

    async def worker(_state: AgentState) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            raise RuntimeError("transient provider error")
        return {"diagnostics": {"worked": True}}

    async def action() -> str:
        nonlocal commits
        commits += 1
        return "saved"

    def route() -> SupervisedRoute:
        return SupervisedRoute(
            mode="interview_generation",
            allowed_roles=lambda state: ("finish",) if state.diagnostics.get("worked") else ("author",),
            select_role=lambda _state, allowed: allowed[0], workers={"author": worker},
            finalize=lambda _state: {"result": {"valid": True}}, max_handoffs=2,
        )

    async def scenario(db: Session) -> str:
        args = dict(
            mode="interview_generation", run_id="supervised-restart",
            original_query="job-hash", checkpointer=saver,
            supervised_route=route(), action=action,
            restart_incomplete=True, restart_completed_failed=True,
        )
        with patch("app.agents.orchestration.workflows.persist_graph_trace"):
            with pytest.raises(GraphExecutionError):
                await run_business_graph(db, **args)
            return await run_business_graph(db, **args)

    try:
        with Session(engine) as db:
            assert asyncio.run(scenario(db)) == "saved"
        assert attempts == 4
        assert commits == 1
    finally:
        engine.dispose()
