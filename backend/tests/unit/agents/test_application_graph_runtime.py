"""Runtime checks for application graph callbacks and bounded decisions."""

from __future__ import annotations

import asyncio

from app.agents.orchestration.graph import AgentState, build_application_graph
from app.agents.orchestration.checkpoint import MemoryCheckpointer


def test_graph_handler_records_business_commit_only_when_confirmed() -> None:
    """A business adapter must explicitly report a committed fact."""

    async def handler(state: AgentState) -> dict:
        return {"status": "committed", "result": {"answer_id": "a-1", "run_id": state.run_id}}

    graph = build_application_graph(handlers={"knowledge_qa": handler})
    output = asyncio.run(graph.ainvoke(AgentState(requested_mode="knowledge_qa", original_query="question")))
    assert output["status"] == "committed"
    assert output["result"]["answer_id"] == "a-1"
    assert sum(event["event_type"] == "business_committed" for event in output["events"]) == 1

    pending = asyncio.run(build_application_graph().ainvoke(AgentState(requested_mode="knowledge_qa")))
    assert pending["status"] == "degraded"
    assert not any(event["event_type"] == "business_committed" for event in pending["events"])


def test_graph_retries_invalid_result_within_budget() -> None:
    """Validation failures loop through dispatch once and preserve one route decision."""
    attempts = 0

    async def answer(_state: AgentState) -> dict:
        nonlocal attempts
        attempts += 1
        return {"result": {"valid": attempts > 1}}

    graph = build_application_graph(stages={"knowledge_qa.answer": answer})
    output = asyncio.run(graph.ainvoke(AgentState(requested_mode="knowledge_qa", max_retries=1)))
    assert attempts == 2
    assert output["retry_count"] == 1
    assert output["status"] == "degraded"
    assert sum(event["event_type"] == "route_decided" for event in output["events"]) == 1
    assert sum(event["event_type"] == "retry_scheduled" for event in output["events"]) == 1


def test_graph_retries_transient_stage_exception_within_budget() -> None:
    """A retryable exception must reach the same bounded decision path as validation errors."""
    attempts = 0

    async def answer(_state: AgentState) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary provider failure")
        return {"result": {"valid": True}}

    output = asyncio.run(build_application_graph(
        stages={"knowledge_qa.answer": answer},
    ).ainvoke(AgentState(requested_mode="knowledge_qa", max_retries=1)))
    assert attempts == 2
    assert output["retry_count"] == 1
    assert output["status"] == "degraded"
    assert sum(event["event_type"] == "retry_scheduled" for event in output["events"]) == 1


def test_handler_runs_only_after_validation_and_decision() -> None:
    """A rejected candidate must never reach the business writer."""
    calls: list[str] = []

    async def candidate(_state: AgentState) -> dict:
        calls.append("candidate")
        return {"result": {"valid": False}}

    async def writer(_state: AgentState) -> dict:
        calls.append("write")
        return {"status": "committed"}

    output = asyncio.run(build_application_graph(
        stages={"knowledge_qa.answer": candidate},
        handlers={"knowledge_qa": writer},
    ).ainvoke(AgentState(requested_mode="knowledge_qa", max_retries=0)))
    assert output["status"] == "failed"
    assert calls == ["candidate"]
    assert not any(event["event_type"] == "business_committed" for event in output["events"])


def test_handler_commit_happens_after_validation() -> None:
    """The compatibility writer sees an accepted candidate at commit_node."""
    calls: list[str] = []

    async def candidate(_state: AgentState) -> dict:
        calls.append("candidate")
        return {"result": {"valid": True}}

    async def writer(state: AgentState) -> dict:
        calls.append(state.current_node)
        assert state.decision["next_action"] == "persist"
        return {"status": "committed"}

    output = asyncio.run(build_application_graph(
        stages={"knowledge_qa.answer": candidate},
        handlers={"knowledge_qa": writer},
    ).ainvoke(AgentState(requested_mode="knowledge_qa")))
    assert output["status"] == "committed"
    assert calls == ["candidate", "commit_node"]
    assert any(event["node"] == "commit_node" and event["event_type"] == "business_committed" for event in output["events"])


def test_stage_cannot_claim_business_commit_before_validation() -> None:
    """A role proposal cannot forge the only side-effect acknowledgement."""

    calls: list[str] = []

    async def candidate(_state: AgentState) -> dict:
        return {"status": "committed", "result": {"committed": True}}

    async def writer(_state: AgentState) -> dict:
        calls.append("write")
        return {"status": "committed"}

    output = asyncio.run(build_application_graph(
        stages={"knowledge_qa.answer": candidate}, commit=writer,
    ).ainvoke(AgentState(requested_mode="knowledge_qa", max_retries=0)))
    assert output["status"] == "failed"
    assert calls == []
    assert not any(event["event_type"] == "business_committed" for event in output["events"])


def test_unknown_stage_patch_field_prevents_business_commit() -> None:
    """A misspelled evidence field must fail before a writer sees the candidate."""

    for field in ("selected_chunk", "citation"):
        writes: list[str] = []

        async def candidate(_state: AgentState) -> dict:
            return {"result": {"valid": True}, field: [{"source_id": "S1"}]}

        async def writer(_state: AgentState) -> dict:
            writes.append("write")
            return {"status": "committed"}

        output = asyncio.run(build_application_graph(
            stages={"knowledge_qa.answer": candidate}, commit=writer,
        ).ainvoke(AgentState(requested_mode="knowledge_qa", max_retries=0)))
        assert output["status"] == "failed"
        assert writes == []
        assert not any(event["event_type"] == "business_committed" for event in output["events"])


def test_stage_diagnostics_accumulate_across_business_nodes() -> None:
    """A later answer diagnostic preserves earlier retrieval observations."""

    async def retrieve(_state: AgentState) -> dict:
        return {"diagnostics": {"candidate_count": 4}}

    async def answer(_state: AgentState) -> dict:
        return {"diagnostics": {"fallback_reason": "provider_timeout"}, "result": {"valid": True}}

    output = asyncio.run(build_application_graph(stages={
        "knowledge_qa.candidate_fusion": retrieve,
        "knowledge_qa.answer": answer,
    }).ainvoke(AgentState(requested_mode="knowledge_qa")))
    assert output["diagnostics"]["candidate_count"] == 4
    assert output["diagnostics"]["fallback_reason"] == "provider_timeout"


def test_policy_gate_rejects_oversized_input_and_cross_route_tool() -> None:
    """A rejected request exits before any business subgraph callback runs."""

    calls: list[str] = []

    async def answer(_state: AgentState) -> dict:
        calls.append("answer")
        return {"result": {"valid": True}}

    graph = build_application_graph(stages={"knowledge_qa.answer": answer})
    large = asyncio.run(graph.ainvoke(AgentState(requested_mode="knowledge_qa", original_query="x" * 4001)))
    forbidden = asyncio.run(graph.ainvoke(AgentState(
        requested_mode="knowledge_qa", plan=[{"tool_name": "hybrid_search"}],
    )))
    invalid_budget = asyncio.run(graph.ainvoke(AgentState(
        requested_mode="knowledge_qa", cost_budget=float("nan"),
    )))
    assert large["error"]["error_code"] == "input_too_large"
    assert forbidden["error"]["error_code"] == "tool_forbidden"
    assert invalid_budget["error"]["error_code"] == "invalid_budget"
    assert calls == []


def test_staged_result_requires_explicit_boolean_validity() -> None:
    async def incomplete(_state: AgentState) -> dict:
        return {"result": {"answer_ready": True}}

    output = asyncio.run(build_application_graph(stages={
        "knowledge_qa.answer": incomplete,
    }).ainvoke(AgentState(requested_mode="knowledge_qa", max_retries=0)))
    assert output["status"] == "failed"
    assert output["error"]["error_code"] == "graph_validation_missing"


def test_checkpoint_saved_event_requires_completed_saver_invoke() -> None:
    state = AgentState(requested_mode="unsupported", thread_id="request:checkpoint-test")
    without = asyncio.run(build_application_graph().ainvoke(state))
    with_saver = asyncio.run(build_application_graph(checkpointer=MemoryCheckpointer()).ainvoke(
        state, config={"configurable": {"thread_id": state.thread_id, "owner_id": state.user_id}},
    ))
    assert not any(event["event_type"] == "checkpoint_saved" for event in without["events"])
    assert any(event["event_type"] == "checkpoint_saved" for event in with_saver["events"])
    assert with_saver["events"][-1]["event_type"] == "run_finished"


def test_graph_failure_uses_safe_public_envelope() -> None:
    import json
    from app.agents.orchestration.workflows import GraphExecutionError
    from app.main import graph_execution_error

    response = asyncio.run(graph_execution_error(None, GraphExecutionError("graph_validation_failed")))
    body = json.loads(response.body)
    assert response.status_code == 503
    assert body["error_code"] == "graph_validation_failed"
    assert body["retryable"] is True
    assert "traceback" not in str(body)


def test_budget_denial_is_not_retried_or_exposed_as_provider_text() -> None:
    import json
    from app.integrations import llm
    from app.agents.orchestration.workflows import GraphExecutionError
    from app.main import graph_execution_error

    attempts = 0

    async def over_budget(_state: AgentState) -> dict:
        nonlocal attempts
        attempts += 1
        raise llm.UsageBudgetError("private provider usage detail")

    outcome = asyncio.run(build_application_graph(stages={
        "knowledge_qa.answer": over_budget,
    }).ainvoke(AgentState(requested_mode="knowledge_qa", max_retries=2)))
    assert attempts == 1
    assert outcome["error"]["error_code"] == "model_budget_exceeded"
    response = asyncio.run(graph_execution_error(None, GraphExecutionError(
        "model_budget_exceeded", retryable=False,
    )))
    assert response.status_code == 429
    assert "private provider" not in json.dumps(json.loads(response.body))


def test_only_executed_stages_emit_paired_timed_events() -> None:
    """No-op topology nodes must not be counted as completed business work."""
    async def answer(_state: AgentState) -> dict:
        return {"result": {"valid": True}}

    output = asyncio.run(build_application_graph(
        stages={"knowledge_qa.answer": answer},
    ).ainvoke(AgentState(requested_mode="knowledge_qa")))
    events = output["events"]
    started = [event for event in events if event["event_type"] == "node_started"]
    finished = [event for event in events if event["event_type"] == "node_finished"]
    for node in ("knowledge_qa.answer", "dispatch_subgraph"):
        starts = [event for event in started if event["node"] == node]
        ends = [event for event in finished if event["node"] == node]
        assert len(starts) == len(ends) == 1
        assert starts[0]["payload"]["attempt"] == ends[0]["payload"]["attempt"] == 1
        assert ends[0]["payload"]["duration_ms"] >= 0
        assert ends[0]["payload"]["status"] == "success"
    assert not any(event["node"] == "knowledge_qa.rerank" for event in started)
    for node in ("load_context", "route_node", "policy_gate", "supervisor_handoff",
                 "observe_node", "validate_node", "decide_node", "checkpoint_node"):
        assert len([event for event in started if event["node"] == node]) == 1
        assert len([event for event in finished if event["node"] == node]) == 1


def test_supervisor_retains_structured_handoff_and_worker_evidence() -> None:
    """A production dispatch keeps the worker proposal after the graph exits."""

    async def answer(_state: AgentState) -> dict:
        return {
            "result": {"valid": True, "answer": "supported"},
            "citations": [{"chunk_id": "chunk-1"}],
            "diagnostics": {
                "observations": [{"tool": "search", "ok": True}],
                "tool_results": [{"tool_name": "search", "ok": True, "data": {"chunk_id": "chunk-1"}}],
            },
        }

    output = asyncio.run(build_application_graph(
        stages={"knowledge_qa.answer": answer},
    ).ainvoke(AgentState(requested_mode="knowledge_qa", session_id="session-1")))
    assert len(output["handoffs"]) == 1
    handoff = output["handoffs"][0]
    assert handoff["task"]["from_agent"] == "supervisor"
    assert handoff["task"]["to_agent"] == "knowledge_qa"
    assert handoff["task"]["input_refs"] == ["session-1"]
    assert handoff["decision"]["output"] == {"valid": True, "answer": "supported"}
    assert handoff["decision"]["evidence_refs"] == ["chunk-1"]
    assert handoff["decision"]["observations"] == [{"tool": "search", "ok": True}]
    assert handoff["decision"]["tool_results"][0]["data"] == {"chunk_id": "chunk-1"}
    assert any(event["event_type"] == "agent_called" for event in output["events"])
    observed = next(event for event in output["events"] if event["event_type"] == "result_observed")
    assert observed["payload"]["handoff_count"] == 1
    assert observed["payload"]["selected_count"] == 0


def test_supervisor_retry_links_tasks_and_has_finite_exit() -> None:
    """A failed proposal is retained when supervisor schedules one retry."""
    attempts = 0

    async def answer(_state: AgentState) -> dict:
        nonlocal attempts
        attempts += 1
        return {"result": {"valid": attempts > 1, "attempt": attempts}}

    output = asyncio.run(build_application_graph(
        stages={"knowledge_qa.answer": answer},
    ).ainvoke(AgentState(requested_mode="knowledge_qa", max_retries=1)))
    assert attempts == 2
    assert len(output["handoffs"]) == 2
    first, second = output["handoffs"]
    assert first["decision"]["status"] == "failed"
    assert first["decision"]["output"]["attempt"] == 1
    assert second["task"]["parent_task_id"] == first["task"]["task_id"]
    assert second["task"]["attempt"] == 2
    assert second["decision"]["output"]["attempt"] == 2
