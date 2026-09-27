"""Report regeneration traces only the precommit command work."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.agents.workflows.evaluation import ReportRegenerationCommandStages
from app.agents.orchestration.graph import AgentState, build_application_graph
from app.agents.orchestration.workflows import GraphExecutionError, run_business_graph


def test_regeneration_command_prechecks_before_commit() -> None:
    order: list[str] = []
    interview = SimpleNamespace(status="ended", report=SimpleNamespace(id="report-1"))
    stages = ReportRegenerationCommandStages(SimpleNamespace(), "iv-1")
    original_validate = stages.validate_command

    async def validate(state):
        order.append("validate")
        return await original_validate(state)

    stages.validate_command = validate

    async def commit():
        order.append("commit")
        return interview

    async def run():
        return await run_business_graph(
            SimpleNamespace(), mode="evaluation_report", interview_id="iv-1",
            stages=stages.callbacks(), stage_sequences=stages.sequences(), action=commit,
        )

    def load(*_args):
        order.append("load")
        return interview

    with patch("app.services.interviews.interview.get_interview", side_effect=load):
        assert asyncio.run(run()) is interview
    assert order == ["load", "validate", "commit"]


def test_regeneration_trace_excludes_background_scoring() -> None:
    stages = ReportRegenerationCommandStages(SimpleNamespace(), "iv-1")

    async def commit(_state):
        return {"status": "committed", "result": {"valid": True}}

    graph = build_application_graph(
        stages=stages.callbacks(), stage_sequences=stages.sequences(), commit=commit,
    )
    with patch("app.services.interviews.interview.get_interview", return_value=SimpleNamespace(status="ended", report=None)):
        output = asyncio.run(graph.ainvoke(AgentState(
            requested_mode="evaluation_report", user_id="anonymous", tenant_id="anonymous",
        )))
    names = [event["node"] for event in output["events"]
             if event["event_type"] == "node_started" and event["node"].startswith("evaluation_report.")]
    assert names == [f"evaluation_report.{name}" for name in stages.STAGE_NAMES]
    assert output["status"] == "committed"


@pytest.mark.parametrize("status", ["missing", "ready", "live", "abandoned"])
def test_invalid_regeneration_does_not_commit(status: str) -> None:
    commits: list[str] = []
    stages = ReportRegenerationCommandStages(SimpleNamespace(), "iv-1")

    async def commit(_state):
        commits.append("write")
        return {"status": "committed"}

    graph = build_application_graph(
        stages=stages.callbacks(), stage_sequences=stages.sequences(), commit=commit,
    )
    found = None if status == "missing" else SimpleNamespace(status=status, report=None)
    with patch("app.services.interviews.interview.get_interview", return_value=found):
        output = asyncio.run(graph.ainvoke(AgentState(
            requested_mode="evaluation_report", max_retries=0,
            user_id="anonymous", tenant_id="anonymous",
        )))
    assert output["status"] == "failed"
    assert output["error"]["error_code"] == "report_regeneration_invalid"
    assert commits == []


def test_invalid_regeneration_uses_client_error_response() -> None:
    from app.main import graph_execution_error

    response = asyncio.run(graph_execution_error(
        None, GraphExecutionError("report_regeneration_invalid", retryable=False),
    ))
    body = json.loads(response.body)
    assert response.status_code == 400
    assert body["error_code"] == "report_regeneration_invalid"
    assert body["retryable"] is False
