"""Tool traces cover the same adapter invoked by the production ToolNode."""

import asyncio

from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agents.roles.loop import _langchain_tool
from app.agents.tools.registry import ToolSpec, validate_question
from app.agents.orchestration.trace import graph_run_scope
from app.core.db import Base
from app.models.platform.audit import LlmCallLog
from app.models.platform.runtime import ToolRun
from app.services.operations.llm_gateway import log_call


def test_tool_node_records_real_call_id_and_redacts_arguments() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[ToolRun.__table__])
    spec = ToolSpec("validate_question", "validate", ("stem",), 0, validate_question,
                    {"stem": {"type": "string"}})
    secret_stem = "candidate private answer for validation"
    with Session(engine) as db, graph_run_scope("graph-123"):
        node = ToolNode([_langchain_tool(spec, db, {})])
        graph = StateGraph(AgentState)
        graph.add_node("tools", node)
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        result = asyncio.run(graph.compile().ainvoke({"messages": [AIMessage(content="", tool_calls=[
            {"name": "validate_question", "args": {"stem": secret_stem}, "id": "model-tool-7"}
        ])]}))
        db.commit()
        row = db.query(ToolRun).one()
        assert row.run_id == "graph-123"
        assert row.tool_call_id == "model-tool-7"
        assert row.status == "success"
        assert result["messages"][-1].tool_call_id == row.tool_call_id
        assert secret_stem not in str(row.__dict__)


def test_tool_trace_records_timeout_without_private_exception() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[ToolRun.__table__])

    async def blocked(_db: Session, _args: dict) -> dict:
        raise TimeoutError("private input must not appear in the trace")

    spec = ToolSpec("blocked", "blocked", (), 0, blocked)
    with Session(engine) as db, graph_run_scope("graph-456"):
        tool = _langchain_tool(spec, db, {})
        asyncio.run(tool.ainvoke({"name": "blocked", "args": {}, "id": "model-tool-8", "type": "tool_call"}))
        db.commit()
        row = db.query(ToolRun).one()
        assert row.status == "timeout"
        assert row.error_code == "tool_timeout"
        assert "private input" not in str(row.__dict__)


def test_tool_exception_message_is_not_returned_to_model() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[ToolRun.__table__])

    async def failed(_db: Session, _args: dict) -> dict:
        raise ValueError("private answer appeared in provider diagnostics")

    spec = ToolSpec("failed", "failed", (), 0, failed)
    with Session(engine) as db, graph_run_scope("graph-error"):
        tool = _langchain_tool(spec, db, {})
        result = asyncio.run(tool.ainvoke({"name": "failed", "args": {}, "id": "tool-error", "type": "tool_call"}))
        db.commit()
        assert "private answer" not in result.content
        assert db.query(ToolRun).one().error_code == "tool_exception"


def test_provider_trace_uses_same_run_and_redacts_exception_text() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[LlmCallLog.__table__])
    with Session(engine) as db, graph_run_scope("graph-789"):
        log_call(db, "author", "provider", "model", "error", 12,
                 "authorization failed: secret-key and private prompt")
        db.commit()
        row = db.query(LlmCallLog).one()
        assert row.run_id == "graph-789"
        assert row.error == "provider_call_failed"
