"""Provider calls and retries respect the shared monotonic request deadline."""

from __future__ import annotations

import asyncio
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from sqlalchemy import Column, Integer, MetaData, Table, create_engine, select
from sqlalchemy.orm import Session

from app.integrations import llm
from app.core.db import graph_commit_deadline
from app.services.materials.rag.providers import _provider_budget
from app.agents.orchestration.graph import AgentState, build_application_graph
from app.agents.contracts.envelopes import AgentTask
from app.agents.contracts.contracts import profile_for
from app.agents.roles.loop import run_agent, run_handoff
from app.agents.orchestration.workflows import GraphExecutionError, run_business_graph


class DeadlineBudgetTests(unittest.TestCase):
    def test_commit_guard_restores_nested_session_scope(self) -> None:
        with Session(create_engine("sqlite://")) as db:
            outer = time.monotonic() + 10
            with graph_commit_deadline(db, outer):
                with graph_commit_deadline(db, outer + 10):
                    self.assertEqual(db.info["graph_deadline_at"], outer)
                self.assertEqual(db.info["graph_deadline_at"], outer)
            self.assertNotIn("graph_deadline_at", db.info)

    def test_http_timeout_is_capped_by_remaining_budget(self) -> None:
        with llm.deadline_scope(time.monotonic() + 0.2):
            timeout = llm._http_timeout(180, connect=15, read=60)
            self.assertLessEqual(timeout.connect, 0.201)
            self.assertLessEqual(timeout.read, 0.201)
            self.assertLessEqual(timeout.write, 0.201)

    def test_nested_scope_cannot_extend_request_deadline(self) -> None:
        with llm.deadline_scope(time.monotonic() + 0.1):
            with llm.deadline_scope(time.monotonic() + 100):
                self.assertLess(llm.remaining_budget(), 0.101)

    def test_expired_budget_prevents_provider_retry(self) -> None:
        with llm.deadline_scope(time.monotonic() - 1):
            with self.assertRaises(TimeoutError):
                llm._http_timeout(180, connect=15)

    def test_rag_adapter_timeout_cannot_exceed_graph_budget(self) -> None:
        with llm.deadline_scope(time.monotonic() + 0.05):
            allowed, remaining = _provider_budget(8.0)
            self.assertIsNotNone(remaining)
            self.assertLessEqual(allowed, remaining)
            self.assertLess(allowed, 0.051)


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_role_timeout_is_nonretryable_deadline_failure(self) -> None:
        cancelled = asyncio.Event()

        class SlowGraph:
            async def ainvoke(self, *_args, **_kwargs) -> dict:
                try:
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                return {"messages": []}

        with patch("app.agents.roles.loop.build_role_agent", return_value=SlowGraph()):
            result = await run_agent(
                object(), profile_for("interviewer"), user="Follow up",
                deadline_at=time.monotonic() + 0.05,
            )
        self.assertTrue(cancelled.is_set())
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "deadline_exceeded")
        self.assertFalse(result["retryable"])

    async def test_handoff_converts_absolute_deadline_to_monotonic_budget(self) -> None:
        async def bounded_role(_db, _profile, **kwargs) -> dict:
            try:
                with llm.deadline_scope(kwargs["deadline_at"]):
                    async with asyncio.timeout(llm.remaining_budget()):
                        await asyncio.sleep(0.5)
            except TimeoutError:
                return {"ok": False, "error_code": "deadline_exceeded", "output": {}}
            return {"ok": True, "output": {"unexpected": True}}

        with patch("app.agents.roles.loop.run_agent", new=bounded_role):
            for offset in (-0.1, 0.1):
                task = AgentTask(
                    run_id="deadline-run", from_agent="supervisor", to_agent="author",
                    route="interview_generation", goal="Generate questions",
                    deadline_at=datetime.now(timezone.utc) + timedelta(seconds=offset),
                )
                started = time.monotonic()
                result = await run_handoff(object(), task)
                self.assertLess(time.monotonic() - started, 0.35)
                self.assertFalse(result["ok"])
                self.assertEqual(result["handoff"]["decision"]["decision"], "deadline_exceeded")

    async def test_sync_business_commit_after_deadline_is_rolled_back(self) -> None:
        engine = create_engine("sqlite://")
        table = Table("deadline_writes", MetaData(), Column("id", Integer, primary_key=True))
        table.create(engine)
        try:
            with Session(engine) as db:
                entered = []
                async def action() -> str:
                    entered.append(True)
                    time.sleep(0.4)
                    db.execute(table.insert().values(id=1))
                    db.commit()
                    return "committed"

                with patch("app.agents.orchestration.workflows.persist_graph_trace"):
                    with self.assertRaises(GraphExecutionError) as caught:
                        await run_business_graph(
                            db, mode="clarification", action=action,
                            deadline_at=time.monotonic() + 0.3,
                            stage_sequences={"clarification": ()},
                        )
                self.assertEqual(entered, [True])
                self.assertEqual(caught.exception.error_code, "deadline_exceeded")
                self.assertFalse(caught.exception.retryable)
                self.assertFalse(db.in_transaction())
                self.assertNotIn("graph_deadline_at", db.info)
            with Session(engine) as check:
                self.assertEqual(check.execute(select(table.c.id)).all(), [])
        finally:
            engine.dispose()

    async def test_post_is_cancelled_at_deadline(self) -> None:
        class SlowClient:
            async def post(self, *_args, **_kwargs):
                await asyncio.sleep(2)

        with llm.deadline_scope(time.monotonic() + 0.01):
            with self.assertRaises(TimeoutError):
                await llm._bounded_post(SlowClient(), "https://example.test")

    async def test_slow_stage_stops_at_absolute_graph_deadline(self) -> None:
        cancelled = asyncio.Event()

        async def slow_stage(_state: AgentState) -> dict:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return {"result": {"valid": True}}

        deadline = datetime.now(timezone.utc) + timedelta(milliseconds=500)
        graph = build_application_graph(stages={"knowledge_qa.answer": slow_stage})
        started = time.monotonic()
        outcome = await graph.ainvoke(AgentState(requested_mode="knowledge_qa", deadline_at=deadline))
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(outcome["error"]["error_code"], "deadline_exceeded")
        self.assertFalse(outcome["error"]["retryable"])

    async def test_slow_commit_cannot_write_after_graph_deadline(self) -> None:
        writes: list[str] = []

        async def slow_commit(_state: AgentState) -> dict:
            await asyncio.sleep(1)
            writes.append("committed")
            return {"status": "committed", "result": {"committed": True}}

        deadline = datetime.now(timezone.utc) + timedelta(milliseconds=200)
        graph = build_application_graph(commit=slow_commit)
        started = time.monotonic()
        outcome = await graph.ainvoke(AgentState(requested_mode="clarification", deadline_at=deadline))
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(writes, [])
        self.assertEqual(outcome["error"]["error_code"], "deadline_exceeded")
        self.assertFalse(outcome["error"]["retryable"])

    async def test_blocking_sync_stage_result_is_rejected_after_deadline(self) -> None:
        calls: list[str] = []

        def blocking_stage(_state: AgentState) -> dict:
            calls.append("entered")
            time.sleep(0.7)
            return {"result": {"valid": True}}

        graph = build_application_graph(stages={"knowledge_qa.answer": blocking_stage})
        outcome = await graph.ainvoke(AgentState(
            requested_mode="knowledge_qa",
            deadline_at=datetime.now(timezone.utc) + timedelta(milliseconds=500),
        ))
        self.assertEqual(calls, ["entered"])
        self.assertEqual(outcome["error"]["error_code"], "deadline_exceeded")
        self.assertFalse(outcome["error"]["retryable"])

    async def test_business_graph_surfaces_nonretryable_deadline_failure(self) -> None:
        writes: list[str] = []

        async def action() -> str:
            await asyncio.sleep(2)
            writes.append("committed")
            return "done"

        with self.assertRaises(GraphExecutionError) as caught:
            await run_business_graph(
                object(), mode="live_interview", action=action,
                deadline_at=time.monotonic() + 0.5,
                stage_sequences={"live_interview": ()},
            )
        self.assertEqual(caught.exception.error_code, "deadline_exceeded")
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(writes, [])
