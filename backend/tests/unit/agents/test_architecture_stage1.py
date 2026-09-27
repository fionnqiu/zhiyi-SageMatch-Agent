"""Stage-one contracts for one route decision and shared request context.

These tests describe the intended boundary while keeping retrieval and routing
deterministic; provider credentials and external services are never consulted.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


class RequestContextContract(unittest.TestCase):
    def test_request_context_carries_stable_turn_identity_and_route(self) -> None:
        """A request context gives API and graph nodes the same typed identifiers."""
        from app.agents.contracts.state import RequestContext

        context = RequestContext(
            request_id="req-test",
            run_id="run-test",
            session_id="session-test",
            interview_id=None,
            route={"route": "knowledge_qa", "confidence": 0.98, "original_query": "什么是缓存击穿？"},
        )
        self.assertEqual(context.route.route, "knowledge_qa")
        self.assertEqual(context.request_id, "req-test")
        self.assertEqual(context.route.original_query, "什么是缓存击穿？")


class RouteNodeContract(unittest.IsolatedAsyncioTestCase):
    async def test_route_node_normalizes_the_four_supported_routes(self) -> None:
        """Routing is explicit and computed once before dispatch."""
        from app.agents.contracts.state import route_node

        async def classify(_db: object, query: str, **_kwargs: object) -> dict:
            del query
            return {"intent": "answer", "confidence": 0.98}

        with patch("app.agents.contracts.state.classify_request", new=classify):
            state = {
                "db": object(),
                "original_query": "解释缓存击穿及预防方式",
                "intent": "knowledge_qa",
                "route": None,
            }
            result = await route_node(state)

        self.assertEqual(result["route"].route, "knowledge_qa")
        self.assertIn(result["route"].route, {
            "knowledge_qa", "interview_generation", "clarification", "unsupported"
        })
        self.assertGreaterEqual(result["route"].confidence, 0)
        self.assertLessEqual(result["route"].confidence, 1)


class KnowledgeQADefaultRetrieval(unittest.IsolatedAsyncioTestCase):
    async def test_knowledge_question_retrieves_even_when_legacy_flag_is_false(self) -> None:
        """The selected route, rather than the legacy perception flag, owns RAG."""
        from app.services.chat.session import prepare_direct_answer

        session = SimpleNamespace(messages=[])
        hits = [{"filename": "缓存说明", "text": "热点 key 失效后并发回源。"}]
        intent = {"intent": "answer", "route": "knowledge_qa", "needs_recall": False, "todos": []}

        with patch("app.services.chat.session.recall_snippets", new=AsyncMock(return_value=hits)) as recall:
            prompt, extra = await prepare_direct_answer(
                object(), session, "什么是缓存击穿？", intent
            )

        recall.assert_awaited_once()
        self.assertIn("热点 key 失效后并发回源。", prompt["user"])
        self.assertEqual(extra["route"], "knowledge_qa")
        self.assertEqual(extra["citations"][0]["filename"], "缓存说明")


class RouteReuseContract(unittest.IsolatedAsyncioTestCase):
    async def test_begin_and_stream_share_the_same_route_decision(self) -> None:
        """Streaming must consume the route resolved for begin, without rerouting."""
        from app.services.chat.session import begin_chat, complete_chat_turn

        db = SimpleNamespace(commits=0, deleted=[], added=[])
        db.add = db.added.append
        db.flush = lambda: None
        db.commit = lambda: None
        decision = {
            "route": "knowledge_qa",
            "intent": "answer",
            "confidence": 0.98,
            "needs_recall": False,
            "todos": [],
        }
        fake_session = SimpleNamespace(id="session-test", messages=[])
        with (
            patch("app.services.chat.session.get_session", return_value=None),
            patch("app.services.chat.session.latest_question_set_for_session", return_value=None),
            patch("app.services.chat.session.ChatSession", return_value=fake_session),
            patch("app.services.chat.session.ChatMessage"),
            patch("app.services.chat.session.resolve_intent", new=AsyncMock(return_value=decision)) as route,
            patch("app.services.chat.session.prepare_direct_answer", new=AsyncMock(return_value=({"system": "s", "user": "u", "fallback": "f"}, {"kind": "answer"}))),
            patch("app.services.chat.session.now", return_value=None),
        ):
            turn = await begin_chat(db, "什么是缓存击穿？", None)
            self.assertEqual(turn["route"]["route"], "knowledge_qa")

            with patch("app.services.chat.session.complete", new=AsyncMock(return_value="说明")):
                await complete_chat_turn(db, turn)

        route.assert_awaited_once()
