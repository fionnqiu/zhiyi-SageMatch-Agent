"""Exercise answer persistence and question progression independently of LLM vendors."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.services.interviews.interview import answer_interview


class AnswerFlowTests(unittest.IsolatedAsyncioTestCase):
    async def _answer(self, *, index: int, followups: int, question_count: int, followup=None):
        questions = [SimpleNamespace(id=f"q-{i}", stem=f"题目{i + 1}") for i in range(question_count)]
        interview = SimpleNamespace(
            id="iv-1",
            status="live",
            question_set=SimpleNamespace(questions=questions),
            current_question_index=index,
            followups_on_question=followups,
            started_at=datetime.now(timezone.utc),
            elapsed_seconds=0,
            turns=[],
        )
        def add(row):
            interview.turns.append(row)
            db.commits_since_answer = 0

        def commit():
            db.commits_since_answer += 1
            pending = next((turn for turn in reversed(interview.turns) if turn.role == "user" and turn.cite == "answer_pending"), None)
            if pending is not None and any(turn.role == "interviewer" for turn in interview.turns):
                pending.cite = None
            if interview.turns and interview.turns[-1].role == "user":
                committed_response.set()

        db = SimpleNamespace(add=add, flush=lambda: None, commit=commit, commits_since_answer=0)
        committed_response = __import__("asyncio").Event()

        def load(_db, _id):
            return interview

        with patch("app.services.interviews.interview.get_interview", side_effect=load), patch(
            "app.services.interviews.interview.followup_line", new=followup or AsyncMock(return_value="追问")
        ):
            task = __import__("asyncio").create_task(answer_interview(db, "iv-1", "我的回答", "text"))
            await committed_response.wait()
            await task
        return interview

    async def test_second_answer_advances_without_waiting_for_interviewer_model(self) -> None:
        async def fail(*_args, **_kwargs):
            raise AssertionError("advance is deterministic and should not call the model")

        interview = await self._answer(index=0, followups=1, question_count=3, followup=fail)
        self.assertEqual(interview.current_question_index, 1)
        self.assertEqual(interview.followups_on_question, 0)
        self.assertEqual(interview.turns[-1].content, "明白。接下来进入下一题：题目2")

    async def test_candidate_answer_and_followup_reference_current_question_primary_key(self) -> None:
        interview = await self._answer(index=0, followups=0, question_count=1)
        question_id = "q-0"
        self.assertEqual(interview.turns[0].question_id, question_id)
        self.assertEqual(interview.turns[-1].question_id, question_id)

    async def test_same_question_response_returns_the_new_interviewer_followup(self) -> None:
        interview = await self._answer(
            index=0,
            followups=0,
            question_count=2,
            followup=AsyncMock(return_value="你为什么选择这个方案？"),
        )
        self.assertEqual(interview.current_question_index, 0)
        self.assertEqual(interview.turns[-1].role, "interviewer")
        self.assertEqual(interview.turns[-1].content, "你为什么选择这个方案？")
        self.assertEqual(interview.turns[0].question_id, "q-0")

    async def test_failed_followup_keeps_answer_and_allows_question_to_advance(self) -> None:
        async def fail(*_args, **_kwargs):
            raise TimeoutError("provider timeout")

        interview = await self._answer(index=1, followups=0, question_count=3, followup=fail)
        self.assertEqual(interview.turns[0].role, "user")
        self.assertEqual(interview.turns[0].content, "我的回答")
        self.assertEqual(interview.turns[-1].content, "请再补充这道题的关键依据、具体步骤或边界情况。")
        self.assertEqual(interview.followups_on_question, 1)

    async def test_followup_timeout_is_bounded(self) -> None:
        from app.services.interviews.interview import followup_line

        async def hang(*_args, **_kwargs):
            import asyncio

            await asyncio.sleep(30)

        with patch("app.services.interviews.interview.FOLLOWUP_TIMEOUT_SECONDS", 0.01), patch(
            "app.services.interviews.interview.llm.llm_available", return_value=True
        ), patch(
            "app.services.interviews.interview.interviewer_followup", new=hang
        ):
            result = await followup_line(
                SimpleNamespace(), "iv-1", SimpleNamespace(stem="缓存设计"), "使用分布式锁", None, []
            )
        self.assertEqual(result, "刚才的回答还偏概括。请给出具体阈值、失败案例和兜底策略。")

    async def test_final_question_produces_finish_guidance(self) -> None:
        interview = await self._answer(index=0, followups=1, question_count=1)
        self.assertEqual(interview.current_question_index, 0)
        self.assertIn("本套问题已完成", interview.turns[-1].content)

    async def test_duplicate_submission_is_rejected_while_previous_answer_waits(self) -> None:
        interview = SimpleNamespace(status="live", turns=[SimpleNamespace(role="user", cite="answer_pending", content="已提交")])
        with patch("app.services.interviews.interview.get_interview", return_value=interview):
            with self.assertRaisesRegex(ValueError, "仍在处理中"):
                await answer_interview(SimpleNamespace(), "iv-1", "重复提交", "text")

    async def test_session_without_questions_still_accepts_answers(self) -> None:
        interview = await self._answer(index=0, followups=0, question_count=0)
        self.assertEqual(interview.turns[0].role, "user")
        self.assertTrue(interview.turns[-1].content)

    async def test_answer_route_converts_persistence_failure_to_retryable_error(self) -> None:
        """A failed commit must rollback the session and return a user-safe 503."""
        from app.api.interviews.interview import answer_interview as answer_route
        from app.schemas.business.interview import InterviewAnswerIn

        db = SimpleNamespace(rollback=__import__("unittest").mock.Mock())
        failure = RuntimeError("database connection dropped")
        # Isolate the route's retry envelope from the graph's SQL preflight.
        with patch("app.api.interviews.interview.services.pending_answer_run_id", return_value=None), patch(
            "app.api.interviews.interview.run_live_answer_graph", new=AsyncMock(side_effect=failure)
        ):
            with self.assertRaises(HTTPException) as raised:
                await answer_route("iv-1", InterviewAnswerIn(content="文字回答"), db)

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail, "回答暂时未提交，请稍后重试")
        self.assertGreaterEqual(db.rollback.call_count, 1)


if __name__ == "__main__":
    unittest.main()
