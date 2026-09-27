"""结束复盘要先停表再返回。直接退出不写报告。复盘在后台补上。"""

from __future__ import annotations

import inspect
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.api.interviews.interview import regenerate_interview_report
from app.services.interviews.interview import abandon_interview, build_report, finish_interview, regenerate_report, resume_or_start


class _Interview:
    def __init__(self) -> None:
        self.id = "iv-1"
        self.title = "后端 · 全真模拟面试"
        self.status = "live"
        self.started_at = datetime.now(timezone.utc) - timedelta(seconds=90)
        self.ended_at = None
        self.elapsed_seconds = 0
        self.summary = ""
        self.turns = [SimpleNamespace(role="user", content="加了锁")]
        self.report = None


class InterviewExitTests(unittest.IsolatedAsyncioTestCase):
    def test_regenerate_report_route_is_async_for_queue_scheduling(self) -> None:
        """The report queue needs the event loop owned by an async FastAPI route."""
        self.assertTrue(inspect.iscoroutinefunction(regenerate_interview_report))

    def test_regenerate_report_removes_old_report_and_resets_summary(self) -> None:
        """The temporary test action must replace the old report and keep the transcript."""
        interview = _Interview()
        interview.status = "ended"
        interview.summary = "旧复盘"
        old_report = SimpleNamespace(id="report-1")
        interview.report = old_report
        deleted: list[object] = []
        db = SimpleNamespace(delete=deleted.append, flush=lambda: None, commit=lambda: None)

        with patch("app.services.interviews.interview.get_interview", side_effect=[interview, interview]):
            result = regenerate_report(db, "iv-1")

        self.assertEqual(deleted, [old_report])
        self.assertIsNone(result.report)
        self.assertEqual(result.summary, "正在重新生成复盘")
        self.assertEqual(result.turns[0].content, "加了锁")

    async def test_finish_stops_the_clock_before_the_report_exists(self) -> None:
        interview = _Interview()
        saved: dict[str, object] = {}

        def commit() -> None:
            saved["status"] = interview.status
            saved["elapsed"] = interview.elapsed_seconds
            saved["has_report"] = interview.report is not None

        db = SimpleNamespace(commit=commit, add=lambda _row: None)

        with patch("app.services.interviews.interview.get_interview", return_value=interview):
            result = await finish_interview(db, "iv-1")

        self.assertEqual(result.status, "ended")
        self.assertGreaterEqual(saved["elapsed"], 90)
        # 停表提交时报告还不存在。复盘是提交之后才开始写的。
        self.assertFalse(saved["has_report"])
        self.assertIsNone(result.report)

    def test_abandon_returns_to_ready_without_a_report(self) -> None:
        interview = _Interview()
        db = SimpleNamespace(commit=lambda: None, add=lambda _row: None)

        with patch("app.services.interviews.interview.get_interview", return_value=interview):
            result = abandon_interview(db, "iv-1")

        self.assertEqual(result.status, "ready")
        self.assertEqual(result.summary, "面试尚未开始")
        self.assertIsNone(result.report)
        self.assertIsNone(result.started_at)
        self.assertIsNone(result.ended_at)
        self.assertEqual(result.elapsed_seconds, 0)
        self.assertEqual(result.current_question_index, 0)
        self.assertEqual(result.followups_on_question, 0)
        self.assertEqual(result.turns, [])

    def test_legacy_abandoned_exit_commits_ready_state(self) -> None:
        """A legacy abandoned row must not revert when the request session closes."""

        interview = _Interview()
        interview.status = "abandoned"
        commits: list[str] = []
        db = SimpleNamespace(commit=lambda: commits.append(interview.status))
        with patch("app.services.interviews.interview.get_interview", return_value=interview):
            result = abandon_interview(db, "iv-1")
        self.assertEqual(result.status, "ready")
        self.assertEqual(commits, ["ready"])

    async def test_ready_interview_can_be_started_again_after_exit(self) -> None:
        interview = _Interview()
        interview.status = "ready"
        interview.started_at = None
        interview.ended_at = None
        interview.turns = []
        qset = SimpleNamespace(questions=[])
        interview.question_set = qset
        db = SimpleNamespace(commit=lambda: None, add=lambda _row: None)

        with patch("app.services.interviews.interview.get_interview", return_value=interview):
            result = resume_or_start(db, "iv-1")

        self.assertEqual(result.status, "live")
        self.assertIsNotNone(result.started_at)
        self.assertEqual(result.turns, [])

    async def test_legacy_abandoned_interview_is_treated_as_ready(self) -> None:
        interview = _Interview()
        interview.status = "abandoned"
        interview.started_at = None
        interview.ended_at = None
        interview.turns = []
        interview.question_set = SimpleNamespace(questions=[])
        db = SimpleNamespace(commit=lambda: None, add=lambda _row: None)

        with patch("app.services.interviews.interview.get_interview", return_value=interview):
            result = resume_or_start(db, "iv-1")

        self.assertEqual(result.status, "live")


    async def test_report_job_runs_after_finish_returns(self) -> None:
        started = AsyncMock()

        async def slow_report(*_args, **_kwargs):
            await started()
            return {"score": 81, "review": "可以", "summary": "达到", "issues": []}

        with patch("app.services.interviews.interview.write_report", new=slow_report), patch(
            "app.services.interviews.interview._store_report"
        ) as store:
            await build_report(object(), "iv-1", "后端", [{"role": "user", "content": "加了锁"}])

        started.assert_awaited()
        store.assert_called_once()

    async def test_report_job_converts_model_failure_to_a_saved_fallback(self) -> None:
        """A worker failure must finish with a readable report, not endless polling."""
        fallback = {"score": 0.0, "review": "复盘服务暂时不可用，已保存保底结果。", "summary": "复盘生成失败，已使用保底结果。", "issues": [], "scoring_status": "unavailable"}

        db = object()
        with patch("app.services.interviews.interview.write_report", new=AsyncMock(side_effect=RuntimeError("provider down"))), patch(
            "app.services.interviews.interview.generation_failed_report", return_value=fallback
        ), patch("app.services.interviews.interview._store_report") as store:
            await build_report(db, "iv-1", "后端", [{"role": "user", "content": "加了锁"}])

        store.assert_called_once_with(db, "iv-1", fallback)
