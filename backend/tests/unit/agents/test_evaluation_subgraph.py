"""Verify scorer output is validated before coaching in the evaluation graph."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agents.orchestration.checkpoint import MemoryCheckpointer, checkpoint_config
from app.agents.workflows.evaluation import EvaluationPipeline, run_evaluation
from app.agents.orchestration.graph import AgentState, build_application_graph
from app.agents.orchestration.workflows import run_business_graph
from app.core.db import Base
from app.models.platform.runtime import GraphCheckpointOwner
from app.services.interviews.interview import _restore_report_pipeline


class EvaluationSubgraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupted_report_rebuilds_pipeline_before_checkpoint_resume(self) -> None:
        """A resumed commit has a freshly computed score after process state is lost."""
        class Interrupted(BaseException):
            pass

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine, tables=[GraphCheckpointOwner.__table__])
        saver = MemoryCheckpointer()
        thread_id = "report:job-1:attempt:1"
        run_id = "report-resume-1"
        calls: list[str] = []

        def pipeline() -> EvaluationPipeline:
            async def scorer(_title, _transcript):
                calls.append("scorer")
                return {"dimensions": {"technical_ability": {"score": 20}}}

            async def coach(_title, _transcript, _score, _dimensions):
                calls.append("coach")
                return {"review": "依据", "summary": "总结", "issues": []}

            return EvaluationPipeline(
                "场次", [{"role": "user", "content": "回答"}], scorer=scorer,
                validate_score=lambda candidate, _: candidate["dimensions"], coach=coach,
                fallback_coach=lambda _: {"review": "保底", "summary": "保底", "issues": []},
                invalid_report=lambda: {"scoring_status": "invalid"},
            )

        first = pipeline()

        def callbacks(current: EvaluationPipeline, *, interrupt: bool = False):
            async def stage(_state, name):
                if interrupt and name == "report_validate":
                    raise Interrupted()
                await current.run_stage(name)
                return {"result": {"valid": True}}

            return {
                f"evaluation_report.{name}": (lambda state, selected=name: stage(state, selected))
                for name in current.stages
            }

        async def commit() -> str:
            self.assertEqual(resumed.report["score"], 20)
            calls.append("commit")
            return "saved"

        with Session(engine) as db:
            with self.assertRaises(Interrupted):
                await run_business_graph(
                    db, mode="evaluation_report", interview_id="interview-1",
                    thread_id=thread_id, run_id=run_id, checkpointer=saver,
                    stages=callbacks(first, interrupt=True), action=commit,
                )
        snapshot = await build_application_graph(checkpointer=saver).compiled.aget_state(
            checkpoint_config(thread_id=thread_id, owner_id="local-user")
        )
        self.assertTrue(snapshot.next)
        resumed = pipeline()
        await _restore_report_pipeline(saver, thread_id, run_id, resumed)
        self.assertEqual(calls.count("scorer"), 2)
        with patch("app.agents.orchestration.workflows.persist_graph_trace"):
            with Session(engine) as db:
                result = await run_business_graph(
                    db, mode="evaluation_report", interview_id="interview-1",
                    thread_id=thread_id, run_id=run_id, checkpointer=saver,
                    stages=callbacks(resumed), action=commit,
                )
        self.assertEqual(result, "saved")
        self.assertEqual(calls.count("commit"), 1)
        self.assertEqual(calls.count("scorer"), 2)
        engine.dispose()

    async def test_application_graph_runs_each_evaluation_stage_before_commit(self) -> None:
        """A report writer sees frozen scores only after the six named nodes run."""
        calls: list[str] = []

        async def scorer(_title, _transcript):
            calls.append("scorer")
            return {"dimensions": {"technical_ability": {"score": 20}}}

        def validate(candidate, _transcript):
            calls.append("validate")
            return candidate["dimensions"]

        async def coach(_title, _transcript, score, _dimensions):
            calls.append("coach")
            self.assertEqual(score, 20)
            return {"score": 100, "review": "依据充分", "summary": "通过", "issues": []}

        pipeline = EvaluationPipeline(
            "场次", [{"role": "user", "content": "回答"}], scorer=scorer,
            validate_score=validate, coach=coach,
            fallback_coach=lambda _: {"review": "保底", "summary": "保底", "issues": []},
            invalid_report=lambda: {"scoring_status": "invalid"},
        )

        async def stage(_state, name):
            calls.append(name)
            await pipeline.run_stage(name)
            return {"result": {"valid": True}}

        async def commit(state):
            calls.append(state.current_node)
            self.assertEqual(pipeline.report["score"], 20)
            return {"status": "committed", "result": {"valid": True}}

        callbacks = {
            f"evaluation_report.{name}": (lambda state, current=name: stage(state, current))
            for name in pipeline.stages
        }
        outcome = await build_application_graph(stages=callbacks, commit=commit).ainvoke(
            AgentState(requested_mode="evaluation_report", interview_id="interview-1")
        )
        self.assertEqual(outcome["status"], "committed")
        self.assertEqual(calls, [
            "load_transcript", "scorer", "scorer", "score_schema_validate", "validate",
            "score_freeze", "coach", "coach", "report_validate", "commit_node",
        ])
        self.assertNotIn("score", pipeline.report["_coach"])
        self.assertEqual(sum(event["event_type"] == "business_committed" for event in outcome["events"]), 1)

    async def test_valid_dimensions_are_frozen_before_coach(self) -> None:
        seen: list[str] = []

        async def scorer(_title, _transcript):
            seen.append("scorer")
            return {"dimensions": {"technical_ability": {"score": 20}}}

        def validate(candidate, _transcript):
            seen.append("validate")
            self.assertEqual(candidate["dimensions"]["technical_ability"]["score"], 20)
            return {"technical_ability": {"score": 20}}

        async def coach(_title, _transcript, score, _dimensions):
            seen.append("coach")
            self.assertEqual(score, 20)
            return {"score": 100, "review": "依据充分", "summary": "通过", "issues": []}

        report = await run_evaluation("场次", [{"role": "user", "content": "回答"}], scorer=scorer,
                                      validate_score=validate, coach=coach,
                                      fallback_coach=lambda _: {"review": "保底", "summary": "保底", "issues": []},
                                      invalid_report=lambda: {"scoring_status": "invalid"})
        self.assertEqual(seen, ["scorer", "validate", "coach"])
        self.assertEqual(report["score"], 20)
        self.assertNotIn("score", report["_coach"])

    async def test_invalid_score_skips_coach(self) -> None:
        coach = AsyncMock()

        def reject(_candidate, _transcript):
            raise ValueError("missing dimension")

        report = await run_evaluation("场次", [], scorer=AsyncMock(return_value={"dimensions": {}}),
                                      validate_score=reject, coach=coach,
                                      fallback_coach=lambda _: {}, invalid_report=lambda: {"scoring_status": "invalid"})
        self.assertEqual(report["scoring_status"], "invalid")
        coach.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
