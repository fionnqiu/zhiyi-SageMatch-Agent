"""Contracts, the tool loop, routing, governance, memory, and intent fusion."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from app.integrations import llm
from app.agents.roles.authoring import author_questions, review_pack
from app.agents.contracts.contracts import profile_for, validate_role_output
from app.agents.providers.governance import FAILURE_THRESHOLD, ProviderGovernor
from app.agents.roles.intent_fusion import fuse_intent, pattern_intent
from app.agents.roles.loop import _RoleModel, run_agent, run_team
from app.agents.roles.memory import MemoryManager
from app.agents.providers.router import choose_provider
from app.agents.tools.registry import check_duplicate, validate_question
from app.services.chat.intent import resolve_intent
from app.services.interviews.interview import SCORE_DIMENSIONS, _frozen_score, score_band, write_report


class ScoreEval(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_scoring_tracks_each_dimension_sigma(self) -> None:
        from app.services.interviews.eval import run_score_eval

        class Query:
            def join(self, *_args):
                # The score test's query double follows the owner-scoped read.
                return self

            def options(self, *_args):
                return self

            def filter(self, *_args):
                return self

            def order_by(self, *_args):
                return self

            def first(self):
                return interview

            def one_or_none(self):
                return interview

        class Run:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        interview = SimpleNamespace(
            id="iv-1",
            title="后端",
            turns=[SimpleNamespace(role="user", content="解释方案")],
        )
        dimensions = {
            key: {"score": (10 if index == 0 else 20), "evidence": "证据", "advice": "建议"}
            for index, key in enumerate(SCORE_DIMENSIONS)
        }
        added = []
        db = SimpleNamespace(
            query=lambda _model: Query(),
            add=added.append,
            commit=lambda: None,
            refresh=lambda _run: None,
        )
        with patch("app.services.interviews.eval.EvalRun", Run), patch("app.services.interviews.eval.audit"), patch("app.services.interviews.eval.write_report", new_callable=AsyncMock, side_effect=[
            {"score": 70, "dimensions": dimensions},
            {"score": 74, "dimensions": {**dimensions, "technical_ability": {**dimensions["technical_ability"], "score": 14}}},
        ]):
            run = await run_score_eval(db, "iv-1", repeats=2)

        self.assertEqual(added[0].metrics["sigma"], 2.0)
        self.assertEqual(added[0].metrics["dimension_sigma"]["technical_ability"], 2.0)
        self.assertEqual(added[0].metrics["dimension_sigma"]["problem_analysis"], 0.0)
        self.assertEqual(added[0].metrics["kendall_tau"], 1.0)
        self.assertEqual(added[0].metrics["band_consistency"], 1.0)

        varied = {**dimensions, "technical_ability": {**dimensions["technical_ability"], "score": 20}}
        with patch("app.services.interviews.eval.EvalRun", Run), patch("app.services.interviews.eval.audit"), patch("app.services.interviews.eval.write_report", new_callable=AsyncMock, side_effect=[
            {"score": 70, "dimensions": dimensions},
            {"score": 70, "dimensions": varied},
        ]):
            await run_score_eval(db, "iv-1", repeats=2)
        self.assertEqual(added[1].metrics["sigma"], 0.0)
        self.assertGreater(added[1].metrics["dimension_sigma"]["technical_ability"], 2)
        self.assertEqual(added[1].metrics["kendall_tau"], 0.0)

        with patch("app.services.interviews.eval.EvalRun", Run), patch("app.services.interviews.eval.audit"), patch("app.services.interviews.eval.write_report", new_callable=AsyncMock, side_effect=[
            {"score": 64, "dimensions": dimensions},
            {"score": 80, "dimensions": dimensions},
        ]):
            await run_score_eval(db, "iv-1", repeats=2)
        self.assertEqual(added[2].metrics["band_consistency"], 0.5)


class _Row:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _Db:
    """Just enough session for units that never touch SQL."""

    def __init__(self, providers: list | None = None, bindings: list | None = None) -> None:
        self.providers = list(providers or [])
        self.bindings = list(bindings or [])
        self.added: list = []
        self.health: dict = {}

    def add(self, row: object) -> None:
        self.added.append(row)
        if hasattr(row, "provider_id"):
            self.health[row.provider_id] = row

    def flush(self) -> None:
        return None

    def get(self, model: type, key: str) -> object | None:
        name = getattr(model, "__name__", "")
        if name == "ProviderConfig":
            return next((row for row in self.providers if row.id == key), None)
        if name == "ProviderHealth":
            return self.health.get(key)
        if name == "UserProfileMemory":
            return getattr(self, "profiles", {}).get(key)
        return None

    def query(self, model: type) -> "_Query":
        return _Query(self, model)


class _Query:
    def __init__(self, db: _Db, model: type) -> None:
        self.db = db
        self.model = model
        self._filters: list = []

    def filter(self, *args: object) -> "_Query":
        self._filters.extend(args)
        return self

    def order_by(self, *_args: object) -> "_Query":
        return self

    def all(self) -> list:
        if getattr(self.model, "__name__", "") == "ProviderConfig":
            return [row for row in self.db.providers if row.capability == "llm"]
        return []

    def one_or_none(self) -> object | None:
        if getattr(self.model, "__name__", "") == "RoleBinding":
            return self.db.bindings[0] if self.db.bindings else None
        return None

    def first(self) -> object | None:
        rows = self.all()
        return rows[0] if rows else None


class Contracts(unittest.TestCase):
    def test_finish_output_rejects_unknown_fields(self) -> None:
        """A model cannot smuggle undeclared fields into a role decision."""
        with self.assertRaises(ValueError):
            validate_role_output("interviewer", {"text": "继续", "unexpected": "hidden"})

    def test_role_outputs_preserve_declared_payloads(self) -> None:
        self.assertEqual(validate_role_output("critic", {"pass": True, "reason": ""}), {"pass": True, "reason": ""})
        self.assertEqual(validate_role_output("interviewer", {"text": "继续"}), {"text": "继续"})
        self.assertEqual(validate_role_output("coach", {"review": "复盘", "summary": "摘要", "issues": []})["review"], "复盘")
        with self.assertRaises(ValueError):
            validate_role_output("scorer", {"dimensions": ScoreFreeze._dimensions(), "score": 99})

    def test_interviewer_cannot_search(self) -> None:
        """Live turns quote speech. Retrieval would turn the interviewer into a grader."""
        profile = profile_for("interviewer")
        self.assertIn("get_turn_quote", profile.tool_scope)
        self.assertNotIn("hybrid_search", profile.tool_scope)
        self.assertEqual(profile.max_steps, 1)
        self.assertEqual(profile.max_tokens, 240)

    def test_author_budget_is_three_steps(self) -> None:
        profile = profile_for("author")
        self.assertTrue(profile.autonomous)
        self.assertEqual(profile.max_steps, 3)
        # Eight to twelve mixed questions do not fit the old 1800-token pack.
        self.assertGreaterEqual(profile.max_tokens, 3600)
        self.assertIn("hybrid_search", profile.tool_scope)

    def test_scorer_contract_matches_evidence_dimensions(self) -> None:
        """The default scorer contract must leave room for four evidence objects."""
        profile = profile_for("scorer")
        self.assertIn("四项", profile.mission)
        self.assertGreaterEqual(profile.max_tokens, 1600)


class ToolRules(unittest.IsolatedAsyncioTestCase):
    async def test_validate_and_duplicate_are_deterministic(self) -> None:
        bad = await validate_question(None, {"stem": "短", "kind": "open", "options": []})
        self.assertFalse(bad["success"])
        # Live interviews are spoken. A choice, or options on an open question, does not pass.
        spoken = await validate_question(None, {"stem": "你会如何拆分模型调用和检索？", "kind": "open"})
        self.assertTrue(spoken["success"])
        choice = await validate_question(
            None,
            {"stem": "两级缓存的主要收益是什么？", "kind": "choice", "options": [{"key": "A", "text": "降延迟"}]},
        )
        self.assertFalse(choice["success"])
        with_options = await validate_question(
            None, {"stem": "你会如何拆分模型调用和检索？", "kind": "open", "options": [{"key": "A", "text": "拆三层"}]}
        )
        self.assertFalse(with_options["success"])
        dup = await check_duplicate(None, {"stems": ["缓存击穿后如何回源", "缓存击穿后如何回源保护"]})
        self.assertGreater(dup["duplicate_rate"], 0)


class ToolLoop(unittest.IsolatedAsyncioTestCase):
    async def test_budget_error_does_not_retry_tool_call_as_json(self) -> None:
        """A hard usage limit must not trigger another model request."""
        model = _RoleModel(role="author")
        model._session = _Db()
        model._tools = []
        provider = AsyncMock(side_effect=llm.UsageBudgetError("budget exhausted"))
        with patch("app.services.operations.llm_gateway.complete_with", new=provider):
            with self.assertRaises(llm.UsageBudgetError):
                await model._complete([HumanMessage(content="出题")])
        self.assertEqual(provider.await_count, 1)

    async def test_interviewer_budget_error_is_not_an_offline_success(self) -> None:
        """Budget exhaustion cannot be presented as a completed follow-up."""
        provider = AsyncMock(side_effect=llm.UsageBudgetError("budget exhausted"))
        with patch("app.services.operations.llm_gateway.complete_with", new=provider):
            result = await run_agent(_Db(), profile_for("interviewer"), user="追问", seed={"turns": []})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "model_budget_exceeded")
        self.assertFalse(result["retryable"])
        self.assertEqual(provider.await_count, 1)

    async def test_supervisor_budget_error_is_nonretryable(self) -> None:
        """A team handoff cannot reclassify budget exhaustion as a vendor retry."""
        graph = SimpleNamespace(ainvoke=AsyncMock(side_effect=llm.UsageBudgetError("budget exhausted")))
        with patch("app.agents.roles.loop.build_supervisor", return_value=graph):
            result = await run_team(_Db(), ("author",), user="出题")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "model_budget_exceeded")
        self.assertFalse(result["retryable"])

    async def test_unknown_finish_field_fails_at_role_boundary(self) -> None:
        """Reject an uncontracted finish call before its decision enters a handoff."""
        class Graph:
            async def ainvoke(self, *_args, **_kwargs):
                return {"messages": [AIMessage(content="", tool_calls=[{
                    "name": "finish", "args": {"text": "继续", "unexpected": "hidden"}, "id": "bad-finish",
                }])]}

        with patch("app.agents.roles.loop.build_role_agent", return_value=Graph()):
            result = await run_agent(_Db(), profile_for("interviewer"), user="追问")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "invalid_role_output")

    async def test_supervisor_hands_authoring_to_the_named_roles(self) -> None:
        """The team graph is a LangGraph supervisor, not the retired step loop."""
        seen: list[tuple[str, ...]] = []

        def fake_supervisor(db, roles, seed=None, on_thought=None):
            del db, seed, on_thought
            seen.append(tuple(roles))

            class _Graph:
                async def ainvoke(self, state, config=None):
                    del state, config
                    return {"messages": []}

            return _Graph()

        with patch("app.agents.roles.loop.build_supervisor", new=fake_supervisor):
            result = await run_team(_Db(), ("author", "critic"), user="出题")
        self.assertEqual(seen, [("author", "critic")])
        self.assertTrue(result["ok"])

    async def test_unknown_tool_is_rejected_then_finish_works(self) -> None:
        model = AsyncMock(
            side_effect=[
                "请给出具体阈值。",
            ]
        )
        with patch("app.services.operations.llm_gateway.complete_with", new=model):
            result = await run_agent(_Db(), profile_for("interviewer"), user="追问", seed={"turns": []})
        self.assertTrue(result["ok"])
        self.assertEqual(result["output"]["text"], "请给出具体阈值。")
        self.assertEqual(result["observations"][0]["tool"], "finish")

    async def test_loop_stops_at_the_contract_budget(self) -> None:
        model = AsyncMock(return_value="先说明你选择方案的依据。")
        with patch("app.services.operations.llm_gateway.complete_with", new=model):
            result = await run_agent(
                _Db(), profile_for("interviewer"), user="追问", seed={"turns": [{"role": "user", "content": "加了锁"}]}
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["output"]["text"], "先说明你选择方案的依据。")
        self.assertEqual(model.await_count, 1)


class Routing(unittest.TestCase):
    def test_open_breaker_falls_through_to_fallback_role(self) -> None:
        sick = _Row(id="p1", name="主", capability="llm", created_at=1)
        spare = _Row(id="p2", name="备", capability="llm", created_at=2)
        binding = _Row(role="author", provider_id="p1", model="m", temperature=0.4)
        db = _Db([sick, spare], [binding])
        health = _Row(
            provider_id="p1",
            provider_name="主",
            state="open",
            consecutive_fails=3,
            total=3,
            success=0,
            total_ms=0,
            opened_at=10**12,
        )
        db.health["p1"] = health
        choice = choose_provider(db, "author")
        self.assertEqual(choice.provider.id, "p2")
        self.assertTrue(choice.degraded)
        self.assertIn("analyst", choice.reason)

    def test_three_failures_open_the_breaker(self) -> None:
        db = _Db()
        governor = ProviderGovernor(db, "p1", "主")
        for _ in range(FAILURE_THRESHOLD):
            governor.record_failure(100)
        self.assertEqual(db.health["p1"].state, "open")
        self.assertFalse(governor.allow())


class MemoryLayers(unittest.TestCase):
    def test_working_memory_keeps_only_the_latest_turns(self) -> None:
        turns = [_Row(role="user", content=str(i)) for i in range(8)]
        memory = MemoryManager(_Db(), interview_id="iv1")
        recent = memory.working(turns)
        self.assertEqual([item["content"] for item in recent], ["3", "4", "5", "6", "7"])

    def test_episode_slot_advances_after_one_followup(self) -> None:
        db = _Db()
        db.query = lambda *_args, **_kwargs: SimpleNamespace(filter=lambda *_a, **_k: SimpleNamespace(one_or_none=lambda: None))  # type: ignore[method-assign]
        memory = MemoryManager(db, interview_id="iv1")
        slots = memory.advance_interview_slot(question_index=0, quote="加了分布式锁", followups_on_question=1)
        self.assertEqual(slots["followups_on_question"], 1)
        self.assertTrue(any(getattr(row, "scope", "") == "interview" for row in db.added))


def _question(stem: str) -> dict:
    return {
        "stem": stem,
        "options": [{"key": "A", "text": "只让一个线程回源"}, {"key": "B", "text": "锁能保证强一致"}],
        "answer": "A",
        "explanation": "互斥锁限制的是并发回源，不是一致性方案。",
    }


class CriticGate(unittest.IsolatedAsyncioTestCase):
    async def test_first_pass_ends_graph_without_revision(self) -> None:
        calls: list[str] = []

        async def fake_run(_db, profile, **_kwargs):
            calls.append(profile.role)
            if profile.role == "critic":
                return {"ok": True, "output": {"pass": True, "reason": ""}}
            return {
                "ok": True, "output": {"questions": [_question("缓存击穿如何处理")]},
                "steps": 1, "observations": [{"tool": "hybrid_search", "ok": True, "data": {"chunk_id": "c1"}}],
                "evidence_refs": ["c1"],
            }

        with patch("app.agents.roles.loop.run_agent", new=fake_run), patch(
            "app.agents.roles.authoring.MemoryManager", return_value=SimpleNamespace(render=lambda **_k: "", remember_profile=lambda *_a, **_k: None, update_episode=lambda **_k: None)
        ):
            result = await author_questions(object(), "后端岗位")
        self.assertEqual(calls, ["author", "critic"])
        self.assertEqual(result["_agent"]["verdict"], "passed")
        handoffs = result["_agent"]["handoffs"]
        self.assertEqual([item["task"]["to_agent"] for item in handoffs], ["author", "critic"])
        self.assertEqual(handoffs[1]["task"]["parent_task_id"], handoffs[0]["task"]["task_id"])
        self.assertEqual(handoffs[0]["decision"]["output"]["questions"], result["questions"])
        self.assertEqual(handoffs[0]["decision"]["evidence_refs"], ["c1"])
        self.assertEqual(handoffs[0]["decision"]["tool_results"][0]["data"], {"chunk_id": "c1"})

    async def test_failed_author_skips_critic(self) -> None:
        calls: list[str] = []

        async def fake_run(_db, profile, **_kwargs):
            calls.append(profile.role)
            return {"ok": False, "output": {}, "steps": 1, "observations": ["provider unavailable"]}

        with patch("app.agents.roles.loop.run_agent", new=fake_run), patch(
            "app.agents.roles.authoring.MemoryManager", return_value=SimpleNamespace(render=lambda **_k: "", remember_profile=lambda *_a, **_k: None, update_episode=lambda **_k: None)
        ):
            result = await author_questions(object(), "后端岗位")
        self.assertEqual(calls, ["author"])
        self.assertFalse(result["_agent"]["ok"])

    async def test_critic_rejects_without_rewriting_and_author_gets_one_more_try(self) -> None:
        """A veto names the problem. It never returns a replacement question."""
        rejected = {"questions": [_question("缓存"), _question("缓存")]}
        accepted = {
            "job_title": "后端",
            "summary": "围绕缓存",
            "focus": ["击穿"],
            "reply": "已出题",
            "questions": [_question("缓存击穿时为什么用互斥锁"), _question("看门狗为什么要续期")],
        }

        calls: list[str] = []

        async def fake_run(_db, profile, **_kwargs):
            role = profile.role
            calls.append(role)
            if role == "critic":
                return {"ok": True, "output": {"pass": False, "reason": "题干重复", "questions": [_question("被改写的题")]}}
            return {"ok": True, "output": rejected if calls.count("author") == 1 else accepted, "steps": 1, "observations": []}

        with patch("app.agents.roles.loop.run_agent", new=fake_run), patch(
            "app.agents.roles.authoring.MemoryManager", return_value=SimpleNamespace(render=lambda **_k: "", remember_profile=lambda *_a, **_k: None, update_episode=lambda **_k: None)
        ):
            result = await author_questions(object(), "后端岗位")
        self.assertEqual(result["questions"], accepted["questions"])
        self.assertEqual(result["_agent"]["verdict"], "rejected")
        self.assertEqual(result["_agent"]["reason"], "题干重复")
        self.assertEqual(calls, ["author", "critic", "author", "critic"])

    async def test_second_rejection_does_not_trigger_a_third_author_pass(self) -> None:
        async def fake_run(_db, profile, **_kwargs):
            role = profile.role
            fake_run.roles.append(role)
            if role == "author":
                return {"ok": True, "output": {"questions": [_question("缓存击穿时为什么用互斥锁")]}, "steps": 1, "observations": []}
            return {"ok": True, "output": {"pass": False, "reason": "缺解析"}}

        fake_run.roles = []
        with patch("app.agents.roles.loop.run_agent", new=fake_run), patch(
            "app.agents.roles.authoring.MemoryManager", return_value=SimpleNamespace(render=lambda **_k: "", remember_profile=lambda *_a, **_k: None, update_episode=lambda **_k: None)
        ):
            await author_questions(object(), "后端岗位")
        # The explicit workflow performs two author attempts, each followed by one critic verdict.
        self.assertEqual(fake_run.roles, ["author", "critic", "author", "critic"])

    def test_critic_payload_cannot_carry_questions(self) -> None:
        verdict = review_pack({"pass": False, "reason": "题干重复", "questions": [_question("替换题")]})
        self.assertNotIn("questions", verdict)
        self.assertFalse(verdict["pass"])


class ScoreFreeze(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _dimensions(score: float = 20) -> dict:
        return {
            key: {"score": score, "evidence": "候选人说明了处理步骤。", "advice": "补充边界条件。"}
            for key in SCORE_DIMENSIONS
        }

    async def test_coach_prose_cannot_replace_the_frozen_score(self) -> None:
        """The code sums scorer dimensions, and coach prose cannot replace that total."""

        async def fake_complete(_db, role, _system, _user, **_kwargs):
            if role == "scorer":
                return {"dimensions": self._dimensions()}
            if role == "coach":
                return {"score": 99, "review": "表述清楚，但缺少阈值。", "summary": "达到进一步", "issues": []}
            raise AssertionError(role)

        with patch("app.services.interviews.interview.complete", new=fake_complete), patch("app.integrations.llm.llm_available", return_value=True):
            report = await write_report(object(), "后端场次", [{"role": "user", "content": "加了锁"}])
        self.assertEqual(report["score"], 80)
        self.assertEqual(set(report["dimensions"]), set(SCORE_DIMENSIONS))
        self.assertIn("复盘文字暂未生成", report["review"])
        self.assertEqual(report["scoring_status"], "valid")
        self.assertNotIn("score", report["_coach"])

    async def test_coach_failure_keeps_frozen_score_and_uses_local_review(self) -> None:
        async def scorer_then_fail_coach(_db, role, _system, _user, **_kwargs):
            if role == "scorer":
                return {"dimensions": self._dimensions()}
            raise TimeoutError("coach timed out")

        with patch("app.services.interviews.interview.complete", new=scorer_then_fail_coach), patch("app.integrations.llm.llm_available", return_value=True):
            report = await write_report(object(), "后端场次", [{"role": "user", "content": "加了锁"}])
        # The scorer already produced four valid dimensions. A coach outage must
        # not erase that evidence or turn a valid score into a misleading zero.
        self.assertEqual(report["score"], 80)
        self.assertEqual(report["scoring_status"], "valid")
        self.assertIn("复盘文字暂未生成", report["review"])
        self.assertEqual(report["issues"], [])

    async def test_coach_payload_must_contain_parseable_review_fields(self) -> None:
        async def scorer_then_incomplete_coach(_db, role, _system, _user, **_kwargs):
            if role == "scorer":
                return {"dimensions": self._dimensions()}
            return {"review": "", "summary": "", "issues": "not-a-list"}

        with patch("app.services.interviews.interview.complete", new=scorer_then_incomplete_coach), patch("app.integrations.llm.llm_available", return_value=True):
            report = await write_report(object(), "后端场次", [{"role": "user", "content": "加了锁"}])
        self.assertEqual(report["scoring_status"], "valid")
        self.assertIn("复盘文字暂未生成", report["review"])
        self.assertEqual(report["score"], 80)

    async def test_dimension_scores_must_be_complete_and_in_range(self) -> None:
        async def missing(_db, _role, _system, _user, **_kwargs):
            return {"dimensions": {"technical_ability": {"score": 26, "evidence": "x", "advice": "y"}}}

        with patch("app.services.interviews.interview.complete", new=missing):
            with self.assertRaisesRegex(ValueError, "dimensions"):
                await _frozen_score(object(), "后端场次", [{"role": "user", "content": "加了锁"}])

        async def out_of_range(_db, _role, _system, _user, **_kwargs):
            dimensions = self._dimensions()
            dimensions["technical_ability"]["score"] = 26
            return {"dimensions": dimensions}

        with patch("app.services.interviews.interview.complete", new=out_of_range):
            with self.assertRaisesRegex(ValueError, "invalid score or evidence"):
                await _frozen_score(object(), "后端场次", [{"role": "user", "content": "加了锁"}])

    async def test_invalid_score_falls_back_to_unscored_report(self) -> None:
        async def fail(_db, _role, _system, _user, **_kwargs):
            raise ValueError("bad model response")

        with patch("app.services.interviews.interview.complete", new=fail), patch("app.integrations.llm.llm_available", return_value=True):
            report = await write_report(object(), "后端场次", [{"role": "user", "content": "加了锁"}])
        self.assertEqual(report["score"], 0)
        self.assertIn("有效面试评估", report["summary"])
        self.assertEqual(report["scoring_status"], "invalid")

    async def test_unanswered_dimensions_are_zero_and_never_reach_hiring_line(self) -> None:
        async def fail_if_called(*_args, **_kwargs):
            raise AssertionError("empty interviews must not ask the model for a score")

        transcript = [{"role": "interviewer", "content": "请介绍一下你的项目。"}]
        with patch("app.services.interviews.interview.complete", new=fail_if_called), patch("app.integrations.llm.llm_available", return_value=True):
            report = await write_report(object(), "空场次", transcript)
        self.assertLessEqual(report["score"], 20)
        self.assertIn("未作答", report["summary"])
        self.assertTrue(all(item["score"] == 0 for item in report["dimensions"].values()))

    def test_score_bands_keep_the_existing_recommendation_line(self) -> None:
        self.assertEqual(score_band(80), "达到建议线")
        self.assertEqual(score_band(65), "接近建议线")
        self.assertEqual(score_band(64.9), "尚未达到建议线")


class IntentFusion(unittest.IsolatedAsyncioTestCase):
    def test_pattern_and_embedding_can_override_a_wrong_label(self) -> None:
        """A greeting the model calls a job description is pulled back to answer."""
        decision = {"intent": "generate_interview", "needs_recall": False, "todos": [], "actions": ["finish"], "source": "agent"}
        fused = fuse_intent("你好", decision, embedding_scores={"answer": 0.9, "generate_interview": 0.1})
        self.assertEqual(fused["intent"], "answer")
        self.assertIn("pattern", fused["source_scores"])

    def test_agreement_keeps_the_model_choice(self) -> None:
        decision = {"intent": "generate_interview", "source": "agent", "todos": [], "actions": ["finish"]}
        fused = fuse_intent("这是一份后端岗位 JD，请出题", decision)
        self.assertEqual(pattern_intent("这是一份后端岗位 JD，请出题")[0], "generate_interview")
        self.assertEqual(fused["intent"], "generate_interview")

    async def test_resolve_intent_returns_fused_decision(self) -> None:
        with patch(
            "app.services.chat.intent.complete",
            new=AsyncMock(return_value={"action": "finish", "intent": "generate_interview", "needs_recall": False, "todos": ["出题"]}),
        ):
            result = await resolve_intent(
                object(),
                "你好",
                recall=AsyncMock(),
            )
        # Routing accepts the classifier label without running the old fusion
        # lanes; retrieval belongs to the RAG business subgraph.
        self.assertEqual(result["route"], "interview_generation")
        self.assertEqual(result["source"], "llm")
        self.assertNotIn("source_scores", result)
