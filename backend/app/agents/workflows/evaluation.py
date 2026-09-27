"""Executable evaluation subgraph with deterministic score ownership."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph



class EvaluationState(TypedDict, total=False):
    title: str
    transcript: list[dict[str, str]]
    candidate: dict[str, Any]
    dimensions: dict[str, dict[str, Any]]
    score: float
    coach: dict[str, Any]
    report: dict[str, Any]
    scoring_status: str


class ReportRegenerationCommandStages:
    """Precheck one report command before its transactional service commit."""

    STAGE_NAMES = ("load_interview", "validate_command", "decide_command")

    def __init__(self, db: Any, interview_id: str) -> None:
        self.db = db
        self.interview_id = interview_id
        self.status = "missing"
        self.report_present = False
        self.precheck_passed = False

    async def load_interview(self, _state: Any) -> dict[str, Any]:
        # The service query applies the current owner filter. No transcript or
        # report body needs to enter the graph checkpoint for this command.
        from app.services.interviews import interview as interview_service

        interview = interview_service.get_interview(self.db, self.interview_id)
        self.status = interview.status if interview is not None else "missing"
        self.report_present = interview is not None and interview.report is not None
        return {"diagnostics": {"interview_status": self.status,
                                "report_present": self.report_present}}

    async def validate_command(self, _state: Any) -> dict[str, Any]:
        # The write service rechecks under a row lock after this advisory read.
        self.precheck_passed = self.status == "ended"
        return {"diagnostics": {"command_precheck_passed": self.precheck_passed}}

    async def decide_command(self, _state: Any) -> dict[str, Any]:
        return {"result": {"valid": self.precheck_passed, "retryable": False,
                           "error_code": "report_regeneration_invalid" if not self.precheck_passed else None,
                           "command": "regenerate_report"},
                "diagnostics": {"commit_requires_service_validation": True}}

    def callbacks(self) -> dict[str, Any]:
        return {f"evaluation_report.{name}": getattr(self, name) for name in self.STAGE_NAMES}

    def sequences(self) -> dict[str, tuple[str, ...]]:
        return {"evaluation_report": self.STAGE_NAMES}


class EvaluationPipeline:
    """Keep provider work in named stages and the frozen score in code-owned state."""

    stages = ("load_transcript", "scorer", "score_schema_validate", "score_freeze", "coach", "report_validate")

    def __init__(self, title: str, transcript: list[dict[str, str]], *,
                 scorer: Callable[[str, list[dict[str, str]]], Awaitable[dict[str, Any]]],
                 validate_score: Callable[[dict[str, Any], list[dict[str, str]]], dict[str, dict[str, Any]]],
                 coach: Callable[[str, list[dict[str, str]], float, dict[str, dict[str, Any]]], Awaitable[dict[str, Any]]],
                 fallback_coach: Callable[[dict[str, dict[str, Any]]], dict[str, Any]],
                 invalid_report: Callable[[], dict[str, Any]],
                 preset_report: dict[str, Any] | None = None) -> None:
        self.state: EvaluationState = {"title": title, "transcript": transcript}
        self.scorer = scorer
        self.validate_score = validate_score
        self.coach = coach
        self.fallback_coach = fallback_coach
        self.invalid_report = invalid_report
        self.preset_report = preset_report
        self.completed_stages: set[str] = set()

    async def run_stage(self, name: str) -> dict[str, Any]:
        """Execute one stage; provider failures become explicit report fallbacks."""
        state = self.state
        if name not in self.stages:
            raise ValueError(f"unknown evaluation stage: {name}")
        if name in self.completed_stages:
            return {"stage": name, "scoring_status": state.get("scoring_status", "pending")}
        if self.preset_report is not None:
            if name == "report_validate":
                state["report"] = self.preset_report
            self.completed_stages.add(name)
            return {"stage": name, "scoring_status": self.preset_report.get("scoring_status", "unavailable")}
        if name == "scorer":
            try:
                state["candidate"] = await self.scorer(state["title"], state["transcript"])
            except Exception:
                state["scoring_status"] = "invalid"
        elif name == "score_schema_validate" and state.get("scoring_status") != "invalid":
            try:
                state["dimensions"] = self.validate_score(state["candidate"], state["transcript"])
            except Exception:
                state["scoring_status"] = "invalid"
        elif name == "score_freeze":
            if state.get("scoring_status") == "invalid":
                state["report"] = self.invalid_report()
            else:
                # Only validated dimensions can set the final total.
                state["score"] = round(sum(item["score"] for item in state["dimensions"].values()), 1)
                state["scoring_status"] = "valid"
        elif name == "coach" and state.get("scoring_status") == "valid":
            try:
                prose = await self.coach(state["title"], state["transcript"], state["score"], state["dimensions"])
            except Exception:
                prose = self.fallback_coach(state["dimensions"])
            state["coach"] = {key: value for key, value in prose.items() if key != "score"}
        elif name == "report_validate" and state.get("scoring_status") == "valid":
            prose = state["coach"]
            if (not isinstance(prose.get("review"), str) or not prose["review"].strip()
                    or not isinstance(prose.get("summary"), str) or not prose["summary"].strip()
                    or not isinstance(prose.get("issues"), list)):
                prose = self.fallback_coach(state["dimensions"])
            state["report"] = {"score": state["score"], "dimensions": state["dimensions"],
                               "review": prose["review"], "summary": prose["summary"],
                               "issues": prose["issues"], "scoring_status": "valid", "_coach": prose}
        self.completed_stages.add(name)
        return {"stage": name, "scoring_status": state.get("scoring_status", "pending")}

    @property
    def report(self) -> dict[str, Any]:
        """Expose only a validated report after the final stage has run."""
        if "report" not in self.state:
            raise RuntimeError("evaluation report is not ready")
        return self.state["report"]


async def run_evaluation(
    title: str,
    transcript: list[dict[str, str]],
    *,
    scorer: Callable[[str, list[dict[str, str]]], Awaitable[dict[str, Any]]],
    validate_score: Callable[[dict[str, Any], list[dict[str, str]]], dict[str, dict[str, Any]]],
    coach: Callable[[str, list[dict[str, str]], float, dict[str, dict[str, Any]]], Awaitable[dict[str, Any]]],
    fallback_coach: Callable[[dict[str, dict[str, Any]]], dict[str, Any]],
    invalid_report: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run scorer, code-owned validation and freeze, then explanatory coaching."""
    pipeline = EvaluationPipeline(title, transcript, scorer=scorer, validate_score=validate_score,
                                  coach=coach, fallback_coach=fallback_coach, invalid_report=invalid_report)
    graph = StateGraph(EvaluationState)
    for name in pipeline.stages:
        async def run_stage(_state: EvaluationState, *, current: str = name) -> EvaluationState:
            await pipeline.run_stage(current)
            return dict(pipeline.state)
        graph.add_node(name, run_stage)
    graph.add_edge(START, pipeline.stages[0])
    for source, target in zip(pipeline.stages, pipeline.stages[1:]):
        graph.add_edge(source, target)
    graph.add_edge(pipeline.stages[-1], END)
    await graph.compile().ainvoke({"title": title, "transcript": transcript})
    return pipeline.report
