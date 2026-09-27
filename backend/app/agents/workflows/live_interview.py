"""Production stage callbacks for one live interview answer command."""

from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy.orm import Session

from app.agents.orchestration.workflows import run_business_graph
from app.models import Interview, InterviewTurn
from app.services.interviews import interview as interview_service


class LiveInterviewStages:
    """Keep answer text and ORM facts outside the checkpoint payload."""

    def __init__(self, db: Session, interview_id: str, content: str, answer_mode: str, run_id: str | None) -> None:
        self.db = db
        self.interview_id = interview_id
        self.content = content
        self.answer_mode = answer_mode
        self.run_id = run_id
        self.progress: interview_service.AnswerProgress | None = None
        self.followup = ""
        self.error: Exception | None = None

    async def load_interview(self, _state) -> dict[str, Any]:
        interview = self.db.get(Interview, self.interview_id)
        return {"diagnostics": {"interview_found": interview is not None,
                                "interview_status": interview.status if interview else "missing"}}

    async def ask_question(self, _state) -> dict[str, Any]:
        # The opening or preceding interviewer turn is already a business fact.
        return {"diagnostics": {"question_source": "interview_turn"}}

    async def wait_for_answer(self, _state) -> dict[str, Any]:
        return {"diagnostics": {"answer_received": bool(self.content.strip()),
                                "answer_sha256": hashlib.sha256(self.content.encode("utf-8")).hexdigest()}}

    async def persist_answer(self, _state) -> dict[str, Any]:
        # This node proposes the write; the first graph's commit_node owns it.
        return {"result": {"valid": bool(self.content.strip()), "retryable": False,
                           "error_code": "answer_empty" if not self.content.strip() else None},
                "diagnostics": {"answer_commit_ready": bool(self.content.strip())}}

    def commit_answer(self) -> interview_service.AnswerProgress:
        """Persist the candidate fact before any provider-dependent stage runs."""
        try:
            self.progress = interview_service.persist_interview_answer(
                self.db, self.interview_id, self.content, self.answer_mode, run_id=self.run_id,
            )
        except Exception as exc:
            self.db.rollback()
            self.error = exc
            raise
        return self.progress

    async def decide_followup(self, _state) -> dict[str, Any]:
        assert self.progress is not None
        return {"diagnostics": {"advance_question": self.progress.advance,
                                "question_id": self.progress.question_id}}

    async def interviewer_followup(self, _state) -> dict[str, Any]:
        assert self.progress is not None
        self.followup = await interview_service.propose_interview_followup(
            self.db, self.progress, self.content,
        )
        return {"diagnostics": {"followup_prepared": bool(self.followup)}}

    async def advance_question(self, _state) -> dict[str, Any]:
        # Progress was committed with the answer, ahead of provider work.
        assert self.progress is not None
        return {"diagnostics": {"question_progress_committed": True,
                                "current_question_index": self.progress.interview.current_question_index}}

    async def finish_or_continue(self, _state) -> dict[str, Any]:
        assert self.progress is not None
        return {"result": {"valid": True, "interview_id": self.interview_id},
                "diagnostics": {"interview_status": self.progress.interview.status}}

    def callbacks(self) -> dict[str, Any]:
        prefix = "live_interview."
        return {prefix + name: getattr(self, name) for name in (
            "load_interview", "ask_question", "wait_for_answer", "persist_answer",
            "decide_followup", "interviewer_followup", "advance_question", "finish_or_continue",
        )}

    def commit(self) -> Interview:
        """Finalize only after the graph has observed and validated the proposal."""
        if self.progress is None:
            if self.error is not None:
                raise self.error
            raise RuntimeError("answer was not persisted")
        return interview_service.finalize_interview_answer(self.db, self.progress, self.followup)


async def run_live_answer_graph(db: Session, interview_id: str, content: str,
                                answer_mode: str, run_id: str, *, checkpointer=None) -> Interview:
    """Use one graph run with an answer commit before provider follow-up."""
    if not content.strip():
        raise ValueError("请先输入或说出回答")
    stages = LiveInterviewStages(db, interview_id, content, answer_mode, run_id)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    existing = db.get(InterviewTurn, run_id)
    if existing is not None and (existing.interview_id != interview_id or existing.role != "user"
                                 or existing.content != content or existing.answer_mode != answer_mode):
        raise ValueError("同一请求标识对应另一条回答")
    if existing is not None and existing.cite != "answer_pending":
        interview = interview_service.get_interview(db, interview_id)
        if interview is None:
            raise ValueError("面试不存在")
        return interview

    async def persist():
        return stages.commit_answer()

    async def finalize():
        return stages.commit()

    return await run_business_graph(
        db, mode="live_interview", interview_id=interview_id,
        original_query=digest, run_id=run_id, action=finalize,
        checkpointer=checkpointer, stages=stages.callbacks(),
        first_commit=persist, first_commit_after="persist_answer",
        restart_incomplete=True, restart_completed_failed=True,
    )


class LiveInterviewCommandStages:
    """Read-only command proposal; the service rechecks it under a row lock."""

    STAGE_NAMES = ("load_interview", "validate_command", "decide_command")

    def __init__(self, db: Session, command: str, *, interview_id: str | None = None,
                 session_id: str | None = None, question_set_id: str | None = None) -> None:
        self.db = db
        self.command = command
        self.interview_id = interview_id
        self.session_id = session_id
        self.question_set_id = question_set_id
        self.found = False
        self.status = "missing"
        self.precheck_passed = False

    async def load_interview(self, _state) -> dict[str, Any]:
        if self.command == "create":
            qset = interview_service.resolve_question_set(
                self.db, self.session_id, self.question_set_id,
            )
            self.found = qset is not None
            self.status = "question_set_ready" if self.found else "missing"
        else:
            interview = interview_service.get_interview(self.db, self.interview_id)
            self.found = interview is not None
            self.status = interview.status if interview else "missing"
        return {"diagnostics": {"command": self.command, "interview_found": self.found,
                                "interview_status": self.status}}

    async def validate_command(self, _state) -> dict[str, Any]:
        # This is advisory: the service owns validation after acquiring its
        # serialization lock, including a concurrent transition since load.
        allowed = {
            "create": {"question_set_ready"},
            "resume": {"ready", "live", "abandoned"},
            "end": {"live", "ended"},
            "abandon": {"ready", "live", "abandoned", "ended"},
        }[self.command]
        self.precheck_passed = self.status in allowed
        return {"diagnostics": {"command_precheck_passed": self.precheck_passed}}

    async def decide_command(self, _state) -> dict[str, Any]:
        # The graph validates a compact proposal; no business fact is written
        # until commit_node invokes the existing transactional service.
        return {"result": {"valid": self.precheck_passed, "retryable": False,
                           "error_code": "interview_command_invalid" if not self.precheck_passed else None,
                           "command": self.command},
                "diagnostics": {"commit_requires_service_validation": True}}

    def callbacks(self) -> dict[str, Any]:
        return {f"live_interview.{name}": getattr(self, name) for name in self.STAGE_NAMES}

    def sequences(self) -> dict[str, tuple[str, ...]]:
        return {"live_interview": self.STAGE_NAMES}
