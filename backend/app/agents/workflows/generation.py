"""Stage callbacks for one bounded interview question-set generation run."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sqlalchemy.orm import Session

from app.integrations import llm
from app.agents.roles.authoring import author_question_candidate, critique_question_candidate
from app.agents.roles.memory import MemoryManager
from app.services.materials.recall import recall_snippets
from app.services.chat.session import stub_pack


def _candidate_error(candidate: dict[str, Any]) -> str:
    """Reject packs that the live interviewer cannot ask as spoken questions."""
    questions = candidate.get("questions")
    if not isinstance(questions, list) or not 8 <= len(questions) <= 12:
        return "题目数量须在 8 到 12 道之间"
    for item in questions:
        if not isinstance(item, dict) or item.get("kind") not in {"open", "scenario"}:
            return "题目类型不符合口头面试要求"
        if not all(str(item.get(key) or "").strip() for key in ("stem", "answer", "explanation")):
            return "题干、答案和解析必须完整"
        if item.get("options"):
            return "口头面试不能包含选项"
    return ""


def _compact_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    """Retain role evidence without persisting prompts, question text or tool data."""
    task = handoff.get("task") or {}
    decision = handoff.get("decision") or {}
    output = decision.get("output") or {}
    serialized = json.dumps(output, ensure_ascii=False, sort_keys=True, default=str)
    def safe_ref(value: Any) -> str:
        raw = str(value)
        return raw if re.fullmatch(r"[A-Za-z0-9_:-]{1,120}", raw) else hashlib.sha256(raw.encode("utf-8")).hexdigest()

    return {
        "task_id": str(task.get("task_id") or ""),
        "agent": str(decision.get("agent") or task.get("to_agent") or ""),
        "status": str(decision.get("status") or ""),
        "decision": str(decision.get("decision") or ""),
        "output_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "question_count": len(output.get("questions") or []) if isinstance(output, dict) else 0,
        "evidence_refs": [safe_ref(ref) for ref in (decision.get("evidence_refs") or [])[:8]],
        "observations": [
            {key: (observation.get(key) if key in {"ok", "cached"} else safe_ref(observation.get(key)))
             for key in ("tool_call_id", "tool", "tool_name", "ok", "error_code", "cached")
             if key in observation}
            for observation in (decision.get("observations") or [])[:12]
            if isinstance(observation, dict)
        ],
    }


class InterviewGenerationStages:
    """Keep raw JD, model candidates and recall text outside checkpoint state."""

    def __init__(self, db: Session, content: str, run_id: str, on_thought=None) -> None:
        self.db = db
        self.content = content.strip()
        self.run_id = run_id
        self.on_thought = on_thought
        self.hits: list[dict[str, Any]] = []
        self.context = ""
        self.candidate: dict[str, Any] = {}
        self.verdict: dict[str, Any] = {"pass": True, "reason": ""}
        self.validation_error = ""
        self.revisions = 0
        self.parent_task_id: str | None = None
        self.payload: dict[str, Any] = {}
        self.handoffs: list[dict[str, Any]] = []

    async def retrieve_context(self, _state) -> dict[str, Any]:
        self.hits = await recall_snippets(self.db, self.content)
        self.context = MemoryManager(self.db).render()
        if self.hits:
            self.context += "\n\n[预取知识]\n" + "\n".join(
                str(hit.get("text") or "")[:200] for hit in self.hits[:4]
            )
        return {"diagnostics": {"generation_recall_count": len(self.hits)}}

    async def load_job_profile(self, _state) -> dict[str, Any]:
        # The profile row is created at commit with the question set. This
        # stage validates the source text before any provider or recall work.
        if len(self.content) < 8:
            raise ValueError("请先写下岗位描述")
        return {"diagnostics": {"job_profile_source": "request"}}

    async def author(self, _state) -> dict[str, Any]:
        if not llm.llm_available():
            self.candidate = stub_pack(self.content)
            return {"diagnostics": {"generation_fallback": "provider_unavailable"}}
        result = await author_question_candidate(
            self.db, self.content, context=self.context, hits=self.hits,
            run_id=self.run_id, attempt=1, on_thought=self.on_thought,
        )
        self.candidate = dict(result.get("output") or {}) if result.get("ok") else {}
        if result.get("handoff"):
            self.handoffs.append(_compact_handoff(result["handoff"]))
        self.parent_task_id = (result.get("handoff") or {}).get("task", {}).get("task_id")
        return {"diagnostics": {"generation_author_ok": bool(self.candidate)}}

    async def deterministic_validate(self, _state) -> dict[str, Any]:
        self.validation_error = _candidate_error(self.candidate)
        return {"diagnostics": {"generation_candidate_valid": not self.validation_error}}

    async def critic(self, _state) -> dict[str, Any]:
        if self.validation_error or not llm.llm_available():
            self.verdict = {"pass": not self.validation_error, "reason": self.validation_error}
            return {}
        reviewed = await critique_question_candidate(
            self.db, self.candidate, run_id=self.run_id,
            parent_task_id=self.parent_task_id, on_thought=self.on_thought,
        )
        self.verdict = reviewed["verdict"]
        if reviewed.get("handoff"):
            self.handoffs.append(_compact_handoff(reviewed["handoff"]))
        return {"diagnostics": {"generation_critic_pass": self.verdict["pass"]}}

    async def revise_once(self, _state) -> dict[str, Any]:
        if self.verdict["pass"] or not llm.llm_available():
            return {}
        self.revisions = 1
        result = await author_question_candidate(
            self.db, self.content, context=self.context, hits=self.hits,
            run_id=self.run_id, attempt=2,
            objection=self.verdict["reason"] or "题目未通过质检",
            parent_task_id=self.parent_task_id, on_thought=self.on_thought,
        )
        self.candidate = dict(result.get("output") or {}) if result.get("ok") else {}
        if result.get("handoff"):
            self.handoffs.append(_compact_handoff(result["handoff"]))
        self.validation_error = _candidate_error(self.candidate)
        if not self.validation_error:
            reviewed = await critique_question_candidate(
                self.db, self.candidate, run_id=self.run_id,
                parent_task_id=(result.get("handoff") or {}).get("task", {}).get("task_id"),
                on_thought=self.on_thought,
            )
            self.verdict = reviewed["verdict"]
            if reviewed.get("handoff"):
                self.handoffs.append(_compact_handoff(reviewed["handoff"]))
        return {"diagnostics": {"generation_revisions": 1}}

    async def question_set_validate(self, _state) -> dict[str, Any]:
        self.validation_error = _candidate_error(self.candidate)
        if not self.validation_error and not self.verdict["pass"]:
            self.validation_error = self.verdict["reason"] or "题目未通过质检"
        # One rejected revision has a deterministic endpoint. The fallback is
        # deliberately explicit in diagnostics and never invokes another model.
        self.payload = stub_pack(self.content) if self.validation_error else self.candidate
        return {"result": {"valid": True, "question_count": len(self.payload["questions"])},
                "diagnostics": {"generation_fallback": self.validation_error or "",
                                "generation_revisions": self.revisions,
                                "generation_handoffs": self.handoffs}}

    def callbacks(self) -> dict[str, Any]:
        prefix = "interview_generation."
        return {prefix + name: getattr(self, name) for name in (
            "load_job_profile", "retrieve_context", "author", "deterministic_validate", "critic",
            "revise_once", "question_set_validate",
        )}
