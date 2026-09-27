"""Classify a request into one business route without performing tool work."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy.orm import Session

from app.agents.roles.intent_fusion import pattern_intent
from app.services.operations.llm_gateway import complete

Recall = Callable[[str], Awaitable[list[dict[str, Any]]]]
History = Callable[[], str]
MODE_ALIASES = {
    "answer": "knowledge_qa", "knowledge": "knowledge_qa", "knowledge_qa": "knowledge_qa",
    "interview": "interview_generation", "generate_interview": "interview_generation",
    "interview_generation": "interview_generation", "clarify": "clarification",
    "clarification": "clarification", "unsupported": "unsupported",
}
INTERVIEW_REQUEST = re.compile(r"(?:模拟面试|生成.{0,16}面试题|出.{0,12}题|开始.{0,8}面试|面试.{0,8}开始)", re.I)


async def resolve_intent(
    db: Session, content: str, *, mode: str | None = None,
    recall: Recall | None = None, history: History | None = None,
) -> dict[str, Any]:
    """Choose a route once; legacy tool arguments are accepted but never run."""
    del recall, history
    query = content.strip()
    explicit = MODE_ALIASES.get(str(mode or "").strip().lower())
    if explicit:
        return _decision(explicit, query, 1.0, "explicit_mode", "explicit_mode")
    if INTERVIEW_REQUEST.search(query):
        return _decision("interview_generation", query, 0.98, "pattern", "interview_request")
    try:
        data = await complete(
            db, "analyst", "你是业务路由分类器。只输出 JSON，不调用工具、不回答问题。",
            _prompt(query), max_tokens=180, expect_json=True,
        )
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    route = MODE_ALIASES.get(str(data.get("route") or data.get("intent") or "").strip().lower())
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.8 if route else 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    if route and confidence >= 0.45:
        result = _decision(route, query, confidence, "llm", "classifier")
        labels = [str(item).strip()[:18] for item in data.get("todos", []) if str(item).strip()] if isinstance(data.get("todos"), list) else []
        result["todos"] = [{"id": f"intent-{index}", "label": label, "status": "complete"} for index, label in enumerate(labels[:4])]
        if route == "clarification":
            questions = _clarification_questions(data.get("questions"))
            reply = str(data.get("reply") or "").strip()
            if questions and reply:
                result.update(questions=questions, reply=reply[:200])
        return result
    # Weak classifier output can be corrected by a clear rule match.
    pattern, strength = pattern_intent(query)
    if strength >= 0.8:
        return _decision(MODE_ALIASES[pattern], query, strength, "rule", "pattern_match")
    return _decision("clarification", query, 0.0, "fallback", "low_confidence")


def _prompt(query: str) -> str:
    return (
        "固定路由仅有 knowledge_qa、interview_generation、clarification、unsupported。"
        "知识问题与问候走 knowledge_qa；明确要求出题或模拟面试走 interview_generation；"
        "需要补充关键信息走 clarification；无法处理才走 unsupported。"
        "只输出 route、confidence、reason_code；可选 todos、reply、questions。\n"
        f"用户：{query[:2000]}"
    )


def _decision(route: str, query: str, confidence: float, source: str, reason: str) -> dict[str, Any]:
    """Expose graph fields and compatibility fields for existing chat callers."""
    legacy = {"knowledge_qa": "answer", "interview_generation": "generate_interview", "clarification": "clarify", "unsupported": "answer"}
    return {
        "route": route, "confidence": confidence, "source": source,
        "original_query": query, "reason_code": reason,
        "intent": legacy[route], "needs_recall": route == "knowledge_qa",
        "todos": [], "actions": ["finish"], "observations": [],
    }


def _clarification_questions(raw: object) -> list[dict[str, Any]]:
    """Pass through only valid model-authored clarification choices."""
    if not isinstance(raw, list):
        return []
    questions: list[dict[str, Any]] = []
    for index, item in enumerate(raw[:2]):
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or "").strip()
        options = []
        for option_index, option in enumerate(item.get("options") or []):
            if isinstance(option, dict) and str(option.get("label") or "").strip():
                options.append({"id": str(option.get("id") or f"o{option_index + 1}"), "label": str(option["label"]).strip()[:24]})
        if prompt and len(options) >= 2:
            questions.append({"id": str(item.get("id") or f"q{index + 1}"), "prompt": prompt[:40], "options": options[:4]})
    return questions
