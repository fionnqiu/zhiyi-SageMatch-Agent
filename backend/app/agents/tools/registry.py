"""Tools an autonomous role may call. Handlers do the work; the loop only dispatches.

hybrid_search is for authoring. The interviewer is not in its allow-list — live
turns quote earlier speech instead of looking up the knowledge base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sqlalchemy.orm import Session

from app.materials import knowledge

Handler = Callable[[Session, dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    """Tool contract shared by local validation and provider function calling."""

    name: str
    description: str
    required: tuple[str, ...]
    cache_ttl: float
    handler: Handler
    properties: dict[str, dict[str, Any]] | None = None
    # The metadata is part of governance rather than prompt text.  Defaults
    # preserve the original six-argument registry declarations while allowing
    # the application graph to reject unsafe side effects before execution.
    args_schema: dict[str, Any] | None = None
    result_schema: dict[str, Any] | None = None
    permission: str = "agent"
    side_effect: str = "read"
    timeout_seconds: float = 10.0
    retry_policy: dict[str, Any] = field(default_factory=lambda: {"max_attempts": 1})
    idempotency_key_builder: Callable[[dict[str, Any]], str] | None = None
    cache_scope: str = "user"
    sensitive_fields: tuple[str, ...] = ()

    def schema(self) -> dict[str, Any]:
        """Return the OpenAI function schema from the same contract used locally."""
        parameters = self.args_schema or {
            "type": "object",
            "properties": self.properties or {key: {"type": "string"} for key in self.required},
            "required": list(self.required),
            "additionalProperties": False,
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def idempotency_key(self, args: dict[str, Any]) -> str:
        """Build a stable key for a side-effecting call without persisting secrets."""
        if self.idempotency_key_builder is not None:
            return self.idempotency_key_builder(args)
        import hashlib
        import json

        safe = {key: value for key, value in args.items() if key not in self.sensitive_fields}
        raw = json.dumps({"tool": self.name, "args": safe}, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def hybrid_search(db: Session, args: dict[str, Any]) -> dict[str, Any]:
    # Imported here so agents.tools can load before the service package finishes.
    from app.services.materials.recall import recall_snippets

    query = str(args.get("query") or "").strip()
    hits = await recall_snippets(db, query) if query else []
    compact = [
        {"filename": hit.get("filename"), "text": str(hit.get("text") or "")[:240], "score": hit.get("score")}
        for hit in hits[:4]
    ]
    return {"success": True, "hits": compact}


async def validate_question(db: Session, args: dict[str, Any]) -> dict[str, Any]:
    """Deterministic checks only. Semantic quality stays with the critic role."""
    del db
    stem = str(args.get("stem") or "").strip()
    kind = str(args.get("kind") or "open").strip()
    options = args.get("options") or []
    problems: list[str] = []
    if len(stem) < 8:
        problems.append("题干过短")
    if kind not in {"open", "scenario"}:
        problems.append("只能是问答题")
    if isinstance(options, list) and options:
        problems.append("问答题不能带选项")
    return {"success": not problems, "problems": problems}


async def check_duplicate(db: Session, args: dict[str, Any]) -> dict[str, Any]:
    del db
    stems = [str(item) for item in args.get("stems") or [] if str(item).strip()]
    dup = 0
    for i, left in enumerate(stems):
        # 2-grams, not the 2-character dictionary cut, so a repeated sentence still overlaps.
        left_terms = _shingles(left)
        for right in stems[i + 1 :]:
            right_terms = _shingles(right)
            if not left_terms or not right_terms:
                continue
            if len(left_terms & right_terms) / len(left_terms | right_terms) > 0.6:
                dup += 1
    pairs = len(stems) * (len(stems) - 1) / 2
    rate = round(dup / pairs, 4) if pairs else 0.0
    return {"success": True, "duplicate_rate": rate, "pairs": int(pairs)}


async def get_turn_quote(db: Session, args: dict[str, Any]) -> dict[str, Any]:
    """Return one earlier utterance. The interviewer uses this instead of retrieval."""
    del db
    turns = list(args.get("turns") or [])
    needle = str(args.get("query") or "").strip()
    if needle:
        for turn in reversed(turns):
            content = str(turn.get("content") or "")
            if needle in content:
                return {"success": True, "quote": content[:180], "role": turn.get("role")}
    if turns:
        last = turns[-1]
        return {"success": True, "quote": str(last.get("content") or "")[:180], "role": last.get("role")}
    return {"success": False, "quote": "", "error": "没有可引用的原话"}


async def finish_tool(db: Session, args: dict[str, Any]) -> dict[str, Any]:
    """Stop the loop. The payload is the role's decision."""
    del db
    return {"success": True, "final": True, "payload": args}


TOOLS: dict[str, ToolSpec] = {
    "hybrid_search": ToolSpec("hybrid_search", "按查询检索知识片段", ("query",), 120.0, hybrid_search, {"query": {"type": "string"}}),
    "validate_question": ToolSpec("validate_question", "检查题干和选项是否齐", ("stem",), 0.0, validate_question, {"stem": {"type": "string"}, "kind": {"type": "string", "enum": ["open", "scenario"]}, "options": {"type": "array", "items": {"type": "string"}}}),
    "check_duplicate": ToolSpec("check_duplicate", "计算题干之间的重复率", ("stems",), 0.0, check_duplicate, {"stems": {"type": "array", "items": {"type": "string"}}}),
    "get_turn_quote": ToolSpec("get_turn_quote", "引用候选人说过的原话", (), 0.0, get_turn_quote, {"turns": {"type": "array", "items": {"type": "object"}}, "query": {"type": "string"}}),
    # finish intentionally allows role-specific payload keys; LangGraph still enforces the role allow-list.
    # finish is the only role tool that proposes a business result; the
    # application graph still performs the actual database commit separately.
    "finish": ToolSpec(
        "finish",
        "结束循环并提交决定",
        (),
        0.0,
        finish_tool,
        {"text": {"type": "string"}, "questions": {"type": "array", "items": {"type": "object"}}},
        permission="commit",
        side_effect="proposal",
        cache_scope="run",
    ),
}


def tools_for(scope: tuple[str, ...]) -> dict[str, ToolSpec]:
    return {name: TOOLS[name] for name in scope if name in TOOLS}


def tool_schemas_for(scope: tuple[str, ...], *, role: str | None = None) -> list[dict[str, Any]]:
    """Give finish the same fields the role output validator will require."""
    from app.agents.contracts.contracts import ROLE_OUTPUTS

    schemas = [spec.schema() for spec in tools_for(scope).values()]
    if role in ROLE_OUTPUTS:
        for schema in schemas:
            if schema["function"]["name"] == "finish":
                schema["function"]["parameters"] = ROLE_OUTPUTS[role].model_json_schema(by_alias=True)
    return schemas


def _shingles(text: str) -> set[str]:
    chars = [ch for ch in text.lower() if not ch.isspace()]
    if len(chars) < 2:
        return set(chars)
    return {"".join(chars[i : i + 2]) for i in range(len(chars) - 1)} | knowledge.terms(text)
