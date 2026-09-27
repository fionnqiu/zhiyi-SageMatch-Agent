"""Typed request and graph state shared by every Agent Loop boundary.

The application graph keeps this object deliberately boring: it is a JSON-safe
snapshot of orchestration, while SQL rows remain the source of truth for chat,
interview, question-set and report facts.  Keeping the two concerns separate is
what makes a checkpoint useful after a process restart without duplicating
business records in the graph store.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


RouteName = Literal["knowledge_qa", "interview_generation", "clarification", "unsupported"]
GraphStatus = Literal[
    "running",
    "waiting",
    "committed",
    "success",
    "degraded",
    "clarification",
    "unsupported",
    "failed",
]


class RouteDecision(BaseModel):
    """One stable business route reused by streaming, execution, and persistence."""

    route: RouteName
    confidence: float = 0.0
    source: str = "unknown"
    original_query: str = ""
    reason_code: str = ""

    model_config = ConfigDict(extra="forbid")


class RetrievalResult(BaseModel):
    """Structured RAG output kept separate from the prompt text sent to an LLM."""

    original_query: str = ""
    search_queries: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    selected_chunks: list[dict[str, Any]] = Field(default_factory=list)
    context_text: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_status: Literal["ok", "empty", "degraded", "failed"] = "empty"
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Expose stable counts even when a caller only supplies selected chunks."""
        self.diagnostics.setdefault("candidate_count", len(self.candidates))
        self.diagnostics.setdefault("selected_count", len(self.selected_chunks))


class AgentState(BaseModel):
    """JSON-serializable state for the application graph and its subgraphs.

    ``db`` and live model objects intentionally do not belong here.  Runtime
    dependencies are supplied to node callables; putting them in a checkpoint
    would make recovery impossible and could persist credentials accidentally.
    """

    # This must match CheckpointPolicy.schema_version so the same snapshot can
    # pass the owner-aware store before LangGraph persists its channels.
    schema_version: str = "agent-state.v1"
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    thread_id: str = ""
    user_id: str = "local-user"
    tenant_id: str = "local"
    session_id: str | None = None
    interview_id: str | None = None
    entrypoint: str = "chat"
    route: RouteDecision | None = None
    route_confidence: float = 0.0
    original_query: str = ""
    normalized_query: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    memory_snapshot: dict[str, Any] = Field(default_factory=dict)
    memory_refs: list[dict[str, Any]] = Field(default_factory=list)
    plan: list[dict[str, Any]] = Field(default_factory=list)
    current_node: str = ""
    active_agent: str = ""
    completed_nodes: list[str] = Field(default_factory=list)
    loop_count: int = 0
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    selected_chunks: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    question_set_id: str | None = None
    report_id: str | None = None
    decision: dict[str, Any] = Field(default_factory=dict)
    # Each dispatch retains its task and worker decision across checkpoint recovery.
    handoffs: list[dict[str, Any]] = Field(default_factory=list)
    next_action: str = ""
    status: GraphStatus = "running"
    error: dict[str, Any] | None = None
    retry_count: int = 0
    repair_count: int = 0
    replan_count: int = 0
    max_graph_steps: int = 32
    max_agent_steps: int = 8
    max_retries: int = 2
    max_repairs: int = 1
    max_replans: int = 1
    deadline_at: datetime | None = None
    token_budget: int | None = None
    cost_budget: float | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")

    def can_continue(self, now: datetime | None = None) -> bool:
        """Return whether this run still has graph and deadline budget."""
        if self.status in {"committed", "success", "degraded", "clarification", "unsupported", "failed"}:
            return False
        if self.loop_count >= self.max_graph_steps:
            return False
        if self.deadline_at is not None and now is not None and now >= self.deadline_at:
            return False
        return True

    def as_checkpoint(self) -> dict[str, Any]:
        """Return only JSON-safe state; callers may persist this verbatim."""
        return self.model_dump(mode="json")


class RequestContext(BaseModel):
    """Request-scoped envelope used to correlate route, Agent, tool, and RAG work."""

    request_id: str
    run_id: str
    thread_id: str = ""
    user_id: str = "local-user"
    tenant_id: str = "local"
    session_id: str | None = None
    interview_id: str | None = None
    entrypoint: str = "chat"
    original_query: str = ""
    normalized_query: str = ""
    route: RouteDecision | None = None
    retrieval: RetrievalResult | None = None
    deadline_at: datetime | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class AgentResult(BaseModel):
    """Stable result envelope for role, supervisor, and tool boundaries."""

    ok: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    retryable: bool = False
    trace_id: str = ""
    observations: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def failure(cls, error_code: str, *, retryable: bool, trace_id: str) -> "AgentResult":
        return cls(ok=False, error_code=error_code, retryable=retryable, trace_id=trace_id)


async def classify_request(db: Any, query: str, **kwargs: Any) -> dict[str, Any]:
    """Adapt the existing perception service to the explicit route contract."""
    from app.services.chat.intent import resolve_intent

    return await resolve_intent(db, query, **kwargs)


def _route_from_intent(intent: dict[str, Any], query: str) -> RouteDecision:
    """Normalize legacy intent values without adding another business mode."""
    explicit = str(intent.get("route") or intent.get("mode") or "")
    explicit = {
        "answer": "knowledge_qa",
        "knowledge": "knowledge_qa",
        "interview": "interview_generation",
        "generate_interview": "interview_generation",
        "clarify": "clarification",
    }.get(explicit, explicit)
    if explicit in {"knowledge_qa", "interview_generation", "clarification", "unsupported"}:
        route = explicit
    else:
        route = {
            "generate_interview": "interview_generation",
            "clarify": "clarification",
            "answer": "knowledge_qa",
        }.get(str(intent.get("intent") or "answer"), "unsupported")
    confidence = max(0.0, min(1.0, float(intent.get("confidence") or 0.0)))
    source = str(intent.get("source") or "legacy")
    # A low-confidence classifier result is a clarification, but explicit API
    # modes and unmistakable interview requests remain authoritative.
    if not explicit and "confidence" in intent and confidence < 0.45 and route == "knowledge_qa" and query.strip():
        route = "clarification"
        source = f"{source}:low-confidence"
    return RouteDecision(
        route=route,
        confidence=confidence,
        source=source,
        original_query=query,
        reason_code=str(intent.get("reason_code") or intent.get("intent") or route),
    )


async def route_node(state: dict[str, Any]) -> dict[str, Any]:
    """Compute one route decision for all later stream and execution nodes."""
    query = str(state.get("original_query") or "")
    existing = state.get("route")
    if existing is not None:
        route = existing if isinstance(existing, RouteDecision) else RouteDecision.model_validate(existing)
        return {"route": route, "route_confidence": route.confidence, "intent": state.get("intent") or {}}
    intent = state.get("intent")
    if not isinstance(intent, dict):
        # An API mode is authoritative and must not incur a classifier call.
        intent = await classify_request(state["db"], query, mode=state.get("mode"))
    route = _route_from_intent(intent, query)
    return {"route": route, "route_confidence": route.confidence, "intent": intent}
