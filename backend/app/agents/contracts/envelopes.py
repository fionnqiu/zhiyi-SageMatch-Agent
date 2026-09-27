"""Typed communication envelopes shared by the application graph and workers.

The existing role agents still expose dictionaries for backwards compatibility.  These
models make the graph boundary explicit: a worker can propose a decision and evidence,
while only graph/commit nodes are allowed to turn that proposal into a side effect.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


# The public API has four routes; live interview and report generation are internal
# execution modes selected by an entrypoint/requested_mode and share the same graph.
RouteName = Literal[
    "knowledge_qa",
    "interview_generation",
    "clarification",
    "unsupported",
    "live_interview",
    "evaluation_report",
]

TerminalAction = Literal[
    "continue",
    "retry",
    "repair",
    "clarification",
    "unsupported",
    "persist",
    "fail",
]


class _Envelope(BaseModel):
    """Base configuration that rejects accidental protocol field drift."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ErrorEnvelope(_Envelope):
    """Stable error information with a safe user-facing serialization boundary."""

    error_code: str
    retryable: bool = False
    user_message: str = "请求暂时无法完成，请稍后重试"
    # The detail is retained for logs/checkpoint diagnostics but never belongs in a
    # response assembled from ``public_dict``.
    internal_detail: str | None = None
    trace_id: str = ""
    node: str | None = None

    @field_validator("error_code", "user_message")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("error fields must not be empty")
        return value

    def public_dict(self) -> dict[str, Any]:
        """Return only fields safe to expose to a caller or SSE event."""
        return self.model_dump(exclude={"internal_detail"})


class AgentTask(_Envelope):
    """A bounded handoff from one role/node to another role/node."""

    task_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    parent_task_id: str | None = None
    from_agent: str
    to_agent: str
    route: RouteName
    goal: str
    input_refs: list[str] = Field(default_factory=list)
    deadline_at: datetime | None = None
    attempt: int = Field(default=1, ge=1)
    trace_id: str = ""

    @field_validator("task_id", "run_id", "from_agent", "to_agent", "goal")
    @classmethod
    def _required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("task identity fields must not be empty")
        return value


class ToolResult(_Envelope):
    """Result of a tool invocation, including retry and trace metadata."""

    tool_call_id: str = Field(default_factory=lambda: uuid4().hex)
    tool_name: str
    ok: bool
    data: Any = None
    error_code: str | None = None
    retryable: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)
    cached: bool = False
    trace_id: str = ""

    @field_validator("tool_name")
    @classmethod
    def _tool_name_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("tool_name must not be empty")
        return value


class AgentDecision(_Envelope):
    """Structured worker output; it proposes the next action without committing facts."""

    task_id: str
    agent: str
    status: Literal["success", "waiting", "degraded", "failed", "rejected"]
    decision: str
    output: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    next_action: TerminalAction | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    trace_id: str = ""


class AgentEvent(_Envelope):
    """Append-only event emitted at every graph/worker boundary."""

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    node: str
    agent: str | None = None
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "agent-event.v1"
    trace_id: str = ""

    @field_validator("event_type", "node", "run_id")
    @classmethod
    def _event_identity_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("event identity fields must not be empty")
        return value

    @field_validator("timestamp")
    @classmethod
    def _timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


__all__ = [
    "AgentDecision",
    "AgentEvent",
    "AgentTask",
    "ErrorEnvelope",
    "RouteName",
    "TerminalAction",
    "ToolResult",
]
