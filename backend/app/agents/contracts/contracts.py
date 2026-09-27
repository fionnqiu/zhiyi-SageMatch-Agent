"""Frozen role contracts.

A contract says what a role may decide, which tools it may call, and when it
must stop. Provider and model stay in RoleBinding — this file never names a vendor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _RolePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleInput(_RolePayload):
    """Bound the task text while keeping retrieved seed data as context."""

    goal: str
    context: str = ""
    seed: dict[str, Any] = Field(default_factory=dict)


class AnalystInput(RoleInput):
    pass


class AuthorInput(RoleInput):
    pass


class CriticInput(RoleInput):
    pass


class InterviewerInput(RoleInput):
    pass


class ScorerInput(RoleInput):
    pass


class CoachInput(RoleInput):
    pass


class JudgeInput(RoleInput):
    pass


class AnalystOutput(_RolePayload):
    text: str


class AuthorQuestion(_RolePayload):
    kind: Literal["open", "scenario"] | None = None
    stem: str
    answer: str
    explanation: str
    options: list[Any] = Field(default_factory=list)


class AuthorOutput(_RolePayload):
    job_title: str | None = None
    summary: str | None = None
    focus: list[str] = Field(default_factory=list)
    reply: str | None = None
    questions: list[AuthorQuestion]


class CriticOutput(_RolePayload):
    passed: bool = Field(alias="pass")
    reason: str = ""


class InterviewerOutput(_RolePayload):
    text: str


class ScoreDimension(_RolePayload):
    score: float
    evidence: str
    advice: str


class ScoreDimensions(_RolePayload):
    technical_ability: ScoreDimension
    problem_analysis: ScoreDimension
    solution_tradeoffs: ScoreDimension
    communication: ScoreDimension


class ScorerOutput(_RolePayload):
    dimensions: ScoreDimensions


class CoachIssue(_RolePayload):
    issue: str
    quote: str
    advice: str


class CoachOutput(_RolePayload):
    review: str
    summary: str
    issues: list[CoachIssue]


class JudgeOutput(_RolePayload):
    text: str


ROLE_OUTPUTS: dict[str, type[_RolePayload]] = {
    "analyst": AnalystOutput,
    "author": AuthorOutput,
    "critic": CriticOutput,
    "interviewer": InterviewerOutput,
    "scorer": ScorerOutput,
    "coach": CoachOutput,
    "judge": JudgeOutput,
}

ROLE_INPUTS: dict[str, type[RoleInput]] = {
    "analyst": AnalystInput,
    "author": AuthorInput,
    "critic": CriticInput,
    "interviewer": InterviewerInput,
    "scorer": ScorerInput,
    "coach": CoachInput,
    "judge": JudgeInput,
}


def validate_role_input(role: str, payload: dict[str, Any]) -> RoleInput:
    """Validate the shared task wire shape with its role-specific model."""
    schema = ROLE_INPUTS.get(role)
    if schema is None:
        raise ValueError(f"unknown role: {role}")
    return schema.model_validate(payload)


def validate_role_output(role: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Reject fields outside a role's decision before any business commit."""
    schema = ROLE_OUTPUTS.get(role)
    if schema is None:
        raise ValueError(f"unknown role: {role}")
    return schema.model_validate(payload).model_dump(by_alias=True, exclude_unset=True)


@dataclass(frozen=True)
class AgentProfile:
    """One role's boundary. Frozen so a prompt edit cannot silently widen tool access."""

    role: str
    mission: str
    # Autonomous roles run the tool loop. Modules are a single structured call.
    autonomous: bool
    tool_scope: tuple[str, ...]
    max_steps: int
    temperature: float
    max_tokens: int
    # Used when the bound provider is open or missing.
    fallback_role: str | None = None


# Order matches the admin role table. speech is a capability binding, not an agent.
PROFILES: dict[str, AgentProfile] = {
    "analyst": AgentProfile(
        role="analyst",
        mission="把岗位描述整理成考察要点，或直接回答知识问题。不发起工具循环。",
        autonomous=False,
        tool_scope=(),
        max_steps=1,
        temperature=0.2,
        max_tokens=700,
        fallback_role="author",
    ),
    "author": AgentProfile(
        role="author",
        mission="按岗位描述出问答题。实时对话，不出选择题。题量按职责覆盖，不固定 5 道。可以检索、自检题干、查重，然后结束。",
        autonomous=True,
        tool_scope=("hybrid_search", "validate_question", "check_duplicate", "finish"),
        max_steps=3,
        temperature=0.4,
        # 8 到 12 道混合题的 JSON 放不进 1800。留出解析和参考要点。
        max_tokens=3600,
        fallback_role="analyst",
    ),
    "critic": AgentProfile(
        role="critic",
        mission="只判断题目是否重复、是否缺解析。不改写题目。",
        autonomous=True,
        tool_scope=("validate_question", "check_duplicate", "finish"),
        max_steps=2,
        temperature=0.0,
        max_tokens=600,
        fallback_role="author",
    ),
    "interviewer": AgentProfile(
        role="interviewer",
        mission="基于候选人原话追问。在线只能引用历史原话，不查知识库。",
        autonomous=True,
        tool_scope=("get_turn_quote", "finish"),
        max_steps=1,
        temperature=0.5,
        max_tokens=240,
        fallback_role="analyst",
    ),
    "scorer": AgentProfile(
        role="scorer",
        mission="给技术能力、问题分析、方案权衡、表达沟通四项分数，并为每项提供证据和建议。不写给用户看的复盘。",
        autonomous=False,
        tool_scope=(),
        max_steps=1,
        temperature=0.0,
        # Four dimensions each carry evidence and advice; 400 tokens truncates the JSON object.
        max_tokens=1600,
    ),
    "coach": AgentProfile(
        role="coach",
        mission="读已冻结的分数，写复盘文字。不再改分。",
        autonomous=False,
        tool_scope=(),
        max_steps=1,
        temperature=0.3,
        max_tokens=1200,
        fallback_role="analyst",
    ),
    "judge": AgentProfile(
        role="judge",
        mission="复用业务链路计算指标，不另写一套出题结果。",
        autonomous=False,
        tool_scope=("check_duplicate",),
        max_steps=1,
        temperature=0.0,
        max_tokens=400,
        fallback_role="coach",
    ),
}


def profile_for(role: str) -> AgentProfile:
    """Return the contract, or a locked-down analyst if the role is unknown."""
    return PROFILES.get(role, PROFILES["analyst"])
