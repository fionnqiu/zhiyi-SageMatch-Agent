"""Request and response shapes for the mock interview and its recap."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class QuestionOut(BaseModel):
    id: str
    ordinal: int
    stem: str
    options: list[dict[str, str]]
    explanation: str | None = None
    generated_by: str = "system"


class InterviewTurnOut(BaseModel):
    id: str
    role: Literal["interviewer", "user"]
    content: str
    answer_mode: str | None = None
    cite: str | None = None
    question_id: str | None = None
    created_at: datetime


class ReportIssue(BaseModel):
    issue: str
    quote: str
    advice: str


class ReportDimension(BaseModel):
    score: float
    evidence: str
    advice: str


class ReportOut(BaseModel):
    id: str
    score: float
    review: str
    issues: list[ReportIssue]
    dimensions: dict[str, ReportDimension] | None = None
    scoring_status: Literal["valid", "unavailable", "invalid", "legacy"] = "legacy"
    created_at: datetime


class InterviewOut(BaseModel):
    id: str
    title: str
    status: str
    current_question_index: int = 0
    started_at: datetime | None = None
    ended_at: datetime | None = None
    elapsed_seconds: int = 0
    tags: list[str] = Field(default_factory=list)
    summary: str | None = None
    score: float | None = None
    created_at: datetime
    current_question: QuestionOut | None = None
    report: ReportOut | None = None


class InterviewDetail(InterviewOut):
    turns: list[InterviewTurnOut] = Field(default_factory=list)


class InterviewAnswerIn(BaseModel):
    content: str
    answer_mode: Literal["text", "voice"] = "text"


class InterviewCreateIn(BaseModel):
    session_id: str | None = None
    question_set_id: str | None = None


class InterviewGenerateIn(BaseModel):
    """面试页单独出题。岗位描述不经过会话。"""

    content: str = ""
