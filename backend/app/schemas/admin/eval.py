"""Request and response shapes for admin quality-eval runs."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class EvalRunOut(BaseModel):
    id: str
    kind: str
    status: str
    input_text: str
    metrics: dict[str, Any] | None = None
    detail: dict[str, Any] | None = None
    created_at: datetime


class EvalQuestionIn(BaseModel):
    job_text: str


class EvalScoreIn(BaseModel):
    interview_id: str | None = None
    repeats: int = 5
