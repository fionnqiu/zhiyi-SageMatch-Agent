"""Quality-eval table for admin question-generation and scoring-consistency runs."""

from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class EvalRun(Base):
    """One admin quality-eval job (question generation or scoring consistency)."""

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(40))  # question | score
    status: Mapped[str] = mapped_column(String(20), default="done")
    input_text: Mapped[str] = mapped_column(Text, default="")
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
