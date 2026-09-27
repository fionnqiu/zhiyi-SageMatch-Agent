"""Interview tables: generated question sets, live turns, and the recap report."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.business.session import JobProfile


class QuestionSet(Base):
    """One generation run for a job profile."""

    __tablename__ = "question_sets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    profile_id: Mapped[str] = mapped_column(ForeignKey("job_profiles.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    profile: Mapped[JobProfile] = relationship(back_populates="question_sets")
    questions: Mapped[list["Question"]] = relationship(
        back_populates="question_set", cascade="all, delete-orphan", order_by="Question.ordinal"
    )
    interviews: Mapped[list["Interview"]] = relationship(back_populates="question_set")


class Question(Base):
    """A generated interview question; shown only when asked."""

    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    question_set_id: Mapped[str] = mapped_column(ForeignKey("question_sets.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    stem: Mapped[str] = mapped_column(Text)
    options: Mapped[list] = mapped_column(JSONB, default=lambda: [])
    # Open and scenario answers are a rubric, not a single option letter.
    answer: Mapped[str] = mapped_column(String(200), default="")
    explanation: Mapped[str] = mapped_column(Text, default="")
    generated_by: Mapped[str] = mapped_column(String(40), default="system")

    question_set: Mapped[QuestionSet] = relationship(back_populates="questions")


class Interview(Base):
    """One mock-interview session."""

    __tablename__ = "interviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    profile_id: Mapped[str | None] = mapped_column(ForeignKey("job_profiles.id"), nullable=True)
    question_set_id: Mapped[str | None] = mapped_column(ForeignKey("question_sets.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="ready")  # ready | live | ended
    current_question_index: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    elapsed_seconds: Mapped[int] = mapped_column(Integer, default=0)
    followups_on_question: Mapped[int] = mapped_column(Integer, default=0)
    tags: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    question_set: Mapped[QuestionSet | None] = relationship(back_populates="interviews")
    turns: Mapped[list["InterviewTurn"]] = relationship(
        back_populates="interview", cascade="all, delete-orphan", order_by="InterviewTurn.created_at"
    )
    report: Mapped["Report | None"] = relationship(back_populates="interview", uselist=False)


class InterviewTurn(Base):
    """One interviewer or candidate utterance."""

    __tablename__ = "interview_turns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    interview_id: Mapped[str] = mapped_column(ForeignKey("interviews.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # interviewer | user
    content: Mapped[str] = mapped_column(Text)
    answer_mode: Mapped[str | None] = mapped_column(String(20), nullable=True)  # text | voice
    cite: Mapped[str | None] = mapped_column(String(80), nullable=True)
    question_id: Mapped[str | None] = mapped_column(ForeignKey("questions.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    interview: Mapped[Interview] = relationship(back_populates="turns")


class Report(Base):
    """Overall score and evidence-backed dimension breakdown for one interview."""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    interview_id: Mapped[str] = mapped_column(ForeignKey("interviews.id"), unique=True)
    score: Mapped[float] = mapped_column(Float)
    review: Mapped[str] = mapped_column(Text)
    issues: Mapped[list] = mapped_column(JSONB, default=lambda: [])
    dimensions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    scoring_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    interview: Mapped[Interview] = relationship(back_populates="report")
