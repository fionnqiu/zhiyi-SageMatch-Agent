"""Session cockpit tables: conversation, messages, and the job profile extracted from them."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class ChatSession(Base):
    """One conversation in the session cockpit."""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(200), default="新会话")
    job_title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # chat 是历史对话。interview 是创建面试时的内部会话，不进历史列表。
    origin: Mapped[str] = mapped_column(String(20), default="chat")
    user_id: Mapped[str] = mapped_column(String(64), default="local-user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )
    profiles: Mapped[list["JobProfile"]] = relationship(back_populates="session")


class ChatMessage(Base):
    """A single turn in the session cockpit."""

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # user | assistant | system
    content: Mapped[str] = mapped_column(Text)
    extra: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class JobProfile(Base):
    """A submitted job description and its internal analysis.

    Lives with the session because a profile is born from a chat turn, even though
    question sets and interviews later hang off it.
    """

    __tablename__ = "job_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("chat_sessions.id"), nullable=True)
    user_id: Mapped[str] = mapped_column(String(64), default="local-user")
    raw_text: Mapped[str] = mapped_column(Text)
    job_title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    analysis: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[ChatSession | None] = relationship(back_populates="profiles")
    question_sets: Mapped[list["QuestionSet"]] = relationship(back_populates="profile")
