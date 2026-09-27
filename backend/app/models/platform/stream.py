"""Durable client event stream records used for reconnect replay."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base
from app.models.platform.runtime import RuntimeJSON


class StreamRun(Base):
    """One owner-scoped stream; sequence allocation survives worker restarts."""

    __tablename__ = "stream_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(120), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    business_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="running")
    next_seq: Mapped[int] = mapped_column(Integer, default=1)
    # The lease makes an interrupted HTTP worker claimable by another process.
    lease_owner: Mapped[str | None] = mapped_column(String(120), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Only text requests store the route and minimal turn state. File text stays out.
    recovery: Mapped[dict | None] = mapped_column(RuntimeJSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class StreamEvent(Base):
    """A committed, client-visible frame; heartbeat and reasoning are excluded."""

    __tablename__ = "stream_events"
    __table_args__ = (UniqueConstraint("run_id", "seq", name="uq_stream_events_run_seq"),)

    event_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("stream_runs.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(30))
    payload: Mapped[dict] = mapped_column(RuntimeJSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
