"""Admin provider tables: vendor credentials and the role → model routing table."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class ProviderConfig(Base):
    """One LLM / ASR / TTS vendor. Keys are stored server-side and masked in APIs."""

    __tablename__ = "provider_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    protocol: Mapped[str] = mapped_column(String(40))
    base_url: Mapped[str] = mapped_column(String(300), default="")
    capability: Mapped[str] = mapped_column(String(40), default="llm")
    status: Mapped[str] = mapped_column(String(40), default="unknown")
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    api_key: Mapped[str] = mapped_column(Text, default="")
    models: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class RoleBinding(Base):
    """Maps an Agent role to a provider + model so user-facing code stays unchanged."""

    __tablename__ = "role_bindings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    role: Mapped[str] = mapped_column(String(40), unique=True)
    label: Mapped[str] = mapped_column(String(80))
    provider_id: Mapped[str | None] = mapped_column(ForeignKey("provider_configs.id"), nullable=True)
    model: Mapped[str] = mapped_column(String(120), default="")
    # 语音角色专用：ASR 走 provider_id/model，TTS 走下面两列。
    # 拆开是因为识别和合成经常不是同一家、也不是同一个模型，但面试里要成对使用。
    tts_provider_id: Mapped[str | None] = mapped_column(ForeignKey("provider_configs.id"), nullable=True)
    tts_model: Mapped[str] = mapped_column(String(120), default="")
    temperature: Mapped[float] = mapped_column(Float, default=0.4)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
