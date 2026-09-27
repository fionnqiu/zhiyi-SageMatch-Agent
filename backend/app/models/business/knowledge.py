"""Knowledge-base tables: uploaded materials and the chunks used for recall."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class Material(Base):
    """An uploaded knowledge document. Chunks are deleted with the parent."""

    __tablename__ = "materials"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(260))
    source: Mapped[str] = mapped_column(String(200), default="upload")
    mime: Mapped[str] = mapped_column(String(80), default="text/plain")
    status: Mapped[str] = mapped_column(String(20), default="ready")  # pending | ready | failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["MaterialChunk"]] = relationship(
        back_populates="material", cascade="all, delete-orphan", order_by="MaterialChunk.ordinal"
    )


class MaterialChunk(Base):
    """One retrieval unit. RAG numeric params live in backend config, not the admin UI."""

    __tablename__ = "material_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    material_id: Mapped[str] = mapped_column(ForeignKey("materials.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    embedding_model: Mapped[str] = mapped_column(Text, default="")

    material: Mapped[Material] = relationship(back_populates="chunks")
