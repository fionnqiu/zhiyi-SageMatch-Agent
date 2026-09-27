"""Request and response shapes for materials and recall."""

from datetime import datetime

from pydantic import BaseModel, Field


class MaterialOut(BaseModel):
    id: str
    filename: str
    source: str
    mime: str
    status: str
    error: str | None = None
    size_bytes: int
    chunk_count: int
    created_at: datetime


class ChunkOut(BaseModel):
    id: str
    ordinal: int
    text: str
    token_estimate: int


class MaterialDetail(MaterialOut):
    chunks: list[ChunkOut] = Field(default_factory=list)


class RecallHit(BaseModel):
    chunk_id: str
    material_id: str
    filename: str
    ordinal: int
    score: float
    text: str


class RecallOut(BaseModel):
    query: str
    hits: list[RecallHit]
    index_status: str
