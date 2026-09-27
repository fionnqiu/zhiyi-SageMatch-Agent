"""Load RAG tuning from YAML and model connections from the root .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.config import get_settings

BACKEND_ROOT = Path(__file__).resolve().parents[2]
RAG_CONFIG_PATH = BACKEND_ROOT / "config" / "rag.yaml"


class ChunkSettings(BaseModel):
    # size / overlap 单位是 token，见 knowledge.tokenize。
    size: int = 720
    overlap: int = 120


class RecallSettings(BaseModel):
    top_k: int = 6
    fusion: Literal["weighted", "rrf"] = "weighted"
    lexical_top_k: int = Field(default=40, ge=1)
    vector_top_k: int = Field(default=40, ge=1)
    candidate_k: int = Field(default=40, ge=1)
    max_chunks_per_material: int = Field(default=4, ge=1)
    max_materials: int | None = Field(default=None, ge=1)
    merge_adjacent: bool = True
    lexical_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    score_floor: float = 0.0
    exact_match_bonus: float = 0.35


class ContextSettings(BaseModel):
    """Bound both prompt evidence volume and each source excerpt."""

    max_chunks: int = Field(default=8, ge=1)
    max_tokens: int = Field(default=5000, ge=1)
    max_chunk_chars: int = Field(default=1200, ge=1)
    include_source_id: bool = True


class GenerationSettings(BaseModel):
    temperature: float = 0.4
    top_p: float = 1.0
    max_tokens: int = 900


class EmbeddingSettings(BaseModel):
    """Embedding settings are sourced from runtime Settings, not tracked YAML."""

    enabled: bool = True
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    dimensions: int = 1024
    batch_size: int = 32
    similarity: str = "cosine"
    lexical_weight: float = 0.35


class RerankSettings(BaseModel):
    enabled: bool = False
    top_n: int = 20
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout_seconds: float = 8.0
    batch_size: int = 16
    fail_open: bool = True


class QueryRewriteSettings(BaseModel):
    enabled: bool = False
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout_seconds: float = 8.0
    max_queries: int = 3
    max_query_chars: int = 2000


class RagConfig(BaseModel):
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    recall: RecallSettings = Field(default_factory=RecallSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    query_rewrite: QueryRewriteSettings = Field(default_factory=QueryRewriteSettings)


def _coerce(raw: str) -> Any:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~", ""}:
        return None
    try:
        return int(text, 10)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """ponytail: indent/key YAML subset (no lists/anchors). Switch to PyYAML if the file grows."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, rest = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = rest.strip()
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce(value)
    return root


@lru_cache
def get_rag_config() -> RagConfig:
    data = _parse_simple_yaml(RAG_CONFIG_PATH.read_text(encoding="utf-8")) if RAG_CONFIG_PATH.exists() else {}
    # YAML owns numeric tuning; model connection fields are never read there.
    model_fields = {"enabled", "model", "base_url", "api_key"}
    tuning = {
        section: ({key: value for key, value in values.items() if key not in model_fields}
                  if section in {"embedding", "rerank", "query_rewrite"} and isinstance(values, dict)
                  else values)
        for section, values in data.items()
    }
    config = RagConfig.model_validate(tuning)
    settings = get_settings()
    for section in ("embedding", "rerank", "query_rewrite"):
        target = getattr(config, section)
        for field in model_fields:
            setattr(target, field, getattr(settings, f"sagematch_rag_{section}_{field}"))
    return config
