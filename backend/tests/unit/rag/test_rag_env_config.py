"""Model settings come from the root dotenv, never from tracked RAG YAML."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.core.config import Settings
from app.core import rag_config
from app.services.materials.rag.live_eval import safe_config_snapshot
from app.services.materials.rag.providers import rewrite_with_provider


def test_model_settings_use_dotenv_and_ignore_yaml_model_sections(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "SAGEMATCH_RAG_EMBEDDING_ENABLED=true\n"
        "SAGEMATCH_RAG_EMBEDDING_MODEL=dotenv-embedding\n"
        "SAGEMATCH_RAG_EMBEDDING_BASE_URL=https://credential@example.test/embeddings\n"
        "SAGEMATCH_RAG_EMBEDDING_API_KEY=dotenv-embedding-key\n"
        "SAGEMATCH_RAG_RERANK_ENABLED=true\n"
        "SAGEMATCH_RAG_RERANK_MODEL=dotenv-reranker\n"
        "SAGEMATCH_RAG_RERANK_BASE_URL=https://credential@example.test/rerank\n"
        "SAGEMATCH_RAG_RERANK_API_KEY=dotenv-rerank-key\n"
        "SAGEMATCH_RAG_QUERY_REWRITE_ENABLED=true\n"
        "SAGEMATCH_RAG_QUERY_REWRITE_MODEL=dotenv-rewriter\n"
        "SAGEMATCH_RAG_QUERY_REWRITE_BASE_URL=https://credential@example.test/rewrite\n"
        "SAGEMATCH_RAG_QUERY_REWRITE_API_KEY=dotenv-rewrite-key\n"
        "SAGEMATCH_RAG_GENERATION_TEMPERATURE=0.25\n",
        encoding="utf-8",
    )
    yaml = tmp_path / "rag.yaml"
    yaml.write_text(
        "recall:\n  top_k: 7\n"
        "embedding:\n  model: obsolete-yaml-model\n  api_key: obsolete-yaml-key\n"
        "rerank:\n  enabled: false\n  top_n: 13\n"
        "query_rewrite:\n  max_queries: 2\n"
        "generation:\n  temperature: 0.95\n",
        encoding="utf-8",
    )
    for name in (
        "SAGEMATCH_RAG_EMBEDDING_MODEL", "SAGEMATCH_RAG_EMBEDDING_API_KEY",
        "SAGEMATCH_RAG_RERANK_MODEL", "SAGEMATCH_RAG_RERANK_API_KEY",
        "SAGEMATCH_RAG_QUERY_REWRITE_MODEL", "SAGEMATCH_RAG_QUERY_REWRITE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=dotenv)
    with patch.object(rag_config, "RAG_CONFIG_PATH", yaml), patch.object(
        rag_config, "get_settings", return_value=settings
    ):
        rag_config.get_rag_config.cache_clear()
        config = rag_config.get_rag_config()
        assert config.recall.top_k == 7
        assert config.embedding.model == "dotenv-embedding"
        assert config.rerank.enabled and config.rerank.model == "dotenv-reranker"
        assert config.rerank.top_n == 13
        assert config.query_rewrite.enabled and config.query_rewrite.model == "dotenv-rewriter"
        assert config.query_rewrite.max_queries == 2
        assert config.generation.temperature == 0.95
        with patch("app.services.materials.rag.live_eval.get_rag_config", return_value=config):
            snapshot = str(safe_config_snapshot())
        assert "dotenv-embedding-key" not in snapshot
        assert "dotenv-rerank-key" not in snapshot
        assert "dotenv-rewrite-key" not in snapshot
        assert "credential@example.test" not in snapshot
    rag_config.get_rag_config.cache_clear()


def test_process_environment_overrides_dotenv(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("SAGEMATCH_RAG_EMBEDDING_MODEL=dotenv-model\n", encoding="utf-8")
    monkeypatch.setenv("SAGEMATCH_RAG_EMBEDDING_MODEL", "deployment-model")
    assert Settings(_env_file=dotenv).sagematch_rag_embedding_model == "deployment-model"


def test_rewrite_adapter_reads_dotenv_key(tmp_path, monkeypatch) -> None:
    """The HTTP adapter must use the Settings-loaded key, not only os.getenv."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("SAGEMATCH_RAG_QUERY_REWRITE_API_KEY=dotenv-only-key\n", encoding="utf-8")
    monkeypatch.delenv("SAGEMATCH_RAG_QUERY_REWRITE_API_KEY", raising=False)
    headers_seen: list[bool] = []

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": '{"queries":["alternate"]}'}}]}

    class Client:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            pass

        async def post(self, _url: str, *, headers: dict, json: dict) -> Response:
            headers_seen.append(headers.get("Authorization") == "Bearer dotenv-only-key")
            return Response()

    with patch("app.services.materials.rag.providers.get_settings", return_value=Settings(_env_file=dotenv)), patch(
        "app.services.materials.rag.providers.httpx.AsyncClient", Client
    ):
        result = asyncio.run(rewrite_with_provider(
            "question", model="rewrite-model", base_url="https://example.test/v1", timeout=2,
        ))
    assert result["queries"] == ["alternate"]
    assert headers_seen == [True]
