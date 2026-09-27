"""Measured provider usage controls request budgets and remains auditable."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.integrations import llm
from app.models.platform.audit import LlmCallLog
from app.services.operations.llm_gateway import log_call


def test_token_budget_reserves_before_dispatch_and_reconciles_provider_usage() -> None:
    with llm.usage_budget_scope(token_budget=26) as ledger:
        with llm.model_request_budget("system", "user", 10, "model"):
            assert llm.record_provider_usage({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, "model")
            with pytest.raises(llm.UsageBudgetError):
                with llm.model_request_budget("system", "user", 10, "model"):
                    pass
        assert ledger.snapshot()["total_tokens"] == 5
        with llm.model_request_budget("system", "user", 10, "model"):
            llm.record_provider_usage({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, "model")
        assert ledger.snapshot()["total_tokens"] == 10


def test_missing_usage_blocks_budgeted_followup() -> None:
    with llm.usage_budget_scope(token_budget=100):
        with pytest.raises(llm.UsageBudgetError, match="usage unavailable"):
            with llm.model_request_budget("s", "u", 10, "model"):
                pass
        with pytest.raises(llm.UsageBudgetError, match="usage unavailable"):
            with llm.model_request_budget("s", "u", 10, "model"):
                pass


def test_missing_usage_remains_unknown_without_configured_budget() -> None:
    """A vendor omitting usage cannot break otherwise unrestricted chat."""

    with llm.usage_budget_scope() as ledger:
        with llm.model_request_budget("s", "u", 10, "model"):
            pass
        assert ledger.snapshot()["total_tokens"] is None
        with llm.model_request_budget("s", "u", 10, "model"):
            llm.record_provider_usage({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, "model")
        assert ledger.snapshot()["total_tokens"] is None


def test_cost_requires_explicit_pricing_and_uses_provider_counters() -> None:
    with patch("app.integrations.llm.get_settings", return_value=SimpleNamespace(llm_pricing_json="{}")):
        assert llm.record_provider_usage({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, "model").estimated_cost is None
        with llm.usage_budget_scope(cost_budget=1):
            with pytest.raises(llm.UsageBudgetError, match="pricing unavailable"):
                with llm.model_request_budget("s", "u", 10, "model"):
                    pass
    pricing = json.dumps({"model": {"input_per_million": 1, "output_per_million": 2}})
    with patch("app.integrations.llm.get_settings", return_value=SimpleNamespace(llm_pricing_json=pricing)):
        with llm.usage_budget_scope(cost_budget=0.0001) as ledger:
            with llm.model_request_budget("s", "u", 10, "model"):
                llm.record_provider_usage({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, "model")
            assert ledger.snapshot()["estimated_cost"] == pytest.approx(0.000008)


def test_usage_is_written_without_prompts_or_credentials() -> None:
    engine = create_engine("sqlite://")
    LlmCallLog.__table__.create(engine)
    with Session(engine) as db:
        usage = llm.ModelUsage(2, 3, 5, None)
        log_call(db, "author", "vendor", "model", "ok", 12, None, usage=usage)
        db.commit()
        row = db.query(LlmCallLog).one()
        assert (row.prompt_tokens, row.completion_tokens, row.total_tokens, row.estimated_cost) == (2, 3, 5, None)


def test_stream_terminal_usage_is_captured() -> None:
    class Response:
        async def aiter_lines(self):
            yield 'data: {"choices": [{"delta": {"content": "hi"}}]}'
            yield 'data: {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}'
            yield "data: [DONE]"

    async def consume():
        with llm.call_usage_scope() as rows:
            parts = [part async for part in llm._iter_chat_parts(Response(), model="model")]
            return parts, rows

    parts, rows = asyncio.run(consume())
    assert parts == [("content", "hi")]
    assert llm.combine_usage(rows).total_tokens == 6


def test_business_graph_applies_configured_usage_budget_and_records_measurement() -> None:
    from app.agents.orchestration.workflows import run_business_graph

    captured = []

    async def action() -> str:
        with llm.model_request_budget("system", "question", 10, "model"):
            llm.record_provider_usage({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, "model")
        return "committed"

    engine = create_engine("sqlite://")
    with Session(engine) as db, patch(
        "app.agents.orchestration.workflows.get_settings",
        return_value=SimpleNamespace(graph_token_budget=100, graph_cost_budget=None),
    ), patch("app.agents.orchestration.workflows.persist_graph_trace", side_effect=lambda _db, outcome: captured.append(outcome)):
        assert asyncio.run(run_business_graph(db, mode="live_interview", action=action)) == "committed"
    assert captured[0]["diagnostics"]["token_count"] == 5


def test_bound_ledger_is_shared_across_tasks_and_restores_context() -> None:
    async def worker(ledger):
        with llm.bind_usage_ledger(ledger):
            with llm.model_request_budget("s", "u", 0, "model"):
                llm.record_provider_usage({"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2}, "model")

    async def run():
        with llm.usage_budget_scope(token_budget=20) as ledger:
            await asyncio.create_task(worker(ledger), context=__import__("contextvars").Context())
            assert ledger.snapshot()["total_tokens"] == 2
            assert llm.current_usage()["total_tokens"] == 2
        assert llm.current_usage() is None

    asyncio.run(run())


def test_embedding_batches_count_provider_usage_and_refuse_missing_usage() -> None:
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class Client:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, *_args, **_kwargs):
            return Response({"data": [{"embedding": [1.0]}], "usage": {"prompt_tokens": 3, "total_tokens": 3}})

    with patch("app.integrations.llm.httpx.AsyncClient", Client):
        with llm.usage_budget_scope(token_budget=20) as ledger:
            assert asyncio.run(llm.embed_texts(["a", "b"], api_key="test", base_url="https://example.test/v1", model="embed", batch_size=1)) == [[1.0], [1.0]]
            assert ledger.snapshot()["total_tokens"] == 6

    async def missing(*_args, **_kwargs):
        return Response({"data": [{"embedding": [1.0]}]})

    with patch.object(Client, "post", missing), patch("app.integrations.llm.httpx.AsyncClient", Client):
        with llm.usage_budget_scope(token_budget=20):
            with pytest.raises(llm.UsageBudgetError, match="usage unavailable"):
                asyncio.run(llm.embed_texts(["a"], api_key="test", base_url="https://example.test/v1", model="embed"))
        with llm.usage_budget_scope() as ledger:
            assert asyncio.run(llm.embed_texts(["a"], api_key="test", base_url="https://example.test/v1", model="embed")) == [[1.0]]
            assert ledger.snapshot()["total_tokens"] is None


def test_embedding_failure_log_uses_adapter_name_instead_of_credentialed_url() -> None:
    """Operational call logs must not persist a configured endpoint URL."""
    from app.services.materials.recall import embed_or_empty

    config = SimpleNamespace(embedding=SimpleNamespace(
        enabled=True, model="embed", api_key="private", batch_size=1,
        base_url="https://user:secret@example.test/v1?token=secret", dimensions=2,
    ))
    with patch("app.services.materials.recall.get_rag_config", return_value=config), patch(
        "app.services.materials.recall.llm.embed_texts", new=AsyncMock(side_effect=RuntimeError("failed"))
    ), patch("app.services.materials.recall.log_call") as logged:
        assert asyncio.run(embed_or_empty(MagicMock(), ["query"])) == ([], "")
        assert logged.call_args.args[2] == "embedding"

    with patch("app.services.materials.recall.get_rag_config", return_value=config), patch(
        "app.services.materials.recall.llm.embed_texts", new=AsyncMock(return_value=[[1.0]])
    ), patch("app.services.materials.recall.log_call") as logged:
        assert asyncio.run(embed_or_empty(MagicMock(), ["query"])) == ([], "")
        assert logged.call_args.args[2] == "embedding"


def test_embedding_success_adds_measured_call_to_provider_denominator() -> None:
    """A successful vector call supplies one log row with actual provider usage."""
    from app.services.materials.recall import embed_or_empty

    config = SimpleNamespace(embedding=SimpleNamespace(
        enabled=True, model="embed", api_key="private", batch_size=1,
        base_url="https://example.test/v1", dimensions=1,
    ))

    async def embed(_texts, **_kwargs):
        with llm.model_request_budget("embed", "query", 0, "embed"):
            llm.record_provider_usage({"prompt_tokens": 3, "completion_tokens": 0,
                                       "total_tokens": 3}, "embed")
        return [[1.0]]

    with patch("app.services.materials.recall.get_rag_config", return_value=config), patch(
        "app.services.materials.recall.llm.embed_texts", new=embed
    ), patch("app.services.materials.recall.log_call") as logged:
        assert asyncio.run(embed_or_empty(MagicMock(), ["query"])) == ([[1.0]], "embed")
    assert logged.call_count == 1
    assert logged.call_args.args[2:5] == ("embedding", "embed", "ok")
    assert logged.call_args.kwargs["usage"].total_tokens == 3


def test_rag_budget_denial_never_becomes_fallback() -> None:
    from app.services.materials.rag.query_rewrite import rewrite_queries
    from app.services.materials.rag.ranking import rerank_candidates
    from app.services.materials.rag.stages import StagedRetrieval
    from app.services.materials.recall import embed_or_empty

    async def denied(*_args, **_kwargs):
        raise llm.UsageBudgetError("request token budget exceeded")

    with pytest.raises(llm.UsageBudgetError):
        asyncio.run(rewrite_queries("query", provider=denied, enabled=True))
    with pytest.raises(llm.UsageBudgetError):
        asyncio.run(rerank_candidates("query", [{"chunk_id": "a"}], reranker=denied))
    pipeline = StagedRetrieval(object(), "query")
    pipeline._rows_loaded = True
    with patch("app.services.materials.rag.stages.embed_or_empty", side_effect=denied):
        with pytest.raises(llm.UsageBudgetError):
            asyncio.run(pipeline.vector_retrieve())
    with patch("app.services.materials.recall.embedding_ready", return_value=True), patch("app.services.materials.recall.llm.embed_texts", side_effect=denied):
        with pytest.raises(llm.UsageBudgetError):
            asyncio.run(embed_or_empty(MagicMock(), ["query"]))


def test_rag_adapters_account_for_provider_usage_and_missing_usage() -> None:
    from app.services.materials.rag.providers import rerank_with_provider, rewrite_with_provider

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class Client:
        payload = {}
        calls = 0

        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, *_args, **_kwargs):
            Client.calls += 1
            return Response(Client.payload)

    env = {"SAGEMATCH_RAG_QUERY_REWRITE_API_KEY": "test", "SAGEMATCH_RAG_RERANK_API_KEY": "test"}
    with patch.dict("os.environ", env), patch("app.services.materials.rag.providers.httpx.AsyncClient", Client):
        Client.payload = {"choices": [{"message": {"content": '{"queries": ["alternate"]}'}}],
                          "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}
        with llm.usage_budget_scope(token_budget=1000) as ledger:
            assert asyncio.run(rewrite_with_provider("query", model="rewrite", base_url="https://example.test/v1", timeout=2))["queries"] == ["alternate"]
            assert ledger.snapshot()["total_tokens"] == 6
        Client.payload = {"results": [{"index": 0, "relevance_score": 0.8}],
                          "usage": {"input_tokens": 3, "total_tokens": 3}}
        with llm.usage_budget_scope(token_budget=1000) as ledger:
            assert asyncio.run(rerank_with_provider("query", [{"text": "document"}], model="rerank", base_url="https://example.test/v1", timeout=2)) == [0.8]
            assert ledger.snapshot()["total_tokens"] == 3
        Client.payload = {"results": [{"index": 0, "relevance_score": 0.8}]}
        with llm.usage_budget_scope(token_budget=1000):
            with pytest.raises(llm.UsageBudgetError, match="usage unavailable"):
                asyncio.run(rerank_with_provider("query", [{"text": "document"}], model="rerank", base_url="https://example.test/v1", timeout=2))
        with llm.usage_budget_scope() as ledger:
            assert asyncio.run(rerank_with_provider("query", [{"text": "document"}], model="rerank", base_url="https://example.test/v1", timeout=2)) == [0.8]
            assert ledger.snapshot()["total_tokens"] is None


def test_rag_adapter_attempt_logs_are_bounded_and_redacted() -> None:
    """A malformed rewrite retry and reranker failure are distinct call outcomes."""
    from app.services.materials.rag.providers import rerank_with_provider, rewrite_with_provider

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class Client:
        payloads = []

        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, *_args, **_kwargs):
            return Response(Client.payloads.pop(0))

    endpoint = "https://user:secret@example.test/v1?token=secret"
    env = {"SAGEMATCH_RAG_QUERY_REWRITE_API_KEY": "private", "SAGEMATCH_RAG_RERANK_API_KEY": "private"}
    db = MagicMock()
    with patch.dict("os.environ", env), patch(
        "app.services.materials.rag.providers.httpx.AsyncClient", Client
    ), patch("app.services.materials.rag.providers.log_call") as logged:
        Client.payloads = [
            {"choices": [{"message": {"content": "not json"}}],
             "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
            {"choices": [{"message": {"content": '{"queries":["fixed"]}'}}],
             "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
        ]
        assert asyncio.run(rewrite_with_provider("query", model="rewrite", base_url=endpoint,
                                                 timeout=2, db=db))["queries"] == ["fixed"]
        assert [call.args[4] for call in logged.call_args_list] == ["error", "ok"]
        assert all(call.args[1:3] == ("query_rewrite", "query_rewrite")
                   for call in logged.call_args_list)
        assert all(call.kwargs["usage"].total_tokens == 3 for call in logged.call_args_list)
        assert "secret" not in str(logged.call_args_list)

        logged.reset_mock()
        Client.payloads = [{"results": [{"index": 0, "relevance_score": 0.7}],
                            "usage": {"input_tokens": 2, "total_tokens": 2}}]
        assert asyncio.run(rerank_with_provider("query", [{"text": "document"}],
                                                model="rerank", base_url=endpoint,
                                                timeout=2, db=db)) == [0.7]
        assert logged.call_args.args[1:4] == ("reranker", "reranker", "rerank")
        assert logged.call_args.args[4] == "ok"
        assert logged.call_args.kwargs["usage"].total_tokens == 2

        logged.reset_mock()
        Client.payloads = [{"results": []}]
        with pytest.raises(ValueError, match="count"):
            asyncio.run(rerank_with_provider("query", [{"text": "document"}],
                                               model="rerank", base_url=endpoint,
                                               timeout=2, db=db))
        assert logged.call_args.args[4] == "error"
        assert "secret" not in str(logged.call_args_list)
