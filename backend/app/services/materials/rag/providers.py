"""Bounded OpenAI-compatible adapters for RAG rewrite and reranking."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.integrations import llm
from app.core.config import get_settings
from app.services.operations.llm_gateway import log_call


def _log_adapter_call(db: Session | None, role: str, model: str, status: str,
                      started: float, measured: list[llm.ModelUsage], error: Exception | None = None) -> None:
    """Persist attempt metadata without endpoint URLs, prompts, or vendor errors."""
    if db is not None:
        log_call(db, role, role, model, status,
                 int((time.perf_counter() - started) * 1000),
                 type(error).__name__ if error is not None else None,
                 usage=llm.combine_usage(measured))


def _provider_budget(timeout: float) -> tuple[float, float | None]:
    """Cap an adapter's own timeout by the enclosing graph deadline."""

    remaining = llm.remaining_budget()
    requested = max(0.1, float(timeout))
    return (min(requested, remaining) if remaining is not None else requested, remaining)


async def rewrite_with_provider(query: str, *, model: str, base_url: str, timeout: float,
                                db: Session | None = None) -> dict[str, Any]:
    """Ask for retrieval phrases; the caller validates and retains the original.

    The compatible endpoint does not reliably honour ``response_format``, so the
    first parse failure is retried once with an explicit JSON-only instruction.
    A retry is a bounded recovery for a flaky vendor, never a silent downgrade:
    if the second attempt also fails the caller still records ``rewrite_failed``.
    """

    key = os.getenv("SAGEMATCH_RAG_QUERY_REWRITE_API_KEY") or get_settings().sagematch_rag_query_rewrite_api_key
    if not key or not model or not base_url:
        raise RuntimeError("query rewrite provider is not configured")
    system = ("Return JSON {\"queries\": [string, ...]} with up to two alternate search phrases. "
              "Do not answer the question or add facts.")
    plain_system = (system + " Respond with the JSON object only: no prose, no markdown fence, "
                    "no code block delimiters.")
    last_error: Exception | None = None
    for attempt in range(2):
        started = time.perf_counter()
        with llm.call_usage_scope() as measured:
            try:
                content = await _post_chat_completion(
                    model=model, base_url=base_url, timeout=timeout, key=key,
                    system=plain_system if attempt else system, user=query,
                )
                result = _parse_json_object(content)
                if not isinstance(result, dict) or not isinstance(result.get("queries"), list):
                    raise ValueError("rewrite response must contain a queries list")
            except llm.UsageBudgetError:
                raise
            except ValueError as exc:
                # A malformed response consumes an attempt even when the next
                # prompt repairs it; retain both outcomes in the call log.
                _log_adapter_call(db, "query_rewrite", model, "error", started, measured, exc)
                last_error = exc
            except Exception as exc:
                _log_adapter_call(db, "query_rewrite", model, "error", started, measured, exc)
                raise
            else:
                _log_adapter_call(db, "query_rewrite", model, "ok", started, measured)
                return result
    raise last_error if last_error is not None else ValueError("rewrite provider returned no usable JSON")


async def _post_chat_completion(*, model: str, base_url: str, timeout: float, key: str,
                                system: str, user: str) -> str:
    """Send one bounded rewrite request and return the raw assistant content."""

    body = {
        "model": model,
        "temperature": 0,
        # 768 而不是 160：该兼容端点的模型带 reasoning_tokens，预算太小时推理会
        # 吃掉全部额度，正文 JSON 被截断，json.loads 失败并让 rewrite_failed 误报。
        # 该值仍低于现有 1000-token 单请求测试门槛（连同提示估算后可预留）。
        "max_tokens": 768,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    allowed, remaining = _provider_budget(timeout)
    async with httpx.AsyncClient(timeout=allowed) as client:
        with llm.model_request_budget(system, user, body["max_tokens"], model):
            async with asyncio.timeout(remaining):
                response = await client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=body,
                )
            response.raise_for_status()
            payload = response.json()
            llm.record_provider_usage(payload.get("usage") if isinstance(payload, dict) else None, model)
            return str(payload["choices"][0]["message"]["content"])


def _parse_json_object(content: str) -> Any:
    """Parse a JSON object, tolerating a fenced or prose-wrapped provider answer."""

    import json

    text = content.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    # Some vendors wrap the object in a markdown fence or add surrounding prose.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("rewrite response is not JSON")
    return json.loads(text[start:end + 1])


def _rerank_url(base_url: str) -> str:
    """Resolve the rerank endpoint for both adapter styles.

    An OpenAI-compatible base already points at a ``.../v1`` root, so ``/rerank``
    is appended. DashScope-style hosts publish rerank under a services path, so a
    bare host is routed there instead of producing a 404 on ``/rerank``.
    """

    root = base_url.rstrip("/")
    if root.endswith("/v1") or root.endswith("/compatible-mode/v1"):
        return f"{root}/rerank"
    return f"{root}/api/v1/services/rerank/text-rerank/text-rerank"


def _rerank_entries(payload: Any) -> Any:
    """Accept both top-level ``results`` and DashScope's nested ``output.results``."""

    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("results"), list):
        return payload["results"]
    output = payload.get("output")
    if isinstance(output, dict) and isinstance(output.get("results"), list):
        return output["results"]
    return None


async def rerank_with_provider(
    query: str, candidates: list[dict[str, Any]], *, model: str, base_url: str, timeout: float,
    db: Session | None = None,
) -> list[float]:
    """Use a rerank endpoint and restore scores to original candidate order."""

    key = os.getenv("SAGEMATCH_RAG_RERANK_API_KEY") or get_settings().sagematch_rag_rerank_api_key
    if not key or not model or not base_url:
        raise RuntimeError("rerank provider is not configured")
    body = {"model": model, "query": query, "documents": [str(item.get("text") or "") for item in candidates]}
    allowed, remaining = _provider_budget(timeout)
    started = time.perf_counter()
    with llm.call_usage_scope() as measured:
        try:
            async with httpx.AsyncClient(timeout=allowed) as client:
                # A rerank request has no generated output; unknown vendor usage stays unknown.
                with llm.model_request_budget(query, "\n".join(body["documents"]), 0, model):
                    async with asyncio.timeout(remaining):
                        response = await client.post(
                            _rerank_url(base_url),
                            headers={"Authorization": f"Bearer {key}"},
                            json=body,
                        )
                    response.raise_for_status()
                    payload = response.json()
                    raw_usage = payload.get("usage") if isinstance(payload, dict) else None
                    if isinstance(raw_usage, dict) and "completion_tokens" not in raw_usage:
                        prompt = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens"))
                        total = raw_usage.get("total_tokens")
                        if type(prompt) is int and type(total) is int and prompt == total:
                            raw_usage = {"prompt_tokens": prompt, "completion_tokens": 0, "total_tokens": total}
                    llm.record_provider_usage(raw_usage, model)
            entries = _rerank_entries(payload)
            if not isinstance(entries, list) or len(entries) != len(candidates):
                raise ValueError("rerank response count does not match candidates")
            scores: list[float | None] = [None] * len(candidates)
            for entry in entries:
                index = entry.get("index") if isinstance(entry, dict) else None
                value = entry.get("relevance_score") if isinstance(entry, dict) else None
                if not isinstance(index, int) or not 0 <= index < len(scores) or scores[index] is not None:
                    raise ValueError("rerank response has invalid indices")
                scores[index] = float(value)
            if any(score is None for score in scores):
                raise ValueError("rerank response has missing scores")
        except llm.UsageBudgetError:
            raise
        except Exception as exc:
            _log_adapter_call(db, "reranker", model, "error", started, measured, exc)
            raise
        _log_adapter_call(db, "reranker", model, "ok", started, measured)
        return [float(score) for score in scores if score is not None]
