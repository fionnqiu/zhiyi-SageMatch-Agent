"""OpenAI Chat Completions client for streaming chat and structured JSON roles."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import AsyncIterator
from threading import Lock
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.rag_config import get_rag_config

# 普通文本对话只走 OpenAI Chat Completions，思考和流式固定开启；结构化 JSON 角色走独立的非流式路径。
CHAT_PROTOCOL = "openai_chat"

# A monotonic deadline follows the current graph task through provider retries and tools.
_deadline_at: ContextVar[float | None] = ContextVar("llm_deadline_at", default=None)


class UsageBudgetError(RuntimeError):
    """A configured request budget cannot authorize another model request."""


@dataclass(frozen=True)
class ModelUsage:
    """Provider-reported tokens and a cost computed only from configured rates."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float | None


@dataclass
class ModelReservation:
    """A provider-rejected request has no generated response to account for."""

    rejected: bool = False


class UsageLedger:
    """Share measured usage and in-flight reservations across one graph request."""

    def __init__(self, token_budget: int | None, cost_budget: float | None) -> None:
        if token_budget is not None and token_budget < 0:
            raise ValueError("token_budget must be nonnegative")
        if cost_budget is not None and cost_budget < 0:
            raise ValueError("cost_budget must be nonnegative")
        self.token_budget = token_budget
        self.cost_budget = Decimal(str(cost_budget)) if cost_budget is not None else None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cost = Decimal(0)
        self.cost_known = True
        self.unknown_usage = False
        self._reserved_tokens = 0
        self._reserved_cost = Decimal(0)
        self._lock = Lock()

    def snapshot(self) -> dict[str, int | float | None]:
        """Expose measured totals; unknown price never becomes a fabricated zero."""
        with self._lock:
            return {
                "prompt_tokens": None if self.unknown_usage else self.prompt_tokens,
                "completion_tokens": None if self.unknown_usage else self.completion_tokens,
                "total_tokens": None if self.unknown_usage else self.total_tokens,
                "estimated_cost": float(self.cost) if self.cost_known and not self.unknown_usage else None,
            }

    def reserve(self, tokens: int, cost: Decimal | None) -> None:
        with self._lock:
            if self.unknown_usage and (self.token_budget is not None or self.cost_budget is not None):
                raise UsageBudgetError("provider usage unavailable for configured budget")
            if self.token_budget is not None and self.total_tokens + self._reserved_tokens + tokens > self.token_budget:
                raise UsageBudgetError("request token budget exceeded")
            if self.cost_budget is not None:
                if cost is None or not self.cost_known:
                    raise UsageBudgetError("model pricing unavailable for configured cost budget")
                if self.cost + self._reserved_cost + cost > self.cost_budget:
                    raise UsageBudgetError("request cost budget exceeded")
            self._reserved_tokens += tokens
            if cost is not None:
                self._reserved_cost += cost

    def settle(self, reserved_tokens: int, reserved_cost: Decimal | None, usage: ModelUsage | None) -> None:
        with self._lock:
            self._reserved_tokens -= reserved_tokens
            if reserved_cost is not None:
                self._reserved_cost -= reserved_cost
            if usage is None:
                self.unknown_usage = True
                return
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
            if usage.estimated_cost is None:
                self.cost_known = False
            else:
                self.cost += Decimal(str(usage.estimated_cost))

    def release(self, reserved_tokens: int, reserved_cost: Decimal | None) -> None:
        """Release a protocol-rejected request that produced no model output."""
        with self._lock:
            self._reserved_tokens -= reserved_tokens
            if reserved_cost is not None:
                self._reserved_cost -= reserved_cost


_usage_ledger: ContextVar[UsageLedger | None] = ContextVar("llm_usage_ledger", default=None)
_call_usage: ContextVar[list[ModelUsage] | None] = ContextVar("llm_call_usage", default=None)
_request_usage: ContextVar[list[ModelUsage] | None] = ContextVar("llm_request_usage", default=None)


@contextmanager
def bind_usage_ledger(ledger: UsageLedger):
    """Carry one thread-safe ledger into a detached stream worker's context."""
    token = _usage_ledger.set(ledger)
    try:
        yield ledger
    finally:
        _usage_ledger.reset(token)


@contextmanager
def usage_budget_scope(*, token_budget: int | None = None, cost_budget: float | None = None):
    """Bind one request ledger; nested callers cannot replace an active budget."""
    parent = _usage_ledger.get()
    if parent is not None:
        if token_budget is not None and (parent.token_budget is None or token_budget < parent.token_budget):
            raise UsageBudgetError("nested token budget cannot replace the active ledger")
        if cost_budget is not None and (parent.cost_budget is None or Decimal(str(cost_budget)) < parent.cost_budget):
            raise UsageBudgetError("nested cost budget cannot replace the active ledger")
        yield parent
        return
    ledger = UsageLedger(token_budget, cost_budget)
    token = _usage_ledger.set(ledger)
    try:
        yield ledger
    finally:
        _usage_ledger.reset(token)


@contextmanager
def call_usage_scope():
    """Capture every provider attempt, including a JSON repair retry, for one audit row."""
    captured: list[ModelUsage] = []
    token = _call_usage.set(captured)
    try:
        yield captured
    finally:
        _call_usage.reset(token)


def current_usage() -> dict[str, int | float | None] | None:
    ledger = _usage_ledger.get()
    return ledger.snapshot() if ledger is not None else None


def pricing_for(model: str) -> tuple[Decimal, Decimal] | None:
    """Load explicit model prices per million tokens without assuming a vendor tariff."""
    try:
        configured = json.loads(get_settings().llm_pricing_json)
        row = configured.get(model) if isinstance(configured, dict) else None
        if not isinstance(row, dict):
            return None
        input_rate = Decimal(str(row["input_per_million"]))
        output_rate = Decimal(str(row["output_per_million"]))
        if not input_rate.is_finite() or not output_rate.is_finite() or min(input_rate, output_rate) < 0:
            return None
        return input_rate, output_rate
    except (ValueError, KeyError, TypeError, InvalidOperation):
        return None


@contextmanager
def model_request_budget(system: str, user: str, max_tokens: int, model: str):
    """Reserve a conservative prompt-byte estimate and maximum output before dispatch."""
    ledger = _usage_ledger.get()
    if ledger is None:
        yield ModelReservation()
        return
    # UTF-8 byte count is a conservative admission estimate for common BPE
    # tokenizers; provider-reported counts replace it after the response.
    prompt_estimate = len(system.encode("utf-8")) + len(user.encode("utf-8"))
    reserved_tokens = prompt_estimate + max_tokens
    price = pricing_for(model)
    reserved_cost = (Decimal(prompt_estimate) * price[0] + Decimal(max_tokens) * price[1]) / 1_000_000 if price else None
    ledger.reserve(reserved_tokens, reserved_cost)
    observed: list[ModelUsage] = []
    reservation = ModelReservation()
    token = _request_usage.set(observed)
    clean = False
    try:
        yield reservation
        clean = True
    finally:
        _request_usage.reset(token)
        usage = combine_usage(observed)
        if reservation.rejected and usage is None:
            # 4xx protocol rejection happens before generation; the fallback
            # request receives its own reservation and usage measurement.
            ledger.release(reserved_tokens, reserved_cost)
        else:
            ledger.settle(reserved_tokens, reserved_cost, usage)
        if clean and usage is None and not reservation.rejected and (
            ledger.token_budget is not None or ledger.cost_budget is not None
        ):
            raise UsageBudgetError("provider usage unavailable for configured budget")
        if clean and ((ledger.token_budget is not None and ledger.total_tokens > ledger.token_budget) or
                      (ledger.cost_budget is not None and (not ledger.cost_known or ledger.cost > ledger.cost_budget))):
            raise UsageBudgetError("request model budget exceeded")


def combine_usage(rows: list[ModelUsage]) -> ModelUsage | None:
    if not rows:
        return None
    return ModelUsage(
        sum(row.prompt_tokens for row in rows), sum(row.completion_tokens for row in rows),
        sum(row.total_tokens for row in rows),
        sum(row.estimated_cost for row in rows) if all(row.estimated_cost is not None for row in rows) else None,
    )


def record_provider_usage(raw: Any, model: str) -> ModelUsage | None:
    """Accept only complete, nonnegative provider counters as measured usage."""
    if not isinstance(raw, dict):
        return None
    prompt, completion, total = (raw.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
    if any(type(value) is not int or value < 0 for value in (prompt, completion, total)) or total < prompt + completion:
        return None
    price = pricing_for(model)
    cost = float((Decimal(prompt) * price[0] + Decimal(completion) * price[1]) / 1_000_000) if price else None
    usage = ModelUsage(prompt, completion, total, cost)
    for bucket in (_call_usage.get(), _request_usage.get()):
        if bucket is not None:
            bucket.append(usage)
    return usage


@contextmanager
def deadline_scope(deadline_at: float | None):
    """Apply the earlier of a caller deadline and the active request deadline."""
    current = _deadline_at.get()
    effective = min(current, deadline_at) if current is not None and deadline_at is not None else (current or deadline_at)
    token = _deadline_at.set(effective)
    try:
        yield
    finally:
        _deadline_at.reset(token)


def remaining_budget() -> float | None:
    """Return seconds left, raising before another retry when the budget is gone."""
    deadline = _deadline_at.get()
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("request deadline exceeded")
    return remaining


def _http_timeout(total: float, *, connect: float, read: float | None = None) -> httpx.Timeout:
    remaining = remaining_budget()
    allowed = min(total, remaining) if remaining is not None else total
    return httpx.Timeout(allowed, connect=min(connect, allowed), read=min(read or total, allowed))


async def _bounded_post(client: httpx.AsyncClient, *args: Any, **kwargs: Any) -> httpx.Response:
    remaining = remaining_budget()
    if remaining is None:
        return await client.post(*args, **kwargs)
    async with asyncio.timeout(remaining):
        return await client.post(*args, **kwargs)


def llm_available(api_key: str | None = None) -> bool:
    settings = get_settings()
    return bool((api_key if api_key is not None else settings.llm_api_key) and settings.llm_model)


async def complete_json(
    system: str,
    user: str,
    *,
    max_tokens: int = 1800,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    top_p: float | None = None,
    protocol: str = CHAT_PROTOCOL,
) -> dict[str, Any]:
    """Request a bounded JSON object without spending tokens on reasoning.

    Structured roles such as scorer and coach need a complete object more than
    they need token-by-token UI output. They therefore use a non-streaming call
    with thinking disabled, while ordinary chat continues through the streaming
    path below.
    """
    del protocol
    finish_reason: str | None = None
    try:
        raw, finish_reason = await _complete_json_once(
            system,
            user,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            top_p=top_p,
            use_response_format=True,
        )
        return _extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        # Compatible providers can still emit malformed JSON. Retry once with a
        # stricter instruction and no optional response_format field so providers
        # that only partially implement the OpenAI contract get a second chance.
        retry_user = (
            f"{user}\n上一轮输出无法解析。请重新生成完整 JSON 对象，只输出 JSON，"
            "不要 Markdown、解释、截断或额外文本。"
        )
        retry_raw, retry_finish_reason = await _complete_json_once(
            system,
            retry_user,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.0,
            top_p=top_p,
            use_response_format=False,
        )
        try:
            return _extract_json(retry_raw)
        except (ValueError, json.JSONDecodeError) as exc:
            reason = retry_finish_reason or finish_reason
            if reason:
                raise ValueError(f"LLM JSON response is invalid (finish_reason={reason})") from exc
            raise


async def _complete_json_once(
    system: str,
    user: str,
    *,
    max_tokens: int,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
    temperature: float,
    top_p: float | None,
    use_response_format: bool,
) -> tuple[str, str | None]:
    """Make one non-streaming JSON request and return content plus finish reason."""
    settings = get_settings()
    gen = get_rag_config().generation
    key = api_key if api_key is not None else settings.llm_api_key
    url = base_url if base_url is not None else settings.llm_base_url
    mdl = model or settings.llm_model
    nucleus = gen.top_p if top_p is None else top_p
    if not key:
        raise RuntimeError("LLM_API_KEY is empty")

    body: dict[str, Any] = {
        "model": mdl,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        # Scoring output must not share its token budget with hidden reasoning.
        "enable_thinking": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if 0 < nucleus <= 1:
        body["top_p"] = nucleus
    if use_response_format:
        body["response_format"] = {"type": "json_object"}

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    timeout = _http_timeout(180.0, connect=15.0, read=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        with model_request_budget(system, user, max_tokens, mdl) as reservation:
            res = await _bounded_post(client, f"{openai_root(url)}/chat/completions", headers=headers, json=body)
            if use_response_format and getattr(res, "status_code", 200) in {400, 404, 422}:
                reservation.rejected = True
            else:
                res.raise_for_status()
                data = res.json()
                record_provider_usage(data.get("usage") if isinstance(data, dict) else None, mdl)
        if reservation.rejected:
            # Only a documented protocol rejection can skip usage accounting.
            retry_body = {key: value for key, value in body.items() if key != "response_format"}
            with model_request_budget(system, user, max_tokens, mdl):
                res = await _bounded_post(client, f"{openai_root(url)}/chat/completions", headers=headers, json=retry_body)
                res.raise_for_status()
                data = res.json()
                record_provider_usage(data.get("usage") if isinstance(data, dict) else None, mdl)

    choices = data.get("choices") if isinstance(data, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        # A few compatible endpoints return content parts instead of one string.
        content = "".join(
            str(part.get("text") or part.get("content") or "") if isinstance(part, dict) else str(part)
            for part in content
        )
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"LLM returned no JSON content (finish_reason={finish_reason or 'unknown'})")
    return content.strip(), str(finish_reason) if finish_reason else None


async def complete_tool_call(
    system: str,
    user: str,
    *,
    tools: list[dict[str, Any]],
    max_tokens: int = 900,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    top_p: float | None = None,
) -> dict[str, Any] | None:
    """Call the OpenAI tool protocol and normalize its first function call."""
    settings = get_settings()
    gen = get_rag_config().generation
    key = api_key if api_key is not None else settings.llm_api_key
    url = base_url if base_url is not None else settings.llm_base_url
    mdl = model or settings.llm_model
    nucleus = gen.top_p if top_p is None else top_p
    if not key:
        raise RuntimeError("LLM_API_KEY is empty")
    body: dict[str, Any] = {
        "model": mdl,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "tools": tools,
        "tool_choice": "auto",
    }
    if 0 < nucleus <= 1:
        body["top_p"] = nucleus
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    timeout = _http_timeout(180.0, connect=15.0, read=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        with model_request_budget(system, user, max_tokens, mdl):
            res = await _bounded_post(client, f"{openai_root(url)}/chat/completions", headers=headers, json=body)
            res.raise_for_status()
            data = res.json()
            record_provider_usage(data.get("usage") if isinstance(data, dict) else None, mdl)
    choices = data.get("choices") if isinstance(data, dict) else None
    message = choices[0].get("message") if choices else None
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    call = calls[0] if calls else None
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or not function.get("name"):
        return None
    raw_args = function.get("arguments") or "{}"
    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be a JSON object")
    return {"id": str(call.get("id") or "tool-call"), "name": str(function["name"]), "arguments": args}


async def complete_text(
    system: str,
    user: str,
    *,
    max_tokens: int = 900,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    top_p: float | None = None,
    protocol: str = CHAT_PROTOCOL,
) -> str:
    settings = get_settings()
    gen = get_rag_config().generation
    key = api_key if api_key is not None else settings.llm_api_key
    url = base_url if base_url is not None else settings.llm_base_url
    mdl = model or settings.llm_model
    nucleus = gen.top_p if top_p is None else top_p
    if not key:
        raise RuntimeError("LLM_API_KEY is empty")
    # 文本对话不再看供应商上的协议名。思考和流式都在 stream_text 里写死。
    del protocol
    parts: list[str] = []
    async for delta in stream_text(
        system,
        user,
        max_tokens=max_tokens,
        api_key=key,
        base_url=url,
        model=mdl,
        temperature=temperature,
        top_p=nucleus,
    ):
        parts.append(delta)
    return "".join(parts).strip()


def openai_root(base_url: str) -> str:
    root = (base_url or "https://api.openai.com/v1").rstrip("/")
    if not root.endswith("/v1"):
        root = root + "/v1"
    return root


async def _ping_chat(api_key: str, base_url: str, model: str) -> None:
    """One short non-streaming completion. Thinking stays off so the probe can finish."""
    if not api_key:
        raise RuntimeError("LLM_API_KEY is empty")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
        "enable_thinking": False,
        "messages": [{"role": "user", "content": "回复 ok"}],
    }
    # 管理端按钮不应等满正式对话的 60 秒读超时。连不上就在 25 秒内失败。
    timeout = httpx.Timeout(25.0, connect=8.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        res = await client.post(f"{openai_root(base_url)}/chat/completions", headers=headers, json=body)
        res.raise_for_status()


async def list_models(protocol: str, api_key: str, base_url: str) -> list[str]:
    """Probe the OpenAI-compatible /models list."""
    if not api_key:
        raise ValueError("请先填写 API Key")
    if protocol.startswith("websocket"):
        raise ValueError("WebSocket 音频协议不提供模型列表，请手填模型名")
    if protocol not in {CHAT_PROTOCOL, "openai_embeddings", "openai_embed"} and not protocol.startswith("openai"):
        raise ValueError("只支持 OpenAI Chat Completions")
    headers = {"Authorization": f"Bearer {api_key}", "x-api-key": api_key, "Content-Type": "application/json"}
    urls = [f"{openai_root(base_url)}/models"]
    last_err = "无法获取模型列表"
    async with httpx.AsyncClient(timeout=20.0) as client:
        for url in urls:
            try:
                res = await client.get(url, headers=headers)
                res.raise_for_status()
                data = res.json()
                names = _parse_model_names(data)
                if names:
                    return names
                last_err = "供应商返回了空的模型列表"
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)[:240]
    raise ValueError(last_err)


def _parse_model_names(data: Any) -> list[str]:
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    names: list[str] = []
    for item in rows:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            ident = item.get("id") or item.get("name") or item.get("model")
            if ident:
                names.append(str(ident))
    return sorted(set(names))


async def ping_provider(
    protocol: str,
    api_key: str,
    base_url: str,
    model: str,
    *,
    capability: str = "llm",
) -> tuple[bool, int, str]:
    """Tiny connectivity check used by the admin ping button.

    ASR / TTS must not be probed with /chat/completions. Those models reject a
    text chat (400/500) even when the key and speech model id are valid.
    """
    started = time.perf_counter()
    try:
        if capability in {"asr", "tts"}:
            # 只验证密钥和地址。语音模型经常不出现在 /models，缺席不算失败。
            await list_models(CHAT_PROTOCOL, api_key, base_url)
        elif protocol in {"openai_embeddings", "openai_embed"}:
            await embed_texts(["ping"], api_key=api_key, base_url=base_url, model=model or "text-embedding-3-small")
        elif protocol == CHAT_PROTOCOL or protocol.startswith("openai") or protocol.startswith("anthropic"):
            # 探测只确认密钥、地址和模型能回一个字。正式对话的思考和流式不放进这次短请求，
            # 否则思考模型在 60 秒读超时里还没吐出首块，按钮就会一直失败。
            await _ping_chat(api_key, base_url, model or "gpt-4o-mini")
        else:
            return False, 0, "WebSocket 音频协议本期不测连通性"
        ms = int((time.perf_counter() - started) * 1000)
        return True, ms, "ok"
    except Exception as exc:  # noqa: BLE001 — surface vendor error text to admin
        ms = int((time.perf_counter() - started) * 1000)
        return False, ms, str(exc)[:240]


async def stream_text(
    system: str,
    user: str,
    *,
    max_tokens: int = 900,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    top_p: float | None = None,
) -> AsyncIterator[str]:
    """只交出正文。整段调用不需要思考过程。"""
    async for kind, delta in stream_chat_parts(
        system,
        user,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        top_p=top_p,
    ):
        if kind == "content" and delta:
            yield delta


async def stream_chat_parts(
    system: str,
    user: str,
    *,
    max_tokens: int = 900,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    top_p: float | None = None,
) -> AsyncIterator[tuple[str, str]]:
    """逐块交出 reasoning 或 content。思考不混进正式回答。"""
    settings = get_settings()
    gen = get_rag_config().generation
    key = api_key if api_key is not None else settings.llm_api_key
    url = base_url if base_url is not None else settings.llm_base_url
    mdl = model or settings.llm_model
    nucleus = gen.top_p if top_p is None else top_p
    if not key:
        raise RuntimeError("LLM_API_KEY is empty")
    root = openai_root(url)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body: dict[str, Any] = {
        "model": mdl,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        # 千问兼容接口用 extra 字段开关思考。思考文本只在流里拼接，不返回给调用方。
        "enable_thinking": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if 0 < nucleus <= 1:
        body["top_p"] = nucleus
    # 思考模型首 token 可能很晚。读超时按块计算，不按整段回答计算。
    timeout = _http_timeout(180.0, connect=15.0, read=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        remaining = remaining_budget()
        async with asyncio.timeout(remaining):
            with model_request_budget(system, user, max_tokens, mdl):
                async with client.stream("POST", f"{root}/chat/completions", headers=headers, json=body) as res:
                    res.raise_for_status()
                    async for kind, delta in _iter_chat_parts(res, model=mdl):
                        yield kind, delta


async def _iter_chat_parts(res: httpx.Response, *, model: str = "") -> AsyncIterator[tuple[str, str]]:
    """思考和正文分开。reasoning_content 只作为思考过程，不进入回答。"""
    async for line in res.aiter_lines():
        payload = line.strip()
        if not payload.startswith("data:"):
            continue
        data = payload[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(chunk, dict) and chunk.get("usage") is not None:
            record_provider_usage(chunk["usage"], model)
        choices = chunk.get("choices") if isinstance(chunk, dict) else None
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        choice = choices[0] or {}
        # thinking 是模型当下正在想什么，reasoning 是推理过程。同一块里两者都要交出去，不能因为先读到一个就把另一个丢掉。
        thinking = delta.get("thinking") or choice.get("thinking")
        reasoning = (
            delta.get("reasoning_content")
            or delta.get("reasoning")
            or delta.get("reasoning_text")
            or choice.get("reasoning_content")
            or choice.get("reasoning")
            or _reasoning_details_text(delta.get("reasoning_details") or choice.get("reasoning_details"))
        )
        content = delta.get("content")
        if thinking:
            yield "thinking", str(thinking)
        if reasoning:
            yield "reasoning", str(reasoning)
        if content:
            yield "content", str(content)


def _reasoning_details_text(raw: Any) -> str:
    """兼容接口会把思考放进列表，而不是 reasoning_content 字符串。"""
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return ""
    parts: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("reasoning") or ""
            if str(text).strip():
                parts.append(str(text))
    return "".join(parts)


def _extract_json(raw: str) -> dict[str, Any]:
    """Extract strict JSON and repair only the common provider wrapper failure.

    Some compatible endpoints wrap JSON in markdown fences or prepend prose;
    those wrappers are safe to remove. Python-literal conversion is deliberately
    excluded because it can silently accept malformed model output and weaken
    the scorer/coach contract.
    """
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"LLM did not return JSON: {raw[:240]}")
    payload = text[start : end + 1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        # A frequent Qwen response shape uses a trailing comma. Remove commas
        # immediately before a closing object/array, then retry strict JSON.
        repaired = re.sub(r",\s*([}\]])", r"\1", payload)
        if repaired == payload:
            raise exc
        return json.loads(repaired)


def _parse_embedding_vectors(data: Any) -> list[list[float]]:
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    out: list[list[float]] = []
    for item in rows:
        vec = item.get("embedding") if isinstance(item, dict) else item
        if isinstance(vec, list) and vec:
            out.append([float(x) for x in vec])
    return out


async def embed_texts(
    texts: list[str],
    *,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int = 32,
) -> list[list[float]]:
    """OpenAI-compatible /embeddings. Used by ingest and vector recall."""
    if not api_key:
        raise RuntimeError("embedding API Key is empty")
    if not model:
        raise RuntimeError("embedding model is empty")
    root = openai_root(base_url)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    vectors: list[list[float]] = []
    step = max(1, batch_size)
    async with httpx.AsyncClient(timeout=_http_timeout(45.0, connect=15.0)) as client:
        for i in range(0, len(texts), step):
            body = {"model": model, "input": texts[i : i + step]}
            # Reserve each batch separately so later batches cannot bypass the graph budget.
            with model_request_budget("", "\n".join(body["input"]), 0, model):
                res = await _bounded_post(client, f"{root}/embeddings", headers=headers, json=body)
                res.raise_for_status()
                payload = res.json()
                raw_usage = payload.get("usage") if isinstance(payload, dict) else None
                if isinstance(raw_usage, dict) and "completion_tokens" not in raw_usage:
                    prompt = raw_usage.get("prompt_tokens")
                    total = raw_usage.get("total_tokens")
                    if type(prompt) is int and type(total) is int and prompt == total:
                        raw_usage = {**raw_usage, "completion_tokens": 0}
                record_provider_usage(raw_usage, model)
                batch = _parse_embedding_vectors(payload)
            if len(batch) != len(texts[i : i + step]):
                raise RuntimeError("embedding 返回条数与输入不一致")
            vectors.extend(batch)
    return vectors
