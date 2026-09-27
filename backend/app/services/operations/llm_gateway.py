"""Role → provider routing and the call log written around every model request."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.orm import Session

from app.integrations import llm
from app.agents.contracts.contracts import profile_for
from app.agents.orchestration.trace import current_run_id
from app.integrations.llm import CHAT_PROTOCOL
from app.agents.providers.governance import ProviderGovernor
from app.agents.providers.router import choose_provider
from app.core.config import get_settings
from app.models import LlmCallLog, ProviderConfig, RoleBinding
from app.core.rag_config import get_rag_config
from app.services.shared.common import new_id


def route(db: Session, role: str) -> tuple[ProviderConfig | None, RoleBinding | None]:
    """Role binding, then health, then the contract fallback. Same three layers as choose_provider."""
    choice = choose_provider(db, role)
    return choice.provider, choice.binding


async def complete(
    db: Session,
    role: str,
    system: str,
    user: str,
    max_tokens: int | None = None,
    expect_json: bool = False,
    temperature: float | None = None,
    deadline_at: float | None = None,
) -> Any:
    """Call the routed model and fold the result into that vendor's breaker."""
    return await complete_with(
        db, role, system, user, max_tokens=max_tokens, expect_json=expect_json, temperature=temperature,
        deadline_at=deadline_at,
    )


async def complete_with(
    db: Session,
    role: str,
    system: str,
    user: str,
    *,
    max_tokens: int | None = None,
    expect_json: bool = False,
    temperature: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    deadline_at: float | None = None,
) -> Any:
    """Route one call; standard tool calls are optional so legacy providers still work."""
    with llm.deadline_scope(deadline_at):
        return await _complete_with(db, role, system, user, max_tokens=max_tokens,
                                    expect_json=expect_json, temperature=temperature, tools=tools)


async def _complete_with(
    db: Session, role: str, system: str, user: str, *, max_tokens: int | None,
    expect_json: bool, temperature: float | None, tools: list[dict[str, Any]] | None,
) -> Any:
    """Execute and log the routed request inside the shared deadline scope."""
    llm.remaining_budget()
    choice = choose_provider(db, role)
    provider, binding = choice.provider, choice.binding
    # An open breaker already redirected choose_provider. Record against the vendor we actually call.
    settings = get_settings()
    api_key = (provider.api_key if provider else "") or settings.llm_api_key
    base_url = (provider.base_url if provider else "") or settings.llm_base_url
    model = (binding.model if binding and binding.model else None) or settings.llm_model
    # 文本角色固定走 Chat Completions。供应商上残留的旧协议名不再决定请求格式。
    protocol = CHAT_PROTOCOL
    gen = get_rag_config().generation
    contract = profile_for(choice.role)
    if temperature is not None:
        temp = temperature
    elif binding is not None:
        temp = binding.temperature
    else:
        temp = contract.temperature
    token_budget = gen.max_tokens if max_tokens is None else max_tokens
    governor = ProviderGovernor(db, provider.id if provider else None, provider.name if provider else "env")
    if provider is not None and not governor.allow():
        raise RuntimeError(f"供应商 {provider.name} 熔断中")
    started = time.perf_counter()
    with llm.call_usage_scope() as measured:
        try:
            if tools:
                # Providers receive a real function schema; callers still get the internal normalized shape.
                data = await llm.complete_tool_call(
                    system, user, tools=tools, max_tokens=token_budget,
                    api_key=api_key, base_url=base_url, model=model,
                    temperature=temp, top_p=gen.top_p,
                )
                ms = int((time.perf_counter() - started) * 1000)
                log_call(db, choice.role, provider.name if provider else "env", model, "ok", ms, None, usage=llm.combine_usage(measured))
                governor.record_success(ms)
                return data
            if expect_json:
                data = await llm.complete_json(
                    system, user, max_tokens=token_budget,
                    api_key=api_key, base_url=base_url, model=model,
                    temperature=temp, top_p=gen.top_p, protocol=protocol,
                )
                ms = int((time.perf_counter() - started) * 1000)
                log_call(db, choice.role, provider.name if provider else "env", model, "ok", ms, None, usage=llm.combine_usage(measured))
                governor.record_success(ms)
                return data
            answer = await llm.complete_text(
                system, user, max_tokens=token_budget,
                api_key=api_key, base_url=base_url, model=model,
                temperature=temp, top_p=gen.top_p, protocol=protocol,
            )
            ms = int((time.perf_counter() - started) * 1000)
            log_call(db, choice.role, provider.name if provider else "env", model, "ok", ms, None, usage=llm.combine_usage(measured))
            governor.record_success(ms)
            return answer
        except Exception as exc:
            ms = int((time.perf_counter() - started) * 1000)
            log_call(db, choice.role, provider.name if provider else "env", model, "error", ms, str(exc)[:240], usage=llm.combine_usage(measured))
            if not isinstance(exc, llm.UsageBudgetError):
                governor.record_failure(ms)
            raise


async def stream_parts(
    db: Session,
    role: str,
    system: str,
    user: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    deadline_at: float | None = None,
) -> AsyncIterator[tuple[str, str]]:
    """Yield reasoning and answer deltas as the vendor sends them.

    成功或失败都在流结束时记一次调用。中途失败不再补一条整段调用，避免同一轮记两次。
    """
    with llm.deadline_scope(deadline_at):
        async for part in _stream_parts(db, role, system, user, max_tokens, temperature):
            yield part


async def _stream_parts(
    db: Session, role: str, system: str, user: str,
    max_tokens: int | None, temperature: float | None,
) -> AsyncIterator[tuple[str, str]]:
    """Execute a streaming provider request inside the shared deadline scope."""
    llm.remaining_budget()
    choice = choose_provider(db, role)
    provider, binding = choice.provider, choice.binding
    settings = get_settings()
    api_key = (provider.api_key if provider else "") or settings.llm_api_key
    base_url = (provider.base_url if provider else "") or settings.llm_base_url
    model = (binding.model if binding and binding.model else None) or settings.llm_model
    gen = get_rag_config().generation
    contract = profile_for(choice.role)
    if temperature is not None:
        temp = temperature
    elif binding is not None:
        temp = binding.temperature
    else:
        temp = contract.temperature
    token_budget = gen.max_tokens if max_tokens is None else max_tokens
    governor = ProviderGovernor(db, provider.id if provider else None, provider.name if provider else "env")
    if provider is not None and not governor.allow():
        raise RuntimeError(f"供应商 {provider.name} 熔断中")
    started = time.perf_counter()
    with llm.call_usage_scope() as measured:
        try:
            async for kind, delta in llm.stream_chat_parts(
                system,
                user,
                max_tokens=token_budget,
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temp,
                top_p=gen.top_p,
            ):
                yield kind, delta
        except Exception as exc:
            ms = int((time.perf_counter() - started) * 1000)
            log_call(db, choice.role, provider.name if provider else "env", model, "error", ms, str(exc)[:240], usage=llm.combine_usage(measured))
            if not isinstance(exc, llm.UsageBudgetError):
                governor.record_failure(ms)
            raise
        ms = int((time.perf_counter() - started) * 1000)
        log_call(db, choice.role, provider.name if provider else "env", model, "ok", ms, None, usage=llm.combine_usage(measured))
        governor.record_success(ms)


def log_call(
    db: Session, role: str, provider_name: str, model: str, status: str, latency_ms: int, error: str | None,
    *, usage: llm.ModelUsage | None = None,
) -> None:
    # Provider exceptions can echo headers and prompt fragments. Persist a
    # stable category only; the operational trace already has role and status.
    safe_error = "provider_call_failed" if error else None
    db.add(
        LlmCallLog(
            id=new_id(),
            run_id=current_run_id(),
            role=role,
            provider_name=provider_name,
            model=model,
            status=status,
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            estimated_cost=usage.estimated_cost if usage else None,
            error=safe_error,
        )
    )
