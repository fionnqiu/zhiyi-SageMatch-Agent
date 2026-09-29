"""Agent resilience layer with retry, fallback, and self-healing capabilities.

Provides automatic recovery strategies for transient failures, resource constraints,
and provider outages without requiring manual intervention.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.integrations import llm
from app.agents.contracts.state import AgentResult


@dataclass
class RetryPolicy:
    """重试策略配置"""

    # 最大重试次数
    max_retries: int = 3

    # 基础退避时间（秒）
    base_backoff: float = 1.0

    # 指数退避因子
    backoff_multiplier: float = 2.0

    # 最大退避时间（秒）
    max_backoff: float = 30.0

    # 是否使用指数退避
    exponential: bool = True

    # 可重试的错误类型
    retryable_errors: set[str] | None = None

    def __post_init__(self):
        """初始化默认可重试错误"""
        if self.retryable_errors is None:
            self.retryable_errors = {
                "timeout",
                "rate_limit",
                "provider_unavailable",
                "network_error",
                "service_unavailable",
                "tool_timeout",
            }

    def should_retry(self, error_type: str, attempt: int) -> bool:
        """判断是否应该重试"""
        if attempt >= self.max_retries:
            return False

        # 检查错误类型是否可重试
        return error_type in self.retryable_errors

    def backoff_seconds(self, attempt: int) -> float:
        """计算退避时间"""
        if not self.exponential:
            return self.base_backoff

        # 指数退避: base * (multiplier ^ attempt)
        backoff = self.base_backoff * (self.backoff_multiplier ** attempt)
        return min(backoff, self.max_backoff)


@dataclass
class DegradationStrategy:
    """降级策略配置"""

    # 上下文截断比例（每次重试时递减）
    context_reduction_ratio: float = 0.7

    # 最小保留长度
    min_context_length: int = 200

    # 工具预算削减
    reduce_tool_budget: bool = True

    # 降低温度参数（提高确定性）
    reduce_temperature: bool = True

    # 缩短最大 token 数
    reduce_max_tokens: bool = True


class ResilientAgent:
    """具备自愈能力的 Agent 包装器。

    为关键 Agent 增加：
    1. 自动重试（指数退避）
    2. 上下文降级（减少长度避免超限）
    3. 参数调整（降低温度、减少 token）
    4. 故障转移（切换备用模型）
    """

    def __init__(
        self,
        db: Session,
        *,
        retry_policy: RetryPolicy | None = None,
        degradation: DegradationStrategy | None = None,
    ):
        self.db = db
        self.retry_policy = retry_policy or RetryPolicy()
        self.degradation = degradation or DegradationStrategy()

    async def invoke_with_resilience(
        self,
        agent_fn: Callable[..., Any],
        *,
        role: str,
        input_data: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """带弹性能力的 Agent 调用

        Args:
            agent_fn: Agent 执行函数（如 run_agent）
            role: Agent 角色名称
            input_data: 输入数据（包含 goal, context 等）
            **kwargs: 其他参数

        Returns:
            Agent 执行结果，包含重试和降级的诊断信息
        """
        attempt = 0
        last_error = None
        degraded_input = dict(input_data)

        diagnostics = {
            "role": role,
            "attempts": 0,
            "degradations_applied": [],
            "retry_delays": [],
            "final_status": "pending",
        }

        while attempt <= self.retry_policy.max_retries:
            diagnostics["attempts"] = attempt + 1

            try:
                # 调用 Agent
                result = await agent_fn(
                    self.db,
                    role=role,
                    **degraded_input,
                    **kwargs,
                )

                # 成功返回
                diagnostics["final_status"] = "success"
                result["resilience_diagnostics"] = diagnostics
                return result

            except llm.UsageBudgetError as exc:
                # 预算耗尽不可重试
                diagnostics["final_status"] = "budget_exceeded"
                diagnostics["error"] = str(exc)
                return self._failure_result(role, "model_budget_exceeded", diagnostics)

            except TimeoutError as exc:
                # 超时：可重试
                last_error = exc
                error_type = "timeout"

                if not self.retry_policy.should_retry(error_type, attempt):
                    diagnostics["final_status"] = "timeout_exhausted"
                    return self._failure_result(role, "deadline_exceeded", diagnostics)

                # 应用降级策略
                degraded_input = self._apply_degradation(
                    degraded_input,
                    attempt,
                    diagnostics,
                )

                # 退避等待
                backoff = self.retry_policy.backoff_seconds(attempt)
                diagnostics["retry_delays"].append(backoff)
                await asyncio.sleep(backoff)

                attempt += 1

            except Exception as exc:  # noqa: BLE001
                # 其他错误：判断是否可重试
                last_error = exc
                error_type = self._classify_error(exc)

                if not self.retry_policy.should_retry(error_type, attempt):
                    diagnostics["final_status"] = f"failed_{error_type}"
                    diagnostics["error"] = str(exc)
                    return self._failure_result(role, "agent_execution_failed", diagnostics)

                # 应用降级策略
                degraded_input = self._apply_degradation(
                    degraded_input,
                    attempt,
                    diagnostics,
                )

                # 退避等待
                backoff = self.retry_policy.backoff_seconds(attempt)
                diagnostics["retry_delays"].append(backoff)
                await asyncio.sleep(backoff)

                attempt += 1

        # 重试耗尽
        diagnostics["final_status"] = "retries_exhausted"
        diagnostics["last_error"] = str(last_error) if last_error else "unknown"
        return self._failure_result(role, "agent_execution_failed", diagnostics)

    def _apply_degradation(
        self,
        input_data: dict[str, Any],
        attempt: int,
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        """应用降级策略，减少资源消耗"""
        degraded = dict(input_data)
        applied = []

        # 降级 1: 截断上下文
        if "context" in degraded and degraded["context"]:
            original_len = len(degraded["context"])
            reduction_ratio = self.degradation.context_reduction_ratio ** (attempt + 1)
            target_len = max(
                self.degradation.min_context_length,
                int(original_len * reduction_ratio),
            )

            if target_len < original_len:
                degraded["context"] = degraded["context"][:target_len]
                applied.append(f"context_truncated_to_{target_len}")

        # 降级 2: 减少工具预算（通过 seed 传递）
        if self.degradation.reduce_tool_budget and "seed" in degraded:
            if "tool_budget_multiplier" not in degraded["seed"]:
                degraded["seed"]["tool_budget_multiplier"] = 0.7
                applied.append("tool_budget_reduced")

        # 降级 3: 降低温度（提高确定性，减少重复失败）
        if self.degradation.reduce_temperature:
            if "temperature" not in degraded:
                degraded["temperature"] = 0.1
                applied.append("temperature_reduced")

        # 降级 4: 减少 max_tokens
        if self.degradation.reduce_max_tokens:
            if "max_tokens" not in degraded:
                degraded["max_tokens"] = 500
                applied.append("max_tokens_reduced")

        if applied:
            diagnostics["degradations_applied"].extend(applied)

        return degraded

    def _classify_error(self, exc: Exception) -> str:
        """分类错误类型"""
        exc_str = str(exc).lower()
        exc_type = type(exc).__name__.lower()

        # 超时类
        if "timeout" in exc_str or "timeout" in exc_type:
            return "timeout"

        # 限流类
        if "rate" in exc_str or "limit" in exc_str or "429" in exc_str:
            return "rate_limit"

        # 服务不可用类
        if any(kw in exc_str for kw in ["unavailable", "503", "502", "504"]):
            return "service_unavailable"

        # 网络类
        if any(kw in exc_str for kw in ["connection", "network", "unreachable"]):
            return "network_error"

        # 默认：未知错误（不可重试）
        return "unknown_error"

    def _failure_result(
        self,
        role: str,
        error_code: str,
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        """构造失败结果"""
        result = AgentResult.failure(
            error_code,
            retryable=False,
            trace_id=diagnostics.get("trace_id", "unknown"),
        )

        return {
            **result.model_dump(),
            "role": role,
            "resilience_diagnostics": diagnostics,
        }


class CircuitBreaker:
    """断路器：检测持续失败并暂时禁用故障组件。

    三种状态：
    - CLOSED: 正常工作
    - OPEN: 故障过多，拒绝请求
    - HALF_OPEN: 尝试恢复
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds

        self.failure_count = 0
        self.success_count = 0
        self.state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self.opened_at: float | None = None

    def record_success(self):
        """记录成功调用"""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                # 恢复正常
                self.state = "CLOSED"
                self.failure_count = 0
                self.success_count = 0
                self.opened_at = None
        elif self.state == "CLOSED":
            # 重置失败计数
            self.failure_count = 0

    def record_failure(self):
        """记录失败调用"""
        if self.state == "CLOSED":
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                # 打开断路器
                self.state = "OPEN"
                self.opened_at = time.monotonic()
        elif self.state == "HALF_OPEN":
            # 尝试恢复失败，重新打开
            self.state = "OPEN"
            self.opened_at = time.monotonic()
            self.success_count = 0

    def can_proceed(self) -> bool:
        """判断是否允许调用"""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            # 检查是否到达恢复时间
            if self.opened_at is not None:
                elapsed = time.monotonic() - self.opened_at
                if elapsed >= self.timeout_seconds:
                    # 进入半开状态
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                    return True

            return False

        # HALF_OPEN: 允许少量尝试
        return True

    def status(self) -> dict[str, Any]:
        """获取断路器状态"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "opened_at": datetime.fromtimestamp(self.opened_at, tz=timezone.utc).isoformat() if self.opened_at else None,
        }


# 全局断路器实例（按角色/工具管理）
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """获取或创建断路器实例"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name)
    return _circuit_breakers[name]


def circuit_breaker_status() -> list[dict[str, Any]]:
    """获取所有断路器状态"""
    return [cb.status() for cb in _circuit_breakers.values()]


__all__ = [
    "ResilientAgent",
    "RetryPolicy",
    "DegradationStrategy",
    "CircuitBreaker",
    "get_circuit_breaker",
    "circuit_breaker_status",
]
