"""Tests for resilient agent execution."""

import asyncio
import pytest

from app.agents.orchestration.resilience import (
    ResilientAgent,
    RetryPolicy,
    DegradationStrategy,
    CircuitBreaker,
)
from app.agents.contracts.state import AgentResult

# 使用 anyio 后端运行所有异步测试
pytestmark = pytest.mark.anyio


class MockDB:
    """模拟数据库会话"""
    pass


async def test_retry_policy_should_retry():
    """测试重试策略：判断是否应该重试"""
    policy = RetryPolicy(max_retries=3)

    # 可重试错误
    assert policy.should_retry("timeout", 0) is True
    assert policy.should_retry("rate_limit", 1) is True
    assert policy.should_retry("network_error", 2) is True

    # 超过最大重试次数
    assert policy.should_retry("timeout", 3) is False

    # 不可重试错误
    assert policy.should_retry("invalid_input", 0) is False


def test_retry_policy_backoff():
    """测试退避时间计算"""
    policy = RetryPolicy(base_backoff=2.0, backoff_multiplier=2.0, exponential=True)

    # 指数退避: 2.0 * (2.0 ^ attempt)
    assert policy.backoff_seconds(0) == 2.0
    assert policy.backoff_seconds(1) == 4.0
    assert policy.backoff_seconds(2) == 8.0


def test_retry_policy_max_backoff():
    """测试最大退避时间"""
    policy = RetryPolicy(base_backoff=10.0, max_backoff=20.0, exponential=True)

    # 不应超过 max_backoff
    assert policy.backoff_seconds(5) == 20.0


async def test_resilient_agent_success_on_first_try():
    """测试首次调用成功"""
    db = MockDB()
    agent = ResilientAgent(db)

    async def mock_agent_fn(db_session, **kwargs):
        return {"ok": True, "output": {"result": "success"}}

    result = await agent.invoke_with_resilience(
        mock_agent_fn,
        role="author",
        input_data={"goal": "test"},
    )

    assert result["ok"] is True
    assert result["resilience_diagnostics"]["attempts"] == 1
    assert result["resilience_diagnostics"]["final_status"] == "success"


async def test_resilient_agent_retry_on_timeout():
    """测试超时重试"""
    db = MockDB()
    agent = ResilientAgent(db, retry_policy=RetryPolicy(max_retries=2, base_backoff=0.1))

    call_count = 0

    async def mock_agent_fn(db_session, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            raise TimeoutError("timeout")

        return {"ok": True, "output": {"result": "success"}}

    result = await agent.invoke_with_resilience(
        mock_agent_fn,
        role="author",
        input_data={"goal": "test"},
    )

    # 应该重试 2 次后成功
    assert call_count == 3
    assert result["ok"] is True
    assert result["resilience_diagnostics"]["attempts"] == 3


async def test_resilient_agent_context_degradation():
    """测试上下文降级"""
    db = MockDB()
    degradation = DegradationStrategy(context_reduction_ratio=0.5)
    agent = ResilientAgent(db, degradation=degradation)

    call_count = 0
    received_contexts = []

    async def mock_agent_fn(db_session, **kwargs):
        nonlocal call_count
        call_count += 1
        received_contexts.append(kwargs.get("context", ""))

        if call_count < 2:
            raise TimeoutError("timeout")

        return {"ok": True, "output": {"result": "success"}}

    result = await agent.invoke_with_resilience(
        mock_agent_fn,
        role="author",
        input_data={"goal": "test", "context": "A" * 1000},
    )

    # 第二次调用的上下文应该更短
    assert len(received_contexts[1]) < len(received_contexts[0])
    assert "context_truncated" in result["resilience_diagnostics"]["degradations_applied"][0]


async def test_resilient_agent_budget_exceeded():
    """测试预算耗尽（不可重试）"""
    from app.integrations import llm

    db = MockDB()
    agent = ResilientAgent(db)

    async def mock_agent_fn(db_session, **kwargs):
        raise llm.UsageBudgetError("budget exceeded")

    result = await agent.invoke_with_resilience(
        mock_agent_fn,
        role="author",
        input_data={"goal": "test"},
    )

    assert result["ok"] is False
    assert result["error_code"] == "model_budget_exceeded"
    assert result["resilience_diagnostics"]["attempts"] == 1  # 不重试


def test_circuit_breaker_closed_state():
    """测试断路器：关闭状态（正常）"""
    breaker = CircuitBreaker("test", failure_threshold=3)

    assert breaker.state == "CLOSED"
    assert breaker.can_proceed() is True

    # 记录失败，但未达到阈值
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == "CLOSED"
    assert breaker.can_proceed() is True


def test_circuit_breaker_open_state():
    """测试断路器：打开状态（拒绝请求）"""
    breaker = CircuitBreaker("test", failure_threshold=3, timeout_seconds=1.0)

    # 记录失败，达到阈值
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == "OPEN"
    assert breaker.can_proceed() is False


async def test_circuit_breaker_half_open_state():
    """测试断路器：半开状态（尝试恢复）"""
    breaker = CircuitBreaker("test", failure_threshold=2, timeout_seconds=0.1)

    # 打开断路器
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == "OPEN"

    # 等待恢复时间
    await asyncio.sleep(0.15)

    # 应该进入半开状态
    assert breaker.can_proceed() is True

    # 记录成功，应该关闭
    breaker.record_success()
    breaker.record_success()

    assert breaker.state == "CLOSED"


def test_circuit_breaker_half_open_to_open():
    """测试断路器：半开状态失败后重新打开"""
    breaker = CircuitBreaker("test", failure_threshold=2, timeout_seconds=0.0)

    # 打开断路器
    breaker.record_failure()
    breaker.record_failure()

    # 立即进入半开状态（timeout=0）
    breaker.state = "HALF_OPEN"

    # 半开状态下失败，应该重新打开
    breaker.record_failure()

    assert breaker.state == "OPEN"


def test_circuit_breaker_success_resets_failures():
    """测试断路器：成功调用重置失败计数"""
    breaker = CircuitBreaker("test", failure_threshold=3)

    breaker.record_failure()
    breaker.record_failure()

    assert breaker.failure_count == 2

    # 成功调用应该重置计数
    breaker.record_success()

    assert breaker.failure_count == 0
    assert breaker.state == "CLOSED"
