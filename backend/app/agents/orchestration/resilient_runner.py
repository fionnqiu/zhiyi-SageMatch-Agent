"""Integration wrapper for resilient agent execution in the orchestration layer.

Provides drop-in replacements for run_agent and run_handoff with automatic
retry, degradation, and circuit breaker capabilities.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy.orm import Session

from app.agents.contracts.contracts import profile_for
from app.agents.orchestration.resilience import ResilientAgent, RetryPolicy, DegradationStrategy
from app.agents.roles.loop import run_agent as _original_run_agent


async def run_agent_resilient(
    db: Session,
    role: str,
    *,
    user: str,
    context: str = "",
    seed: dict[str, Any] | None = None,
    on_thought=None,
    thread_id: str | None = None,
    checkpointer: Any | None = None,
    deadline_at: float | None = None,
    enable_retry: bool = True,
) -> dict[str, Any]:
    """带弹性能力的 Agent 执行入口。

    Args:
        db: 数据库会话
        role: Agent 角色名称
        user: 用户输入（goal）
        context: 上下文信息
        seed: 种子数据
        on_thought: 思考回调
        thread_id: 线程 ID
        checkpointer: Checkpoint 存储
        deadline_at: 截止时间（monotonic）
        enable_retry: 是否启用重试（默认 True）

    Returns:
        Agent 执行结果（包含弹性诊断信息）
    """
    profile = profile_for(role)

    # 关键角色启用重试，非关键角色直接调用
    critical_roles = {"author", "critic", "scorer", "interviewer"}

    if not enable_retry or role not in critical_roles:
        # 直接调用原始函数
        return await _original_run_agent(
            db, profile,
            user=user, context=context, seed=seed,
            on_thought=on_thought, thread_id=thread_id,
            checkpointer=checkpointer, deadline_at=deadline_at,
        )

    # 使用弹性包装器
    resilient = ResilientAgent(
        db,
        retry_policy=RetryPolicy(
            max_retries=2 if role == "interviewer" else 3,  # Interviewer 重试少一些
            base_backoff=1.0,
            exponential=True,
        ),
        degradation=DegradationStrategy(
            context_reduction_ratio=0.7,
            min_context_length=200,
            reduce_tool_budget=True,
            reduce_temperature=True,
            reduce_max_tokens=True,
        ),
    )

    # 构造参数字典
    input_data = {
        "user": user,
        "context": context,
        "seed": seed,
        "on_thought": on_thought,
        "thread_id": thread_id,
        "checkpointer": checkpointer,
        "deadline_at": deadline_at,
    }

    # 定义 Agent 调用函数
    async def agent_fn(db_session: Session, **kwargs):
        return await _original_run_agent(db_session, profile, **kwargs)

    # 执行弹性调用
    result = await resilient.invoke_with_resilience(
        agent_fn,
        role=role,
        input_data=input_data,
    )

    return result


# 便捷函数：直接用角色名调用
async def run_author_resilient(db: Session, user: str, context: str = "", **kwargs) -> dict[str, Any]:
    """弹性调用 Author Agent"""
    return await run_agent_resilient(db, "author", user=user, context=context, **kwargs)


async def run_critic_resilient(db: Session, user: str, context: str = "", **kwargs) -> dict[str, Any]:
    """弹性调用 Critic Agent"""
    return await run_agent_resilient(db, "critic", user=user, context=context, **kwargs)


async def run_scorer_resilient(db: Session, user: str, context: str = "", **kwargs) -> dict[str, Any]:
    """弹性调用 Scorer Agent"""
    return await run_agent_resilient(db, "scorer", user=user, context=context, **kwargs)


async def run_interviewer_resilient(db: Session, user: str, context: str = "", **kwargs) -> dict[str, Any]:
    """弹性调用 Interviewer Agent"""
    return await run_agent_resilient(db, "interviewer", user=user, context=context, **kwargs)


__all__ = [
    "run_agent_resilient",
    "run_author_resilient",
    "run_critic_resilient",
    "run_scorer_resilient",
    "run_interviewer_resilient",
]
