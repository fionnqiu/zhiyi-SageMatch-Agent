"""Tool governance enhancements with budget tracking and circuit breaker integration.

Extends the existing governance module with fine-grained resource control,
per-session budget tracking, and automatic tool disabling on repeated failures.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.agents.orchestration.resilience import get_circuit_breaker


@dataclass
class ToolBudget:
    """单次对话的工具调用预算"""

    # 工具级别预算（工具名 -> 剩余次数）
    tool_limits: dict[str, int] = field(default_factory=lambda: {
        "hybrid_search": 5,      # RAG 检索最多 5 次
        "rerank": 3,             # 重排最多 3 次
        "get_turn_quote": 10,    # 引用查询最多 10 次
        "finish": 1,             # finish 只能调用 1 次
    })

    # 角色级别预算（角色名 -> 剩余 LLM 调用次数）
    role_limits: dict[str, int] = field(default_factory=lambda: {
        "author": 3,        # Author 最多 3 次 LLM 调用
        "critic": 2,        # Critic 最多 2 次
        "interviewer": 15,  # Interviewer 可以多次追问
        "scorer": 5,        # Scorer 每个维度 1 次，共 4-5 次
        "coach": 2,         # Coach 最多 2 次
    })

    # 全局 token 预算
    total_token_budget: int = 100000  # 单次对话 10 万 token
    tokens_used: int = 0

    # 调用计数器
    tool_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    role_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def can_call_tool(self, tool_name: str) -> tuple[bool, str]:
        """检查工具是否可调用

        Returns:
            (是否允许, 拒绝原因)
        """
        # 检查工具级预算
        limit = self.tool_limits.get(tool_name)
        if limit is not None:
            used = self.tool_calls[tool_name]
            if used >= limit:
                return False, f"tool_budget_exceeded:{tool_name}:{used}/{limit}"

        # 检查断路器
        breaker = get_circuit_breaker(f"tool:{tool_name}")
        if not breaker.can_proceed():
            return False, f"circuit_breaker_open:{tool_name}"

        return True, ""

    def can_call_role(self, role: str) -> tuple[bool, str]:
        """检查角色是否可调用"""
        # 检查角色级预算
        limit = self.role_limits.get(role)
        if limit is not None:
            used = self.role_calls[role]
            if used >= limit:
                return False, f"role_budget_exceeded:{role}:{used}/{limit}"

        # 检查断路器
        breaker = get_circuit_breaker(f"role:{role}")
        if not breaker.can_proceed():
            return False, f"circuit_breaker_open:{role}"

        return True, ""

    def can_spend_tokens(self, estimated_tokens: int) -> tuple[bool, str]:
        """检查 token 预算"""
        if self.tokens_used + estimated_tokens > self.total_token_budget:
            return False, f"token_budget_exceeded:{self.tokens_used}/{self.total_token_budget}"
        return True, ""

    def record_tool_call(self, tool_name: str, *, success: bool = True):
        """记录工具调用"""
        self.tool_calls[tool_name] += 1

        # 更新断路器
        breaker = get_circuit_breaker(f"tool:{tool_name}")
        if success:
            breaker.record_success()
        else:
            breaker.record_failure()

    def record_role_call(self, role: str, *, success: bool = True):
        """记录角色调用"""
        self.role_calls[role] += 1

        # 更新断路器
        breaker = get_circuit_breaker(f"role:{role}")
        if success:
            breaker.record_success()
        else:
            breaker.record_failure()

    def record_tokens(self, tokens: int):
        """记录 token 消耗"""
        self.tokens_used += tokens

    def summary(self) -> dict[str, Any]:
        """获取预算使用摘要"""
        return {
            "tool_usage": dict(self.tool_calls),
            "role_usage": dict(self.role_calls),
            "tokens_used": self.tokens_used,
            "tokens_remaining": max(0, self.total_token_budget - self.tokens_used),
            "budget_exhausted": self.tokens_used >= self.total_token_budget,
        }


# 会话级预算管理器（内存存储，进程重启后重置）
_session_budgets: dict[str, ToolBudget] = {}


def get_session_budget(session_id: str) -> ToolBudget:
    """获取或创建会话预算"""
    if session_id not in _session_budgets:
        _session_budgets[session_id] = ToolBudget()
    return _session_budgets[session_id]


def clear_session_budget(session_id: str):
    """清除会话预算（会话结束时调用）"""
    _session_budgets.pop(session_id, None)


class EnhancedToolGovernor:
    """增强的工具治理器，集成预算和断路器"""

    def __init__(self, db: Session, session_id: str | None = None):
        self.db = db
        self.session_id = session_id or "default"
        self.budget = get_session_budget(self.session_id)

    def check_tool_call(self, tool_name: str, role: str) -> tuple[bool, str]:
        """调用前检查（工具 + 角色 + token 预算）"""
        # 检查工具预算
        can_tool, reason_tool = self.budget.can_call_tool(tool_name)
        if not can_tool:
            return False, reason_tool

        # 检查角色预算
        can_role, reason_role = self.budget.can_call_role(role)
        if not can_role:
            return False, reason_role

        # 估算 token 消耗（简化估计）
        estimated_tokens = self._estimate_tokens(tool_name)
        can_tokens, reason_tokens = self.budget.can_spend_tokens(estimated_tokens)
        if not can_tokens:
            return False, reason_tokens

        return True, ""

    def record_call(
        self,
        tool_name: str,
        role: str,
        *,
        success: bool,
        tokens_used: int = 0,
    ):
        """调用后记录"""
        self.budget.record_tool_call(tool_name, success=success)
        self.budget.record_role_call(role, success=success)

        if tokens_used > 0:
            self.budget.record_tokens(tokens_used)

    def _estimate_tokens(self, tool_name: str) -> int:
        """估算工具调用的 token 消耗"""
        # 粗略估计（实际消耗在调用后更新）
        estimates = {
            "hybrid_search": 2000,   # RAG 检索 + embedding
            "rerank": 500,           # 重排模型
            "get_turn_quote": 100,   # 简单查询
            "finish": 0,             # 不消耗 token
        }
        return estimates.get(tool_name, 500)

    def get_budget_summary(self) -> dict[str, Any]:
        """获取预算摘要"""
        return self.budget.summary()


def tool_governance_status(session_id: str | None = None) -> dict[str, Any]:
    """获取工具治理状态（用于管理端监控）"""
    if session_id:
        budget = get_session_budget(session_id)
        return {
            "session_id": session_id,
            **budget.summary(),
        }

    # 返回所有会话的摘要
    return {
        "active_sessions": len(_session_budgets),
        "sessions": {
            sid: budget.summary()
            for sid, budget in _session_budgets.items()
        },
    }


__all__ = [
    "ToolBudget",
    "EnhancedToolGovernor",
    "get_session_budget",
    "clear_session_budget",
    "tool_governance_status",
]
