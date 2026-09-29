"""Bounded model decisions shared by the interview generation entry points."""

from __future__ import annotations

from typing import Any

from app.integrations import llm


async def choose_generation_revision(db: Any, reason: str) -> tuple[str, str]:
    """Choose one safe exit after a critic veto, defaulting to the prior rewrite path."""
    # Service initialization imports role modules, so resolve the gateway only
    # when a model decision is actually needed.
    from app.services.operations.llm_gateway import complete_with

    try:
        response = await complete_with(
            db, "analyst",
            "你是出题流程主管。只输出 JSON 对象，next_role 只能是 author 或 finish。"
            "author 表示再重写一次；finish 表示结束并使用确定性兜底题包。",
            f"质检未通过：{reason[:200]}。请选择下一角色。",
            expect_json=True, temperature=0, max_tokens=80,
        )
        selected = response.get("next_role") if isinstance(response, dict) else None
        if selected in {"author", "finish"}:
            return selected, ""
        return "author", "invalid_decision"
    except (llm.UsageBudgetError, TimeoutError):
        raise
    except Exception:
        # A provider outage must not turn a rejected candidate into a commit.
        return "author", "provider_failed"
