"""Production adapters that run business writes inside the application graph."""

from __future__ import annotations

import uuid
import logging
import time
from functools import wraps
from datetime import datetime, timedelta, timezone
from collections.abc import Awaitable, Callable
from typing import Any, Mapping, TypeVar

from sqlalchemy.orm import Session

from app.agents.orchestration.checkpoint import (
    CheckpointValidationError, authorize_checkpoint, checkpoint_config, derive_thread_id,
)
from app.agents.orchestration.graph import AgentState, StageCallback, SupervisedRoute, build_application_graph
from app.agents.orchestration.observability import persist_graph_trace
from app.agents.orchestration.trace import graph_run_scope
from app.integrations import llm
from app.core.config import get_settings
from app.core.db import graph_commit_deadline
from app.services.shared.common import ANON

T = TypeVar("T")
logger = logging.getLogger(__name__)


def _exclusive_checkpoint_run(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Serialize a PostgreSQL thread before inspecting its recovery snapshot."""
    @wraps(fn)
    async def guarded(db: Session, **kwargs: Any) -> T:
        saver = kwargs.get("checkpointer")
        if saver is None or not hasattr(saver, "exclusive_run"):
            return await fn(db, **kwargs)
        run_id = kwargs.get("run_id") or uuid.uuid4().hex
        kwargs["run_id"] = run_id
        thread_id = kwargs.get("thread_id") or derive_thread_id(
            session_id=kwargs.get("session_id"), interview_id=kwargs.get("interview_id"), request_id=run_id,
        )
        async with saver.exclusive_run(thread_id):
            return await fn(db, **kwargs)
    return guarded


class GraphExecutionError(RuntimeError):
    """Safe public failure when a graph exits without a business commit."""

    def __init__(self, error_code: str = "business_graph_failed", *, retryable: bool = True) -> None:
        super().__init__("business graph did not commit")
        self.error_code = error_code[:80]
        self.retryable = retryable


@_exclusive_checkpoint_run
async def run_business_graph(
    db: Session,
    *,
    mode: str,
    action: Callable[[], Awaitable[T]],
    original_query: str = "",
    session_id: str | None = None,
    interview_id: str | None = None,
    thread_id: str | None = None,
    checkpointer: Any | None = None,
    run_id: str | None = None,
    deadline_at: float | None = None,
    stages: Mapping[str, StageCallback] | None = None,
    stage_sequences: Mapping[str, tuple[str, ...]] | None = None,
    supervised_route: SupervisedRoute | None = None,
    restart_incomplete: bool = False,
    first_commit: Callable[[], Awaitable[Any]] | None = None,
    first_commit_after: str | None = None,
    restart_completed_failed: bool = False,
) -> T:
    """Run a bounded graph with controlled business writes at commit nodes.

    The callback captures the ORM session outside graph state.  The checkpoint
    carries only IDs and a compact result, while existing services remain the
    fact writers during migration to finer-grained business subgraph nodes.
    """
    run_id = run_id or uuid.uuid4().hex
    deadline_at = deadline_at if deadline_at is not None else time.monotonic() + 300.0
    # Durable workers may need one checkpoint thread per attempt even when
    # several runs refer to the same interview fact.
    thread_id = thread_id or derive_thread_id(session_id=session_id, interview_id=interview_id, request_id=run_id)
    state = AgentState(
        request_id=run_id,
        run_id=run_id,
        thread_id=thread_id,
        user_id=ANON,
        session_id=session_id,
        interview_id=interview_id,
        entrypoint=mode,
        requested_mode=mode,
        original_query=original_query[:4000],
        normalized_query=original_query[:4000].strip(),
        token_budget=get_settings().graph_token_budget,
        cost_budget=get_settings().graph_cost_budget,
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=max(0.0, deadline_at - time.monotonic())),
    )
    result: dict[str, Any] = {}

    async def commit(_state: AgentState) -> dict[str, Any]:
        try:
            if isinstance(db, Session):
                with graph_commit_deadline(db, deadline_at):
                    value = await action()
            else:
                value = await action()
            result["value"] = value
            return {"status": "committed", "result": {"valid": True, "run_id": run_id}}
        except Exception as exc:  # Preserve API error semantics after graph traces failure.
            if isinstance(db, Session):
                db.rollback()
            result["error"] = exc
            raise

    async def commit_first(_state: AgentState) -> dict[str, Any]:
        assert first_commit is not None
        try:
            if isinstance(db, Session):
                with graph_commit_deadline(db, deadline_at):
                    await first_commit()
            else:
                await first_commit()
            return {"diagnostics": {"answer_persisted": True}}
        except Exception as exc:
            if isinstance(db, Session):
                db.rollback()
            result["error"] = exc
            raise

    if checkpointer is not None:
        authorize_checkpoint(
            db, thread_id, state, owner_id=ANON,
            session_id=session_id, interview_id=interview_id,
        )
        db.commit()
    graph = build_application_graph(checkpointer=checkpointer, stages=stages,
                                    stage_sequences=stage_sequences, commit=commit,
                                    supervised_route=supervised_route,
                                    first_commit=commit_first if first_commit else None,
                                    first_commit_after=first_commit_after)
    config = checkpoint_config(thread_id=thread_id, owner_id=ANON)
    recovery_attempted = False
    with graph_run_scope(run_id), llm.deadline_scope(deadline_at), llm.usage_budget_scope(
        token_budget=state.token_budget, cost_budget=state.cost_budget,
    ) as usage:
        if checkpointer is None:
            outcome = await graph.ainvoke(state, config=config)
        else:
            snapshot = await graph.compiled.aget_state(config)
            previous = snapshot.values.get("payload") if snapshot.values else None
            if previous is not None:
                previous_run = previous.get("run_id")
                if snapshot.next and previous_run != run_id:
                    raise CheckpointValidationError("checkpoint has an unfinished run")
                if previous_run == run_id:
                    if not snapshot.next:
                        # A failed answer graph may have already committed its
                        # candidate turn. Its idempotent first commit can safely
                        # reconstruct the missing follow-up on the same run.
                        if not (restart_completed_failed and previous.get("status") == "failed"):
                            raise CheckpointValidationError("run already completed; load its business result")
                    # A caller must supply the original command when resuming;
                    # reusing a run ID with different input could execute the
                    # new callback against an unrelated saved decision.
                    if any(previous.get(key) != expected for key, expected in (
                        ("requested_mode", mode),
                        ("original_query", state.original_query),
                        ("session_id", session_id),
                        ("interview_id", interview_id),
                    )):
                        raise CheckpointValidationError("checkpoint request mismatch")
                    recovery_attempted = True
                    # Some business candidates intentionally live outside the
                    # compact checkpoint. Restart their read-only stages from
                    # the original request before the idempotent commit.
                    if restart_incomplete or (restart_completed_failed and not snapshot.next):
                        outcome = await graph.ainvoke(state, config=config)
                    else:
                        resumed = await graph.compiled.ainvoke(None, config=config)
                        outcome = graph.finalize_output(resumed)
                else:
                    outcome = await graph.ainvoke(state, config=config)
            else:
                outcome = await graph.ainvoke(state, config=config)
        measured = usage.snapshot()
        outcome["diagnostics"] = {
            **(outcome.get("diagnostics") or {}),
            "token_count": measured["total_tokens"],
            "cost": measured["estimated_cost"],
            **({"checkpoint_recovered": outcome["status"] == "committed" and "value" in result}
               if recovery_attempted else {}),
        }
    if isinstance(db, Session):
        try:
            persist_graph_trace(db, outcome)
        except Exception:
            db.rollback()
            logger.exception("failed to persist graph trace: run_id=%s", run_id)
    if "error" in result:
        original = result["error"]
        if isinstance(original, TimeoutError):
            raise GraphExecutionError("deadline_exceeded", retryable=False) from original
        if isinstance(original, (ValueError, llm.UsageBudgetError)):
            raise original
        raise GraphExecutionError("business_commit_failed") from original
    if outcome["status"] != "committed" or "value" not in result:
        error = outcome.get("error") or {}
        raise GraphExecutionError(
            str(error.get("error_code") or "business_result_missing"),
            retryable=bool(error.get("retryable", True)),
        )
    return result["value"]


__all__ = ["GraphExecutionError", "run_business_graph"]
