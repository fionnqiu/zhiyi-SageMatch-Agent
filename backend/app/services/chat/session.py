"""Session cockpit: history, first-turn question packs, and later knowledge follow-ups."""

from __future__ import annotations

import re
import uuid
import logging
import time
from datetime import datetime, timedelta, timezone
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

# 用户已经点名要一场面试。回答模型无权再解释成「不能出题」。
_INTERVIEW_REQUEST = re.compile(r"(模拟面试|面试题|出题|生成.{0,8}面试|开始.{0,8}面试)")

from sqlalchemy import exists, or_
from sqlalchemy.orm import Session, selectinload

from app.materials import knowledge
from app.integrations import llm
from app.models import (
    ChatMessage,
    ChatSession,
    Interview,
    InterviewTurn,
    JobProfile,
    Question,
    QuestionSet,
    Report,
)
from app.models.platform.stream import StreamRun
from app.agents.roles.authoring import author_questions
from app.agents.orchestration.checkpoint import authorize_checkpoint, checkpoint_config, derive_thread_id
from app.agents.orchestration.graph import AgentState as GraphAgentState, build_application_graph
from app.agents.orchestration.observability import persist_graph_trace
from app.agents.orchestration.trace import graph_run_scope
from app.agents.roles.memory import MemoryManager
from app.core.config import get_settings
from app.services.shared.common import ANON, audit, new_id, now
from app.services.chat.intent import resolve_intent
from app.services.operations.llm_gateway import complete, stream_parts

# 知识回答和追问要能写完整段分析。700 会在简历分析这类长回答中途被供应商截断。
ANSWER_MAX_TOKENS = 4096
logger = logging.getLogger(__name__)
from app.services.materials.recall import recall_snippets
from app.services.materials.rag.citations import repair_citations, validate_citations
from app.services.materials.rag.stages import StagedRetrieval
from app.agents.contracts.state import RetrievalResult, _route_from_intent


def list_sessions(db: Session) -> list[ChatSession]:
    # 空草稿不进历史。创建面试写的内部会话也不进历史。
    has_turns = exists().where(ChatMessage.session_id == ChatSession.id)
    return (
        db.query(ChatSession)
        .filter(has_turns, ChatSession.origin != "interview", ChatSession.user_id == ANON)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )


def get_session(db: Session, session_id: str) -> ChatSession | None:
    return (
        db.query(ChatSession)
        .options(selectinload(ChatSession.messages), selectinload(ChatSession.profiles))
        .filter(ChatSession.id == session_id, ChatSession.user_id == ANON)
        .one_or_none()
    )


def create_session(db: Session, title: str = "新会话") -> ChatSession:
    session = ChatSession(id=new_id(), title=title, user_id=ANON)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def clear_session(db: Session, session_id: str) -> ChatSession:
    session = get_session(db, session_id)
    if not session:
        raise ValueError("session not found")
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    session.job_title = None
    session.title = "新会话"
    db.commit()
    return get_session(db, session_id)  # type: ignore[return-value]


def delete_interviews(db: Session, interview_ids: list[str]) -> None:
    """Remove interviews after their turns and reports.

    Turns also point at questions, so they have to go before any question rows.
    Bulk delete skips the ORM cascade, which does not cover reports.
    Shared with the interview service so a hub-card delete uses the same order.
    """
    if not interview_ids:
        return
    owned_ids = {
        row.id for row in db.query(Interview.id)
        .join(JobProfile, JobProfile.id == Interview.profile_id)
        .filter(Interview.id.in_(interview_ids), JobProfile.user_id == ANON)
    }
    if owned_ids != set(interview_ids):
        # Reject inconsistent cross-owner links before any bulk cascade runs.
        raise ValueError("面试不存在")
    db.query(Report).filter(Report.interview_id.in_(interview_ids)).delete(synchronize_session=False)
    db.query(InterviewTurn).filter(InterviewTurn.interview_id.in_(interview_ids)).delete(synchronize_session=False)
    db.query(Interview).filter(Interview.id.in_(interview_ids)).delete(synchronize_session=False)


def delete_session(db: Session, session_id: str) -> None:
    """Drop a history session and everything generated from it.

    Question sets hang off the session's job profile, and interviews hang off those
    sets. Leaving them would either break the foreign keys or leave a hub card that
    can no longer be started.
    """
    session = get_session(db, session_id)
    if not session:
        raise ValueError("会话不存在")
    if db.query(JobProfile.id).filter(JobProfile.session_id == session_id, JobProfile.user_id != ANON).first():
        raise ValueError("会话归属不一致")
    profiles = db.query(JobProfile).filter(JobProfile.session_id == session_id).all()
    profile_ids = [row.id for row in profiles]
    set_ids = (
        [row.id for row in db.query(QuestionSet.id).filter(QuestionSet.profile_id.in_(profile_ids)).all()]
        if profile_ids
        else []
    )
    interview_ids: list[str] = []
    if profile_ids or set_ids:
        filters = []
        if profile_ids:
            filters.append(Interview.profile_id.in_(profile_ids))
        if set_ids:
            filters.append(Interview.question_set_id.in_(set_ids))
        interview_ids = [row.id for row in db.query(Interview.id).filter(or_(*filters)).all()]
    title = session.title
    delete_interviews(db, interview_ids)
    if set_ids:
        db.query(Question).filter(Question.question_set_id.in_(set_ids)).delete(synchronize_session=False)
        db.query(QuestionSet).filter(QuestionSet.id.in_(set_ids)).delete(synchronize_session=False)
    if profile_ids:
        db.query(JobProfile).filter(JobProfile.id.in_(profile_ids)).delete(synchronize_session=False)
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete(synchronize_session=False)
    db.delete(session)
    audit(db, "session.delete", title, {"interviews": len(interview_ids)})
    db.commit()


def latest_question_set_for_session(db: Session, session_id: str) -> QuestionSet | None:
    if get_session(db, session_id) is None:
        return None
    profile = (
        db.query(JobProfile)
        .options(selectinload(JobProfile.question_sets).selectinload(QuestionSet.questions))
        .filter(JobProfile.session_id == session_id, JobProfile.user_id == ANON)
        .order_by(JobProfile.created_at.desc())
        .first()
    )
    if profile and profile.question_sets:
        return profile.question_sets[-1]
    return None


async def send_chat(
    db: Session,
    content: str,
    session_id: str | None,
    answers: list[dict[str, Any]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
) -> ChatSession:
    """Generate a pack on first JD; later turns are knowledge follow-ups, not a new pack.

    answers 只在上一轮留下澄清题时使用：把它并回岗位描述再出题，避免模型凭空猜方向。
    """
    turn = await begin_chat(db, content, session_id, answers, attachments)
    reply, extra = await complete_chat_turn(db, turn)
    return await finish_chat(db, turn, reply, extra)


async def send_chat_graph(
    db: Session,
    content: str,
    session_id: str | None,
    answers: list[dict[str, Any]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    *,
    checkpointer: Any | None = None,
) -> ChatSession:
    """Keep the classifier and business graph in one request usage budget."""

    settings = get_settings()
    with llm.usage_budget_scope(
        token_budget=settings.graph_token_budget,
        cost_budget=settings.graph_cost_budget,
    ):
        return await _send_chat_graph_impl(
            db, content, session_id, answers, attachments, checkpointer=checkpointer,
        )


async def _send_chat_graph_impl(
    db: Session,
    content: str,
    session_id: str | None,
    answers: list[dict[str, Any]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    *,
    checkpointer: Any | None = None,
) -> ChatSession:
    """Run an ordinary chat turn through one graph and one commit node.

    The graph callbacks hold the SQLAlchemy session in their closure, keeping
    it out of checkpoints. The route is computed once before invocation, and
    the commit callback writes the assistant fact or clarification result.
    """
    deadline_at = time.monotonic() + 180.0
    with llm.deadline_scope(deadline_at):
        routed = await route_chat(db, content, session_id, answers, attachments)
    run_id = routed["run_id"]
    thread_id = derive_thread_id(session_id=session_id, request_id=run_id)
    state = GraphAgentState(
        request_id=run_id,
        run_id=run_id,
        thread_id=thread_id,
        user_id=ANON,
        session_id=session_id,
        original_query=content,
        normalized_query=routed.get("content") or content,
        route=routed["route"],
        requested_mode=routed["route"]["route"],
        entrypoint="chat",
        token_budget=get_settings().graph_token_budget,
        cost_budget=get_settings().graph_cost_budget,
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=max(0.0, deadline_at - time.monotonic())),
    )
    # A user's message is a durable fact before model work begins. The graph
    # owns the generated candidate and its checks; only commit writes the reply.
    turn = await begin_chat(
        db, content, session_id, answers, attachments,
        precomputed_intent=routed["intent"], run_id=run_id,
        defer_retrieval=routed["route"]["route"] == "knowledge_qa",
    )
    prepared: dict[str, Any] = {"turn": turn}
    retrieval_stages = staged_chat_retrieval_callbacks(db, turn)

    async def answer(_state: GraphAgentState) -> dict[str, Any]:
        reply, extra = await complete_chat_turn(db, turn)
        prepared.update(reply=reply, extra=extra)
        return {"result": {"valid": True, "mode": turn["mode"], "answer_ready": True}}

    async def citation_validate(_state: GraphAgentState) -> dict[str, Any]:
        retrieval = (prepared.get("extra") or {}).get("retrieval")
        selected = retrieval.get("selected_chunks") if isinstance(retrieval, dict) else None
        if turn["mode"] in {"answer", "followup"} and isinstance(selected, list) and selected:
            prepared["citation_validation"] = validate_citations(
                prepared["reply"], selected, require_citation=True,
            )
        return {"diagnostics": {"citation_valid": prepared.get("citation_validation", {}).get("valid")}}

    async def repair_once(_state: GraphAgentState) -> dict[str, Any]:
        retrieval = (prepared.get("extra") or {}).get("retrieval")
        selected = retrieval.get("selected_chunks") if isinstance(retrieval, dict) else None
        if turn["mode"] in {"answer", "followup"} and isinstance(selected, list) and selected:
            reply, extra = await _validated_knowledge_reply(
                db, prepared["reply"], prepared["extra"],
                initial=prepared.get("citation_validation"),
            )
            prepared.update(reply=reply, extra=extra)
        return {"result": {"valid": True, "mode": turn["mode"], "answer_ready": True}}

    async def commit(_state: GraphAgentState) -> dict[str, Any]:
        if "reply" not in prepared:
            # Clarification is a controlled waiting branch without a subgraph.
            await answer(_state)
        session = await finish_chat(
            db, turn, prepared["reply"], prepared["extra"],
            citation_checked=turn["mode"] in {"answer", "followup"},
        )
        prepared["session"] = session
        waiting = routed["mode"] == "blocked"
        return {
            "status": "waiting" if waiting else "committed",
            "result": {"valid": True, "committed": True, "session_id": session.id},
        }

    graph = build_application_graph(
        checkpointer=checkpointer,
        stages={
            **retrieval_stages,
            "knowledge_qa.answer": answer,
            "knowledge_qa.citation_validate": citation_validate,
            "knowledge_qa.repair_once": repair_once,
        } if routed["route"]["route"] == "knowledge_qa" else {},
        commit=commit,
    )
    if checkpointer is not None:
        authorize_checkpoint(
            db,
            thread_id,
            state,
            owner_id=ANON,
            session_id=session_id,
        )
        db.commit()
    with graph_run_scope(run_id), llm.deadline_scope(deadline_at), llm.usage_budget_scope(
        token_budget=state.token_budget, cost_budget=state.cost_budget,
    ) as usage:
        outcome = await graph.ainvoke(
            state,
            config=checkpoint_config(thread_id=thread_id, owner_id=ANON),
        )
        measured = usage.snapshot()
        outcome["diagnostics"] = {
            **(outcome.get("diagnostics") or {}),
            "token_count": measured["total_tokens"],
            "cost": measured["estimated_cost"],
        }
    if isinstance(db, Session):
        try:
            persist_graph_trace(db, outcome)
        except Exception:
            # A committed chat response must not be retried merely because its
            # secondary trace sink failed after the business transaction.
            db.rollback()
            logger.exception("failed to persist graph trace: run_id=%s", run_id)
    if outcome["status"] not in {"committed", "waiting"} or "session" not in prepared:
        from app.agents.orchestration.workflows import GraphExecutionError

        error = outcome.get("error") or {}
        raise GraphExecutionError(
            str(error.get("error_code") or "chat_result_missing"),
            retryable=bool(error.get("retryable", True)),
        )
    return prepared["session"]


def staged_chat_retrieval_callbacks(db: Session, turn: dict[str, Any]) -> dict[str, Any]:
    """Share the six evidence stages between whole-response and SSE chat runs."""
    previous_messages = list(getattr(turn["session"], "messages", None) or [])
    if previous_messages and getattr(previous_messages[-1], "role", None) == "user":
        # begin_chat already persisted the current user turn; rewrite context
        # must contain prior dialogue only to resolve omitted references.
        previous_messages = previous_messages[:-1]
    retrieval_pipeline = StagedRetrieval(
        db, turn.get("content") or "",
        history=[{"role": item.role, "content": item.content} for item in previous_messages[-4:]
                 if isinstance(getattr(item, "content", None), str)],
    ) if turn.get("deferred_retrieval") else None

    async def query_rewrite(_state: GraphAgentState) -> dict[str, Any]:
        if retrieval_pipeline is not None:
            await retrieval_pipeline.query_rewrite()
        return {}

    async def parallel_retrieve(_state: GraphAgentState) -> dict[str, Any]:
        if retrieval_pipeline is not None:
            await retrieval_pipeline.retrieve_parallel()
            return {"diagnostics": {
                "lexical_failed": retrieval_pipeline.lexical_failed,
                "vector_failed": retrieval_pipeline.vector_failed,
                "lexical_latency_ms": retrieval_pipeline.timings.get("lexical_retrieve"),
                "vector_latency_ms": retrieval_pipeline.timings.get("vector_retrieve"),
            }}
        return {}

    async def candidate_fusion(_state: GraphAgentState) -> dict[str, Any]:
        if retrieval_pipeline is not None:
            retrieval_pipeline.candidate_fusion()
            return {"diagnostics": {"candidate_count": len(retrieval_pipeline.fused)}}
        return {}

    async def deduplicate_and_diversify(_state: GraphAgentState) -> dict[str, Any]:
        if retrieval_pipeline is not None:
            retrieval_pipeline.deduplicate_and_diversify()
        return {}

    async def rerank(_state: GraphAgentState) -> dict[str, Any]:
        if retrieval_pipeline is not None:
            await retrieval_pipeline.rerank()
        return {}

    async def context_select(_state: GraphAgentState) -> dict[str, Any]:
        if retrieval_pipeline is not None:
            retrieval = retrieval_pipeline.context_select()
            if turn["mode"] == "followup":
                turn["prompt"], turn["extra"] = await prepare_followup(
                    db, turn["session"], turn["question_set"], turn["content"],
                    retrieval=retrieval,
                )
            else:
                turn["prompt"], turn["extra"] = await prepare_direct_answer(
                    db, turn["session"], turn["content"], turn["intent"],
                    retrieval=retrieval,
                )
        retrieval = (turn.get("extra") or {}).get("retrieval")
        if not isinstance(retrieval, dict):
            return {}
        # Keep prompt text and raw chunks in this request's closure; checkpoint
        # metadata carries only evidence identifiers for audit.
        return {
            "selected_chunks": [
                {key: chunk.get(key) for key in ("source_id", "chunk_id", "material_id", "filename", "ordinal")}
                for chunk in retrieval.get("selected_chunks", [])
            ],
            "citations": retrieval.get("citations", []),
            "diagnostics": {
                "retrieval_status": retrieval.get("retrieval_status"),
                "fallback_reason": next(iter((retrieval.get("diagnostics") or {}).get("fallback_reasons") or []), ""),
            },
        }

    return {
        "knowledge_qa.query_rewrite": query_rewrite,
        "knowledge_qa.parallel_retrieve": parallel_retrieve,
        "knowledge_qa.candidate_fusion": candidate_fusion,
        "knowledge_qa.deduplicate_and_diversify": deduplicate_and_diversify,
        "knowledge_qa.rerank": rerank,
        "knowledge_qa.context_select": context_select,
    }


def validate_stream_answer(turn: dict[str, Any], reply: str, extra: dict[str, Any]) -> dict[str, Any] | None:
    """Validate the assembled answer against the same selected evidence as whole-response chat."""
    retrieval = extra.get("retrieval")
    selected = retrieval.get("selected_chunks") if isinstance(retrieval, dict) else None
    if turn.get("mode") in {"answer", "followup"} and isinstance(selected, list) and selected:
        return validate_citations(reply, selected, require_citation=True)
    return None


async def repair_stream_answer(
    db: Session, turn: dict[str, Any], reply: str, extra: dict[str, Any],
    initial: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Spend at most one citation repair attempt before the streamed turn commits."""
    if turn.get("mode") in {"answer", "followup"} and initial is not None:
        return await _validated_knowledge_reply(db, reply, extra, initial=initial)
    return reply, extra


async def begin_chat(
    db: Session,
    content: str,
    session_id: str | None,
    answers: list[dict[str, Any]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    precomputed_intent: dict[str, Any] | None = None,
    run_id: str | None = None,
    defer_retrieval: bool = False,
) -> dict[str, Any]:
    """写入用户消息并决定这一轮怎么回答。流式接口只接能逐块生成的知识回答和追问。"""
    session = get_session(db, session_id) if session_id else None
    if session_id and session is None:
        raise ValueError("session not found")
    turn_run_id = run_id or uuid.uuid4().hex
    created = False
    if session is None:
        session = ChatSession(id=new_id(), title="新会话", user_id=ANON)
        db.add(session)
        db.flush()
        created = True

    # Explicit interview creation is a navigation action, not a conversational turn.
    if _asks_for_interview(content):
        return {
            "session": session,
            "content": content,
            "intent": {"intent": "answer", "needs_recall": False, "todos": [], "actions": ["redirect"], "source": "redirect"},
            "first_turn": False,
            "opening": "",
            "mode": "redirect",
            "prompt": None,
            "extra": None,
            "user_message": None,
            "run_id": turn_run_id,
        }

    pending = _pending_clarification(session)
    user_message = None
    # 标题只看第一条用户消息。澄清选项和后续追问不再改历史里的名字。
    first_turn = not any(message.role == "user" for message in (session.messages or []))
    opening = ""
    if pending and answers:
        content = _apply_clarification(pending, answers)
        if first_turn:
            opening = content
        user_message = ChatMessage(
            id=new_id(),
            session_id=session.id,
            role="user",
            content=_answers_text(pending, answers),
            extra={"kind": "clarification_reply", "answers": answers},
        )
        db.add(user_message)
    else:
        # 气泡只留用户自己写的字和文件名。抽出的正文放进模型上下文，不回写到 content。
        visible, model_content, files = _compose_turn(content, attachments or [])
        content = model_content
        if first_turn:
            opening = visible or "、".join(item["name"] for item in files)
        user_message = ChatMessage(
            id=new_id(),
            session_id=session.id,
            role="user",
            content=visible,
            extra={"kind": "attachment", "files": files} if files else None,
        )
        db.add(user_message)
    # 用户这句话先落库。后面的模型流可以中断，刷新后仍能看到自己发过什么。
    session_id_saved = session.id
    db.commit()
    # 新建会话还没进查询结果时，继续用刚写入的对象。已有会话提交后重新加载消息。
    if not created:
        reloaded = get_session(db, session_id_saved)
        if reloaded is not None:
            session = reloaded

    existing = latest_question_set_for_session(db, session.id)
    # 出题已经挪到面试页。会话里点名要面试时不再跑感知，也不再生成题目。
    if _asks_for_interview(content):
        intent = {"intent": "answer", "needs_recall": False, "todos": [], "actions": ["redirect"], "source": "redirect"}
        return {
            "session": session,
            "content": content,
            "intent": intent,
            "first_turn": first_turn,
            "opening": opening,
            "mode": "redirect",
            "prompt": None,
            "extra": None,
            "user_message": user_message,
            "run_id": turn_run_id,
        }
    # 选项只是把方向补进这句话。无论有没有选项，这一轮都由感知代理决定。
    intent = precomputed_intent or await resolve_intent(
        db,
        content,
        recall=lambda query: recall_snippets(db, query),
        history=lambda: MemoryManager(db, session_id=session.id).render(),
    )
    if answers:
        intent = {**intent, "source": "clarification"}
    route = _route_from_intent(intent, content)
    mode = "blocked"
    prompt: dict[str, str] | None = None
    extra: dict[str, Any] | None = None
    if intent["intent"] == "clarify" and not answers:
        mode = "clarify"
    elif intent["intent"] == "generate_interview":
        # 感知仍可能把岗位描述认成出题。会话只负责把人带到面试页。
        mode = "redirect"
    elif existing and existing.questions:
        mode = "followup"
        if not defer_retrieval:
            prompt, extra = await prepare_followup(db, session, existing, content)
    else:
        mode = "answer"
        if not defer_retrieval:
            prompt, extra = await prepare_direct_answer(db, session, content, intent)
    return {
        "session": session,
        "content": content,
        "intent": intent,
        "route": route.model_dump(mode="json"),
        "first_turn": first_turn,
        "opening": opening,
        "mode": mode,
        "prompt": prompt,
        "extra": extra,
        "user_message": user_message,
        "run_id": turn_run_id,
        "deferred_retrieval": defer_retrieval and mode in {"answer", "followup"},
        "question_set": existing if defer_retrieval and mode == "followup" else None,
    }


async def route_chat(
    db: Session,
    content: str,
    session_id: str | None,
    answers: list[dict[str, Any]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve a chat route once so streaming and execution share the same decision."""
    session = get_session(db, session_id) if session_id else None
    if session_id and session is None:
        raise ValueError("session not found")
    run_id = uuid.uuid4().hex
    if _asks_for_interview(content):
        intent = {"intent": "generate_interview", "needs_recall": False, "todos": [], "source": "pattern"}
        return {"intent": intent, "route": _route_from_intent(intent, content).model_dump(mode="json"), "mode": "redirect", "run_id": run_id}
    pending = _pending_clarification(session)
    model_content = content
    if pending and answers:
        model_content = _apply_clarification(pending, answers)
    elif attachments:
        _visible, model_content, _files = _compose_turn(content, attachments)
    intent = await resolve_intent(
        db,
        model_content,
        recall=lambda query: recall_snippets(db, query),
        history=lambda: MemoryManager(db, session_id=session.id).render() if session is not None else "",
    )
    if intent["intent"] == "clarify" and not answers:
        mode = "blocked"
    elif intent["intent"] == "generate_interview":
        mode = "redirect"
    else:
        mode = "stream"
    return {
        "intent": intent,
        "route": _route_from_intent(intent, model_content).model_dump(mode="json"),
        "mode": mode,
        "content": model_content,
        "run_id": run_id,
    }


def recover_chat_turn(db: Session, recovery: dict[str, Any], run_id: str) -> dict[str, Any] | None:
    """Rebuild a staged text turn without repeating classification or user writes."""
    session_id = recovery.get("session_id")
    session = get_session(db, session_id) if isinstance(session_id, str) else None
    if session is None or session.user_id != ANON:
        return None
    message_id = recovery.get("user_message_id")
    matching = [index for index, item in enumerate(session.messages)
                if item.id == message_id and item.role == "user"]
    if not matching:
        return None
    # A later user request changes the conversation context. Replaying the
    # older answer after it would write history in the wrong order.
    if any(item.role == "user" for item in session.messages[matching[0] + 1:]):
        return None
    intent = recovery.get("intent")
    mode = recovery.get("mode")
    content = recovery.get("content")
    if not isinstance(intent, dict) or mode not in {"answer", "followup", "clarify"} or not isinstance(content, str):
        return None
    return {
        "session": session,
        "content": content,
        "intent": intent,
        "route": recovery.get("route") or {},
        "first_turn": bool(recovery.get("first_turn")),
        "opening": str(recovery.get("opening") or ""),
        "mode": mode,
        "prompt": None,
        "extra": None,
        "user_message": None,
        "run_id": run_id,
        "deferred_retrieval": mode in {"answer", "followup"},
        "question_set": latest_question_set_for_session(db, session.id) if mode == "followup" else None,
    }


def iter_turn_parts(db: Session, turn: dict[str, Any]) -> AsyncIterator[tuple[str, str]]:
    """只对知识回答和追问逐块产出。出题不再从会话流里生成。"""
    prompt = turn.get("prompt")
    if turn.get("mode") not in {"answer", "followup"} or not isinstance(prompt, dict):
        raise RuntimeError("这一轮不能逐块输出")
    return _iter_parts_or_fallback(db, prompt)


async def complete_chat_turn(db: Session, turn: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """整段接口的后半段。能流的模式也从这里收齐，保证和 SSE 用同一份提示。"""
    session = turn["session"]
    content = turn["content"]
    intent = turn["intent"]
    mode = turn["mode"]
    if mode == "clarify":
        # 只有模型判断这一句确实不够出题时才追问，问题和选项都用它这一轮写的。
        return _clarification_from_intent(content, intent)
    if mode in {"answer", "followup"} and turn["prompt"] is not None:
        prompt = turn["prompt"]
        try:
            reply = await complete(db, "analyst", prompt["system"], prompt["user"], max_tokens=ANSWER_MAX_TOKENS)
        except Exception:
            reply = prompt["fallback"]
        return reply, {**turn["extra"], "intent": intent["intent"]}
    if mode == "redirect":
        return _redirect_reply(), {"kind": "redirect", "intent": "generate_interview"}
    reply, extra = await generate_and_store(db, session, content)
    return reply, {**extra, "intent": intent["intent"]}


async def _validated_knowledge_reply(
    db: Session,
    reply: str,
    extra: dict[str, Any],
    *,
    initial: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Check answer citations and spend at most one model repair attempt."""
    retrieval = extra.get("retrieval")
    selected = retrieval.get("selected_chunks") if isinstance(retrieval, dict) else None
    if not isinstance(selected, list) or not selected:
        return reply, extra

    async def repair_once(answer: str, sources: str) -> str:
        """Rephrase against the selected evidence without introducing new facts."""
        return str(await complete(
            db,
            "analyst",
            "只修正回答中的来源引用。仅可使用提供的 [Sx]，保留原回答中有依据的内容；没有证据的断言要删去，不得编造材料。只返回修正后的正文。",
            f"原回答：\n{answer[:6000]}\n\n可用证据：\n{sources[:12000]}",
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=0.0,
        ))

    if initial and initial.get("valid"):
        citation_result = {**initial, "repair_attempted": False}
    else:
        citation_result = await repair_citations(
            reply, selected, repair_provider=repair_once, require_citation=True,
        )
    repaired = citation_result.get("repaired_answer")
    if citation_result.get("valid") and isinstance(repaired, str) and repaired.strip():
        reply = repaired.strip()
    return reply, {
        **extra, "citation_validation": citation_result,
        "evidence_coverage": citation_result["evidence_coverage"],
    }


async def finish_chat(
    db: Session,
    turn: dict[str, Any],
    reply: str,
    extra: dict[str, Any],
    *,
    citation_checked: bool = False,
    worker_id: str | None = None,
) -> ChatSession:
    """助手消息和标题都在正文收齐后写入。流式中途不落半句。"""
    session = turn["session"]
    # One run identifier survives both streaming and non-streaming paths so
    # the persisted answer can be correlated with the route and SSE events.
    extra = {**extra, "run_id": turn.get("run_id") or uuid.uuid4().hex, "route": (turn.get("route") or {}).get("route", extra.get("route"))}
    if turn["mode"] == "redirect":
        if worker_id is not None:
            _lock_chat_stream_lease(db, turn["run_id"], worker_id)
        # Remove only this transient navigation turn; older conversation stays intact.
        discard_redirect_turn(db, turn)
        # The API caller still needs a redirect marker, but it must never reach the database.
        redirect_message = SimpleNamespace(
            id="redirect",
            session_id=session.id,
            role="assistant",
            content=reply,
            extra=extra,
            created_at=now(),
        )
        if not hasattr(db, "query"):
            # Lightweight service doubles retain the historical return identity.
            session.messages = [*(session.messages or []), redirect_message]
            return session
        return SimpleNamespace(
            id=session.id, title=session.title, job_title=session.job_title,
            created_at=session.created_at, updated_at=session.updated_at,
            messages=[*(session.messages or []), redirect_message],
        )
    if turn["first_turn"] and turn["opening"]:
        # 标题仍是一次短调用。正文流结束后再起名，避免和回答抢同一轮输出。
        session.title = await session_title(db, turn["opening"])
    # Citation validation happens after streaming has collected the full answer,
    # so a provider cannot smuggle an unknown source id into durable history.
    if turn.get("mode") in {"answer", "followup"} and not citation_checked:
        reply, extra = await _validated_knowledge_reply(db, reply, extra)
    if worker_id is not None:
        _lock_chat_stream_lease(db, turn["run_id"], worker_id)
    # A crash between the assistant commit and terminal frame can replay this
    # run. The locked stream row serializes current-owner retries.
    if turn.get("run_id") and hasattr(db, "query"):
        prior = next((row for row in db.query(ChatMessage).filter(
            ChatMessage.session_id == session.id, ChatMessage.role == "assistant",
        ).all() if isinstance(row.extra, dict) and row.extra.get("run_id") == turn["run_id"]), None)
        if prior is not None:
            return get_session(db, session.id)  # type: ignore[return-value]
    session.updated_at = now()
    db.add(
        ChatMessage(
            id=new_id(),
            session_id=session.id,
            role="assistant",
            content=reply,
            extra=extra,
        )
    )
    db.commit()
    return get_session(db, session.id)  # type: ignore[return-value]


def _lock_chat_stream_lease(db: Session, run_id: str, worker_id: str) -> None:
    """Fence the next business write while retaining the row lock to commit."""
    lease = (db.query(StreamRun).filter(
        StreamRun.id == run_id, StreamRun.status == "running",
        StreamRun.lease_owner == worker_id,
        StreamRun.lease_until > datetime.now(timezone.utc),
    ).with_for_update().one_or_none())
    if lease is None:
        raise PermissionError("chat stream lease lost")


def discard_redirect_turn(db: Session, turn: dict[str, Any]) -> None:
    """Discard the chat row staged before intent resolution selected interview navigation."""
    user_message = turn.get("user_message")
    if user_message is not None:
        db.delete(user_message)
        db.commit()


def prepare_chat_file(filename: str, data: bytes) -> str:
    """抽出附件正文，留给输入框。发送时才和用户写的字一起进会话。"""
    text = knowledge.parse_bytes(filename, data)
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("无法从该文件提取文本，请改用 TXT、MD、PDF 或 DOCX")
    return cleaned[:12000]


def _compose_turn(content: str, attachments: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]]]:
    """可见内容不含文件正文。模型侧把正文接在用户这句话后面。"""
    files: list[dict[str, Any]] = []
    bodies: list[str] = []
    for item in attachments:
        name = str(item.get("name") or "附件").strip() or "附件"
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        files.append({"name": name[:180], "size": int(item.get("size") or 0)})
        bodies.append(f"【附件 {name}】\n{text[:12000]}")
    visible = content.strip()
    # 只有附件时，模型仍要看到正文，不能拿到空字符串。
    model_content = "\n\n".join(part for part in [visible, *bodies] if part) or "请阅读附件。"
    return visible, model_content, files


async def generate_and_store(
    db: Session, session: ChatSession, content: str, on_thought=None,
    *, prepared_hits: list[dict[str, Any]] | None = None,
    prepared_payload: dict[str, Any] | None = None,
    generation_trace: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    # The generation graph supplies a validated candidate; ordinary chat still
    # uses the service's own retrieval and authoring path.
    hits = prepared_hits if prepared_hits is not None else await recall_snippets(db, content)
    payload = prepared_payload if prepared_payload is not None else await generate_question_pack(db, content, hits, session.id, on_thought=on_thought)
    job_title = payload.get("job_title") or title_from(content)
    session.job_title = job_title
    # 历史标题已经按首轮对话定过。岗位名留在 job_title，不再把侧边栏改成「岗位（N题）」。
    if not session.title or session.title == "新会话":
        session.title = f"{job_title} ({len(payload.get('questions') or [])}题)"

    profile = JobProfile(
        id=new_id(),
        session_id=session.id,
        user_id=ANON,
        raw_text=content,
        job_title=job_title,
        analysis={"focus": payload.get("focus", []), "summary": payload.get("summary", "")},
    )
    db.add(profile)
    db.flush()

    qset = QuestionSet(
        id=new_id(),
        profile_id=profile.id,
        status="ready",
        coverage=payload.get("coverage"),
        snapshot={"source": "chat", "recall": [h["filename"] for h in hits[:3]],
                  **({"generation_handoffs": generation_trace} if generation_trace is not None else {})},
    )
    db.add(qset)
    db.flush()

    questions = payload.get("questions") or []
    for i, item in enumerate(questions, start=1):
        db.add(
            Question(
                id=new_id(),
                question_set_id=qset.id,
                ordinal=i,
                stem=item.get("stem") or f"题目 {i}",
                options=_stored_options(item),
                answer=str(item.get("answer") or "")[:200] or "见解析",
                explanation=item.get("explanation") or "",
                generated_by="system",
            )
        )

    # Ready card on the interview hub, without starting the live stage.
    interview = Interview(
        id=new_id(),
        profile_id=profile.id,
        question_set_id=qset.id,
        title=f"{job_title} · 全真模拟面试",
        status="ready",
        tags=payload.get("focus") or [],
        summary=payload.get("summary") or "基于岗位要求生成 · 预计时长 30 分钟",
    )
    db.add(interview)

    # The generated pack and its derived memories share one transaction. If
    # the caller rolls back question persistence, these memories roll back too.
    memory = MemoryManager(db, session_id=session.id)
    memory.remember_profile(
        "local-user",
        {"job_title": job_title, "focus": payload.get("focus") or []},
    )
    memory.update_episode(
        summary=str(payload.get("summary") or "")[:400],
        slots={"question_count": len(questions)},
    )

    extra = {
        "kind": "question_pack",
        "question_set_id": qset.id,
        "profile_id": profile.id,
        # 出题清单跟这一份岗位和实际检索结果走，不再套「阅读 / 检索 / 出题 / 核对」四步。
        "todos": _todos_for_pack(job_title, payload.get("focus") or [], hits, bool(questions)),
        "questions": [{"stem": q.get("stem"), "ordinal": i} for i, q in enumerate(questions, start=1)],
        "actions": payload.get("actions") or ["直接发起一场 30 分钟全真模拟面试实战"],
    }
    reply = payload.get("reply") or "已分析该岗位的核心要求，正在为你生成针对性题目。"
    # 面试页要拿到这场面试的 id。会话旧路径仍只用 reply 和 extra。
    extra["interview_id"] = interview.id
    return reply, extra


async def answer_directly(
    db: Session, session: ChatSession, content: str, intent: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Answer without creating questions. Retrieval and checklist come from perception."""
    prompt, extra = await prepare_direct_answer(db, session, content, intent)
    try:
        reply = await complete(db, "analyst", prompt["system"], prompt["user"], max_tokens=ANSWER_MAX_TOKENS)
    except Exception:
        reply = prompt["fallback"]
    return reply, extra


async def prepare_direct_answer(
    db: Session, session: ChatSession, content: str, intent: dict[str, Any],
    *, retrieval: dict[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build a knowledge answer from one structured retrieval result.

    The legacy needs_recall flag remains in the intent payload for compatibility,
    but the explicit knowledge route always performs retrieval so stream and
    non-stream answers share the same evidence contract.
    """
    if retrieval is None:
        hits = await recall_snippets(db, content)
        selected = [dict(hit) for hit in hits]
    else:
        selected = list(retrieval.get("selected_chunks") or [])
    citations = [
        {
            "source_id": f"S{index}",
            "chunk_id": hit.get("chunk_id"),
            "material_id": hit.get("material_id"),
            "filename": hit.get("filename"),
            "ordinal": hit.get("ordinal"),
        }
        for index, hit in enumerate(selected, start=1)
    ]
    retrieval_result = RetrievalResult.model_validate(retrieval) if retrieval is not None else RetrievalResult(
        original_query=content,
        search_queries=[content],
        candidates=selected,
        selected_chunks=selected,
        context_text="\n\n".join(
            f"[{citation['source_id']}] {hit.get('text', '')[:400]}"
            for citation, hit in zip(citations, selected, strict=True)
        )
        or "（知识库暂无命中）",
        citations=citations,
        retrieval_status="ok" if selected else "empty",
        diagnostics={"legacy_needs_recall": bool(intent.get("needs_recall"))},
    )
    knowledge_block = retrieval_result.context_text or "（知识库暂无命中）"
    history = _history_lines(session, 6)
    prompt = {
        "system": "你是面试知识助手。用户这一轮只是提问，直接回答。不要生成面试题。若用户要出题或开始面试，告诉对方去模拟面试页。每个可验证事实都要在句末引用上下文中的来源编号，例如 [S1]；不要编造来源编号。",
        "user": f"知识片段：\n{knowledge_block}\n\n最近对话：\n{history}\n\n用户：{content}",
        "fallback": "这一轮先按知识问题回答；模型暂时不可用。你可以稍后再问，或明确说出要准备的岗位。",
    }
    extra = {
        "kind": "answer",
        "intent": "answer",
        # Preserve the old todo field while exposing the structured retrieval result.
        "todos": intent.get("todos") or _todos_for_answer(content, True, selected),
        "retrieval": retrieval_result.model_dump(mode="json"),
        "citations": citations,
        "route": "knowledge_qa",
    }
    return prompt, extra


async def chat_followup(
    db: Session, session: ChatSession, qset: QuestionSet, content: str
) -> tuple[str, dict[str, Any]]:
    prompt, extra = await prepare_followup(db, session, qset, content)
    try:
        reply = await complete(db, "analyst", prompt["system"], prompt["user"], max_tokens=ANSWER_MAX_TOKENS)
    except Exception:
        reply = prompt["fallback"]
    return reply, extra


async def prepare_followup(
    db: Session, session: ChatSession, qset: QuestionSet, content: str,
    *, retrieval: dict[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    questions = list(qset.questions)
    stems = "\n".join(f"{q.ordinal}. {q.stem}" for q in questions)
    if retrieval is None:
        hits = await recall_snippets(db, content)
        knowledge_block = "\n\n".join(h["text"][:400] for h in hits) if hits else "（知识库暂无命中）"
    else:
        knowledge_block = retrieval.get("context_text") or "（知识库暂无命中）"
    history = _history_lines(session, 8)
    prompt = {
        "system": "你是模拟面试训练助手。回答用户对题目或岗位知识的追问。不要重新出一整套题。不要使用「对弈」「博弈」「棋」等字眼。",
        "user": f"已有题目：\n{stems}\n\n知识片段：\n{knowledge_block}\n\n最近对话：\n{history}\n\n用户：{content}",
        "fallback": "这套题目已经生成。你可以追问某一题在考什么，或直接发起模拟面试。",
    }
    extra = {
        "kind": "followup",
        "question_set_id": qset.id,
        "actions": ["直接发起一场 30 分钟全真模拟面试实战"],
    }
    if retrieval is not None:
        extra["retrieval"] = retrieval
        extra["citations"] = retrieval.get("citations", [])
    return prompt, extra


async def _iter_parts_or_fallback(db: Session, prompt: dict[str, str]) -> AsyncIterator[tuple[str, str]]:
    """模型流中断时改交兜底句。已经吐出的半句由调用方决定是否保留。"""
    try:
        async for kind, delta in stream_parts(db, "analyst", prompt["system"], prompt["user"], max_tokens=ANSWER_MAX_TOKENS):
            yield kind, delta
    except Exception:
        yield "content", prompt["fallback"]


def _redirect_reply() -> str:
    """会话不再出题。这句话只负责把人送到面试页，避免回答模型再说自己不能出题。"""
    return "出题已改到模拟面试页。把岗位描述发到那里，生成完成后可以直接开始面试。"


def _recent_dialogue(session: ChatSession) -> str:
    """Give the perception agent only the previous turns, not the current one."""
    lines = _history_lines(session, 6).splitlines()
    return "\n".join(lines[:-1])


def _history_lines(session: ChatSession, limit: int) -> str:
    """只拼接真正的文本消息。测试替身或空正文不进提示词。"""
    lines: list[str] = []
    for message in list(session.messages or [])[-limit:]:
        content = getattr(message, "content", "")
        role = getattr(message, "role", "")
        if not isinstance(content, str) or not isinstance(role, str):
            continue
        lines.append(f"{role}: {content[:400]}")
    return "\n".join(lines)


def _pending_clarification(session: ChatSession | None) -> dict[str, Any] | None:
    """最近一条助手消息若还在等用户选方向，就把它当作未完成的澄清。"""
    if session is None:
        return None
    messages = list(session.messages or [])
    if not messages:
        return None
    last = messages[-1]
    extra = last.extra if isinstance(last.extra, dict) else None
    if last.role == "assistant" and extra and extra.get("kind") == "clarification":
        return extra
    return None


def _clarification_from_intent(content: str, intent: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Render the questions the perception agent wrote. No shared card."""
    questions = intent.get("questions") or []
    todos = list(intent.get("todos") or [])
    if todos:
        todos[-1] = {**todos[-1], "status": "active"}
    return (
        str(intent.get("reply") or "").strip(),
        {
            "kind": "clarification",
            "source": content,
            "todos": todos,
            "questions": questions,
        },
    )


def _apply_clarification(pending: dict[str, Any], answers: list[dict[str, Any]]) -> str:
    """把选项拼回原始请求。出题仍走同一条 JD 路径，不另开一套提示词。"""
    picked = "；".join(
        f"{item.get('prompt') or item.get('id')}：{item.get('label') or item.get('option_id')}"
        for item in answers
        if item.get("label") or item.get("option_id")
    )
    source = str(pending.get("source") or "").strip()
    return f"{source}\n面试方向补充：{picked}".strip()


def _answers_text(pending: dict[str, Any], answers: list[dict[str, Any]]) -> str:
    labels = [str(item.get("label") or item.get("option_id") or "") for item in answers]
    labels = [label for label in labels if label]
    source = str(pending.get("source") or "").strip()
    chosen = "、".join(labels) or "已选择"
    return f"按「{source}」准备，方向是{chosen}。" if source else f"方向是{chosen}。"


def _short_label(text: str, limit: int = 18) -> str:
    """清单一行只留这一轮能对上的短语，避免把整段岗位描述塞进侧栏。"""
    label = " ".join(str(text).split())
    return label[:limit]


def _todos_for_answer(content: str, needs_recall: bool, hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    """知识问答的兜底清单。只记这一问和真实检索，不再排三条固定步骤。"""
    subject = _short_label(content, 12) or "这一问"
    items = [{"id": "answer-0", "label": f"回答「{subject}」", "status": "complete"}]
    if needs_recall:
        sources = [str(hit.get("filename") or "") for hit in hits[:2] if hit.get("filename")]
        if sources:
            items.append({"id": "answer-1", "label": f"对照{'、'.join(sources)}"[:18], "status": "complete"})
        else:
            items.append({"id": "answer-1", "label": "知识库没有命中", "status": "cancelled"})
    return items


def _todos_for_pack(
    job_title: str,
    focus: list[Any],
    hits: list[dict[str, Any]],
    has_questions: bool,
) -> list[dict[str, str]]:
    """出题清单用岗位、考点和命中资料拼出来，而不是每次都显示同一套步骤。"""
    title = _short_label(job_title, 10) or "这个岗位"
    points = [str(item).strip() for item in focus if str(item).strip()]
    sources = [str(hit.get("filename") or "") for hit in hits[:2] if hit.get("filename")]
    done = "complete" if has_questions else "cancelled"
    items = [{"id": "pack-0", "label": f"拆解「{title}」", "status": "complete"}]
    if points:
        items.append({"id": "pack-1", "label": f"围绕{'、'.join(points[:2])}"[:18], "status": "complete"})
    if sources:
        items.append({"id": "pack-2", "label": f"对照{'、'.join(sources)}"[:18], "status": "complete"})
    else:
        items.append({"id": "pack-2", "label": "没有命中资料", "status": "cancelled"})
    items.append({"id": "pack-3", "label": "写出这一套题" if has_questions else "这一套题没有写出", "status": done})
    return items[:4]


async def session_title(db: Session, content: str) -> str:
    """Name a new history row from the first thing the user said.

    The model only supplies a short label. A bad or empty result falls back to
    the trimmed opening line, so the sidebar never stays on 「新会话」.
    """
    system = "你给会话起一个简短标题。只返回标题本身，4 到 16 个字，不要引号、句号或解释。"
    user = f"用户第一条消息：\n{content.strip()[:500]}"
    try:
        title = str(await complete(db, "analyst", system, user, max_tokens=40)).strip().splitlines()[0]
    except Exception:
        title = ""
    title = title.strip("「」\"'“” 。.").strip()[:16]
    return title or title_from(content)


def title_from(text: str) -> str:
    # Keep sidebar titles short; never use a full JD or question stem.
    # 澄清选项可能没有正文，空文本不能拿第一行。
    lines = text.strip().splitlines()
    line = lines[0] if lines else ""
    for prefix in ("目标岗位 JD：", "目标岗位JD：", "岗位：", "【上传文件"):
        if line.startswith(prefix):
            line = line.split("】")[-1] if "】" in line else line[len(prefix) :]
    line = line.split("。")[0].split("，")[0].strip()[:18]
    return line or "新会话"


async def generate_question_pack(
    db: Session,
    job_text: str,
    hits: list[dict[str, Any]],
    session_id: str | None = None,
    on_thought=None,
) -> dict[str, Any]:
    """Author a question pack through the author contract. Eval reuses this same path."""
    if not llm.llm_available():
        return stub_pack(job_text)
    try:
        data = await author_questions(db, job_text, session_id=session_id, hits=hits, on_thought=on_thought)
        if not data.get("questions"):
            return stub_pack(job_text)
        data.pop("_agent", None)
        return data
    except Exception:
        return stub_pack(job_text)


def _asks_for_interview(content: str) -> bool:
    """A named interview request is enough. A bare job title without this ask still goes to the model."""
    return bool(_INTERVIEW_REQUEST.search(content or ""))


def _stored_options(item: dict[str, Any]) -> list:
    """Live interviews are spoken. Options are dropped even if a model still returns them."""
    del item
    return []


def stub_pack(job_text: str) -> dict[str, Any]:
    title = "资深分布式系统架构师" if "架构" in job_text else title_from(job_text)
    # Offline packs follow the same band as the author: eight spoken-heavy questions, not five choices.
    questions = [
        {
            "kind": "open",
            "stem": "这个岗位要你设计一套可扩展的 AI 应用架构。你会如何拆分模型调用、检索和工具调用？",
            "options": [],
            "answer": "按调用、检索、工具三层拆分，并说明扩展点和失败边界",
            "explanation": "听候选人是否能把 LLM、RAG 和工具调用拆开，而不是堆在一个接口里。",
        },
        {
            "kind": "scenario",
            "stem": "检索结果经常答非所问。你会怎么定位是切块、召回还是提示词的问题？",
            "options": [],
            "answer": "先看召回片段是否相关，再看提示是否约束了引用",
            "explanation": "排查要有顺序：先验证检索命中，再谈生成。",
        },
        {
            "kind": "open",
            "stem": "如果要接入 MCP 或 Function Call，你如何决定一个工具该不该交给模型？",
            "options": [],
            "answer": "只把边界清楚、可校验、失败可回退的动作做成工具",
            "explanation": "工具不是越多越好，关键是权限和失败后的退路。",
        },
        {
            "kind": "scenario",
            "stem": "线上回答突然变慢，用户开始超时。你会先看哪一层，并如何临时止血？",
            "options": [],
            "answer": "先分清模型延迟、检索延迟和下游工具，再限流或降级",
            "explanation": "并发瓶颈要先定位，再决定降级，而不是一律加大超时。",
        },
        {
            "kind": "open",
            "stem": "长对话里模型开始忘记前面的约束。你会怎么区分该进上下文的内容和该外置的记忆？",
            "options": [],
            "answer": "近期原话留在上下文，稳定事实外置，并说明何时回读",
            "explanation": "看候选人是否把工作记忆和长期记忆分开，而不是无限加长提示。",
        },
        {
            "kind": "scenario",
            "stem": "同一个岗位描述连续两次出题，题目高度重复。你怎么在生成链路里发现并停下来？",
            "options": [],
            "answer": "用题干重叠率做闸门，超阈值则带原因重出一次后停止",
            "explanation": "质检要有停止条件，不能为了换题无限循环。",
        },
        {
            "kind": "open",
            "stem": "让你和后端、产品一起落地一个 Agent 功能。你如何划清谁决定工具权限、谁验收结果？",
            "options": [],
            "answer": "权限由契约约束，验收看可观察结果，不由模型自行放宽",
            "explanation": "协作题看边界，而不是只讲个人能写什么代码。",
        },
        {
            "kind": "open",
            "stem": "为什么热点读会用本地缓存加 Redis，而不是只靠其中一层？一致性你怎么处理？",
            "options": [],
            "answer": "本地挡重复读，Redis 做跨实例共享；一致性靠失效和锁，不是本地更强",
            "explanation": "问答题听候选人讲分层和一致性，不再用选项暗示答案。",
        },
    ]
    return {
        "job_title": title,
        "summary": "高并发、缓存一致性与故障应急是本岗位的核心考察要求。",
        "focus": ["高并发架构", "网络协议", "一致性算法", "故障排查"],
        "coverage": 0.92,
        "reply": "已分析该岗位的核心要求，正在为你生成针对性题目。生成完成后可直接发起模拟面试：",
        "actions": [
            questions[0]["stem"],
            "直接发起一场 30 分钟全真模拟面试实战",
        ],
        "questions": questions,
    }
