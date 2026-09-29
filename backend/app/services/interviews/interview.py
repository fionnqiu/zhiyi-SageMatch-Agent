"""Live interview: start, answer, end, and the recap written when a session closes."""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy.orm import Session, selectinload

from app.integrations import llm
from app.agents.roles.authoring import interviewer_followup
from app.agents.workflows.evaluation import EvaluationPipeline, run_evaluation
from app.agents.orchestration.checkpoint import authorize_checkpoint, checkpoint_config
from app.agents.contracts.contracts import validate_role_input, validate_role_output
from app.agents.orchestration.graph import AgentState as GraphAgentState, build_application_graph
from app.agents.roles.memory import MemoryManager
from app.models import ChatMessage, ChatSession, DurableJob, Interview, InterviewCreateReceipt, InterviewTurn, JobProfile, Question, QuestionSet, Report
from app.core.db import SessionLocal
from app.services.shared.common import ANON, audit, new_id, now
from app.services.operations.jobs import JobWorker, complete_job, create_job, enqueue_report_job, fail_job, keep_job_lease
from app.services.operations.llm_gateway import complete
from app.services.chat.session import delete_interviews, generate_and_store, get_session, latest_question_set_for_session

# The task handle is process-local, but report work is persisted in durable_jobs
# and can be reclaimed by another process after a restart.
_report_worker: asyncio.Task[None] | None = None
_report_worker_id = f"report-{os.getpid()}"
_report_checkpointer: Any | None = None
# Keep candidate-answer responses bounded even when an interviewer provider stalls.
FOLLOWUP_TIMEOUT_SECONDS = 12.0


def _lock_interview(db: Session, interview_id: str) -> Interview | None:
    """Serialize state transitions before callers inspect the current status.

    PostgreSQL holds this row lock until commit/rollback. SQLite ignores
    ``FOR UPDATE``, which keeps lightweight local tests compatible while real
    multi-worker deployments receive the required concurrency boundary.
    """

    # A few focused service tests use a tiny session double; production always
    # supplies SQLAlchemy Session and therefore always takes the lock path.
    if not hasattr(db, "query"):
        return get_interview(db, interview_id)
    return (
        db.query(Interview)
        # Ownership comes from the generating profile; unattributed legacy rows
        # need an explicit migration before they can be exposed to this user.
        .join(JobProfile, JobProfile.id == Interview.profile_id)
        .filter(Interview.id == interview_id, JobProfile.user_id == ANON)
        .with_for_update()
        .one_or_none()
    )


def list_interviews(db: Session) -> list[Interview]:
    return (
        db.query(Interview)
        .join(JobProfile, JobProfile.id == Interview.profile_id)
        .options(selectinload(Interview.report), selectinload(Interview.question_set).selectinload(QuestionSet.questions))
        .filter(JobProfile.user_id == ANON)
        .order_by(Interview.created_at.desc())
        .all()
    )


def get_interview(db: Session, interview_id: str) -> Interview | None:
    return (
        db.query(Interview)
        .join(JobProfile, JobProfile.id == Interview.profile_id)
        .options(
            selectinload(Interview.turns),
            selectinload(Interview.report),
            selectinload(Interview.question_set).selectinload(QuestionSet.questions),
        )
        .filter(Interview.id == interview_id, JobProfile.user_id == ANON)
        .one_or_none()
    )


def pending_answer_run_id(db: Session, interview_id: str, content: str, answer_mode: str) -> str | None:
    """Recover the durable run for an answer whose graph did not finish.

    The owned interview query is the authorization boundary. Only the latest
    candidate turn can be resumed, so an older pending marker cannot rewind a
    later answer after an interrupted request.
    """
    interview = get_interview(db, interview_id)
    if interview is None:
        return None
    latest = (db.query(InterviewTurn).filter(
        InterviewTurn.interview_id == interview.id, InterviewTurn.role == "user",
    ).order_by(InterviewTurn.created_at.desc(), InterviewTurn.id.desc()).first())
    if latest is None or latest.cite != "answer_pending":
        return None
    if latest.content != content or latest.answer_mode != answer_mode:
        raise ValueError("上一条回答仍在处理中，请稍后重试")
    return latest.id


async def prepare_interview(
    db: Session, content: str, on_thought=None, *, run_id: str | None = None,
    prepared_hits: list[dict] | None = None, prepared_payload: dict | None = None,
    generation_trace: list[dict] | None = None,
) -> Interview:
    """在面试页出题。思考原文只经回调推给当次页面，不写进面试记录。"""
    text = content.strip()
    if len(text) < 8:
        raise ValueError("请先写下岗位描述")
    if run_id:
        # The internal session and generated interview commit together. A
        # replay after that commit returns the same result even if LangGraph
        # had not yet checkpointed its commit node before a process exit.
        existing = db.get(ChatSession, run_id)
        if existing is not None:
            if existing.origin != "interview" or existing.user_id != ANON:
                raise ValueError("运行标识已被其他会话使用")
            original = next((message.content for message in existing.messages if message.role == "user"), None)
            if original != text:
                raise ValueError("运行标识与原岗位描述不匹配")
            ready = (
                db.query(Interview)
                .join(JobProfile, JobProfile.id == Interview.profile_id)
                .filter(JobProfile.session_id == run_id)
                .order_by(Interview.created_at.desc())
                .first()
            )
            if ready is None:
                # Request acceptance staged this internal session before the
                # model ran. Finish its question set in the commit transaction.
                session = existing
            else:
                return {"id": ready.id, "title": ready.title, "status": ready.status}
    # 题目仍要挂在会话上才能保存。origin 把它排除出历史对话。
    if not run_id or not db.get(ChatSession, run_id):
        session = ChatSession(id=run_id or new_id(), title="新会话", user_id=ANON, origin="interview")
        db.add(session)
        db.flush()
        db.add(ChatMessage(id=new_id(), session_id=session.id, role="user", content=text))
    _reply, extra = await generate_and_store(
        db, session, text, on_thought=on_thought,
        prepared_hits=prepared_hits, prepared_payload=prepared_payload,
        generation_trace=generation_trace,
    )
    db.commit()
    interview_id = str(extra.get("interview_id") or "")
    ready = get_interview(db, interview_id) if interview_id else None
    if ready is None:
        raise ValueError("题目没有生成")
    # 页面只需要这场面试的标识就能跳过去。思考原文不在这份结果里。
    return {"id": ready.id, "title": ready.title, "status": ready.status}


def _create_receipt(db: Session, key: str, session_id: str | None,
                    question_set_id: str | None) -> tuple[InterviewCreateReceipt, bool]:
    """Claim a caller key in the current transaction so concurrent retries wait."""
    if not key or len(key) > 255:
        raise ValueError("Idempotency-Key 长度必须为 1 到 255 个字符")
    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    values = dict(owner_id=ANON, key_hash=key_hash, session_id=session_id,
                  question_set_id=question_set_id)
    table = InterviewCreateReceipt.__table__
    if db.bind.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    inserted = db.execute(insert(table).values(**values).on_conflict_do_nothing(
        index_elements=[table.c.owner_id, table.c.key_hash],
    ))
    receipt = db.get(InterviewCreateReceipt, (ANON, key_hash))
    assert receipt is not None
    if receipt.session_id != session_id or receipt.question_set_id != question_set_id:
        raise ValueError("Idempotency-Key 已用于不同的面试创建请求")
    return receipt, bool(inserted.rowcount)


def find_interview_create_replay(db: Session, key: str, session_id: str | None,
                                 question_set_id: str | None) -> Interview | None:
    """Return a committed result before graph validation, including ended runs."""
    if not key or len(key) > 255:
        raise ValueError("Idempotency-Key 长度必须为 1 到 255 个字符")
    receipt = db.get(InterviewCreateReceipt, (ANON, hashlib.sha256(key.encode("utf-8")).hexdigest()))
    if receipt is None:
        return None
    if receipt.session_id != session_id or receipt.question_set_id != question_set_id:
        raise ValueError("Idempotency-Key 已用于不同的面试创建请求")
    interview = get_interview(db, receipt.interview_id) if receipt.interview_id else None
    if interview is None:
        raise ValueError("原面试已不存在")
    return interview


def start_interview(db: Session, session_id: str | None, question_set_id: str | None,
                    *, idempotency_key: str | None = None) -> Interview:
    receipt = None
    if idempotency_key is not None:
        receipt, claimed = _create_receipt(db, idempotency_key, session_id, question_set_id)
        if not claimed:
            interview = get_interview(db, receipt.interview_id) if receipt.interview_id else None
            if interview is None:
                raise ValueError("原面试已不存在")
            return interview
    qset = resolve_question_set(db, session_id, question_set_id)
    if qset is None:
        raise ValueError("还没有可面试的题目，请先在模拟面试页生成")

    # The question-set row is the stable serialization key when no interview
    # exists yet; locking only a candidate interview cannot prevent two inserts.
    db.query(QuestionSet).filter(QuestionSet.id == qset.id).with_for_update().one()
    live = (
        db.query(Interview)
        .filter(Interview.status == "live", Interview.question_set_id == qset.id, Interview.profile_id == qset.profile_id)
        .first()
    )
    if live:
        if receipt is not None:
            receipt.interview_id = live.id
            db.commit()
        return get_interview(db, live.id)  # type: ignore[return-value]

    ready = (
        db.query(Interview)
        .filter(Interview.status == "ready", Interview.question_set_id == qset.id, Interview.profile_id == qset.profile_id)
        .order_by(Interview.created_at.desc())
        .first()
    )
    profile = db.get(JobProfile, qset.profile_id)
    title = f"{(profile.job_title if profile else '目标岗位')} · 全真模拟面试"
    tags = (profile.analysis or {}).get("focus") if profile else []
    if ready:
        interview = ready
        interview.status = "live"
        interview.started_at = now()
        interview.current_question_index = 0
        interview.summary = "面试已开始，题目将随提问逐题出现。"
        interview.tags = tags or interview.tags
    else:
        interview = Interview(
            id=new_id(),
            profile_id=qset.profile_id,
            question_set_id=qset.id,
            title=title,
            status="live",
            current_question_index=0,
            started_at=now(),
            tags=tags,
            summary="面试已开始，题目将随提问逐题出现。",
        )
        db.add(interview)
        db.flush()

    first = question_at(qset, 0)
    opening = opening_line(first)
    db.add(
        InterviewTurn(
            id=new_id(),
            interview_id=interview.id,
            role="interviewer",
            content=opening,
            question_id=first.id if first else None,
        )
    )
    if receipt is not None:
        # The receipt and first turn commit together; a crash after commit can
        # always replay the same interview even after its status changes.
        receipt.interview_id = interview.id
    db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


def delete_interview(db: Session, interview_id: str) -> None:
    """Remove one hub card. The question set stays so the session can start another."""
    # Serialize deletion with answer, start, and finish transitions so a
    # committed candidate turn cannot be removed by a stale read.
    interview = _lock_interview(db, interview_id)
    if not interview:
        raise ValueError("面试不存在")
    title = interview.title
    status = interview.status
    # Keep the receipt as a tombstone: a delayed retry using the old key must
    # not create another interview after the original was explicitly deleted.
    delete_interviews(db, [interview_id])
    # 空对象在审计页会显示成空白。至少留下场次当时的状态。
    audit(db, "interview.delete", title, {"status": status})
    db.commit()


def resume_or_start(db: Session, interview_id: str) -> Interview:
    """Hub card '开启这场面试' either resumes live or flips ready → live."""
    interview = _lock_interview(db, interview_id)
    if not interview:
        raise ValueError("面试不存在")
    if interview.status == "ended":
        raise ValueError("本场已结束，请查看复盘")
    if interview.status == "abandoned":
        # 旧版本把退出写成 abandoned；清理旧进度后按未开始重新开放。
        interview.status = "ready"
        interview.started_at = None
        interview.ended_at = None
        interview.elapsed_seconds = 0
        interview.current_question_index = 0
        interview.followups_on_question = 0
        interview.summary = "面试尚未开始"
        interview.turns.clear()
    if interview.status == "live":
        return interview
    interview.status = "live"
    interview.started_at = interview.started_at or now()
    interview.summary = "面试已开始，题目将随提问逐题出现。"
    qset = interview.question_set
    first = question_at(qset, 0) if qset else None
    interview.current_question_index = 0
    interview.followups_on_question = 0
    if not interview.turns:
        db.add(
            InterviewTurn(
                id=new_id(),
                interview_id=interview.id,
                role="interviewer",
                content=opening_line(first),
                question_id=first.id if first else None,
            )
        )
    db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


@dataclass
class AnswerProgress:
    """Keep the pending answer's decision outside the graph checkpoint."""

    interview: Interview
    current: Question | None
    next_question: Question | None
    question_id: str | None
    fallback: str
    advance: bool
    completed: bool = False


def persist_interview_answer(db: Session, interview_id: str, content: str, answer_mode: str, *, run_id: str | None = None) -> AnswerProgress:
    """Commit the candidate answer and question progress before model work."""
    # Record the visible answer before waiting on the row lock. A different
    # answer committed during that wait makes this concurrently sent command stale.
    previous_answer_id = None
    if hasattr(db, "query"):
        previous = (db.query(InterviewTurn.id).filter(
            InterviewTurn.interview_id == interview_id, InterviewTurn.role == "user"
        ).order_by(InterviewTurn.created_at.desc(), InterviewTurn.id.desc()).first())
        previous_answer_id = previous[0] if previous else None
    interview = _lock_interview(db, interview_id)
    if not interview:
        raise ValueError("面试不存在或已结束")

    # Reload relationships after acquiring the row lock so the decision uses
    # state committed by the preceding request rather than a stale identity-map.
    if hasattr(db, "expire"):
        db.expire(interview)
        interview = get_interview(db, interview_id)
        if interview is None:
            raise ValueError("面试不存在")

    existing = next((turn for turn in interview.turns if turn.id == run_id), None) if run_id else None
    if existing is None and hasattr(db, "query"):
        latest = (db.query(InterviewTurn.id).filter(
            InterviewTurn.interview_id == interview_id, InterviewTurn.role == "user"
        ).order_by(InterviewTurn.created_at.desc(), InterviewTurn.id.desc()).first())
        latest_answer = latest[0] if latest else None
        if latest_answer != previous_answer_id:
            raise ValueError("面试状态已变化，请刷新后重试")
    if existing is not None and (existing.role != "user" or existing.content != content or existing.answer_mode != answer_mode):
        raise ValueError("同一请求标识对应另一条回答")
    if existing is not None and existing.cite != "answer_pending":
        return AnswerProgress(interview, None, None, None, "", False, completed=True)
    if interview.status != "live":
        raise ValueError("面试不存在或已结束")

    last_turn = interview.turns[-1] if interview.turns else None
    if existing is None and last_turn is not None and last_turn.role == "user" and getattr(last_turn, "cite", None) == "answer_pending":
        raise ValueError("上一条回答仍在处理中，请稍后重试")

    if not content.strip():
        raise ValueError("请先输入或说出回答")

    questions = list(interview.question_set.questions) if interview.question_set else []
    if existing is None:
        current = questions[interview.current_question_index] if interview.current_question_index < len(questions) else None
        # Legacy/imported question sets can contain a stale relationship row.
        # Never copy that identifier into the FK column; the answer remains
        # durable while the next request can rebuild the question state.
        if current is not None and hasattr(db, "get") and db.get(Question, current.id) is None:
            current = None
        user_turn = InterviewTurn(
            id=run_id or new_id(), interview_id=interview.id, role="user",
            content=content, answer_mode=answer_mode, cite="answer_pending",
            # The candidate turn is the durable idempotency fact for this request.
            question_id=current.id if current else None,
        )
        db.add(user_turn)
        followups = int(interview.followups_on_question or 0)
        next_index = interview.current_question_index + 1
        advance = followups >= 1 or not questions
        final_question = current is not None and next_index >= len(questions) and advance
        interview.elapsed_seconds = int((now() - (interview.started_at or now())).total_seconds())
        if advance and next_index < len(questions):
            interview.current_question_index = next_index
            interview.followups_on_question = 0
        elif advance:
            interview.followups_on_question = 0
        else:
            interview.followups_on_question = followups + 1
        # Commit the answer and progression together before the provider call.
        db.commit()
    else:
        current = next((question for question in questions if question.id == existing.question_id), None)
        next_index = questions.index(current) + 1 if current in questions else interview.current_question_index + 1
        advance = not questions or interview.followups_on_question == 0
        final_question = current is not None and next_index >= len(questions) and advance

    # Reacquire the lock after the answer commit. Only one retry may generate
    # the follow-up, and a crashed request can finish its durable pending turn.
    interview = _lock_interview(db, interview_id)
    if interview is None:
        raise ValueError("面试不存在")
    if hasattr(db, "expire"):
        db.expire(interview)
        interview = get_interview(db, interview_id)
    if interview is None:
        raise ValueError("面试不存在")
    user_turn = next((turn for turn in interview.turns if turn.id == run_id), None) if run_id else next(
        (turn for turn in reversed(interview.turns) if turn.role == "user" and turn.cite == "answer_pending"), None
    )
    if user_turn is None:
        raise ValueError("回答记录不存在")
    if user_turn.cite != "answer_pending":
        return AnswerProgress(interview, None, None, None, "", False, completed=True)

    if advance and next_index < len(questions):
        next_q = questions[next_index]
        question_id = next_q.id if not hasattr(db, "get") or db.get(Question, next_q.id) is not None else None
        fallback = f"明白。接下来进入下一题：{next_q.stem}"
    elif final_question:
        next_q = None
        question_id = current.id if current and (not hasattr(db, "get") or db.get(Question, current.id) is not None) else None
        fallback = "感谢作答，本套问题已完成。你可以结束面试并生成复盘。"
    else:
        next_q = current
        question_id = current.id if current and (not hasattr(db, "get") or db.get(Question, current.id) is not None) else None
        fallback = "请再补充这道题的关键依据、具体步骤或边界情况。"

    return AnswerProgress(interview, current, next_q, question_id, fallback, advance)


async def propose_interview_followup(db: Session, progress: AnswerProgress, content: str) -> str:
    """Generate at most one follow-up after the answer is durable."""
    if progress.completed or progress.advance:
        return progress.fallback
    try:
        return await followup_line(
            db, progress.interview.id, progress.current, content,
            progress.next_question, progress.interview.turns,
        ) or progress.fallback
    except Exception:
        # A provider failure cannot strand an already committed answer.
        return progress.fallback


def finalize_interview_answer(db: Session, progress: AnswerProgress, followup: str) -> Interview:
    """Commit one interviewer turn and clear the pending answer marker."""
    interview = progress.interview
    if progress.completed:
        return interview
    user_turn = next(
        (turn for turn in reversed(interview.turns) if turn.role == "user" and turn.cite == "answer_pending"), None
    )
    if user_turn is None:
        raise ValueError("回答记录不存在")
    db.add(
        InterviewTurn(
            id=new_id(),
            interview_id=interview.id,
            role="interviewer",
            content=followup or progress.fallback,
            cite="针对上一轮的回答",
            question_id=progress.question_id,
        )
    )
    user_turn.cite = None
    db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


async def answer_interview(db: Session, interview_id: str, content: str, answer_mode: str, *, run_id: str | None = None) -> Interview:
    """Compatibility entry point for callers outside the application graph."""
    progress = persist_interview_answer(db, interview_id, content, answer_mode, run_id=run_id)
    followup = await propose_interview_followup(db, progress, content)
    return finalize_interview_answer(db, progress, followup)


def _stop_clock(interview: Interview) -> None:
    """结束瞬间停表。之后的复盘耗时不能再加进已用时。"""
    ended = now()
    interview.status = "ended"
    interview.ended_at = ended
    started = interview.started_at or ended
    interview.elapsed_seconds = max(0, int((ended - started).total_seconds()))


def abandon_interview(db: Session, interview_id: str) -> Interview:
    """Direct exit returns the attempt to ready without generating a recap."""
    interview = _lock_interview(db, interview_id)
    if not interview:
        raise ValueError("面试不存在")
    if hasattr(db, "expire"):
        db.expire(interview)
        interview = get_interview(db, interview_id)
        if interview is None:
            raise ValueError("面试不存在")
    if interview.status in {"abandoned", "live"}:
        interview.status = "ready"
        interview.started_at = None
        interview.ended_at = None
        interview.elapsed_seconds = 0
        interview.current_question_index = 0
        interview.followups_on_question = 0
        interview.summary = "面试尚未开始"
        # Both current and legacy exit paths persist the reset in one commit.
        interview.turns.clear()
        db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


async def finish_interview(db: Session, interview_id: str, *, enqueue_report: bool = False) -> Interview:
    """Stop one interview and optionally persist its report outbox atomically."""
    interview = _lock_interview(db, interview_id)
    if not interview:
        raise ValueError("面试不存在")
    if interview.status not in {"live", "ended"}:
        raise ValueError("未开始或已中止的面试不能生成复盘")
    if interview.status != "ended":
        _stop_clock(interview)
        interview.summary = "正在生成复盘"
    if enqueue_report and interview.report is None:
        transcript = [{"role": turn.role, "content": turn.content} for turn in interview.turns]
        job = enqueue_report_job(db, interview.id)
        if not job.payload:
            job.payload = {"title": interview.title[:200], "transcript": transcript[-80:]}
    # Status and outbox share this commit so a crash cannot leave an ended
    # interview with no durable report command.
    db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


def regenerate_report(db: Session, interview_id: str) -> Interview:
    """Clear one ended report so the temporary test action can enqueue it again."""
    interview = _lock_interview(db, interview_id)
    if not interview:
        raise ValueError("面试不存在")
    if interview.status != "ended":
        raise ValueError("只有已结束的面试可以重新生成复盘")
    if hasattr(db, "expire"):
        db.expire(interview)
        interview = get_interview(db, interview_id)
        if interview is None:
            raise ValueError("面试不存在")
    if interview.report is not None:
        # The transcript is intentionally retained; only the derived report is replaced.
        generation = interview.report.id
        db.delete(interview.report)
        interview.report = None
        db.flush()
        # The deleted report identifies this regeneration command. Concurrent
        # retries see no report and therefore cannot enqueue another generation.
        if hasattr(db, "begin_nested"):
            create_job(
                db,
                "report_generation",
                interview_id,
                idempotency_key=f"report:{interview_id}:regen:{generation}",
            )
    interview.summary = "正在重新生成复盘"
    db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


async def end_interview(db: Session, interview_id: str) -> Interview:
    """兼容旧调用：停表后就在这条连接里写完复盘。页面改走 finish。"""
    interview = await finish_interview(db, interview_id)
    if interview.report:
        return interview
    transcript = [{"role": t.role, "content": t.content} for t in interview.turns]
    await build_report(db, interview.id, interview.title, transcript)
    return get_interview(db, interview.id)  # type: ignore[return-value]


async def build_report(
    db: Session, interview_id: str, title: str, transcript: list[dict[str, str]]
) -> None:
    """后台补复盘。请求上的会话已经关掉，这里用自己的会话。"""
    try:
        recap = await write_report(db, title, transcript)
    except Exception:
        # The worker must always terminate with a persisted report. Otherwise the
        # UI keeps polling an ended interview forever after a provider or parser failure.
        recap = generation_failed_report()
    _store_report(db, interview_id, recap)


def schedule_report(
    interview_id: str,
    title: str,
    transcript: list[dict[str, str]],
) -> None:
    """Persist an idempotent report job before waking the local poller."""
    db = SessionLocal()
    try:
        # Normal completion is idempotent for the interview. Regeneration is
        # enqueued transactionally by regenerate_report using the old report ID.
        job = enqueue_report_job(db, interview_id)
        # Keep a bounded transcript snapshot for recovery when the source rows
        # are still being read by a separate worker; it is not a new business fact.
        if not job.payload:
            job.payload = {"title": title[:200], "transcript": transcript[-80:]}
        db.commit()
    finally:
        db.close()
    _ensure_report_worker()


def _ensure_report_worker() -> None:
    """Start one local poller; the database owns job state and retry count."""
    global _report_worker
    if _report_worker is not None and not _report_worker.done():
        return
    _report_worker = asyncio.create_task(_run_report_queue())


def start_report_worker(checkpointer: Any | None = None) -> None:
    """Start the durable poller with the lifespan saver when available."""
    global _report_checkpointer
    if checkpointer is not None:
        _report_checkpointer = checkpointer
    _ensure_report_worker()


def report_worker_running() -> bool:
    """Report whether the durable report queue still has an active poller."""
    return _report_worker is not None and not _report_worker.done()


async def stop_report_worker() -> None:
    """Stop the poller before its lifespan-owned saver closes."""
    global _report_worker, _report_checkpointer
    task = _report_worker
    _report_worker = None
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    _report_checkpointer = None


async def _restore_report_pipeline(checkpointer: Any, thread_id: str, run_id: str,
                                   pipeline: EvaluationPipeline) -> None:
    """Recreate closure state when an unfinished graph resumes after a restart."""
    snapshot = await build_application_graph(checkpointer=checkpointer).compiled.aget_state(
        checkpoint_config(thread_id=thread_id, owner_id=ANON)
    )
    previous = snapshot.values.get("payload") if snapshot.values else None
    if snapshot.next and previous and previous.get("run_id") == run_id:
        # Only the graph state survived the process. Rebuild scorer and coach
        # outputs before its pending node can resume from the checkpoint.
        for stage_name in pipeline.stages:
            await pipeline.run_stage(stage_name)


async def _run_report_queue() -> None:
    """Claim and execute persisted report jobs until the process stops."""
    # The graph adapter imports services.common; defer this dependency until
    # the service package has finished initialization.
    from app.agents.orchestration.workflows import run_business_graph

    worker = JobWorker(_report_worker_id)
    while True:
        own = SessionLocal()
        row = None
        try:
            row = worker.claim(own, kinds=["report_generation"])
            if row is None:
                own.commit()
                await asyncio.sleep(0.5)
                continue
            # Persist the lease before awaiting a provider call. A crash then
            # leaves a reclaimable running row instead of an invisible claim.
            own.commit()
            payload = dict(row.payload or {})
            interview_id = row.business_key
            interview = get_interview(own, interview_id)
            title = str(payload.get("title") or (interview.title if interview else "面试复盘"))
            transcript = payload.get("transcript") or (
                [{"role": turn.role, "content": turn.content} for turn in interview.turns]
                if interview
                else []
            )
            async with keep_job_lease(SessionLocal, row.id, worker.worker_id, lease_seconds=worker.lease_seconds) as lease:
                pipeline = _report_pipeline(own, title, transcript)
                # A lease retry gets a new checkpoint thread. Within one attempt,
                # the IDs remain stable so an interrupted graph can resume.
                thread_id = f"report:{row.id}:attempt:{row.attempts}"
                run_id = uuid5(NAMESPACE_URL, thread_id).hex
                if _report_checkpointer is not None:
                    # The PostgreSQL saver rejects even a read until the SQL
                    # owner binding exists. Commit it before the guard connection
                    # inspects a fresh thread, then let run_business_graph renew it.
                    authorize_checkpoint(
                        own, thread_id,
                        GraphAgentState(thread_id=thread_id, run_id=run_id,
                                        request_id=run_id, interview_id=interview_id,
                                        requested_mode="evaluation_report"),
                        owner_id=ANON, interview_id=interview_id,
                    )
                    own.commit()
                    await _restore_report_pipeline(_report_checkpointer, thread_id, run_id, pipeline)

                async def commit_report() -> str:
                    if not lease[0]:
                        raise RuntimeError("report job lease lost")
                    # The lease, UPSERT, and job completion share one commit.
                    current_job = (
                        own.query(DurableJob)
                        .filter(
                            DurableJob.id == row.id,
                            DurableJob.status == "running",
                            DurableJob.worker_id == worker.worker_id,
                            DurableJob.lease_until > now(),
                        )
                        .with_for_update()
                        .one_or_none()
                    )
                    if current_job is None:
                        raise RuntimeError("report job lease lost")
                    _store_report(own, interview_id, pipeline.report, commit=False)
                    completed = complete_job(own, row.id, worker_id=worker.worker_id,
                                             result={"interview_id": interview_id})
                    if completed is None:
                        raise RuntimeError("report job lease lost")
                    # The graph trace is secondary; commit the business fact
                    # before its separate observational transaction begins.
                    own.commit()
                    return interview_id

                async def report_stage(_state, *, name: str) -> dict[str, Any]:
                    status = await pipeline.run_stage(name)
                    return {"result": {"valid": True, **status}}

                callbacks = {
                    f"evaluation_report.{name}":
                    (lambda state, current=name: report_stage(state, name=current))
                    for name in pipeline.stages
                }
                await run_business_graph(
                    own, mode="evaluation_report", interview_id=interview_id,
                    action=commit_report, stages=callbacks,
                    checkpointer=_report_checkpointer, thread_id=thread_id, run_id=run_id,
                )
        except asyncio.CancelledError:
            own.rollback()
            raise
        except Exception as exc:  # noqa: BLE001 - one failed report must not kill the poller
            own.rollback()
            if row is not None:
                # Preserve a terminal graph error's retryability when classifying the job.
                fail_job(own, row.id, exc, worker_id=worker.worker_id)
                own.commit()
        finally:
            own.close()


def _store_report(db: Session, interview_id: str, recap: dict[str, Any], *, commit: bool = True) -> None:
    interview = get_interview(db, interview_id)
    if interview is None:
        return
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    # The unique interview_id arbitrates concurrent workers. The first
    # committed recap wins; retries cannot replace a completed report.
    table = Report.__table__
    values = dict(
        id=new_id(), interview_id=interview.id,
        score=float(recap.get("score") if recap.get("score") is not None else 0),
        review=recap.get("review") or "", issues=recap.get("issues") or [],
        dimensions=recap.get("dimensions"), scoring_status=recap.get("scoring_status") or "legacy",
    )
    if db.get_bind().dialect.name == "postgresql":
        statement = pg_insert(table).values(**values)
    else:
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        statement = sqlite_insert(table).values(**values)
    inserted = db.execute(statement.on_conflict_do_nothing(index_elements=[table.c.interview_id]))
    if inserted.rowcount:
        interview.summary = recap.get("summary") or recap.get("review") or interview.summary
    if commit:
        db.commit()
    else:
        db.flush()


def resolve_question_set(db: Session, session_id: str | None, question_set_id: str | None) -> QuestionSet | None:
    if question_set_id:
        return (
            db.query(QuestionSet)
            .join(JobProfile, JobProfile.id == QuestionSet.profile_id)
            .options(selectinload(QuestionSet.questions))
            .filter(QuestionSet.id == question_set_id, JobProfile.user_id == ANON)
            .one_or_none()
        )
    if session_id:
        if get_session(db, session_id) is None:
            return None
        found = latest_question_set_for_session(db, session_id)
        if found:
            return found
    return (
        db.query(QuestionSet)
        .join(JobProfile, JobProfile.id == QuestionSet.profile_id)
        .options(selectinload(QuestionSet.questions))
        .filter(JobProfile.user_id == ANON)
        .order_by(QuestionSet.created_at.desc())
        .first()
    )


def question_at(qset: QuestionSet, index: int) -> Question | None:
    questions = list(qset.questions)
    if 0 <= index < len(questions):
        return questions[index]
    return None


def current_question(interview: Interview) -> Question | None:
    if not interview.question_set:
        return None
    return question_at(interview.question_set, interview.current_question_index)


def opening_line(question: Question | None) -> str:
    if not question:
        return "你好，欢迎参加本次模拟面试。请先做个自我介绍，并讲讲你最近负责过的核心项目。"
    return f"你好，欢迎参加本次模拟面试。我们先从这道题开始：{question.stem} 请结合你主导过的项目来谈。"


async def followup_line(
    db: Session,
    interview_id: str,
    question: Question | None,
    answer: str,
    next_q: Question | None,
    turns: list[InterviewTurn],
) -> str:
    if not llm.llm_available():
        stem = next_q.stem if next_q else (question.stem if question else "刚才的方案")
        return f"刚才你提到了关键设计。追问：如果出现超时或抖动，{stem} 你会怎么兜底？"
    packed = [{"role": turn.role, "content": turn.content[:280]} for turn in turns[-6:]]
    try:
        text = await asyncio.wait_for(
            interviewer_followup(
                db,
                interview_id,
                question_stem=question.stem if question else "（开场）",
                answer=answer,
                next_stem=next_q.stem if next_q else "",
                turns=packed,
            ),
            timeout=FOLLOWUP_TIMEOUT_SECONDS,
        )
        return text or "刚才的回答还偏概括。请给出具体阈值、失败案例和兜底策略。"
    except Exception:
        return "刚才的回答还偏概括。请给出具体阈值、失败案例和兜底策略。"


# 没有任何实质回答时的封顶分。面试官开场不算作答，不能再落到录用线附近。
NO_ANSWER_SCORE_CAP = 20.0
SCORE_DIMENSIONS = {
    "technical_ability": "技术能力",
    "problem_analysis": "问题分析",
    "solution_tradeoffs": "方案权衡",
    "communication": "表达沟通",
}


def candidate_answers(transcript: list[dict[str, str]]) -> list[str]:
    """只保留候选人说过的话。空白和面试官独白都不构成作答。"""
    return [str(turn.get("content") or "").strip() for turn in transcript if turn.get("role") == "user" and str(turn.get("content") or "").strip()]


def unanswered_report() -> dict[str, Any]:
    """未作答是确定事实，不交给模型，也不使用任何高分样例。"""
    return {
        "score": NO_ANSWER_SCORE_CAP,
        "dimensions": {
            key: {"score": 0.0, "evidence": "候选人未作答，缺少可评分证据。", "advice": "完成至少一道题的回答后再评估此维度。"}
            for key in SCORE_DIMENSIONS
        },
        "summary": "未作答，远低于录用建议线。",
        "review": "候选人没有完成任何一道题的作答，本次不具备有效面试表现，不能给出录用建议。",
        "issues": [
            {
                "issue": "整场没有任何实质回答",
                "quote": "候选人未作答",
                "advice": "至少完成一道题，并给出具体方案、阈值和失败场景后再结束面试。",
            }
        ],
        "scoring_status": "valid",
    }


async def write_report(db: Session, title: str, transcript: list[dict[str, str]]) -> dict[str, Any]:
    """Run the evaluation subgraph while keeping existing report fallbacks."""
    # 没答题时模型会照抄提示里的高分样例。这里直接封顶，避免空场次进入录用区间。
    if not candidate_answers(transcript):
        return unanswered_report()
    if not llm.llm_available():
        return unavailable_report()
    pipeline = _report_pipeline(db, title, transcript)
    return await run_evaluation(title, transcript, scorer=pipeline.scorer,
                                validate_score=pipeline.validate_score, coach=pipeline.coach,
                                fallback_coach=pipeline.fallback_coach,
                                invalid_report=pipeline.invalid_report)


def _report_pipeline(db: Session, title: str, transcript: list[dict[str, str]]) -> EvaluationPipeline:
    """Bind the six graph stages to one report attempt without checkpointing answers."""
    if not candidate_answers(transcript):
        preset = unanswered_report()
    elif not llm.llm_available():
        preset = unavailable_report()
    else:
        preset = None

    async def scorer(stage_title: str, stage_transcript: list[dict[str, str]]) -> dict[str, Any]:
        return await _candidate_score(db, stage_title, stage_transcript)

    async def coach(stage_title: str, stage_transcript: list[dict[str, str]], score: float,
                    dimensions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        # The report graph passes business facts through the coach's input contract.
        role_input = validate_role_input("coach", {"goal": stage_title, "seed": {
            "transcript": stage_transcript, "score": score, "dimensions": dimensions,
        }})
        system = "你是复盘教练。分数已经冻结，不要改分，也不要输出分数。指出具体问题和可执行建议。"
        user = (
            f"场次：{role_input.goal}\n已冻结总分：{role_input.seed['score']}\n"
            f"维度评分及依据：{role_input.seed['dimensions']}\n对话：{role_input.seed['transcript']}\n"
            '返回 JSON：{"review": "...", "summary": "...", '
            '"issues": [{"issue":"...","quote":"...","advice":"..."}]}'
        )
        prose = await complete(db, "coach", system, user, max_tokens=1600, expect_json=True)
        if not isinstance(prose, dict):
            raise ValueError("coach returned a non-object response")
        # Reject a model's attempt to add score or other fields to frozen prose.
        prose = validate_role_output("coach", prose)
        if not isinstance(prose.get("review"), str) or not prose["review"].strip():
            raise ValueError("coach returned incomplete review")
        if not isinstance(prose.get("summary"), str) or not prose["summary"].strip():
            raise ValueError("coach returned incomplete summary")
        if not isinstance(prose.get("issues"), list):
            raise ValueError("coach returned invalid issues")
        return prose

    return EvaluationPipeline(title, transcript, scorer=scorer, validate_score=_validate_dimensions,
                              coach=coach, fallback_coach=local_coach_fallback,
                              invalid_report=invalid_report, preset_report=preset)


async def _frozen_score(db: Session, title: str, transcript: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Require four anchored, evidence-backed scores before the code computes a total."""
    return _validate_dimensions(await _candidate_score(db, title, transcript), transcript)


async def _candidate_score(db: Session, title: str, transcript: list[dict[str, str]]) -> dict[str, Any]:
    """Ask the scorer for candidate dimensions without accepting its total."""
    role_input = validate_role_input("scorer", {"goal": title, "seed": {"transcript": transcript}})
    answers = candidate_answers(role_input.seed["transcript"])
    data = await complete(
        db,
        "scorer",
        "你是模拟技术面试评分员。只评价候选人回答中可观察、与岗位相关的表现，不推断未表达的能力。"
        "分别给技术能力、问题分析、方案权衡、表达沟通四项 0 到 25 分。"
        "评分锚点：0=无相关证据或错误；1-8=明显不足且缺关键内容；9-15=部分正确但浅或不完整；"
        "16-20=正确、具体且推理基本完整；21-25=深入、严谨、能覆盖边界及取舍。"
        "每项必须引用候选人原话片段或准确概述具体行为作为 evidence；没有证据给 0 分并解释缺失。"
        "advice 写出可执行改进；勿把题目或面试官发言当作候选人证据。不要生成总分或复盘。",
        (
            f"场次：{role_input.goal}\n候选人回答数：{len(answers)}\n对话：{role_input.seed['transcript']}\n"
            '只返回 JSON，四项键必须齐全且分数为数字：{"dimensions": {'
            '"technical_ability":{"score":0,"evidence":"缺少候选人证据","advice":"..."},'
            '"problem_analysis":{"score":0,"evidence":"缺少候选人证据","advice":"..."},'
            '"solution_tradeoffs":{"score":0,"evidence":"缺少候选人证据","advice":"..."},'
            '"communication":{"score":0,"evidence":"缺少候选人证据","advice":"..."}}}'
        ),
        # Four dimensions each need evidence and advice; 800 tokens frequently
        # truncates the closing braces, which is the main parse failure in logs.
        max_tokens=1600,
        expect_json=True,
        temperature=0.0,
    )
    # The scorer proposes only four evidence dimensions; the graph computes total score.
    return validate_role_output("scorer", data)


def _validate_dimensions(data: dict[str, Any], transcript: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Validate every dimension before code computes and freezes the total."""
    answers = candidate_answers(transcript)
    raw = data.get("dimensions")
    if not isinstance(raw, dict) or set(raw) != set(SCORE_DIMENSIONS):
        raise ValueError("scorer returned incomplete dimensions")
    normalized: dict[str, dict[str, Any]] = {}
    for key in SCORE_DIMENSIONS:
        item = raw.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"scorer returned invalid dimension: {key}")
        score = float(item.get("score"))
        evidence = str(item.get("evidence") or "").strip()
        advice = str(item.get("advice") or "").strip()
        if not 0 <= score <= 25 or not evidence or not advice:
            raise ValueError(f"scorer returned invalid score or evidence: {key}")
        if not answers:
            score, evidence = 0.0, "候选人未作答，缺少可评分证据。"
        normalized[key] = {"score": round(score, 1), "evidence": evidence, "advice": advice}
    return normalized


def unavailable_report() -> dict[str, Any]:
    """Avoid presenting a canned sample as a real candidate assessment."""
    return {
        "score": 0.0,
        "dimensions": {
            key: {"score": 0.0, "evidence": "评分模型当前不可用，本次没有形成有效评分。", "advice": "配置评分模型后重新完成面试评估。"}
            for key in SCORE_DIMENSIONS
        },
        "summary": "评分暂不可用，未形成有效面试评估。",
        "review": "评分服务当前不可用，本报告不代表候选人的真实能力表现。请检查模型配置后重新评估。",
        "issues": [],
        "scoring_status": "unavailable",
    }


def generation_failed_report() -> dict[str, Any]:
    """Persist a clearly labelled report when the whole generation path crashes."""
    return {
        "score": 0.0,
        "dimensions": {
            key: {
                "score": 0.0,
                "evidence": "复盘生成失败，未形成可验证的评分证据。",
                "advice": "检查模型配置后重新评估本场面试。",
            }
            for key in SCORE_DIMENSIONS
        },
        "summary": "复盘生成失败，已保存保底结果。",
        "review": "复盘服务暂时不可用，已保存保底结果；本报告不代表候选人的真实能力表现。",
        "issues": [],
        "scoring_status": "unavailable",
    }


def local_coach_fallback(dimensions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Explain a valid score without another provider call.

    This keeps the score evidence useful when only the prose coach fails.
    """
    weak = [SCORE_DIMENSIONS[key] for key, item in dimensions.items() if float(item.get("score", 0)) < 15]
    focus = "、".join(weak) if weak else "边界条件和取舍"
    return {
        "review": f"分项评分已完成，但复盘文字暂未生成。建议下一次重点补充：{focus}。",
        "summary": "分项评分已完成，复盘文字使用本地保底提示。",
        "issues": [],
    }


def score_band(score: float) -> str:
    """Keep the existing 80-point line and label the two lower ranges."""
    if score >= 80:
        return "达到建议线"
    if score >= 65:
        return "接近建议线"
    return "尚未达到建议线"


def invalid_report() -> dict[str, Any]:
    """Mark a failed grading attempt invalid instead of assigning synthetic points."""
    return {
        "score": 0.0,
        "dimensions": {
            key: {"score": 0.0, "evidence": "模型未返回完整、有效的评分结果。", "advice": "重试评分后再查看结果。"}
            for key in SCORE_DIMENSIONS
        },
        "summary": "评分失败，未形成有效面试评估。",
        "review": "本次评分未能通过完整性与范围校验，分数无效，请重新评估。",
        "issues": [],
        "scoring_status": "invalid",
    }
