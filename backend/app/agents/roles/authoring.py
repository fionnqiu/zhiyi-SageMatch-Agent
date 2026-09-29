"""Author, critic, and interviewer entry points. Services call these instead of one-shot prompts."""

from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from sqlalchemy.orm import Session

from app.agents.contracts.contracts import profile_for
from app.agents.contracts.envelopes import AgentTask
from app.agents.orchestration.supervisor import choose_generation_revision
from app.agents.roles.loop import run_agent, run_handoff
from app.agents.roles.memory import MemoryManager
from app.agents.orchestration.trace import current_run_id


class _AuthoringState(TypedDict):
    result: dict[str, Any]
    verdict: dict[str, Any]
    objection: str
    revisions: int
    run_id: str
    handoffs: list[dict[str, Any]]
    next_role: str
    supervisor_fallback: str


def _authoring_graph(db: Session, job_text: str, context: str, hits: list[dict[str, Any]], on_thought: Any) -> Any:
    """Keep the single permitted revision in graph state, where routing can enforce it."""
    graph = StateGraph(_AuthoringState)

    async def author(state: _AuthoringState) -> dict[str, Any]:
        parent = state["handoffs"][-1]["task"]["task_id"] if state["handoffs"] else None
        task = AgentTask(
            run_id=state["run_id"], parent_task_id=parent,
            from_agent="supervisor", to_agent="author", route="interview_generation",
            goal=_author_task(job_text, state["objection"]), attempt=state["revisions"] + 1,
        )
        result = await run_handoff(
            db, task,
            context=context, seed={"hits": hits}, on_thought=on_thought,
        )
        return {"result": result, "handoffs": [*state["handoffs"], result["handoff"]]}

    async def critic(state: _AuthoringState) -> dict[str, Any]:
        task = AgentTask(
            run_id=state["run_id"], parent_task_id=state["handoffs"][-1]["task"]["task_id"],
            from_agent="supervisor", to_agent="critic", route="interview_generation",
            goal=_critic_task(state["result"]["output"]), attempt=state["revisions"] + 1,
        )
        result = await run_handoff(db, task, seed={"questions": state["result"]["output"].get("questions") or []}, on_thought=on_thought)
        return {"verdict": review_pack(result.get("output") or {}), "handoffs": [*state["handoffs"], result["handoff"]]}

    async def revise_once(state: _AuthoringState) -> dict[str, Any]:
        return {"objection": state["verdict"]["reason"] or "题目未通过质检", "revisions": 1}

    async def supervisor_decide(state: _AuthoringState) -> dict[str, Any]:
        # A critic veto permits one model-selected rewrite or a validated
        # fallback; the role cannot choose another route or write a question set.
        role, fallback = await choose_generation_revision(db, state["verdict"]["reason"])
        return {"next_role": role, "supervisor_fallback": fallback}

    def after_author(state: _AuthoringState) -> str:
        output = state["result"].get("output") or {}
        return "critic" if state["result"].get("ok") and output.get("questions") else "finish"

    def after_critic(state: _AuthoringState) -> str:
        return "supervisor" if not state["verdict"]["pass"] and state["revisions"] == 0 else "finish"

    def after_supervisor(state: _AuthoringState) -> str:
        return "revise" if state["next_role"] == "author" else "finish"

    graph.add_node("author", author)
    graph.add_node("critic", critic)
    graph.add_node("supervisor_decide", supervisor_decide)
    graph.add_node("revise_once", revise_once)
    graph.add_edge(START, "author")
    graph.add_conditional_edges("author", after_author, {"critic": "critic", "finish": END})
    graph.add_conditional_edges("critic", after_critic, {"supervisor": "supervisor_decide", "finish": END})
    graph.add_conditional_edges("supervisor_decide", after_supervisor,
                                {"revise": "revise_once", "finish": END})
    graph.add_edge("revise_once", "author")
    return graph.compile()


async def author_questions(
    db: Session,
    job_text: str,
    *,
    session_id: str | None = None,
    hits: list[dict[str, Any]] | None = None,
    on_thought=None,
) -> dict[str, Any]:
    """Run the author, then one critic veto. Prefetched hits are context, not a live tool requirement."""
    memory = MemoryManager(db, session_id=session_id)
    context = memory.render()
    if hits:
        context += "\n\n[预取知识]\n" + "\n".join(str(hit.get("text") or "")[:200] for hit in hits[:4])
    state = await _authoring_graph(db, job_text, context, hits or [], on_thought).ainvoke({
        "result": {"ok": False, "output": {}, "steps": 0, "observations": []},
        "verdict": {"pass": True, "reason": ""},
        "objection": "",
        "revisions": 0,
        "run_id": current_run_id() or uuid4().hex,
        "handoffs": [],
        "next_role": "",
        "supervisor_fallback": "",
    })
    result = state["result"]
    verdict = state["verdict"]
    output = result.get("output") or {}
    # Role nodes propose content only. The service commits memory together
    # with the validated question set, so a failed write cannot leave stale
    # profile or episode facts behind.
    output["_agent"] = {
        "ok": result.get("ok"),
        "steps": result.get("steps"),
        "observations": result.get("observations"),
        "verdict": "passed" if verdict["pass"] else "rejected",
        "reason": verdict["reason"],
        "handoffs": state["handoffs"],
        "supervisor_fallback": state["supervisor_fallback"],
    }
    return output


async def author_question_candidate(
    db: Session, job_text: str, *, context: str, hits: list[dict[str, Any]],
    run_id: str, attempt: int, objection: str = "", on_thought=None,
    parent_task_id: str | None = None,
) -> dict[str, Any]:
    """Produce one candidate for the application graph's author or revision stage."""
    task = AgentTask(
        run_id=run_id, parent_task_id=parent_task_id, from_agent="supervisor",
        to_agent="author", route="interview_generation",
        goal=_author_task(job_text, objection), attempt=attempt,
    )
    # The tool adapter fingerprints the ready index only if the author actually
    # calls hybrid_search; most candidate passes can reuse the prepared hits.
    from app.services.shared.common import ANON

    return await run_handoff(
        db, task, context=context,
        seed={"hits": hits, "cache_owner_id": ANON, "cache_session_id": run_id},
        on_thought=on_thought,
    )


async def critique_question_candidate(
    db: Session, candidate: dict[str, Any], *, run_id: str,
    parent_task_id: str | None = None, on_thought=None,
) -> dict[str, Any]:
    """Return a veto only; the critic cannot replace the author's questions."""
    task = AgentTask(
        run_id=run_id, parent_task_id=parent_task_id, from_agent="supervisor",
        to_agent="critic", route="interview_generation",
        goal=_critic_task(candidate), attempt=1,
    )
    result = await run_handoff(
        db, task, seed={"questions": candidate.get("questions") or []}, on_thought=on_thought,
    )
    return {"verdict": review_pack(result.get("output") or {}), "handoff": result["handoff"]}


async def interviewer_followup(
    db: Session,
    interview_id: str,
    *,
    question_stem: str,
    answer: str,
    next_stem: str,
    turns: list[dict[str, str]],
) -> str:
    """One bounded interviewer loop. The only tool besides finish is quoting a past turn."""
    memory = MemoryManager(db, interview_id=interview_id)
    try:
        result = await run_agent(
            db,
            profile_for("interviewer"),
            user=(
                "给出一段简短自然的口头追问，不要评分。finish.arguments 只包含 text。"
                f"\n当前题：{question_stem}\n候选人刚说：{answer}\n下一题：{next_stem or '无'}"
            ),
            context=memory.render(turns=_as_turns(turns)),
            seed={"turns": turns},
        )
        return str((result.get("output") or {}).get("text") or "").strip()
    except Exception:
        # Interview pacing and answer persistence must not depend on provider uptime.
        return f"你提到的做法是“{answer[:80]}”。{_followup_hint(answer)}"


def review_pack(raw: dict[str, Any]) -> dict[str, Any]:
    """Critic verdict only. Replacement questions are dropped before anyone can store them."""
    return {"pass": bool(raw.get("pass")), "reason": str(raw.get("reason") or "")[:200]}


def _author_task(job_text: str, objection: str) -> str:
    """The rewrite sees the veto reason, never a question the critic tried to substitute.

    The interview is a live conversation, so every question is spoken. Choices are not a kind.
    """
    task = (
        "根据岗位描述生成一套实时对话模拟面试题，题量 8 到 12 道，按职责覆盖来定，不要固定成 5 道。"
        "每条独立职责至少一题；职责少也不要少于 8 道，用追问深度补足，不要用同义重复凑数。"
        "面试是实时对话，禁止选择题，也不要给选项。"
        "讲方案、讲设计用 open；排查、权衡、故障用 scenario。kind 只能是这两个。"
        "finish.arguments 必须包含 job_title、summary、focus、reply、questions。"
        "每题含 kind、stem、answer、explanation，options 固定为空数组。"
        "answer 写可核对的要点，不要写成单个字母。"
        f"\n岗位描述：\n{job_text[:3000]}"
    )
    if objection:
        task += f"\n上一套未通过质检，只重写，不要解释：{objection[:200]}"
    return task


def _critic_task(output: dict[str, Any]) -> str:
    """Keep the critic's scope to a verdict while retaining the question context."""
    questions = output.get("questions") or []
    packed = "\n".join(
        f"{index + 1}. {item.get('stem') or ''}｜解析：{item.get('explanation') or ''}"
        for index, item in enumerate(questions)
        if isinstance(item, dict)
    )
    return "只判断是否通过。finish.arguments 只能有 pass 和 reason，不要返回题目。" + f"\n题目：\n{packed[:3000]}"


def _as_turns(turns: list[dict[str, str]]) -> list[Any]:
    class _Turn:
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.content = content

    return [_Turn(str(item.get("role") or ""), str(item.get("content") or "")) for item in turns]


def _followup_hint(answer: str) -> str:
    """Choose a deterministic probe from the answer when interviewer generation fails."""
    hints = ("为什么选择这个方案，还有什么替代方案？", "能补充关键步骤、数据或阈值吗？", "如果出现超时或部分失败，你会如何处理？")
    marker = sum(answer.encode("utf-8")) % len(hints)
    return hints[marker]
