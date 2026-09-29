"""LangGraph supervisor over the existing role contracts.

The self-written step loop is gone. Each autonomous role is a LangGraph ReAct
agent, and a supervisor decides which role runs next. Tool scope still comes
from the contract: a role cannot call a tool that is not on its allow-list.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph_supervisor import create_supervisor
from pydantic import ConfigDict, Field, create_model
from sqlalchemy.orm import Session

from app.integrations import llm
from app.agents.contracts.contracts import AgentProfile, profile_for, validate_role_input, validate_role_output
from app.agents.contracts.envelopes import AgentDecision, AgentTask, ToolResult
from app.agents.providers.governance import ProviderGovernor, cache_get, cache_put
from app.agents.contracts.state import AgentResult
from app.agents.orchestration.trace import current_run_id
from app.agents.tools.registry import ToolSpec, tool_schemas_for, tools_for
from app.models.platform.runtime import ToolRun

# One supervisor step plus one worker step. A veto-and-rewrite is a second call,
# not a longer graph, so the old "one rewrite" rule still holds.
_RECURSION_LIMIT = 8


class _RoleModel(BaseChatModel):
    """Chat model that calls the role's bound vendor through the existing gateway.

    LangGraph owns the tool loop. Routing, the call log, and the breaker stay in
    llm_gateway, so a role still degrades when its vendor is open.
    """

    role: str
    temperature: float = 0.2
    max_tokens: int = 800

    @property
    def _llm_type(self) -> str:
        return f"sagematch-{self.role}"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "_RoleModel":
        """Remember the tools LangGraph expects this role to be able to call."""
        del kwargs
        bound = self.model_copy()
        bound._tools = list(tools)
        # model_copy 只复制声明字段。思考回调挂在实例上，不拷过去出题页就收不到原文。
        bound._session = getattr(self, "_session", None)
        bound._on_thought = getattr(self, "_on_thought", None)
        return bound

    def _generate(self, messages: list[Any], stop: list[str] | None = None, **kwargs: Any) -> Any:
        del stop, kwargs
        return asyncio.run(self._agenerate(messages))

    async def _agenerate(self, messages: list[Any], stop: list[str] | None = None, **kwargs: Any) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult

        del stop, kwargs
        message = await self._complete(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _complete(self, messages: list[Any]) -> AIMessage:
        tools = list(getattr(self, "_tools", []))
        if self.role == "interviewer":
            try:
                return await self._interviewer_decision(messages)
            except llm.UsageBudgetError:
                raise
            except Exception:
                return _offline_interviewer_message(messages)
        # The supervisor's handoff tools carry injected graph state. The JSON gateway cannot fill
        # those, so a transfer is returned as a bare tool call and LangGraph injects the state.
        if any(str(getattr(tool, "name", "")).startswith("transfer_to_") for tool in tools):
            # A worker already returned its decision. Handing off again would loop the same task.
            if _observations(messages):
                return AIMessage(content="")
            text = await self._text(messages, tools)
            name = _transfer_name(text, tools)
            if not name:
                return AIMessage(content=text)
            return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": f"{self.role}-handoff"}])
        # 非流式角色优先使用标准 function calling；不支持时继续走旧 JSON action。
        if getattr(self, "_on_thought", None) is None:
            from app.services.operations.llm_gateway import complete_with

            schemas = tool_schemas_for(profile_for(self.role).tool_scope, role=self.role)
            try:
                data = await complete_with(
                    self._db(),
                    self.role,
                    "你是受契约约束的代理。只能调用列出的工具。不要解释。",
                    _decision_prompt(messages, tools),
                    max_tokens=self.max_tokens,
                    tools=schemas,
                    temperature=self.temperature,
                )
            except llm.UsageBudgetError:
                raise
            except Exception:
                # Legacy endpoints reject tools with 400; retain the proven JSON protocol as a fallback.
                data = await complete_with(
                    self._db(),
                    self.role,
                    "你是受契约约束的代理。只能调用列出的工具。不要解释。",
                    _decision_prompt(messages, tools),
                    max_tokens=self.max_tokens,
                    expect_json=True,
                    temperature=self.temperature,
                )
            if data is None:
                data = await complete_with(
                    self._db(),
                    self.role,
                    "你是受契约约束的代理。只能调用列出的工具。不要解释。",
                    _decision_prompt(messages, tools),
                    max_tokens=self.max_tokens,
                    expect_json=True,
                    temperature=self.temperature,
                )
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                return AIMessage(content="", tool_calls=[{"name": data["name"], "args": data["arguments"], "id": data.get("id", f"{self.role}-call")}])
        else:
            raw = await self._speak(
                messages,
                system="你是受契约约束的代理。只能调用列出的工具。不要解释。",
                user=_decision_prompt(messages, tools),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            from app.integrations.llm import _extract_json

            try:
                # 流式正文可能包在代码块里。和整段 JSON 用同一套提取。
                data = _extract_json(raw)
            except (ValueError, json.JSONDecodeError):
                data = {}
        if not isinstance(data, dict):
            return AIMessage(content="")
        name = str(data.get("tool") or "")
        args = data.get("arguments") if isinstance(data.get("arguments"), dict) else {}
        if self.role == "interviewer" and name == "finish":
            return AIMessage(content="", tool_calls=[{"name": "finish", "args": args, "id": f"{self.role}-call"}])
        if not name:
            return AIMessage(content=json.dumps(data, ensure_ascii=False))
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"{self.role}-call"}],
        )

    async def _interviewer_decision(self, messages: list[Any]) -> AIMessage:
        """Use one bounded text call for conversational follow-ups; tools stay available on explicit need."""
        from app.services.operations.llm_gateway import complete_with

        text = str(await complete_with(
            self._db(),
            "interviewer",
            "你是中文技术面试官。根据候选人刚才的回答提出一个简短、自然、可口头回答的追问。"
            "不要评价或给分，不要重复原题。若已给出下一题，直接转到下一题。",
            _latest_human(messages)[-1800:],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ) or "").strip()
        if text:
            return AIMessage(content="", tool_calls=[{"name": "finish", "args": {"text": text}, "id": "interviewer-finish"}])
        return _offline_interviewer_message(messages)

    async def _text(self, messages: list[Any], tools: list[Any]) -> str:
        catalog = "、".join(_tool_name(tool) for tool in tools)
        # 调度员也在出题这一轮里。有思考监听时不能再用整段调用，否则页面要等到作者才可能看到字。
        return await self._speak(
            messages,
            system="你是调度员。只返回一个工具名，不要解释。",
            user=f"可选：{catalog}\n任务：{_latest_human(messages)[:1000]}",
            max_tokens=40,
            temperature=0.0,
        )

    async def _speak(
        self,
        messages: list[Any],
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """有监听方时流式读。思考原文交出去，正文仍完整返回给工具循环。"""
        del messages
        listener = getattr(self, "_on_thought", None)
        if listener is None:
            from app.services.operations.llm_gateway import complete_with

            text = await complete_with(
                self._db(),
                self.role,
                system,
                user,
                max_tokens=max_tokens,
                expect_json=False,
                temperature=temperature,
            )
            return str(text or "")
        from app.services.operations.llm_gateway import stream_parts

        chunks: list[str] = []
        async for kind, delta in stream_parts(
            self._db(),
            self.role,
            system,
            user,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            if kind in {"thinking", "reasoning"} and delta.strip():
                await listener(delta)
                continue
            if kind == "content" and delta:
                chunks.append(delta)
        return "".join(chunks)

    def _db(self) -> Session:
        db = getattr(self, "_session", None)
        if db is None:
            raise RuntimeError("角色模型没有绑定数据库会话")
        return db


def _decision_prompt(messages: list[Any], tools: list[Any]) -> str:
    """Flatten the graph transcript into the JSON action prompt the gateway already parses."""
    catalog = "\n".join(f"- {_tool_name(tool)}: {_tool_description(tool)}" for tool in tools)
    seen = json.dumps(_observations(messages), ensure_ascii=False, default=str)[:2000]
    task = _latest_human(messages)
    return (
        '只返回 JSON：{"tool":"工具名","arguments":{...}}。'
        "信息足够时 tool 必须是 finish，arguments 放最终结果。\n"
        f"可用工具：\n{catalog or '（无）'}\n\n已有观察：\n{seen or '（还没有）'}\n\n任务：\n{task[:4000]}"
    )


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "")


def _transfer_name(text: str, tools: list[Any]) -> str:
    """Pick the handoff the supervisor named. Unknown text does not become a tool call."""
    names = [_tool_name(tool) for tool in tools if _tool_name(tool).startswith("transfer_to_")]
    for name in names:
        if name in text:
            return name
    return ""


def _tool_description(tool: Any) -> str:
    return str(getattr(tool, "description", "") or "")


def _latest_human(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _offline_interviewer_message(messages: list[Any]) -> AIMessage:
    """Return a deterministic fallback prompt when the interviewer vendor times out."""
    task = _latest_human(messages)
    answer = task.split("候选人刚说：", 1)[-1].split("\n下一题：", 1)[0].strip()
    next_question = task.split("下一题：", 1)[-1].strip() if "下一题：" in task else "无"
    if next_question and next_question != "无":
        text = f"你提到的做法是“{answer[:70]}”。接下来请回答：{next_question}"
    else:
        hints = ("为什么选择这个方案，还有什么替代方案？", "能补充关键步骤、数据或阈值吗？", "如果出现超时或部分失败，你会如何处理？")
        text = f"你提到的做法是“{answer[:70]}”。{hints[sum(answer.encode('utf-8')) % len(hints)]}"
    return AIMessage(content="", tool_calls=[{"name": "finish", "args": {"text": text}, "id": "interviewer-fallback"}])


def _observations(messages: list[Any]) -> list[dict[str, Any]]:
    """Tool results already produced in this graph run. The model sees these, not the raw transcript."""
    found: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            data = _loads(str(message.content))
            ok = not (isinstance(data, dict) and data.get("success") is False) and not str(message.content).startswith("Error")
            found.append({"tool": message.name, "ok": ok, "data": data})
    return found


def _loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _langchain_tool(spec: ToolSpec, db: Session, seed: dict[str, Any]) -> StructuredTool:
    """Adapt a contract tool to LangChain. Caching and the breaker stay here, not in the graph."""

    # LangChain cannot infer individual tool fields from **arguments. Declare
    # them explicitly so ToolNode accepts the same names advertised to the model.
    properties = (spec.args_schema or {}).get("properties") or spec.properties or {}
    required = set((spec.args_schema or {}).get("required") or spec.required)
    fields = {
        name: (Any, Field(...) if name in required else Field(default=None))
        for name in properties
    }
    fields["tool_call_id"] = (Annotated[str, InjectedToolCallId], Field(...))
    args_model = create_model(
        f"{spec.name.title().replace('_', '')}ToolArgs",
        __config__=ConfigDict(extra="allow" if spec.name == "finish" else "forbid"),
        **fields,
    )
    owner_id = str(seed.get("cache_owner_id") or "")
    session_id = str(seed.get("cache_session_id") or "")
    index_version = str(seed.get("cache_index_version") or "")
    # Identity is fixed before the tool runs; the index fingerprint is read
    # lazily for hybrid_search so an unused tool does not scan the corpus.
    cache_scope = f"{owner_id}:{session_id}" if owner_id and session_id else ""

    async def _run(tool_call_id: Annotated[str, InjectedToolCallId], **arguments: Any) -> str:
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        call_id = uuid.uuid4().hex
        status = "success"
        error_code = None
        cached_hit = False
        payload = dict(arguments)
        # Quote lookup needs the live turns. They are seed, not a model argument.
        if spec.name == "get_turn_quote":
            payload["turns"] = seed.get("turns") or []
        try:
            cache_version = index_version
            if spec.name == "hybrid_search" and spec.cache_ttl > 0 and cache_scope and not cache_version:
                from app.services.materials.rag.live_eval import index_version as ready_index_version

                llm.remaining_budget()
                cache_version, _, _ = ready_index_version(db)
                llm.remaining_budget()
            if spec.cache_ttl > 0 and cache_scope and cache_version:
                cached = cache_get(db, spec.name, arguments, scope=cache_scope, index_version=cache_version)
                if cached is not None:
                    cached_hit = True
                    status = "cached"
                    return json.dumps({**cached, "cached": True}, ensure_ascii=False, default=str)
            # Tool work shares the graph deadline; cancellation reaches the handler.
            remaining = llm.remaining_budget()
            async with asyncio.timeout(remaining):
                result = await spec.handler(db, payload)
            if not result.get("success", True):
                status = "error"
                error_code = "tool_rejected"
            if spec.cache_ttl > 0 and cache_scope and cache_version and result.get("success"):
                cache_put(db, spec.name, arguments, result, spec.cache_ttl,
                          scope=cache_scope, index_version=cache_version)
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TimeoutError, asyncio.TimeoutError):
            status = "timeout"
            error_code = "tool_timeout"
            return json.dumps({"success": False, "error": "request deadline exceeded"}, ensure_ascii=False)
        except Exception:  # noqa: BLE001 — a tool crash is an observation, not a graph abort
            status = "error"
            error_code = "tool_exception"
            try:
                _note_provider(db, spec.name, ok=False)
            except Exception:
                # Breaker accounting is secondary; a missing governance row
                # must not replace the safe tool failure returned to the model.
                pass
            return json.dumps({"success": False, "error": "tool unavailable", "error_code": error_code}, ensure_ascii=False)
        finally:
            # The invocation is recorded even when the handler fails. Store only
            # categorical diagnostics; arguments and exception text can be private.
            db.add(ToolRun(
                id=call_id, run_id=current_run_id(), tool_call_id=tool_call_id,
                tool_name=spec.name, status=status, error_code=error_code,
                cached=cached_hit, latency_ms=round((time.perf_counter() - started) * 1000, 2),
                started_at=started_at, completed_at=datetime.now(timezone.utc),
            ))

    return StructuredTool.from_function(
        coroutine=_run,
        name=spec.name,
        description=spec.description,
        args_schema=args_model,
    )


def _note_provider(db: Session, tool: str, *, ok: bool) -> None:
    """A failed retrieval counts against the role's vendor, not against the whole graph."""
    from app.agents.providers.router import choose_provider

    choice = choose_provider(db, "author" if tool == "hybrid_search" else "analyst")
    if choice.provider is None:
        return
    governor = ProviderGovernor(db, choice.provider.id, choice.provider.name)
    if ok:
        governor.record_success(0)
    else:
        governor.record_failure(0)


def _model_for(db: Session, profile: AgentProfile, on_thought=None) -> _RoleModel:
    model = _RoleModel(role=profile.role, temperature=profile.temperature, max_tokens=profile.max_tokens)
    model._session = db
    model._tools = []
    # 只有创建面试这次把思考原文挂上。其它角色调用保持整段 JSON。
    model._on_thought = on_thought
    return model


def build_role_agent(
    db: Session,
    profile: AgentProfile,
    seed: dict[str, Any],
    on_thought=None,
    checkpointer: Any | None = None,
) -> Any:
    """One ReAct worker. Its tools are exactly the contract allow-list.

    finish is a normal tool to LangGraph, so the prebuilt loop would keep calling it
    until the recursion limit. The edge after tools stops the graph on that call.
    """
    allowed = tools_for(profile.tool_scope)
    tools = [_langchain_tool(spec, db, seed) for spec in allowed.values()]
    model = _model_for(db, profile, on_thought).bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_errors=True)

    async def model_node(state: AgentState) -> dict[str, Any]:
        # The contract mission is the system side. The task arrives as the human message.
        reply = await model.ainvoke([HumanMessage(content=profile.mission), *state["messages"]])
        return {"messages": [reply]}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    def after_tools(state: AgentState) -> str:
        # Only an executed finish may stop the worker; rejected tool calls
        # return to the model so it can correct its arguments within budget.
        if _successful_finish_output(state["messages"]) is not None:
            return END
        return "agent"

    steps = {"n": 0}

    async def bounded_model(state: AgentState) -> dict[str, Any]:
        # The contract step budget is the stop, not LangGraph's recursion limit.
        steps["n"] += 1
        if steps["n"] > profile.max_steps:
            return {"messages": [AIMessage(content="") ]}
        return await model_node(state)

    graph = StateGraph(AgentState)
    graph.add_node("agent", bounded_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", END: END})
    return graph.compile(name=profile.role, checkpointer=checkpointer)


def build_supervisor(
    db: Session, roles: tuple[str, ...], seed: dict[str, Any] | None = None, on_thought=None
) -> Any:
    """Supervisor over the named roles. Workers keep their own tools; the supervisor only hands off."""
    workers = [build_role_agent(db, profile_for(role), seed or {}, on_thought) for role in roles]
    graph = create_supervisor(
        workers,
        model=_model_for(db, profile_for("analyst")),
        prompt=(
            "你是模拟面试的调度员。按任务把工作交给唯一合适的角色，然后结束。"
            "出题交给 author，质检交给 critic，追问交给 interviewer，"
            "打分交给 scorer，复盘交给 coach，评测交给 judge。"
            "不要自己作答，也不要改写角色的结果。"
        ),
        supervisor_name="orchestrator",
        output_mode="last_message",
    )
    return graph.compile()


async def run_agent(
    db: Session,
    profile: AgentProfile,
    *,
    user: str,
    context: str = "",
    seed: dict[str, Any] | None = None,
    on_thought=None,
    thread_id: str | None = None,
    checkpointer: Any | None = None,
    deadline_at: float | None = None,
) -> dict[str, Any]:
    """Run one role through the LangGraph ReAct loop until it finishes or the graph stops it."""
    graph = build_role_agent(db, profile, seed or {}, on_thought)
    if checkpointer is not None:
        graph = build_role_agent(db, profile, seed or {}, on_thought, checkpointer=checkpointer)
    role_input = validate_role_input(profile.role, {"goal": user, "context": context, "seed": seed or {}})
    task = f"{role_input.context}\n\n{role_input.goal}" if role_input.context else role_input.goal
    run_id = thread_id or uuid.uuid4().hex
    try:
        with llm.deadline_scope(deadline_at):
            remaining = llm.remaining_budget()
            async with asyncio.timeout(remaining):
                state = await graph.ainvoke(
                    {"messages": [HumanMessage(content=task)]},
                    config={"recursion_limit": _RECURSION_LIMIT, "configurable": {"thread_id": run_id}},
                )
    except llm.UsageBudgetError:
        result = AgentResult.failure("model_budget_exceeded", retryable=False, trace_id=run_id)
        return {**result.model_dump(), "role": profile.role}
    except TimeoutError:
        # Exhausting the shared absolute deadline cannot be repaired by retrying
        # this role; preserve that distinction in its handoff decision.
        result = AgentResult.failure("deadline_exceeded", retryable=False, trace_id=run_id)
        return {**result.model_dump(), "role": profile.role}
    except Exception:  # noqa: BLE001 — budget and vendor failures stay a role result
        result = AgentResult.failure("agent_execution_failed", retryable=True, trace_id=run_id)
        return {**result.model_dump(), "role": profile.role}
    messages = list(state.get("messages") or [])
    try:
        output = validate_role_output(profile.role, _finish_output(messages))
    except ValueError:
        result = AgentResult.failure("invalid_role_output", retryable=False, trace_id=run_id)
        return {**result.model_dump(), "role": profile.role, "steps": len(_observations(messages))}
    result = AgentResult(
        ok=bool(output),
        output=output,
        observations=_observations(messages),
        trace_id=run_id,
    )
    return {**result.model_dump(), "role": profile.role, "steps": len(result.observations)}


async def run_handoff(
    db: Session,
    task: AgentTask,
    *,
    context: str = "",
    seed: dict[str, Any] | None = None,
    on_thought=None,
) -> dict[str, Any]:
    """Execute one supervisor-approved role and preserve its complete decision.

    The business subgraph chooses the next role deterministically. This boundary
    validates the handoff before invoking a worker, so dispatch costs no extra
    model call and a worker cannot silently change its assigned role.
    """
    profile = profile_for(task.to_agent)
    # AgentTask persists a wall-clock deadline, while the model/tool scope
    # compares monotonic time; convert only the remaining duration at handoff.
    deadline_at = None
    if task.deadline_at is not None:
        wall_deadline = task.deadline_at
        if wall_deadline.tzinfo is None:
            wall_deadline = wall_deadline.replace(tzinfo=timezone.utc)
        remaining = max(0.0, (wall_deadline - datetime.now(timezone.utc)).total_seconds())
        deadline_at = time.monotonic() + remaining
    result = await run_agent(
        db, profile, user=task.goal, context=context, seed=seed,
        on_thought=on_thought, thread_id=task.task_id,
        deadline_at=deadline_at,
    )
    observations = [item for item in result.get("observations") or [] if isinstance(item, dict)]
    tool_results = [ToolResult(
        tool_call_id=str(item.get("tool_call_id") or uuid.uuid4().hex),
        tool_name=str(item.get("tool") or item.get("tool_name") or "unknown"),
        ok=bool(item.get("ok")), data=item.get("data"),
        error_code=item.get("error_code"), retryable=bool(item.get("retryable", False)),
        latency_ms=float(item.get("latency_ms") or 0), cached=bool(item.get("cached", False)),
        trace_id=str(result.get("trace_id") or task.trace_id),
    ) for item in observations]
    evidence_refs = [str(ref) for ref in result.get("evidence_refs") or []]
    decision = AgentDecision(
        task_id=task.task_id, agent=task.to_agent,
        status="success" if result.get("ok") else "failed",
        decision="finished" if result.get("ok") else str(result.get("error_code") or "worker_failed"),
        output=dict(result.get("output") or {}), evidence_refs=evidence_refs,
        next_action="continue" if result.get("ok") else "fail",
        confidence=1.0 if result.get("ok") else 0.0,
        observations=observations, tool_results=tool_results,
        trace_id=str(result.get("trace_id") or task.trace_id),
    )
    # The patch is JSON-safe and can be carried by the business subgraph state.
    return {**result, "handoff": {
        "task": task.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
    }}


async def run_team(
    db: Session,
    roles: tuple[str, ...],
    *,
    user: str,
    context: str = "",
    seed: dict[str, Any] | None = None,
    on_thought=None,
    thread_id: str | None = None,
    deadline_at: float | None = None,
) -> dict[str, Any]:
    """Let the supervisor pick among roles. Callers that need one role still use run_agent."""
    graph = build_supervisor(db, roles, seed, on_thought)
    task = f"{context}\n\n{user}" if context else user
    run_id = thread_id or uuid.uuid4().hex
    try:
        with llm.deadline_scope(deadline_at):
            remaining = llm.remaining_budget()
            async with asyncio.timeout(remaining):
                state = await graph.ainvoke(
                    {"messages": [HumanMessage(content=task)]},
                    config={"recursion_limit": _RECURSION_LIMIT, "configurable": {"thread_id": run_id}},
                )
    except llm.UsageBudgetError:
        result = AgentResult.failure("model_budget_exceeded", retryable=False, trace_id=run_id)
        return {**result.model_dump(), "messages": []}
    except Exception:  # Keep supervisor failures in the same API envelope as workers.
        result = AgentResult.failure("supervisor_execution_failed", retryable=True, trace_id=run_id)
        return {**result.model_dump(), "messages": []}
    messages = list(state.get("messages") or [])
    result = AgentResult(
        # A supervisor invocation can legitimately return no final worker
        # payload in a mocked or interrupted handoff; the call itself still
        # succeeded and callers need that distinction for retry decisions.
        ok=True,
        output=_finish_output(messages),
        observations=_observations(messages),
        trace_id=run_id,
    )
    return {**result.model_dump(), "messages": messages}


def _finish_output(messages: list[Any]) -> dict[str, Any]:
    """Return the last accepted finish payload, not the model's unexecuted arguments."""
    return _successful_finish_output(messages) or {}


def _successful_finish_output(messages: list[Any]) -> dict[str, Any] | None:
    """Match finish receipts to calls before trusting the tool's accepted payload."""
    pending: dict[str, set[str]] = {}
    output: dict[str, Any] | None = None
    for message in messages:
        if isinstance(message, AIMessage):
            pending.update({
                str(call["id"]): set(call["args"])
                for call in message.tool_calls or []
                if call.get("name") == "finish" and call.get("id") and isinstance(call.get("args"), dict)
            })
        elif isinstance(message, ToolMessage) and message.name == "finish":
            call_id = str(message.tool_call_id or "")
            if call_id not in pending:
                continue
            submitted_fields = pending.pop(call_id)
            receipt = _loads(str(message.content))
            if (
                message.status != "error"
                and isinstance(receipt, dict)
                and receipt.get("success") is True
                and receipt.get("final") is True
                and isinstance(receipt.get("payload"), dict)
            ):
                # StructuredTool fills absent optional arguments with None;
                # those defaults were not part of the model's decision.
                output = {key: value for key, value in receipt["payload"].items() if key in submitted_fields}
    return output
