"""Finite application LangGraph with four business subgraphs."""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ConfigDict, Field

from app.integrations import llm
from app.agents.contracts.envelopes import AgentDecision, AgentEvent, AgentTask, ErrorEnvelope, ToolResult
from app.agents.contracts.state import AgentState as TypedAgentState


class AgentState(TypedAgentState):
    """JSON-safe state crossing the public application graph boundary."""

    requested_mode: str = ""
    events: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="ignore")

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class _GraphState(TypedDict):
    payload: dict[str, Any]


StageCallback = Callable[[AgentState], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]

STAGES = {
    "knowledge_qa": ("query_rewrite", "parallel_retrieve", "candidate_fusion", "deduplicate_and_diversify", "rerank", "context_select", "answer", "citation_validate", "repair_once"),
    "interview_generation": ("load_job_profile", "retrieve_context", "author", "deterministic_validate", "critic", "revise_once", "question_set_validate"),
    "live_interview": ("load_interview", "ask_question", "wait_for_answer", "persist_answer", "decide_followup", "interviewer_followup", "advance_question", "finish_or_continue"),
    "evaluation_report": ("load_transcript", "scorer", "score_schema_validate", "score_freeze", "coach", "report_validate"),
}


def _emit(state: AgentState, node: str, event_type: str, payload: dict[str, Any]) -> None:
    state.events.append(AgentEvent(run_id=state.run_id, node=node, agent=state.active_agent or None,
                                   event_type=event_type, payload=payload, trace_id=state.run_id).model_dump(mode="json"))


def _fail(state: AgentState, code: str, *, retryable: bool = False) -> None:
    state.error = ErrorEnvelope(error_code=code, retryable=retryable, trace_id=state.run_id).public_dict()
    state.status = "failed"
    state.next_action = "fail"


class ApplicationGraph:
    """Compiled graph; callbacks remain outside serialized checkpoint state."""

    def __init__(self, *, checkpointer: Any | None = None, max_steps: int = 32,
                 stages: Mapping[str, StageCallback] | None = None,
                 stage_sequences: Mapping[str, tuple[str, ...]] | None = None,
                 handlers: Mapping[str, StageCallback] | None = None,
                 commit: StageCallback | None = None,
                 first_commit: StageCallback | None = None,
                 first_commit_after: str | None = None) -> None:
        self.max_steps = max_steps
        self.checkpointer = checkpointer
        self.stages = dict(stages or {})
        self.handlers = dict(handlers or {})
        self.commit = commit
        self.first_commit = first_commit
        # Command runs keep the live interview route but execute only their
        # actual stages; answer-specific nodes must never appear in their trace.
        sequences = {**STAGES, **(stage_sequences or {})}
        followup_names: tuple[str, ...] = ()
        if first_commit is not None:
            if not first_commit_after or first_commit_after not in sequences["live_interview"]:
                raise ValueError("first commit requires a live interview stage boundary")
            names = sequences["live_interview"]
            boundary = names.index(first_commit_after) + 1
            followup_names = names[boundary:]
            sequences["live_interview"] = names[:boundary]
        self.subgraphs = {mode: self._subgraph(mode, names) for mode, names in sequences.items()}
        self.followup_subgraph = self._subgraph("live_interview", followup_names) if followup_names else None
        graph = StateGraph(_GraphState)
        for name, fn in (("load_context", self._load), ("route_node", self._route),
                         ("policy_gate", self._policy), ("supervisor_handoff", self._handoff),
                         ("dispatch_subgraph", self._dispatch),
                         ("answer_commit_node", self._first_commit),
                         ("dispatch_followup", self._dispatch_followup),
                         ("observe_node", self._observe), ("validate_node", self._validate),
                         ("decide_node", self._decide), ("commit_node", self._commit),
                         ("checkpoint_node", self._checkpoint), ("finish_node", self._finish)):
            # Dispatch already brackets its child graph. Finish must remain
            # the terminal event, so the other main nodes share one timer.
            graph.add_node(name, fn if name in {"dispatch_subgraph", "dispatch_followup", "finish_node"} else self._timed_main(name, fn))
        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "route_node")
        graph.add_edge("route_node", "policy_gate")
        graph.add_conditional_edges(
            "policy_gate",
            lambda s: "dispatch" if s["payload"]["status"] == "running" else (
                "commit" if s["payload"]["status"] == "waiting" and self.commit else "finish"
            ),
            {"dispatch": "supervisor_handoff", "commit": "commit_node", "finish": "finish_node"},
        )
        graph.add_edge("supervisor_handoff", "dispatch_subgraph")
        if first_commit is not None:
            graph.add_edge("dispatch_subgraph", "answer_commit_node")
            graph.add_edge("answer_commit_node", "dispatch_followup")
            graph.add_edge("dispatch_followup", "observe_node")
        else:
            graph.add_edge("dispatch_subgraph", "observe_node")
        graph.add_edge("observe_node", "validate_node")
        graph.add_edge("validate_node", "decide_node")
        graph.add_conditional_edges("decide_node", self._decision_edge,
                                    {"retry": "supervisor_handoff", "commit": "commit_node", "checkpoint": "checkpoint_node"})
        graph.add_edge("commit_node", "checkpoint_node")
        graph.add_edge("checkpoint_node", "finish_node")
        graph.add_edge("finish_node", END)
        self.compiled = graph.compile(checkpointer=checkpointer)

    @staticmethod
    def _timed_main(name: str, action: Callable[[_GraphState], Awaitable[_GraphState]]) -> Callable[[_GraphState], Awaitable[_GraphState]]:
        """Record one truthful lifecycle pair around a main-graph node."""

        async def run(data: _GraphState) -> _GraphState:
            state = AgentState.model_validate(data["payload"])
            attempt = state.retry_count + 1
            started = time.perf_counter()
            _emit(state, name, "node_started", {
                "attempt": attempt, "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            try:
                output = await action({"payload": state.to_json_dict()})
                state = AgentState.model_validate(output["payload"])
            except Exception:
                _fail(state, "graph_node_failed", retryable=True)
            _emit(state, name, "node_finished", {
                "attempt": attempt, "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "status": "failed" if state.error or state.status == "failed" else (
                    "success" if state.status == "running" else state.status
                ),
                **({"error_code": state.error["error_code"]} if state.error else {}),
            })
            return {"payload": state.to_json_dict()}

        return run

    def _advance(self, state: AgentState, node: str) -> bool:
        if state.deadline_at is not None and datetime.now(timezone.utc) >= state.deadline_at:
            _fail(state, "deadline_exceeded")
            return False
        if state.loop_count >= min(state.max_graph_steps, self.max_steps) or not state.can_continue(datetime.now(timezone.utc)):
            _fail(state, "graph_budget_exceeded")
            return False
        state.loop_count += 1
        state.current_node = node
        return True

    @staticmethod
    def _patch(state: AgentState, patch: Mapping[str, Any], *, allow_commit: bool = False) -> AgentState:
        if not isinstance(patch, Mapping):
            raise TypeError("graph callback must return a mapping")
        if not allow_commit and (
            patch.get("status") == "committed"
            or isinstance(patch.get("result"), Mapping) and patch["result"].get("committed") is True
        ):
            raise ValueError("only commit_node may confirm a business write")
        # Runtime adapters may change results and decisions, never route or ownership.
        # Rejecting unknown keys keeps misspelled evidence from silently
        # disappearing while a valid-looking result proceeds to commit.
        allowed = {"result", "diagnostics", "candidates", "selected_chunks", "citations",
                   "question_set_id", "report_id", "error", "status", "next_action", "decision"}
        unknown = set(patch) - allowed
        if unknown:
            raise ValueError(f"unknown graph state patch fields: {', '.join(sorted(map(str, unknown)))}")
        value = state.to_json_dict()
        value.update(patch)
        if isinstance(patch.get("diagnostics"), Mapping):
            # Each stage contributes one part of the run's diagnostics. A
            # later citation or provider stage must not erase retrieval facts.
            value["diagnostics"] = {**state.diagnostics, **patch["diagnostics"]}
        return AgentState.model_validate(value)

    async def _callback(self, state: AgentState, callback: StageCallback, *, allow_commit: bool = False) -> AgentState:
        # A graph node may await work outside the LLM client; enforce the same
        # absolute deadline at that boundary so cancellation reaches its task.
        remaining = llm.remaining_budget()
        if state.deadline_at is not None:
            state_remaining = (state.deadline_at - datetime.now(timezone.utc)).total_seconds()
            remaining = min(remaining, state_remaining) if remaining is not None else state_remaining
        if remaining is not None and remaining <= 0:
            raise TimeoutError("graph deadline exceeded")
        patch = callback(state)
        if inspect.isawaitable(patch):
            async with asyncio.timeout(remaining):
                patch = await patch
        # Synchronous adapters cannot be cancelled in-place, but their late
        # result must never be accepted as a successful graph transition.
        llm.remaining_budget()
        if state.deadline_at is not None and datetime.now(timezone.utc) >= state.deadline_at:
            raise TimeoutError("graph deadline exceeded")
        return self._patch(state, patch, allow_commit=allow_commit)

    def _subgraph(self, mode: str, stages: tuple[str, ...]) -> Any:
        graph = StateGraph(_GraphState)
        previous = START
        for stage in stages:
            name = f"{mode}.{stage}"

            async def run(data: _GraphState, *, current: str = name, role: str = stage) -> _GraphState:
                state = AgentState.model_validate(data["payload"])
                if state.status != "running" or not self._advance(state, current):
                    return {"payload": state.to_json_dict()}
                state.active_agent = role if role in {"author", "critic", "scorer", "coach", "interviewer_followup"} else ""
                callback = self.stages.get(current)
                if callback:
                    attempt = state.retry_count + 1
                    started = time.perf_counter()
                    _emit(state, current, "node_started", {
                        "route": mode, "attempt": attempt,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    try:
                        state = await self._callback(state, callback)
                    except llm.UsageBudgetError:
                        _fail(state, "model_budget_exceeded", retryable=False)
                    except TimeoutError:
                        _fail(state, "deadline_exceeded", retryable=False)
                    except Exception:
                        _fail(state, "graph_stage_failed", retryable=True)
                    _emit(state, current, "node_finished", {
                        "route": mode, "attempt": attempt,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                        "status": "failed" if state.error or state.status == "failed" else ("success" if state.status == "running" else state.status),
                        **({"error_code": state.error["error_code"]} if state.error else {}),
                    })
                state.completed_nodes.append(current)
                return {"payload": state.to_json_dict()}

            graph.add_node(name, run)
            graph.add_edge(previous, name)
            previous = name
        graph.add_edge(previous, END)
        return graph.compile()

    async def _load(self, data: _GraphState) -> _GraphState:
        state = AgentState.model_validate(data["payload"])
        if self._advance(state, "load_context"):
            # This timestamp is the authoritative graph start; trace rows are
            # persisted only after business commit and would otherwise start late.
            _emit(state, "load_context", "run_started", {"timestamp": datetime.now(timezone.utc).isoformat()})
        return {"payload": state.to_json_dict()}

    async def _route(self, data: _GraphState) -> _GraphState:
        state = AgentState.model_validate(data["payload"])
        if state.status == "running" and self._advance(state, "route_node"):
            mode = state.requested_mode if state.requested_mode in {*STAGES, "clarification", "unsupported"} else (state.route.route if state.route else "unsupported")
            state.requested_mode = mode
            if not any(e.get("event_type") == "route_decided" for e in state.events):
                # Traces record the route, never the raw question or attachment
                # body; the business message table already owns that content.
                _emit(state, "route_node", "route_decided", {"route": mode})
        return {"payload": state.to_json_dict()}

    async def _policy(self, data: _GraphState) -> _GraphState:
        state = AgentState.model_validate(data["payload"])
        if state.status == "running":
            self._advance(state, "policy_gate")
        if state.status == "running":
            if not state.user_id or not state.tenant_id:
                _fail(state, "owner_missing")
            elif len(state.original_query) > 4000:
                _fail(state, "input_too_large")
            elif ((state.token_budget is not None and state.token_budget < 0)
                  or (state.cost_budget is not None and (
                      not math.isfinite(state.cost_budget) or state.cost_budget < 0
                  ))):
                _fail(state, "invalid_budget")
            else:
                # Worker ToolNode enforces the role allow-list; this gate also
                # rejects a future planner asking another business route to
                # invoke one of those tools before dispatch begins.
                from app.agents.contracts.contracts import profile_for

                roles = {
                    "knowledge_qa": ("analyst",),
                    "interview_generation": ("author", "critic"),
                    "live_interview": ("interviewer",),
                    "evaluation_report": ("scorer", "coach"),
                }.get(state.requested_mode, ())
                allowed = {tool for role in roles for tool in profile_for(role).tool_scope}
                requested = {str(item.get("tool_name")) for item in state.plan
                             if isinstance(item, dict) and item.get("tool_name")}
                if not requested <= allowed:
                    _fail(state, "tool_forbidden")
        if state.status == "running" and state.requested_mode == "clarification":
            state.status, state.next_action = "waiting", "clarification"
        elif state.status == "running" and state.requested_mode == "unsupported":
            state.status, state.next_action = "unsupported", "unsupported"
        return {"payload": state.to_json_dict()}

    async def _handoff(self, data: _GraphState) -> _GraphState:
        """Authorize one bounded business subgraph attempt and retain its parent task."""
        state = AgentState.model_validate(data["payload"])
        if state.status == "running" and self._advance(state, "supervisor_handoff"):
            parent = state.handoffs[-1]["task"]["task_id"] if state.handoffs else None
            refs = [ref for ref in (state.session_id, state.interview_id) if ref]
            task = AgentTask(
                run_id=state.run_id, parent_task_id=parent, from_agent="supervisor",
                to_agent=state.requested_mode, route=state.requested_mode,
                goal=f"execute_{state.requested_mode}", input_refs=refs,
                deadline_at=state.deadline_at, attempt=state.retry_count + 1,
                trace_id=state.run_id,
            )
            state.handoffs.append({"task": task.model_dump(mode="json"), "decision": None})
            _emit(state, "supervisor_handoff", "agent_called", {
                "task_id": task.task_id, "agent": task.to_agent, "attempt": task.attempt,
            })
        return {"payload": state.to_json_dict()}

    async def _dispatch(self, data: _GraphState) -> _GraphState:
        state = AgentState.model_validate(data["payload"])
        if self._advance(state, "dispatch_subgraph"):
            mode = state.requested_mode
            attempt = state.retry_count + 1
            started = time.perf_counter()
            _emit(state, "dispatch_subgraph", "node_started", {
                "node": mode, "attempt": attempt,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            try:
                state = AgentState.model_validate((await self.subgraphs[mode].ainvoke({"payload": state.to_json_dict()}))["payload"])
            except Exception:
                _fail(state, "graph_dispatch_failed", retryable=True)
            _emit(state, "dispatch_subgraph", "node_finished", {
                "node": mode, "attempt": attempt,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "status": "failed" if state.error or state.status == "failed" else ("success" if state.status == "running" else state.status),
                **({"error_code": state.error["error_code"]} if state.error else {}),
            })

            state.completed_nodes.append(mode)
        return {"payload": state.to_json_dict()}

    async def _first_commit(self, data: _GraphState) -> _GraphState:
        """Commit the candidate answer before any follow-up model is invoked."""
        state = AgentState.model_validate(data["payload"])
        if state.status == "running" and self.first_commit and self._advance(state, "answer_commit_node"):
            if state.result.get("valid") is not True:
                _fail(state, "graph_validation_failed", retryable=False)
            else:
                try:
                    state = await self._callback(state, self.first_commit, allow_commit=True)
                    _emit(state, "answer_commit_node", "business_committed", {"route": state.requested_mode,
                                                                              "phase": "answer"})
                except TimeoutError:
                    _fail(state, "deadline_exceeded", retryable=False)
                except Exception:
                    _fail(state, "answer_commit_failed", retryable=True)
        return {"payload": state.to_json_dict()}

    async def _dispatch_followup(self, data: _GraphState) -> _GraphState:
        """Continue the same graph run after its first durable business fact."""
        state = AgentState.model_validate(data["payload"])
        if state.status == "running" and self.followup_subgraph and self._advance(state, "dispatch_followup"):
            try:
                state = AgentState.model_validate((await self.followup_subgraph.ainvoke(
                    {"payload": state.to_json_dict()}
                ))["payload"])
            except Exception:
                _fail(state, "graph_dispatch_failed", retryable=True)
        return {"payload": state.to_json_dict()}

    async def _observe(self, data: _GraphState) -> _GraphState:
        state = AgentState.model_validate(data["payload"])
        if state.status == "running" and self._advance(state, "observe_node"):
            summary = {
                "result_keys": sorted(state.result),
                "candidate_count": state.diagnostics.get("candidate_count", len(state.candidates)),
                "selected_count": len(state.selected_chunks),
                "handoff_count": len(state.handoffs),
            }
            state.diagnostics = {**state.diagnostics, **{
                key: summary[key] for key in ("candidate_count", "selected_count", "handoff_count")
            }}
            _emit(state, "observe_node", "result_observed", summary)
        return {"payload": state.to_json_dict()}

    async def _validate(self, data: _GraphState) -> _GraphState:
        state = AgentState.model_validate(data["payload"])
        if state.status == "running" and self._advance(state, "validate_node"):
            staged = any(name.startswith(f"{state.requested_mode}.") for name in self.stages)
            valid = state.result.get("valid")
            if valid is False or (staged and type(valid) is not bool):
                code = str(state.result.get("error_code") or "graph_validation_failed") if valid is False else "graph_validation_missing"
                state.error = ErrorEnvelope(
                    error_code=code, retryable=bool(state.result.get("retryable", True)), trace_id=state.run_id,
                ).public_dict()
                _emit(state, "validate_node", "validation_failed", {"error_code": code})
        return {"payload": state.to_json_dict()}

    async def _decide(self, data: _GraphState) -> _GraphState:
        state = AgentState.model_validate(data["payload"])
        # Preserve the full worker proposal before retry or commit changes the state.
        if state.handoffs and state.handoffs[-1]["decision"] is None:
            task = AgentTask.model_validate(state.handoffs[-1]["task"])
            evidence = [str(item[key]) for item in state.citations for key in ("chunk_id", "source_id")
                        if isinstance(item, dict) and item.get(key)]
            raw_tools = state.diagnostics.get("tool_results") or []
            tools = [ToolResult.model_validate(item) for item in raw_tools if isinstance(item, dict)]
            observations = state.diagnostics.get("observations") or []
            decision = AgentDecision(
                task_id=task.task_id, agent=task.to_agent,
                status="failed" if state.error or state.status == "failed" else "success",
                decision=str((state.error or {}).get("error_code") or "completed"),
                output=dict(state.result), evidence_refs=list(dict.fromkeys(evidence)),
                next_action="fail" if state.error or state.status == "failed" else "continue",
                confidence=0.0 if state.error or state.status == "failed" else 1.0,
                observations=[item for item in observations if isinstance(item, dict)],
                tool_results=tools, trace_id=state.run_id,
            )
            state.handoffs[-1]["decision"] = decision.model_dump(mode="json")
            _emit(state, "decide_node", "agent_completed", {
                "task_id": task.task_id, "agent": task.to_agent, "status": decision.status,
            })
        if state.status == "running" and self._advance(state, "decide_node"):
            if state.error and state.error.get("retryable") and state.retry_count < state.max_retries and state.can_continue(datetime.now(timezone.utc)):
                state.retry_count += 1
                state.error = None
                state.next_action = "retry"
                _emit(state, "decide_node", "retry_scheduled", {"attempt": state.retry_count})
            elif state.error:
                _fail(state, str(state.error.get("error_code") or "graph_failed"))
            else:
                state.next_action = "persist" if self.commit or state.requested_mode in self.handlers else "continue"
                state.decision = {"next_action": state.next_action, "route": state.requested_mode}
        return {"payload": state.to_json_dict()}

    @staticmethod
    def _decision_edge(data: _GraphState) -> str:
        state = data["payload"]
        if state["status"] == "running" and state["next_action"] == "retry":
            return "retry"
        return "commit" if state["status"] == "running" and state["next_action"] == "persist" else "checkpoint"

    async def _commit(self, data: _GraphState) -> _GraphState:
        state = AgentState.model_validate(data["payload"])
        # Legacy business adapters may write facts, so invoke them only after
        # observation, validation and decision have authorized this commit.
        callback = self.commit or self.handlers.get(state.requested_mode)
        if callback and state.status in {"running", "waiting"} and self._advance(state, "commit_node"):
            try:
                state = await self._callback(state, callback, allow_commit=True)
                if state.status == "committed" or state.result.get("committed") is True:
                    _emit(state, "commit_node", "business_committed", {"route": state.requested_mode})
                elif state.status == "running":
                    state.status = "degraded"
            except llm.UsageBudgetError:
                _fail(state, "model_budget_exceeded", retryable=False)
            except TimeoutError:
                _fail(state, "deadline_exceeded", retryable=False)
            except Exception:
                _fail(state, "business_commit_failed", retryable=True)
        return {"payload": state.to_json_dict()}

    async def _checkpoint(self, data: _GraphState) -> _GraphState:
        state = AgentState.model_validate(data["payload"])
        if state.status == "running":
            state.status = "degraded"
            state.result.setdefault("status", "ready_for_adapter")
        state.current_node = "checkpoint_node"
        _emit(state, "checkpoint_node", "checkpoint_requested", {"thread_id": state.thread_id})
        return {"payload": state.to_json_dict()}

    async def _finish(self, data: _GraphState) -> _GraphState:
        state = AgentState.model_validate(data["payload"])
        _emit(state, "finish_node", "run_finished", {
            "status": state.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return {"payload": state.to_json_dict()}

    async def ainvoke(self, state: AgentState | dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        current = state if isinstance(state, AgentState) else AgentState.model_validate(state)
        graph_config = dict(config or {})
        graph_config.setdefault("configurable", {}).setdefault("thread_id", current.thread_id or current.request_id)
        output = await self.compiled.ainvoke({"payload": current.to_json_dict()}, config=graph_config)
        return self.finalize_output(output)

    def finalize_output(self, output: _GraphState) -> dict[str, Any]:
        """Acknowledge a saver write only after the compiled invoke returns."""

        state = AgentState.model_validate(output["payload"])
        if self.checkpointer is not None:
            saved = AgentEvent(
                run_id=state.run_id, node="checkpoint_node", event_type="checkpoint_saved",
                payload={"thread_id": state.thread_id}, trace_id=state.run_id,
            ).model_dump(mode="json")
            # run_finished stays the terminal trace event. This acknowledgement
            # is external to the saved graph snapshot, which is intentional.
            end = next((index for index, event in enumerate(state.events)
                        if event.get("event_type") == "run_finished"), len(state.events))
            state.events.insert(end, saved)
        return state.to_json_dict()


def build_application_graph(*, checkpointer: Any | None = None, max_steps: int = 32,
                            stages: Mapping[str, StageCallback] | None = None,
                            stage_sequences: Mapping[str, tuple[str, ...]] | None = None,
                            handlers: Mapping[str, StageCallback] | None = None,
                            commit: StageCallback | None = None,
                            first_commit: StageCallback | None = None,
                            first_commit_after: str | None = None) -> ApplicationGraph:
    """Compile the graph with optional business handlers and checkpoint saver."""
    return ApplicationGraph(checkpointer=checkpointer, max_steps=max_steps, stages=stages,
                            stage_sequences=stage_sequences, handlers=handlers, commit=commit,
                            first_commit=first_commit, first_commit_after=first_commit_after)


application_graph = build_application_graph()

__all__ = ["AgentState", "ApplicationGraph", "application_graph", "build_application_graph"]

