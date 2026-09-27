"""创建面试的生成流要把模型思考原文推出去，题目结果不能混进思考区。"""

from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import patch

from app.agents.roles.loop import _RoleModel
from app.api.interviews.interview import _generate_events


def _events(raw: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        line = block.strip()
        if not line.startswith("data:"):
            continue
        found.append(json.loads(line[5:].strip()))
    return found


class _Session:
    """生成流会关掉请求上的会话，再自己开一条。测试不需要真数据库。"""

    def close(self) -> None:
        return None


def _own_session() -> _Session:
    # SessionLocal() 直接返回会话，不是上下文管理器。
    return _Session()


class GenerateStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_thought_text_is_streamed_and_questions_stay_out(self) -> None:
        interview = {"id": "iv-1", "title": "后端 · 全真模拟面试", "status": "ready"}

        async def fake_prepare(_db, _content, on_thought, *, run_id=None):
            # The stream and generated interview share one durable request ID.
            assert run_id is None
            await on_thought("先看岗位要考察缓存。")
            await on_thought("再补一题故障排查。")
            return interview

        with patch("app.api.interviews.interview.SessionLocal", _own_session), patch(
            "app.services.prepare_interview", new=fake_prepare
        ):
            raw = "".join([event async for event in _generate_events(_Session(), "后端工程师，负责缓存与一致性")])

        events = _events(raw)
        if not any(item["type"] == "done" for item in events):
            self.fail(json.dumps(events, ensure_ascii=False))
        # 创建页不再接收思考原文。题干和工具结构都不能出现在这条流里。
        thoughts = [item for item in events if item["type"] == "thought"]
        self.assertEqual(thoughts, [])
        self.assertNotIn("先看岗位要考察缓存", raw)
        done = [item for item in events if item["type"] == "done"]
        self.assertEqual(done[0]["interview"]["id"], "iv-1")

    async def test_prepare_failure_still_emits_an_error_event(self) -> None:
        """出题任务自己报错时，流不能在思考结束后静默断开。"""

        async def broken_prepare(_db, _content, on_thought):
            await on_thought("先看岗位。")
            raise RuntimeError("name 'SessionLocal' is not defined")

        with patch("app.api.interviews.interview.SessionLocal", _own_session), patch(
            "app.services.prepare_interview", new=broken_prepare
        ):
            raw = "".join([event async for event in _generate_events(_Session(), "后端工程师，负责缓存与一致性")])

        events = _events(raw)
        self.assertTrue(any(item["type"] == "error" for item in events))
        self.assertFalse(any(item["type"] == "done" for item in events))

    async def test_prepare_failure_does_not_escape_before_the_error_event(self) -> None:
        """生成任务在第一段思考前失败时，异常要留在错误事件里，不能先打进事件循环。"""

        async def broken_prepare(_db, _content, _on_thought):
            raise RuntimeError("boom")

        with patch("app.api.interviews.interview.SessionLocal", _own_session), patch(
            "app.services.prepare_interview", new=broken_prepare
        ):
            raw = "".join([event async for event in _generate_events(_Session(), "后端工程师，负责缓存与一致性")])

        events = _events(raw)
        self.assertEqual([item["type"] for item in events], ["error"])

    async def test_role_model_forwards_thinking_without_the_question_json(self) -> None:
        seen: list[str] = []

        async def on_thought(text: str) -> None:
            seen.append(text)

        async def fake_stream(*_args, **_kwargs):
            yield "thinking", "先看缓存一致性。"
            yield "content", '{"tool":"finish","arguments":{"questions":[{"stem":"为什么要互斥"}]}}'

        model = _RoleModel(role="author")
        model._session = object()
        model._on_thought = on_thought
        bound = model.bind_tools([])
        with patch("app.services.operations.llm_gateway.stream_parts", new=fake_stream):
            message = await bound._complete([])

        self.assertEqual(seen, ["先看缓存一致性。"])
        self.assertEqual(message.tool_calls[0]["name"], "finish")
        self.assertIn("questions", message.tool_calls[0]["args"])
        self.assertNotIn("questions", "".join(seen))

    async def test_handoff_thinking_is_not_dropped(self) -> None:
        """调度员选角色时的思考也要给创建页。以前这一步被整段调用吃掉。"""
        seen: list[str] = []

        async def on_thought(text: str) -> None:
            seen.append(text)

        async def fake_stream(*_args, **_kwargs):
            yield "thinking", "这是出题，交给作者。"
            yield "content", "transfer_to_author"

        class _Tool:
            name = "transfer_to_author"
            description = "交给作者"

        model = _RoleModel(role="analyst")
        model._session = object()
        model._on_thought = on_thought
        bound = model.bind_tools([_Tool()])
        with patch("app.services.operations.llm_gateway.stream_parts", new=fake_stream):
            message = await bound._complete([{"role": "user", "content": "出一套后端题"}])

        self.assertEqual(seen, ["这是出题，交给作者。"])
        self.assertEqual(message.tool_calls[0]["name"], "transfer_to_author")
