"""Chat calls stay on OpenAI Chat Completions with thinking and streaming fixed on."""

from __future__ import annotations

import unittest
from typing import Any

from app.integrations import llm
from app.api.chat.session import _sse
from app.agents.tools.registry import tool_schemas_for
from app.services.operations.providers import _chat_protocol


class _Lines:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __aiter__(self):
        return self._walk()

    async def _walk(self):
        for line in self._lines:
            yield line


class _StreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status = 200

    def raise_for_status(self) -> None:
        return None

    def aiter_lines(self) -> _Lines:
        return _Lines(self._lines)


class _StreamContext:
    def __init__(self, response: _StreamResponse) -> None:
        self.response = response

    async def __aenter__(self) -> _StreamResponse:
        return self.response

    async def __aexit__(self, *_args: object) -> None:
        return None


class _Client:
    def __init__(self, response: _StreamResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def stream(self, method: str, url: str, **kwargs: Any) -> _StreamContext:
        self.calls.append({"method": method, "url": url, **kwargs})
        return _StreamContext(self.response)

    async def __aenter__(self) -> "_Client":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class _JsonResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _JsonClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_JsonClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> _JsonResponse:
        snapshot = {"url": url, **kwargs}
        if isinstance(snapshot.get("json"), dict):
            snapshot["json"] = dict(snapshot["json"])
        self.calls.append(snapshot)
        return _JsonResponse(self.payload)


class _JsonSequenceClient:
    """Return deterministic non-streaming responses for retry-path tests."""

    def __init__(self, payloads: list[dict[str, Any]], statuses: list[int] | None = None) -> None:
        self.payloads = payloads
        self.statuses = statuses or [200] * len(payloads)
        self.calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_JsonSequenceClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> _JsonResponse:
        snapshot = {"url": url, **kwargs}
        if isinstance(snapshot.get("json"), dict):
            snapshot["json"] = dict(snapshot["json"])
        self.calls.append(snapshot)
        return _JsonResponse(self.payloads.pop(0), self.statuses.pop(0))


class ChatProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_json_uses_non_streaming_json_request_without_thinking(self) -> None:
        """Structured grading must not spend its output budget on streamed reasoning."""
        client = _JsonClient({"choices": [{"message": {"content": '{"review":"ok"}'}}]})
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            result = await llm.complete_json(
                "系统",
                "返回 review JSON",
                api_key="key",
                base_url="https://example.test/v1",
                model="qwen",
            )
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]

        self.assertEqual(result, {"review": "ok"})
        body = client.calls[0]["json"]
        self.assertFalse(body["stream"])
        self.assertFalse(body["enable_thinking"])
        self.assertEqual(body["response_format"], {"type": "json_object"})

    async def test_complete_json_retries_once_after_malformed_output(self) -> None:
        """A transient delimiter error gets one strict low-temperature retry."""
        client = _JsonSequenceClient(
            [
                {"choices": [{"message": {"content": '{"review":"truncated'}}]},
                {"choices": [{"message": {"content": '{"review":"complete"}'}}]},
            ]
        )
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            result = await llm.complete_json(
                "系统",
                "返回 review JSON",
                api_key="key",
                base_url="https://example.test/v1",
                model="qwen",
            )
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]

        self.assertEqual(result, {"review": "complete"})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1]["json"]["temperature"], 0.0)
        self.assertNotIn("response_format", client.calls[1]["json"])

    async def test_complete_json_retries_when_provider_returns_no_content(self) -> None:
        """A length-truncated empty response must use the same bounded retry."""
        client = _JsonSequenceClient(
            [
                {"choices": [{"finish_reason": "length", "message": {"content": ""}}]},
                {"choices": [{"finish_reason": "stop", "message": {"content": '{"review":"ok"}'}}]},
            ]
        )
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            result = await llm.complete_json(
                "系统",
                "返回 review JSON",
                api_key="key",
                base_url="https://example.test/v1",
                model="qwen",
            )
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]

        self.assertEqual(result, {"review": "ok"})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1]["json"]["temperature"], 0.0)

    async def test_complete_json_drops_unsupported_response_format(self) -> None:
        """Compatible endpoints may reject response_format while accepting JSON prompts."""
        payload = {"choices": [{"message": {"content": '{"review":"ok"}'}}]}
        client = _JsonSequenceClient([payload, payload], statuses=[400, 200])
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            result = await llm.complete_json(
                "系统",
                "返回 review JSON",
                api_key="key",
                base_url="https://example.test/v1",
                model="qwen",
            )
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]

        self.assertEqual(result, {"review": "ok"})
        self.assertIn("response_format", client.calls[0]["json"])
        self.assertNotIn("response_format", client.calls[1]["json"])

    def test_role_tool_schema_is_allow_listed_and_strict(self) -> None:
        schemas = tool_schemas_for(("validate_question", "missing"))
        self.assertEqual([item["function"]["name"] for item in schemas], ["validate_question"])
        parameters = schemas[0]["function"]["parameters"]
        self.assertEqual(parameters["required"], ["stem"])
        self.assertFalse(parameters["additionalProperties"])

    def test_finish_schema_matches_author_and_critic_outputs(self) -> None:
        author = next(item for item in tool_schemas_for(("finish",), role="author")
                      if item["function"]["name"] == "finish")
        critic = next(item for item in tool_schemas_for(("finish",), role="critic")
                      if item["function"]["name"] == "finish")
        author_params = author["function"]["parameters"]
        critic_params = critic["function"]["parameters"]
        self.assertIn("questions", author_params["required"])
        self.assertIn("job_title", author_params["properties"])
        self.assertIn("pass", critic_params["required"])
        self.assertIn("reason", critic_params["properties"])
        self.assertNotIn("questions", critic_params["properties"])

    async def test_complete_tool_call_sends_standard_tools(self) -> None:
        original = llm.httpx.AsyncClient
        json_client = _JsonClient({
            "choices": [{"message": {"tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "finish", "arguments": '{"ok":true}'},
            }]}}]
        })
        llm.httpx.AsyncClient = lambda **_kwargs: json_client  # type: ignore[assignment]
        try:
            result = await llm.complete_tool_call(
                "系统", "用户", tools=[{
                    "type": "function",
                    "function": {"name": "finish", "description": "结束", "parameters": {"type": "object"}},
                }], api_key="key", base_url="https://example.test/v1", model="qwen",
            )
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]

        self.assertEqual(result, {"id": "call-1", "name": "finish", "arguments": {"ok": True}})
        body = json_client.calls[0]["json"]
        self.assertEqual(body["tools"][0]["function"]["name"], "finish")
        self.assertEqual(body["tool_choice"], "auto")
        self.assertFalse(body["stream"])

    async def test_chat_always_streams_with_thinking(self) -> None:
        response = _StreamResponse(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"先看岗位"}}]}',
                'data: {"choices":[{"delta":{"content":"回答"}}]}',
                'data: {"choices":[{"delta":{"content":"正文"}}]}',
                "data: [DONE]",
            ]
        )
        client = _Client(response)
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            text = await llm.complete_text(
                "系统",
                "用户",
                api_key="key",
                base_url="https://example.test/v1",
                model="qwen3.8-max",
                protocol="anthropic_messages",
            )
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]

        self.assertEqual(text, "回答正文")
        body = client.calls[0]["json"]
        self.assertTrue(body["stream"])
        self.assertTrue(body["enable_thinking"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertTrue(client.calls[0]["url"].endswith("/chat/completions"))

    async def test_stream_parts_keep_reasoning_out_of_the_answer(self) -> None:
        response = _StreamResponse(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"先看岗位"}}]}',
                'data: {"choices":[{"delta":{"content":"回答"}}]}',
                'data: {"choices":[{"delta":{"content":"正文"}}]}',
                "data: [DONE]",
            ]
        )
        client = _Client(response)
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            parts = [
                item
                async for item in llm.stream_chat_parts(
                    "系统",
                    "用户",
                    api_key="key",
                    base_url="https://example.test/v1",
                    model="qwen3.8-max",
                )
            ]
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]

        self.assertEqual(parts, [("reasoning", "先看岗位"), ("content", "回答"), ("content", "正文")])
        answer = [text for kind, text in parts if kind == "content"]
        self.assertEqual("".join(answer), "回答正文")
        body = client.calls[0]["json"]
        self.assertTrue(body["stream"])
        self.assertTrue(body["enable_thinking"])

    async def test_stream_parts_accept_thinking_alias(self) -> None:
        response = _StreamResponse(['data: {"choices":[{"delta":{"thinking":"先拆岗位"}}]}', "data: [DONE]"])
        client = _Client(response)
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            parts = [item async for item in llm.stream_chat_parts("系统", "用户", api_key="key", base_url="https://example.test/v1", model="qwen")]
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]
        self.assertEqual(parts, [("thinking", "先拆岗位")])

    async def test_stream_parts_keep_thinking_and_reasoning_apart(self) -> None:
        # 同一块里两个字段都要保留。以前用 or 只会交出先读到的那一个。
        response = _StreamResponse(
            [
                'data: {"choices":[{"delta":{"thinking":"先看岗位","reasoning_content":"岗位要求分布式"}}]}',
                "data: [DONE]",
            ]
        )
        client = _Client(response)
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            parts = [item async for item in llm.stream_chat_parts("系统", "用户", api_key="key", base_url="https://example.test/v1", model="qwen")]
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]
        self.assertEqual(parts, [("thinking", "先看岗位"), ("reasoning", "岗位要求分布式")])

    async def test_stream_parts_accept_reasoning_details(self) -> None:
        """有的兼容接口把思考放在 reasoning_details，不放 reasoning_content。"""
        response = _StreamResponse(
            [
                'data: {"choices":[{"delta":{"reasoning_details":[{"text":"先对照岗位职责"}]}}]}',
                "data: [DONE]",
            ]
        )
        client = _Client(response)
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            parts = [item async for item in llm.stream_chat_parts("系统", "用户", api_key="key", base_url="https://example.test/v1", model="qwen")]
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]
        self.assertEqual(parts, [("reasoning", "先对照岗位职责")])

    def test_sse_frame_keeps_chinese_text(self) -> None:
        frame = _sse({"type": "delta", "text": "回答"})
        self.assertTrue(frame.startswith("data: "))
        self.assertIn("回答", frame)
        self.assertTrue(frame.endswith("\n\n"))

    def test_text_provider_cannot_keep_another_protocol(self) -> None:
        self.assertEqual(_chat_protocol("anthropic_messages"), "openai_chat")
        self.assertEqual(_chat_protocol(""), "openai_chat")
        self.assertEqual(_chat_protocol("websocket_audio", "asr"), "websocket_audio")


if __name__ == "__main__":
    unittest.main()
