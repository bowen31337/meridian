"""Tests for OpenRouterProvider — SSE streaming, OTel span, audit log."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from opentelemetry.trace import StatusCode

from meridian_sdk_provider import ModelProvider
from meridian_sdk_provider.errors import (
    ProviderCallError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from meridian_sdk_provider.openrouter import OpenRouterProvider, _convert_message
from meridian_sdk_provider.types import (
    MessageStartEvent,
    MessageStopEvent,
    ModelCallOpts,
    ModelCountReq,
    TextDeltaEvent,
    ToolInputDeltaEvent,
    ToolUseStartEvent,
)

from .conftest import CollectingAuditLog

_API_KEY = "sk-or-test-key"


# ---------------------------------------------------------------------------
# OTel mock
# ---------------------------------------------------------------------------


class MockSpan:
    def __init__(self) -> None:
        self.name: str = ""
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.status: Any = None
        self.recorded_exceptions: list[BaseException] = []
        self.ended: bool = False

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, status: Any) -> None:
        self.status = status

    def record_exception(self, exc: BaseException, **_: Any) -> None:
        self.recorded_exceptions.append(exc)

    def __enter__(self) -> MockSpan:
        return self

    def __exit__(self, *_: Any) -> bool:
        self.ended = True
        return False


class MockTracer:
    def __init__(self) -> None:
        self.span = MockSpan()

    def start_as_current_span(
        self, name: str, *, attributes: dict[str, Any] | None = None, **_: Any
    ) -> MockSpan:
        self.span.name = name
        if attributes:
            self.span.attributes.update(attributes)
        return self.span


@pytest.fixture()
def mock_tracer(monkeypatch: pytest.MonkeyPatch) -> MockTracer:
    tracer = MockTracer()
    monkeypatch.setattr("meridian_sdk_provider.openrouter.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_span(mock_tracer: MockTracer) -> MockSpan:
    return mock_tracer.span


# ---------------------------------------------------------------------------
# HTTP transport helpers
# ---------------------------------------------------------------------------


class _AsyncTransport(httpx.AsyncBaseTransport):
    """Fake async transport that serves fixed responses by path."""

    def __init__(self, routes: dict[str, httpx.Response]) -> None:
        self._routes = routes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in self._routes:
            return self._routes[path]
        return httpx.Response(404, text="not found")


def _sse(*chunks: dict[str, Any]) -> bytes:
    """Build an SSE response body from a sequence of JSON chunks."""
    lines: list[str] = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines).encode()


def _text_stream(
    text: str = "Hello!",
    model: str = "openai/gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> bytes:
    return _sse(
        {
            "id": "chatcmpl-x",
            "model": model,
            "choices": [
                {"delta": {"role": "assistant", "content": ""}, "index": 0, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-x",
            "model": model,
            "choices": [{"delta": {"content": text}, "index": 0, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-x",
            "model": model,
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    )


def _tool_stream(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_id: str = "call_abc",
    model: str = "openai/gpt-4o",
) -> bytes:
    return _sse(
        {
            "model": model,
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": tool_id,
                                "type": "function",
                                "function": {"name": tool_name, "arguments": ""},
                            }
                        ],
                    },
                    "index": 0,
                    "finish_reason": None,
                }
            ],
        },
        {
            "model": model,
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": json.dumps(tool_args)}}
                        ]
                    },
                    "index": 0,
                    "finish_reason": None,
                }
            ],
        },
        {
            "model": model,
            "choices": [{"delta": {}, "index": 0, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


def _make_opts(**kwargs: Any) -> ModelCallOpts:
    defaults: dict[str, Any] = {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    return ModelCallOpts(**defaults)


_CHAT_PATH = "/api/v1/chat/completions"


def _make_provider(
    chat_body: bytes | None = None,
    chat_status: int = 200,
    audit_log: CollectingAuditLog | None = None,
) -> OpenRouterProvider:
    routes: dict[str, httpx.Response] = {}
    if chat_body is not None:
        routes[_CHAT_PATH] = httpx.Response(chat_status, content=chat_body)
    elif chat_status != 200:
        routes[_CHAT_PATH] = httpx.Response(chat_status, text="error body")
    transport = _AsyncTransport(routes)
    http_client = httpx.AsyncClient(transport=transport, base_url="https://openrouter.ai/api/v1")
    return OpenRouterProvider(
        _API_KEY, name="test-openrouter", audit_log=audit_log, _http=http_client
    )


def _sse_no_done(*chunks: dict[str, Any]) -> bytes:
    """SSE body without a trailing ``data: [DONE]`` terminator."""
    lines: list[str] = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")
    return "\n".join(lines).encode()


class TestConvertMessageBlocks:
    def test_unknown_block_type_skipped(self) -> None:
        msg = SimpleNamespace(
            role="user",
            content=[
                SimpleNamespace(type="text", text="hi"),
                SimpleNamespace(type="image", data="ignored"),
            ],
        )
        entry = _convert_message(msg)
        assert entry == {"role": "user", "content": "hi"}


class TestStreamWithoutDoneTerminator:
    async def test_completes_when_stream_exhausts(self, mock_span: MockSpan) -> None:
        # When the SSE stream ends without a [DONE] sentinel, the read loop must
        # exhaust naturally and still emit a final message_stop event.
        body = _sse_no_done(
            {
                "id": "chatcmpl-x",
                "model": "openai/gpt-4o",
                "choices": [{"delta": {"content": "hi"}, "index": 0, "finish_reason": "stop"}],
            }
        )
        provider = _make_provider(chat_body=body)
        events = [e async for e in provider.call(_make_opts())]
        assert any(isinstance(e, MessageStopEvent) for e in events)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_model_provider_protocol(self) -> None:
        assert isinstance(OpenRouterProvider(_API_KEY), ModelProvider)

    def test_kind(self) -> None:
        assert OpenRouterProvider(_API_KEY).kind == "openrouter"

    def test_name_default(self) -> None:
        assert OpenRouterProvider(_API_KEY).name == "openrouter"

    def test_name_custom(self) -> None:
        assert OpenRouterProvider(_API_KEY, name="my-router").name == "my-router"

    def test_capabilities_streaming(self) -> None:
        caps = OpenRouterProvider(_API_KEY).capabilities
        assert caps.streaming is True

    def test_capabilities_cache_control(self) -> None:
        caps = OpenRouterProvider(_API_KEY).capabilities
        assert caps.cache_control is True

    def test_capabilities_thinking_false(self) -> None:
        assert OpenRouterProvider(_API_KEY).capabilities.thinking is False

    def test_capabilities_count_tokens_false(self) -> None:
        assert OpenRouterProvider(_API_KEY).capabilities.count_tokens is False


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_returns_model_entries(self) -> None:
        payload = {
            "data": [
                {
                    "id": "openai/gpt-4o",
                    "context_length": 128000,
                    "architecture": {"input_modalities": ["text", "image"]},
                },
                {
                    "id": "anthropic/claude-3-5-sonnet",
                    "context_length": 200000,
                    "architecture": {"input_modalities": ["text"]},
                },
            ]
        }
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OpenRouterProvider(_API_KEY, name="test-router").list_models()

        assert len(models) == 2
        assert models[0].model == "openai/gpt-4o"
        assert models[0].provider == "test-router"
        assert models[0].context_window == 128000

    def test_vision_flag_set_from_input_modalities(self) -> None:
        payload = {
            "data": [
                {
                    "id": "openai/gpt-4o",
                    "context_length": 128000,
                    "architecture": {"input_modalities": ["text", "image"]},
                },
                {
                    "id": "meta-llama/llama-3",
                    "context_length": 8192,
                    "architecture": {"input_modalities": ["text"]},
                },
            ]
        }
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OpenRouterProvider(_API_KEY).list_models()

        gpt4o = next(m for m in models if m.model == "openai/gpt-4o")
        llama = next(m for m in models if "llama" in m.model)
        assert gpt4o.capabilities.vision is True
        assert llama.capabilities.vision is False

    def test_cache_flag_set_for_anthropic_models(self) -> None:
        payload = {
            "data": [
                {
                    "id": "anthropic/claude-3-5-sonnet",
                    "context_length": 200000,
                    "architecture": {"input_modalities": ["text"]},
                },
                {
                    "id": "openai/gpt-4o",
                    "context_length": 128000,
                    "architecture": {"input_modalities": ["text"]},
                },
            ]
        }
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OpenRouterProvider(_API_KEY).list_models()

        claude = next(m for m in models if "anthropic" in m.model)
        gpt = next(m for m in models if "openai" in m.model)
        assert claude.capabilities.cache is True
        assert gpt.capabilities.cache is False

    def test_thinking_flag_set_for_thinking_models(self) -> None:
        payload = {
            "data": [
                {
                    "id": "anthropic/claude-3-7-sonnet:thinking",
                    "context_length": 200000,
                    "architecture": {"input_modalities": ["text"]},
                },
            ]
        }
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OpenRouterProvider(_API_KEY).list_models()

        assert models[0].capabilities.thinking is True

    def test_skips_entries_with_no_id(self) -> None:
        payload = {
            "data": [
                {"context_length": 128000, "architecture": {}},
                {"id": "openai/gpt-4o", "context_length": 128000, "architecture": {}},
            ]
        }
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OpenRouterProvider(_API_KEY).list_models()
        assert len(models) == 1

    def test_returns_empty_on_connection_error(self) -> None:
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            assert OpenRouterProvider(_API_KEY).list_models() == []

    def test_returns_empty_on_http_error(self) -> None:
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(401, text="unauthorized")
            assert OpenRouterProvider(_API_KEY).list_models() == []

    def test_uses_default_context_window_when_missing(self) -> None:
        payload = {"data": [{"id": "some/model", "architecture": {}}]}
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OpenRouterProvider(_API_KEY).list_models()
        assert models[0].context_window == 128000


# ---------------------------------------------------------------------------
# call() — streaming events
# ---------------------------------------------------------------------------


class TestCallStreamingEvents:
    async def test_yields_message_start(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream("Hi", model="openai/gpt-4o"))
        events = [e async for e in provider.call(_make_opts())]
        assert isinstance(events[0], MessageStartEvent)
        assert events[0].model == "openai/gpt-4o"
        assert events[0].provider == "test-openrouter"

    async def test_yields_text_delta(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream("Hello world"))
        events = [e async for e in provider.call(_make_opts())]
        text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "Hello world"

    async def test_yields_message_stop_with_tokens(self, mock_span: MockSpan) -> None:
        provider = _make_provider(
            chat_body=_text_stream("Hi", prompt_tokens=7, completion_tokens=3)
        )
        events = [e async for e in provider.call(_make_opts())]
        stop = next(e for e in events if isinstance(e, MessageStopEvent))
        assert stop.input_tokens == 7
        assert stop.output_tokens == 3
        assert stop.stop_reason == "stop"

    async def test_empty_content_chunks_not_yielded(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream(""))
        events = [e async for e in provider.call(_make_opts())]
        assert not any(isinstance(e, TextDeltaEvent) for e in events)

    async def test_multiple_text_chunks(self, mock_span: MockSpan) -> None:
        body = _sse(
            {
                "model": "openai/gpt-4o",
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": ""},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            },
            {
                "model": "openai/gpt-4o",
                "choices": [{"delta": {"content": "A"}, "index": 0, "finish_reason": None}],
            },
            {
                "model": "openai/gpt-4o",
                "choices": [{"delta": {"content": "B"}, "index": 0, "finish_reason": None}],
            },
            {
                "model": "openai/gpt-4o",
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )
        provider = _make_provider(chat_body=body)
        events = [e async for e in provider.call(_make_opts())]
        texts = [e.text for e in events if isinstance(e, TextDeltaEvent)]
        assert texts == ["A", "B"]

    async def test_system_prompt_included(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        class CapturingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                captured.append(json.loads(request.content))
                return httpx.Response(200, content=_text_stream())

        http_client = httpx.AsyncClient(
            transport=CapturingTransport(), base_url="https://openrouter.ai/api/v1"
        )
        provider = OpenRouterProvider(_API_KEY, name="test-openrouter", _http=http_client)
        [e async for e in provider.call(_make_opts(system="Be concise."))]

        messages = captured[0]["messages"]
        assert messages[0] == {"role": "system", "content": "Be concise."}

    async def test_temperature_forwarded(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        class CapturingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                captured.append(json.loads(request.content))
                return httpx.Response(200, content=_text_stream())

        http_client = httpx.AsyncClient(
            transport=CapturingTransport(), base_url="https://openrouter.ai/api/v1"
        )
        provider = OpenRouterProvider(_API_KEY, name="test-openrouter", _http=http_client)
        [e async for e in provider.call(_make_opts(temperature=0.3))]
        assert captured[0]["temperature"] == 0.3

    async def test_max_tokens_forwarded(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        class CapturingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                captured.append(json.loads(request.content))
                return httpx.Response(200, content=_text_stream())

        http_client = httpx.AsyncClient(
            transport=CapturingTransport(), base_url="https://openrouter.ai/api/v1"
        )
        provider = OpenRouterProvider(_API_KEY, name="test-openrouter", _http=http_client)
        [e async for e in provider.call(_make_opts(max_tokens=512))]
        assert captured[0]["max_tokens"] == 512


# ---------------------------------------------------------------------------
# call() — tool call events
# ---------------------------------------------------------------------------


class TestCallToolEvents:
    async def test_yields_tool_use_start_and_delta(self, mock_span: MockSpan) -> None:
        provider = _make_provider(
            chat_body=_tool_stream("get_weather", {"city": "NYC"}, tool_id="call_xyz")
        )
        events = [e async for e in provider.call(_make_opts())]

        tool_start = next(e for e in events if isinstance(e, ToolUseStartEvent))
        tool_delta = next(e for e in events if isinstance(e, ToolInputDeltaEvent))

        assert tool_start.name == "get_weather"
        assert tool_start.id == "call_xyz"
        assert tool_delta.id == "call_xyz"
        assert json.loads(tool_delta.partial_json) == {"city": "NYC"}

    async def test_tool_stop_reason_is_tool_calls(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_tool_stream("fn", {}, tool_id="call_1"))
        events = [e async for e in provider.call(_make_opts())]
        stop = next(e for e in events if isinstance(e, MessageStopEvent))
        assert stop.stop_reason == "tool_calls"

    async def test_multiple_tool_calls_indexed(self, mock_span: MockSpan) -> None:
        body = _sse(
            {
                "model": "openai/gpt-4o",
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "type": "function",
                                    "function": {"name": "fn_a", "arguments": ""},
                                },
                                {
                                    "index": 1,
                                    "id": "call_b",
                                    "type": "function",
                                    "function": {"name": "fn_b", "arguments": ""},
                                },
                            ],
                        },
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            },
            {
                "model": "openai/gpt-4o",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": "{}"}},
                                {"index": 1, "function": {"arguments": '{"x":1}'}},
                            ]
                        },
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            },
            {
                "model": "openai/gpt-4o",
                "choices": [{"delta": {}, "index": 0, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )
        provider = _make_provider(chat_body=body)
        events = [e async for e in provider.call(_make_opts())]
        starts = [e for e in events if isinstance(e, ToolUseStartEvent)]
        assert [s.id for s in starts] == ["call_a", "call_b"]
        assert [s.name for s in starts] == ["fn_a", "fn_b"]


# ---------------------------------------------------------------------------
# call() — error handling
# ---------------------------------------------------------------------------


class TestCallErrors:
    async def test_429_raises_rate_limit_error(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_status=429)
        with pytest.raises(ProviderRateLimitError) as exc_info:
            [e async for e in provider.call(_make_opts())]
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider_name == "test-openrouter"

    async def test_500_raises_server_error(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_status=500)
        with pytest.raises(ProviderServerError) as exc_info:
            [e async for e in provider.call(_make_opts())]
        assert exc_info.value.status_code == 500

    async def test_401_raises_provider_call_error(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_status=401)
        with pytest.raises(ProviderCallError):
            [e async for e in provider.call(_make_opts())]

    async def test_timeout_raises_timeout_error(self, mock_span: MockSpan) -> None:
        class _TimeoutTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                raise httpx.TimeoutException("timed out")

        http = httpx.AsyncClient(
            transport=_TimeoutTransport(), base_url="https://openrouter.ai/api/v1"
        )
        provider = OpenRouterProvider(_API_KEY, name="test-openrouter", _http=http)
        with pytest.raises(ProviderTimeoutError):
            [e async for e in provider.call(_make_opts())]

    async def test_connect_error_raises_provider_call_error(self, mock_span: MockSpan) -> None:
        class _ConnectTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("connection refused")

        http = httpx.AsyncClient(
            transport=_ConnectTransport(), base_url="https://openrouter.ai/api/v1"
        )
        provider = OpenRouterProvider(_API_KEY, name="test-openrouter", _http=http)
        with pytest.raises(ProviderCallError):
            [e async for e in provider.call(_make_opts())]


# ---------------------------------------------------------------------------
# call() — audit log
# ---------------------------------------------------------------------------


class TestCallAuditLog:
    async def test_audit_written_on_http_error(self, mock_span: MockSpan) -> None:
        audit = CollectingAuditLog()
        provider = _make_provider(chat_status=500, audit_log=audit)
        with pytest.raises(ProviderServerError):
            [e async for e in provider.call(_make_opts(model="openai/gpt-4o", session_id="s1"))]

        assert len(audit.entries) == 1
        entry = audit.entries[0]
        assert entry.event == "openrouter.call.failed"
        assert entry.level == "error"
        assert entry.provider_name == "test-openrouter"
        assert entry.provider_kind == "openrouter"
        assert entry.model == "openai/gpt-4o"
        assert entry.session_id == "s1"
        assert "error_type" in entry.detail

    async def test_no_audit_on_success(self, mock_span: MockSpan) -> None:
        audit = CollectingAuditLog()
        provider = _make_provider(chat_body=_text_stream(), audit_log=audit)
        [e async for e in provider.call(_make_opts())]
        assert audit.entries == []

    async def test_audit_written_on_timeout(self, mock_span: MockSpan) -> None:
        class _TimeoutTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                raise httpx.TimeoutException("timed out")

        audit = CollectingAuditLog()
        http = httpx.AsyncClient(
            transport=_TimeoutTransport(), base_url="https://openrouter.ai/api/v1"
        )
        provider = OpenRouterProvider(_API_KEY, name="test-openrouter", audit_log=audit, _http=http)
        with pytest.raises(ProviderTimeoutError):
            [e async for e in provider.call(_make_opts())]

        assert len(audit.entries) == 1
        assert audit.entries[0].event == "openrouter.call.failed"


# ---------------------------------------------------------------------------
# call() — OTel span
# ---------------------------------------------------------------------------


class TestCallOTelSpan:
    async def test_span_name(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream())
        [e async for e in provider.call(_make_opts())]
        assert mock_span.name == "openrouter.model.call"

    async def test_span_attributes_provider_name(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream())
        [e async for e in provider.call(_make_opts())]
        assert mock_span.attributes["provider.name"] == "test-openrouter"

    async def test_span_attributes_model(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream())
        [e async for e in provider.call(_make_opts(model="anthropic/claude-3-5-sonnet"))]
        assert mock_span.attributes["model"] == "anthropic/claude-3-5-sonnet"

    async def test_invocation_event_attached(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream())
        [e async for e in provider.call(_make_opts())]
        event_names = [e[0] for e in mock_span.events]
        assert "provider.invocation" in event_names

    async def test_invocation_event_provider_kind(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream())
        [e async for e in provider.call(_make_opts())]
        inv = next(e for e in mock_span.events if e[0] == "provider.invocation")
        assert inv[1]["provider.kind"] == "openrouter"

    async def test_invocation_event_session_id(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream())
        [e async for e in provider.call(_make_opts(session_id="sess-xyz"))]
        inv = next(e for e in mock_span.events if e[0] == "provider.invocation")
        assert inv[1]["session.id"] == "sess-xyz"

    async def test_span_ended_on_success(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream())
        [e async for e in provider.call(_make_opts())]
        assert mock_span.ended

    async def test_span_marked_error_on_failure(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_status=500)
        with pytest.raises(ProviderServerError):
            [e async for e in provider.call(_make_opts())]
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_error_event_attached_on_failure(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_status=429)
        with pytest.raises(ProviderRateLimitError):
            [e async for e in provider.call(_make_opts())]
        error_events = [e for e in mock_span.events if e[0] == "provider.error"]
        assert len(error_events) == 1
        assert error_events[0][1]["provider.name"] == "test-openrouter"

    async def test_exception_recorded_on_failure(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_status=500)
        with pytest.raises(ProviderServerError):
            [e async for e in provider.call(_make_opts())]
        assert len(mock_span.recorded_exceptions) == 1


# ---------------------------------------------------------------------------
# count_tokens / close
# ---------------------------------------------------------------------------


class TestCountTokensAndClose:
    async def test_count_tokens_raises_not_implemented(self) -> None:
        provider = OpenRouterProvider(_API_KEY)
        with pytest.raises(NotImplementedError):
            await provider.count_tokens(ModelCountReq(model="openai/gpt-4o", messages=[]))

    async def test_close_does_not_raise(self) -> None:
        provider = OpenRouterProvider(_API_KEY)
        await provider.close()
