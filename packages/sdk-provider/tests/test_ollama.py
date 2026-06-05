"""Tests for OllamaProvider — HTTP client, streaming, OTel span, audit log."""

from __future__ import annotations

import json
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
from meridian_sdk_provider.ollama import OllamaProvider
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

# ---------------------------------------------------------------------------
# OTel mock (same pattern as test_fake.py)
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
    monkeypatch.setattr("meridian_sdk_provider.ollama.get_tracer", lambda: tracer)
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


def _ndjson(*chunks: dict[str, Any]) -> bytes:
    return ("\n".join(json.dumps(c) for c in chunks) + "\n").encode()


def _text_stream(
    text: str = "Hello!", model: str = "llama3.2", prompt_tokens: int = 10, eval_tokens: int = 5
) -> bytes:
    return _ndjson(
        {"model": model, "message": {"role": "assistant", "content": text}, "done": False},
        {
            "model": model,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": prompt_tokens,
            "eval_count": eval_tokens,
        },
    )


def _make_opts(**kwargs: Any) -> ModelCallOpts:
    defaults: dict[str, Any] = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    return ModelCallOpts(**defaults)


def _make_provider(
    chat_body: bytes | None = None,
    chat_status: int = 200,
    audit_log: CollectingAuditLog | None = None,
) -> OllamaProvider:
    routes: dict[str, httpx.Response] = {}
    if chat_body is not None:
        routes["/api/chat"] = httpx.Response(chat_status, content=chat_body)
    elif chat_status != 200:
        routes["/api/chat"] = httpx.Response(chat_status, text="error body")
    transport = _AsyncTransport(routes)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://localhost:11434")
    return OllamaProvider(name="test-ollama", audit_log=audit_log, _http=http_client)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_model_provider_protocol(self) -> None:
        assert isinstance(OllamaProvider(), ModelProvider)

    def test_kind(self) -> None:
        assert OllamaProvider().kind == "ollama"

    def test_name_default(self) -> None:
        assert OllamaProvider().name == "ollama"

    def test_name_custom(self) -> None:
        assert OllamaProvider(name="local").name == "local"

    def test_capabilities_streaming(self) -> None:
        caps = OllamaProvider().capabilities
        assert caps.streaming is True
        assert caps.thinking is False
        assert caps.cache_control is False
        assert caps.count_tokens is False


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_returns_model_entries(self) -> None:
        payload = {
            "models": [
                {"name": "llama3.2", "size": 1000},
                {"name": "phi4", "size": 2000},
            ]
        }
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OllamaProvider(name="test-ollama").list_models()

        assert len(models) == 2
        assert models[0].model == "llama3.2"
        assert models[0].provider == "test-ollama"
        assert models[0].context_window == 131072
        assert models[0].capabilities.streaming is True
        assert models[1].model == "phi4"

    def test_skips_entries_with_no_name(self) -> None:
        payload = {"models": [{"size": 1000}, {"name": "llama3.2"}]}
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OllamaProvider().list_models()
        assert len(models) == 1

    def test_returns_empty_on_connection_error(self) -> None:
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            assert OllamaProvider().list_models() == []

    def test_returns_empty_on_http_error(self) -> None:
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(500, text="error")
            assert OllamaProvider().list_models() == []

    def test_uses_model_field_fallback(self) -> None:
        payload = {"models": [{"model": "mistral:7b"}]}
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OllamaProvider().list_models()
        assert models[0].model == "mistral:7b"


# ---------------------------------------------------------------------------
# call() — streaming events
# ---------------------------------------------------------------------------


class TestCallStreamingEvents:
    async def test_yields_message_start(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream("Hi"))
        events = [e async for e in provider.call(_make_opts())]
        assert isinstance(events[0], MessageStartEvent)
        assert events[0].model == "llama3.2"
        assert events[0].provider == "test-ollama"

    async def test_yields_text_delta(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream("Hello world"))
        events = [e async for e in provider.call(_make_opts())]
        text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "Hello world"

    async def test_yields_message_stop_with_tokens(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream("Hi", prompt_tokens=7, eval_tokens=3))
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
        body = _ndjson(
            {"model": "llama3.2", "message": {"role": "assistant", "content": "A"}, "done": False},
            {"model": "llama3.2", "message": {"role": "assistant", "content": "B"}, "done": False},
            {
                "model": "llama3.2",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
            },
        )
        provider = _make_provider(chat_body=body)
        events = [e async for e in provider.call(_make_opts())]
        texts = [e.text for e in events if isinstance(e, TextDeltaEvent)]
        assert texts == ["A", "B"]

    async def test_system_prompt_included(self, mock_span: MockSpan) -> None:
        """Verify the system prompt is forwarded (via request body inspection)."""
        captured: list[dict[str, Any]] = []

        class CapturingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                captured.append(json.loads(request.content))
                return httpx.Response(200, content=_text_stream())

        http_client = httpx.AsyncClient(
            transport=CapturingTransport(), base_url="http://localhost:11434"
        )
        provider = OllamaProvider(name="test-ollama", _http=http_client)
        opts = _make_opts(system="Be concise.")
        [e async for e in provider.call(opts)]

        messages = captured[0]["messages"]
        assert messages[0] == {"role": "system", "content": "Be concise."}

    async def test_temperature_forwarded(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        class CapturingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                captured.append(json.loads(request.content))
                return httpx.Response(200, content=_text_stream())

        http_client = httpx.AsyncClient(
            transport=CapturingTransport(), base_url="http://localhost:11434"
        )
        provider = OllamaProvider(name="test-ollama", _http=http_client)
        [e async for e in provider.call(_make_opts(temperature=0.7))]
        assert captured[0]["options"]["temperature"] == 0.7


# ---------------------------------------------------------------------------
# call() — tool call events
# ---------------------------------------------------------------------------


class TestCallToolEvents:
    async def test_yields_tool_use_start_and_delta(self, mock_span: MockSpan) -> None:
        body = _ndjson(
            {
                "model": "llama3.2",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}
                    ],
                },
                "done": False,
            },
            {
                "model": "llama3.2",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
            },
        )
        provider = _make_provider(chat_body=body)
        events = [e async for e in provider.call(_make_opts())]

        tool_start = next(e for e in events if isinstance(e, ToolUseStartEvent))
        tool_delta = next(e for e in events if isinstance(e, ToolInputDeltaEvent))

        assert tool_start.name == "get_weather"
        assert tool_start.id == "call_0"
        assert tool_delta.id == "call_0"
        assert json.loads(tool_delta.partial_json) == {"city": "NYC"}

    async def test_multiple_tool_calls_indexed(self, mock_span: MockSpan) -> None:
        body = _ndjson(
            {
                "model": "llama3.2",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "fn_a", "arguments": {}}},
                        {"function": {"name": "fn_b", "arguments": {"x": 1}}},
                    ],
                },
                "done": True,
                "done_reason": "stop",
            },
        )
        provider = _make_provider(chat_body=body)
        events = [e async for e in provider.call(_make_opts())]
        starts = [e for e in events if isinstance(e, ToolUseStartEvent)]
        assert [s.id for s in starts] == ["call_0", "call_1"]
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
        assert exc_info.value.provider_name == "test-ollama"

    async def test_500_raises_server_error(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_status=500)
        with pytest.raises(ProviderServerError) as exc_info:
            [e async for e in provider.call(_make_opts())]
        assert exc_info.value.status_code == 500

    async def test_404_raises_provider_call_error(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_status=404)
        with pytest.raises(ProviderCallError):
            [e async for e in provider.call(_make_opts())]

    async def test_timeout_raises_timeout_error(self, mock_span: MockSpan) -> None:
        class _TimeoutTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                raise httpx.TimeoutException("timed out")

        http = httpx.AsyncClient(transport=_TimeoutTransport(), base_url="http://localhost:11434")
        provider = OllamaProvider(name="test-ollama", _http=http)
        with pytest.raises(ProviderTimeoutError):
            [e async for e in provider.call(_make_opts())]

    async def test_connect_error_raises_provider_call_error(self, mock_span: MockSpan) -> None:
        class _ConnectTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("connection refused")

        http = httpx.AsyncClient(transport=_ConnectTransport(), base_url="http://localhost:11434")
        provider = OllamaProvider(name="test-ollama", _http=http)
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
            [e async for e in provider.call(_make_opts(model="llama3.2", session_id="s1"))]

        assert len(audit.entries) == 1
        entry = audit.entries[0]
        assert entry.event == "ollama.call.failed"
        assert entry.level == "error"
        assert entry.provider_name == "test-ollama"
        assert entry.provider_kind == "ollama"
        assert entry.model == "llama3.2"
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
        http = httpx.AsyncClient(transport=_TimeoutTransport(), base_url="http://localhost:11434")
        provider = OllamaProvider(name="test-ollama", audit_log=audit, _http=http)
        with pytest.raises(ProviderTimeoutError):
            [e async for e in provider.call(_make_opts())]

        assert len(audit.entries) == 1
        assert audit.entries[0].event == "ollama.call.failed"


# ---------------------------------------------------------------------------
# call() — OTel span
# ---------------------------------------------------------------------------


class TestCallOTelSpan:
    async def test_span_name(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream())
        [e async for e in provider.call(_make_opts())]
        assert mock_span.name == "ollama.model.call"

    async def test_span_attributes_provider_name(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream())
        [e async for e in provider.call(_make_opts())]
        assert mock_span.attributes["provider.name"] == "test-ollama"

    async def test_span_attributes_model(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream())
        [e async for e in provider.call(_make_opts(model="phi4"))]
        assert mock_span.attributes["model"] == "phi4"

    async def test_invocation_event_attached(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream())
        [e async for e in provider.call(_make_opts())]
        event_names = [e[0] for e in mock_span.events]
        assert "provider.invocation" in event_names

    async def test_invocation_event_provider_kind(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream())
        [e async for e in provider.call(_make_opts())]
        inv = next(e for e in mock_span.events if e[0] == "provider.invocation")
        assert inv[1]["provider.kind"] == "ollama"

    async def test_invocation_event_session_id(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chat_body=_text_stream())
        [e async for e in provider.call(_make_opts(session_id="sess-abc"))]
        inv = next(e for e in mock_span.events if e[0] == "provider.invocation")
        assert inv[1]["session.id"] == "sess-abc"

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
        assert error_events[0][1]["provider.name"] == "test-ollama"

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
        provider = OllamaProvider()
        with pytest.raises(NotImplementedError):
            await provider.count_tokens(ModelCountReq(model="llama3.2", messages=[]))

    async def test_close_does_not_raise(self) -> None:
        provider = OllamaProvider()
        await provider.close()
