"""Tests for OpenAIProvider — streaming, tool-use, OTel span, audit log."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from opentelemetry.trace import StatusCode

from meridian_sdk_provider import ModelProvider
from meridian_sdk_provider.errors import (
    ProviderCallError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from meridian_sdk_provider.openai import OpenAIProvider
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

_API_KEY = "sk-test-key"
_MOCK_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


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
    monkeypatch.setattr("meridian_sdk_provider.openai.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_span(mock_tracer: MockTracer) -> MockSpan:
    return mock_tracer.span


# ---------------------------------------------------------------------------
# Mock chunk helpers
# ---------------------------------------------------------------------------


def _mk_fn(name: str | None = None, arguments: str | None = None) -> Any:
    return SimpleNamespace(name=name, arguments=arguments)


def _mk_tc(index: int = 0, id: str | None = None, function: Any = None) -> Any:
    return SimpleNamespace(index=index, id=id, function=function, type="function")


def _mk_usage(prompt_tokens: int, completion_tokens: int) -> Any:
    return SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def _mk_chunk(
    model: str = "gpt-4o",
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    usage: Any | None = None,
) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls or [], role=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    return SimpleNamespace(id="chatcmpl-x", model=model, choices=[choice], usage=usage)


def _text_chunks(
    text: str = "Hello!",
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> list[Any]:
    return [
        _mk_chunk(model=model, content=""),
        _mk_chunk(model=model, content=text),
        _mk_chunk(
            model=model,
            finish_reason="stop",
            usage=_mk_usage(prompt_tokens, completion_tokens),
        ),
    ]


def _tool_chunks(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_id: str = "call_abc",
    model: str = "gpt-4o",
) -> list[Any]:
    return [
        _mk_chunk(
            model=model,
            tool_calls=[_mk_tc(index=0, id=tool_id, function=_mk_fn(name=tool_name, arguments=""))],
        ),
        _mk_chunk(
            model=model,
            tool_calls=[_mk_tc(index=0, function=_mk_fn(arguments=json.dumps(tool_args)))],
        ),
        _mk_chunk(
            model=model,
            finish_reason="tool_calls",
            usage=_mk_usage(10, 5),
        ),
    ]


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _mk_response(status: int) -> httpx.Response:
    resp = httpx.Response(status)
    resp.request = _MOCK_REQUEST  # type: ignore[assignment]
    return resp


def _rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError("rate limited", response=_mk_response(429), body=None)


def _server_error() -> openai.InternalServerError:
    return openai.InternalServerError("internal error", response=_mk_response(500), body=None)


def _auth_error() -> openai.AuthenticationError:
    return openai.AuthenticationError("unauthorized", response=_mk_response(401), body=None)


def _timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=_MOCK_REQUEST)


def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(message="connection refused", request=_MOCK_REQUEST)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def _make_opts(**kwargs: Any) -> ModelCallOpts:
    defaults: dict[str, Any] = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    return ModelCallOpts(**defaults)


def _make_provider(
    chunks: list[Any] | None = None,
    error: Exception | None = None,
    audit_log: CollectingAuditLog | None = None,
) -> OpenAIProvider:
    async def _create(**kwargs: Any) -> Any:
        if error is not None:
            raise error

        async def _gen() -> Any:
            for c in chunks or []:
                yield c

        return _gen()

    mock_client: Any = MagicMock()
    mock_client.chat.completions.create = _create
    mock_client.close = AsyncMock(return_value=None)

    return OpenAIProvider(
        _API_KEY,
        name="test-openai",
        audit_log=audit_log,
        _client=mock_client,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_model_provider_protocol(self) -> None:
        assert isinstance(OpenAIProvider(_API_KEY), ModelProvider)

    def test_kind(self) -> None:
        assert OpenAIProvider(_API_KEY).kind == "openai"

    def test_name_default(self) -> None:
        assert OpenAIProvider(_API_KEY).name == "openai"

    def test_name_custom(self) -> None:
        assert OpenAIProvider(_API_KEY, name="my-openai").name == "my-openai"

    def test_capabilities_streaming(self) -> None:
        assert OpenAIProvider(_API_KEY).capabilities.streaming is True

    def test_capabilities_cache_control_false(self) -> None:
        assert OpenAIProvider(_API_KEY).capabilities.cache_control is False

    def test_capabilities_thinking_false(self) -> None:
        assert OpenAIProvider(_API_KEY).capabilities.thinking is False

    def test_capabilities_count_tokens_false(self) -> None:
        assert OpenAIProvider(_API_KEY).capabilities.count_tokens is False


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_returns_model_entries(self) -> None:
        payload = {
            "data": [
                {"id": "gpt-4o", "owned_by": "openai"},
                {"id": "gpt-3.5-turbo", "owned_by": "openai"},
            ]
        }
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OpenAIProvider(_API_KEY, name="test-openai").list_models()

        assert len(models) == 2
        assert models[0].model == "gpt-4o"
        assert models[0].provider == "test-openai"
        assert models[0].context_window == 128000

    def test_vision_flag_set_for_gpt4o(self) -> None:
        payload = {
            "data": [
                {"id": "gpt-4o", "owned_by": "openai"},
                {"id": "gpt-3.5-turbo", "owned_by": "openai"},
            ]
        }
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OpenAIProvider(_API_KEY).list_models()

        gpt4o = next(m for m in models if m.model == "gpt-4o")
        gpt35 = next(m for m in models if "3.5" in m.model)
        assert gpt4o.capabilities.vision is True
        assert gpt35.capabilities.vision is False

    def test_thinking_flag_set_for_o1_models(self) -> None:
        payload = {
            "data": [
                {"id": "o1-mini", "owned_by": "openai"},
                {"id": "gpt-4o", "owned_by": "openai"},
            ]
        }
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OpenAIProvider(_API_KEY).list_models()

        o1 = next(m for m in models if "o1" in m.model)
        gpt = next(m for m in models if "gpt" in m.model)
        assert o1.capabilities.thinking is True
        assert gpt.capabilities.thinking is False

    def test_tools_always_true(self) -> None:
        payload = {"data": [{"id": "gpt-4o", "owned_by": "openai"}]}
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OpenAIProvider(_API_KEY).list_models()

        assert models[0].capabilities.tools is True

    def test_skips_entries_with_no_id(self) -> None:
        payload = {
            "data": [
                {"owned_by": "openai"},
                {"id": "gpt-4o", "owned_by": "openai"},
            ]
        }
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(200, json=payload)
            models = OpenAIProvider(_API_KEY).list_models()
        assert len(models) == 1

    def test_returns_empty_on_connection_error(self) -> None:
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            assert OpenAIProvider(_API_KEY).list_models() == []

    def test_returns_empty_on_http_error(self) -> None:
        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(401, text="unauthorized")
            assert OpenAIProvider(_API_KEY).list_models() == []


# ---------------------------------------------------------------------------
# call() — streaming events
# ---------------------------------------------------------------------------


class TestCallStreamingEvents:
    async def test_yields_message_start(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chunks=_text_chunks("Hi", model="gpt-4o"))
        events = [e async for e in provider.call(_make_opts())]
        assert isinstance(events[0], MessageStartEvent)
        assert events[0].model == "gpt-4o"
        assert events[0].provider == "test-openai"

    async def test_yields_text_delta(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chunks=_text_chunks("Hello world"))
        events = [e async for e in provider.call(_make_opts())]
        text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "Hello world"

    async def test_yields_message_stop_with_tokens(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chunks=_text_chunks("Hi", prompt_tokens=7, completion_tokens=3))
        events = [e async for e in provider.call(_make_opts())]
        stop = next(e for e in events if isinstance(e, MessageStopEvent))
        assert stop.input_tokens == 7
        assert stop.output_tokens == 3
        assert stop.stop_reason == "stop"

    async def test_empty_content_chunks_not_yielded(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chunks=_text_chunks(""))
        events = [e async for e in provider.call(_make_opts())]
        assert not any(isinstance(e, TextDeltaEvent) for e in events)

    async def test_multiple_text_chunks(self, mock_span: MockSpan) -> None:
        chunks = [
            _mk_chunk(model="gpt-4o", content="A"),
            _mk_chunk(model="gpt-4o", content="B"),
            _mk_chunk(model="gpt-4o", finish_reason="stop", usage=_mk_usage(5, 2)),
        ]
        provider = _make_provider(chunks=chunks)
        events = [e async for e in provider.call(_make_opts())]
        texts = [e.text for e in events if isinstance(e, TextDeltaEvent)]
        assert texts == ["A", "B"]

    async def test_system_prompt_forwarded(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> Any:
                yield _mk_chunk(model="gpt-4o", content="ok")
                yield _mk_chunk(model="gpt-4o", finish_reason="stop", usage=_mk_usage(5, 2))

            return _gen()

        mock_client: Any = MagicMock()
        mock_client.chat.completions.create = _create
        provider = OpenAIProvider(_API_KEY, name="test-openai", _client=mock_client)
        [e async for e in provider.call(_make_opts(system="Be concise."))]

        messages = captured[0]["messages"]
        assert messages[0] == {"role": "system", "content": "Be concise."}

    async def test_temperature_forwarded(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> Any:
                yield _mk_chunk(model="gpt-4o", finish_reason="stop", usage=_mk_usage(5, 2))

            return _gen()

        mock_client: Any = MagicMock()
        mock_client.chat.completions.create = _create
        provider = OpenAIProvider(_API_KEY, name="test-openai", _client=mock_client)
        [e async for e in provider.call(_make_opts(temperature=0.3))]
        assert captured[0]["temperature"] == 0.3

    async def test_max_tokens_forwarded(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> Any:
                yield _mk_chunk(model="gpt-4o", finish_reason="stop", usage=_mk_usage(5, 2))

            return _gen()

        mock_client: Any = MagicMock()
        mock_client.chat.completions.create = _create
        provider = OpenAIProvider(_API_KEY, name="test-openai", _client=mock_client)
        [e async for e in provider.call(_make_opts(max_tokens=512))]
        assert captured[0]["max_tokens"] == 512

    async def test_stream_options_include_usage(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> Any:
                yield _mk_chunk(model="gpt-4o", finish_reason="stop", usage=_mk_usage(5, 2))

            return _gen()

        mock_client: Any = MagicMock()
        mock_client.chat.completions.create = _create
        provider = OpenAIProvider(_API_KEY, name="test-openai", _client=mock_client)
        [e async for e in provider.call(_make_opts())]
        assert captured[0].get("stream_options") == {"include_usage": True}

    async def test_tools_forwarded_when_present(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> Any:
                yield _mk_chunk(model="gpt-4o", finish_reason="stop", usage=_mk_usage(5, 2))

            return _gen()

        mock_client: Any = MagicMock()
        mock_client.chat.completions.create = _create
        provider = OpenAIProvider(_API_KEY, name="test-openai", _client=mock_client)
        opts = _make_opts(
            tools=[{"name": "fn", "description": "does stuff", "input_schema": {"type": "object"}}]
        )
        [e async for e in provider.call(opts)]
        assert "tools" in captured[0]
        assert captured[0].get("tool_choice") == "auto"


# ---------------------------------------------------------------------------
# call() — tool call events
# ---------------------------------------------------------------------------


class TestCallToolEvents:
    async def test_yields_tool_use_start_and_delta(self, mock_span: MockSpan) -> None:
        provider = _make_provider(
            chunks=_tool_chunks("get_weather", {"city": "NYC"}, tool_id="call_xyz")
        )
        events = [e async for e in provider.call(_make_opts())]

        tool_start = next(e for e in events if isinstance(e, ToolUseStartEvent))
        tool_delta = next(e for e in events if isinstance(e, ToolInputDeltaEvent))

        assert tool_start.name == "get_weather"
        assert tool_start.id == "call_xyz"
        assert tool_delta.id == "call_xyz"
        assert json.loads(tool_delta.partial_json) == {"city": "NYC"}

    async def test_tool_stop_reason_is_tool_calls(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chunks=_tool_chunks("fn", {}, tool_id="call_1"))
        events = [e async for e in provider.call(_make_opts())]
        stop = next(e for e in events if isinstance(e, MessageStopEvent))
        assert stop.stop_reason == "tool_calls"

    async def test_multiple_tool_calls_indexed(self, mock_span: MockSpan) -> None:
        chunks = [
            _mk_chunk(
                model="gpt-4o",
                tool_calls=[
                    _mk_tc(index=0, id="call_a", function=_mk_fn(name="fn_a", arguments="")),
                    _mk_tc(index=1, id="call_b", function=_mk_fn(name="fn_b", arguments="")),
                ],
            ),
            _mk_chunk(
                model="gpt-4o",
                tool_calls=[
                    _mk_tc(index=0, function=_mk_fn(arguments="{}")),
                    _mk_tc(index=1, function=_mk_fn(arguments='{"x":1}')),
                ],
            ),
            _mk_chunk(model="gpt-4o", finish_reason="tool_calls", usage=_mk_usage(10, 5)),
        ]
        provider = _make_provider(chunks=chunks)
        events = [e async for e in provider.call(_make_opts())]
        starts = [e for e in events if isinstance(e, ToolUseStartEvent)]
        assert [s.id for s in starts] == ["call_a", "call_b"]
        assert [s.name for s in starts] == ["fn_a", "fn_b"]

    async def test_tool_input_delta_uses_accumulated_id(self, mock_span: MockSpan) -> None:
        chunks = [
            _mk_chunk(
                model="gpt-4o",
                tool_calls=[
                    _mk_tc(index=0, id="call_z", function=_mk_fn(name="do_it", arguments=""))
                ],
            ),
            _mk_chunk(
                model="gpt-4o",
                tool_calls=[_mk_tc(index=0, function=_mk_fn(arguments='{"k":"v"}'))],
            ),
            _mk_chunk(model="gpt-4o", finish_reason="tool_calls", usage=_mk_usage(5, 3)),
        ]
        provider = _make_provider(chunks=chunks)
        events = [e async for e in provider.call(_make_opts())]
        deltas = [e for e in events if isinstance(e, ToolInputDeltaEvent)]
        assert all(d.id == "call_z" for d in deltas)


# ---------------------------------------------------------------------------
# call() — error handling
# ---------------------------------------------------------------------------


class TestCallErrors:
    async def test_429_raises_rate_limit_error(self, mock_span: MockSpan) -> None:
        provider = _make_provider(error=_rate_limit_error())
        with pytest.raises(ProviderRateLimitError) as exc_info:
            [e async for e in provider.call(_make_opts())]
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider_name == "test-openai"

    async def test_500_raises_server_error(self, mock_span: MockSpan) -> None:
        provider = _make_provider(error=_server_error())
        with pytest.raises(ProviderServerError) as exc_info:
            [e async for e in provider.call(_make_opts())]
        assert exc_info.value.status_code == 500

    async def test_401_raises_provider_call_error(self, mock_span: MockSpan) -> None:
        provider = _make_provider(error=_auth_error())
        with pytest.raises(ProviderCallError):
            [e async for e in provider.call(_make_opts())]

    async def test_timeout_raises_timeout_error(self, mock_span: MockSpan) -> None:
        provider = _make_provider(error=_timeout_error())
        with pytest.raises(ProviderTimeoutError):
            [e async for e in provider.call(_make_opts())]

    async def test_connection_error_raises_provider_call_error(self, mock_span: MockSpan) -> None:
        provider = _make_provider(error=_connection_error())
        with pytest.raises(ProviderCallError):
            [e async for e in provider.call(_make_opts())]

    async def test_timeout_error_provider_name(self, mock_span: MockSpan) -> None:
        provider = _make_provider(error=_timeout_error())
        with pytest.raises(ProviderTimeoutError) as exc_info:
            [e async for e in provider.call(_make_opts())]
        assert exc_info.value.provider_name == "test-openai"


# ---------------------------------------------------------------------------
# call() — audit log
# ---------------------------------------------------------------------------


class TestCallAuditLog:
    async def test_audit_written_on_rate_limit_error(self, mock_span: MockSpan) -> None:
        audit = CollectingAuditLog()
        provider = _make_provider(error=_rate_limit_error(), audit_log=audit)
        with pytest.raises(ProviderRateLimitError):
            [e async for e in provider.call(_make_opts(model="gpt-4o", session_id="s1"))]

        assert len(audit.entries) == 1
        entry = audit.entries[0]
        assert entry.event == "openai.call.failed"
        assert entry.level == "error"
        assert entry.provider_name == "test-openai"
        assert entry.provider_kind == "openai"
        assert entry.model == "gpt-4o"
        assert entry.session_id == "s1"
        assert "error_type" in entry.detail

    async def test_audit_written_on_server_error(self, mock_span: MockSpan) -> None:
        audit = CollectingAuditLog()
        provider = _make_provider(error=_server_error(), audit_log=audit)
        with pytest.raises(ProviderServerError):
            [e async for e in provider.call(_make_opts())]
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "openai.call.failed"

    async def test_audit_written_on_timeout(self, mock_span: MockSpan) -> None:
        audit = CollectingAuditLog()
        provider = _make_provider(error=_timeout_error(), audit_log=audit)
        with pytest.raises(ProviderTimeoutError):
            [e async for e in provider.call(_make_opts())]
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "openai.call.failed"

    async def test_no_audit_on_success(self, mock_span: MockSpan) -> None:
        audit = CollectingAuditLog()
        provider = _make_provider(chunks=_text_chunks(), audit_log=audit)
        [e async for e in provider.call(_make_opts())]
        assert audit.entries == []

    async def test_audit_error_type_in_detail(self, mock_span: MockSpan) -> None:
        audit = CollectingAuditLog()
        provider = _make_provider(error=_rate_limit_error(), audit_log=audit)
        with pytest.raises(ProviderRateLimitError):
            [e async for e in provider.call(_make_opts())]
        assert audit.entries[0].detail["error_type"] == "ProviderRateLimitError"


# ---------------------------------------------------------------------------
# call() — OTel span
# ---------------------------------------------------------------------------


class TestCallOTelSpan:
    async def test_span_name(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chunks=_text_chunks())
        [e async for e in provider.call(_make_opts())]
        assert mock_span.name == "openai.model.call"

    async def test_span_attribute_provider_name(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chunks=_text_chunks())
        [e async for e in provider.call(_make_opts())]
        assert mock_span.attributes["provider.name"] == "test-openai"

    async def test_span_attribute_model(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chunks=_text_chunks())
        [e async for e in provider.call(_make_opts(model="gpt-4o-mini"))]
        assert mock_span.attributes["model"] == "gpt-4o-mini"

    async def test_invocation_event_attached(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chunks=_text_chunks())
        [e async for e in provider.call(_make_opts())]
        event_names = [e[0] for e in mock_span.events]
        assert "provider.invocation" in event_names

    async def test_invocation_event_provider_kind(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chunks=_text_chunks())
        [e async for e in provider.call(_make_opts())]
        inv = next(e for e in mock_span.events if e[0] == "provider.invocation")
        assert inv[1]["provider.kind"] == "openai"

    async def test_invocation_event_session_id(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chunks=_text_chunks())
        [e async for e in provider.call(_make_opts(session_id="sess-abc"))]
        inv = next(e for e in mock_span.events if e[0] == "provider.invocation")
        assert inv[1]["session.id"] == "sess-abc"

    async def test_span_ended_on_success(self, mock_span: MockSpan) -> None:
        provider = _make_provider(chunks=_text_chunks())
        [e async for e in provider.call(_make_opts())]
        assert mock_span.ended

    async def test_span_marked_error_on_failure(self, mock_span: MockSpan) -> None:
        provider = _make_provider(error=_server_error())
        with pytest.raises(ProviderServerError):
            [e async for e in provider.call(_make_opts())]
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_error_event_attached_on_failure(self, mock_span: MockSpan) -> None:
        provider = _make_provider(error=_rate_limit_error())
        with pytest.raises(ProviderRateLimitError):
            [e async for e in provider.call(_make_opts())]
        error_events = [e for e in mock_span.events if e[0] == "provider.error"]
        assert len(error_events) == 1
        assert error_events[0][1]["provider.name"] == "test-openai"

    async def test_exception_recorded_on_failure(self, mock_span: MockSpan) -> None:
        provider = _make_provider(error=_server_error())
        with pytest.raises(ProviderServerError):
            [e async for e in provider.call(_make_opts())]
        assert len(mock_span.recorded_exceptions) == 1

    async def test_span_ended_on_failure(self, mock_span: MockSpan) -> None:
        provider = _make_provider(error=_timeout_error())
        with pytest.raises(ProviderTimeoutError):
            [e async for e in provider.call(_make_opts())]
        assert mock_span.ended


# ---------------------------------------------------------------------------
# count_tokens / close
# ---------------------------------------------------------------------------


class TestCountTokensAndClose:
    async def test_count_tokens_raises_not_implemented(self) -> None:
        provider = OpenAIProvider(_API_KEY)
        with pytest.raises(NotImplementedError):
            await provider.count_tokens(ModelCountReq(model="gpt-4o", messages=[]))

    async def test_close_delegates_to_client(self) -> None:
        mock_client: Any = MagicMock()
        mock_client.close = AsyncMock(return_value=None)
        provider = OpenAIProvider(_API_KEY, _client=mock_client)
        await provider.close()
        mock_client.close.assert_awaited_once()
