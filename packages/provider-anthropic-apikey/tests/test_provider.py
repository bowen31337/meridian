"""Tests for AnthropicApiKeyProvider — streaming, tool-use, thinking, OTel span, audit log."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import pytest
from meridian_sdk_provider import ModelProvider
from meridian_sdk_provider.errors import (
    ProviderCallError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from meridian_sdk_provider.types import (
    MessageStartEvent,
    MessageStopEvent,
    ModelCallOpts,
    ModelCountReq,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolInputDeltaEvent,
    ToolUseStartEvent,
)
from opentelemetry.trace import StatusCode

from meridian_provider_anthropic_apikey.provider import (
    AnthropicApiKeyProvider,
    _convert_message,
)

_API_KEY = "sk-ant-test-key"
_MOCK_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


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
    monkeypatch.setattr("meridian_provider_anthropic_apikey.provider.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_span(mock_tracer: MockTracer) -> MockSpan:
    return mock_tracer.span


# ---------------------------------------------------------------------------
# Streaming event helpers
# ---------------------------------------------------------------------------


def _msg_start(model: str = "claude-sonnet-4-6", input_tokens: int = 10) -> Any:
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            model=model,
            usage=SimpleNamespace(input_tokens=input_tokens),
        ),
    )


def _block_start_text(index: int = 0) -> Any:
    return SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=SimpleNamespace(type="text", text=""),
    )


def _block_start_tool(index: int = 0, id: str = "toolu_abc", name: str = "my_tool") -> Any:
    return SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=SimpleNamespace(type="tool_use", id=id, name=name),
    )


def _block_start_thinking(index: int = 0) -> Any:
    return SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=SimpleNamespace(type="thinking", thinking=""),
    )


def _text_delta(text: str = "Hello!", index: int = 0) -> Any:
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _tool_delta(partial_json: str = '{"x":1}', index: int = 0) -> Any:
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type="input_json_delta", partial_json=partial_json),
    )


def _thinking_delta(thinking: str = "Let me think", index: int = 0) -> Any:
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type="thinking_delta", thinking=thinking),
    )


def _block_stop(index: int = 0) -> Any:
    return SimpleNamespace(type="content_block_stop", index=index)


def _msg_delta(stop_reason: str = "end_turn", output_tokens: int = 5) -> Any:
    return SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason=stop_reason),
        usage=SimpleNamespace(output_tokens=output_tokens),
    )


def _msg_stop() -> Any:
    return SimpleNamespace(type="message_stop")


def _ping() -> Any:
    return SimpleNamespace(type="ping")


def _text_events(
    text: str = "Hello!",
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[Any]:
    return [
        _msg_start(model=model, input_tokens=input_tokens),
        _block_start_text(),
        _text_delta(text=text),
        _block_stop(),
        _msg_delta(output_tokens=output_tokens),
        _msg_stop(),
    ]


def _tool_events(
    tool_name: str = "get_weather",
    tool_id: str = "toolu_xyz",
    partial_json: str = '{"city":"NYC"}',
) -> list[Any]:
    return [
        _msg_start(),
        _block_start_tool(id=tool_id, name=tool_name),
        _tool_delta(partial_json=partial_json),
        _block_stop(),
        _msg_delta(stop_reason="tool_use"),
        _msg_stop(),
    ]


def _thinking_events(thinking_text: str = "Let me think carefully") -> list[Any]:
    return [
        _msg_start(),
        _block_start_thinking(index=0),
        _thinking_delta(thinking=thinking_text, index=0),
        _block_stop(index=0),
        _block_start_text(index=1),
        _text_delta(text="Answer", index=1),
        _block_stop(index=1),
        _msg_delta(),
        _msg_stop(),
    ]


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _mk_response(status: int) -> httpx.Response:
    resp = httpx.Response(status)
    resp.request = _MOCK_REQUEST  # type: ignore[assignment]
    return resp


def _rate_limit_error() -> anthropic.RateLimitError:
    return anthropic.RateLimitError(
        "rate limited",
        response=_mk_response(429),
        body=None,
    )


def _server_error() -> anthropic.InternalServerError:
    return anthropic.InternalServerError(
        "internal error",
        response=_mk_response(500),
        body=None,
    )


def _auth_error() -> anthropic.AuthenticationError:
    return anthropic.AuthenticationError(
        "unauthorized",
        response=_mk_response(401),
        body=None,
    )


def _timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=_MOCK_REQUEST)


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(message="connection refused", request=_MOCK_REQUEST)


# ---------------------------------------------------------------------------
# CollectingAuditLog
# ---------------------------------------------------------------------------


class CollectingAuditLog:
    def __init__(self) -> None:
        from meridian_sdk_provider.audit import AuditLogEntry

        self.entries: list[AuditLogEntry] = []

    def write(self, entry: Any) -> None:
        self.entries.append(entry)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def _make_opts(**kwargs: Any) -> ModelCallOpts:
    defaults: dict[str, Any] = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    return ModelCallOpts(**defaults)


def _make_provider(
    events: list[Any] | None = None,
    error: Exception | None = None,
    count_tokens_result: Any | None = None,
    audit_log: CollectingAuditLog | None = None,
) -> AnthropicApiKeyProvider:
    async def _create(**kwargs: Any) -> Any:
        if error is not None:
            raise error

        async def _gen() -> Any:
            for e in events or []:
                yield e

        return _gen()

    mock_messages: Any = MagicMock()
    mock_messages.create = _create
    mock_messages.count_tokens = AsyncMock(
        return_value=count_tokens_result or SimpleNamespace(input_tokens=20)
    )

    mock_client: Any = MagicMock()
    mock_client.messages = mock_messages
    mock_client.close = AsyncMock(return_value=None)

    return AnthropicApiKeyProvider(
        _API_KEY,
        name="test-anthropic",
        audit_log=audit_log,
        _client=mock_client,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_model_provider_protocol(self) -> None:
        assert isinstance(AnthropicApiKeyProvider(_API_KEY), ModelProvider)

    def test_kind(self) -> None:
        assert AnthropicApiKeyProvider(_API_KEY).kind == "anthropic"

    def test_name_default(self) -> None:
        assert AnthropicApiKeyProvider(_API_KEY).name == "anthropic"

    def test_name_custom(self) -> None:
        assert AnthropicApiKeyProvider(_API_KEY, name="my-anthropic").name == "my-anthropic"

    def test_capabilities_streaming(self) -> None:
        assert AnthropicApiKeyProvider(_API_KEY).capabilities.streaming is True

    def test_capabilities_thinking(self) -> None:
        assert AnthropicApiKeyProvider(_API_KEY).capabilities.thinking is True

    def test_capabilities_cache_control(self) -> None:
        assert AnthropicApiKeyProvider(_API_KEY).capabilities.cache_control is True

    def test_capabilities_count_tokens(self) -> None:
        assert AnthropicApiKeyProvider(_API_KEY).capabilities.count_tokens is True


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_returns_known_models(self) -> None:
        models = AnthropicApiKeyProvider(_API_KEY).list_models()
        assert len(models) >= 4

    def test_provider_name_set(self) -> None:
        models = AnthropicApiKeyProvider(_API_KEY, name="test-ant").list_models()
        assert all(m.provider == "test-ant" for m in models)

    def test_claude_sonnet_present(self) -> None:
        models = AnthropicApiKeyProvider(_API_KEY).list_models()
        ids = [m.model for m in models]
        assert any("sonnet" in mid for mid in ids)

    def test_thinking_flag_for_opus_and_sonnet4(self) -> None:
        models = AnthropicApiKeyProvider(_API_KEY).list_models()
        opus = next((m for m in models if "opus-4" in m.model), None)
        assert opus is not None
        assert opus.capabilities.thinking is True

    def test_cache_always_true(self) -> None:
        models = AnthropicApiKeyProvider(_API_KEY).list_models()
        assert all(m.capabilities.cache is True for m in models)

    def test_tools_always_true(self) -> None:
        models = AnthropicApiKeyProvider(_API_KEY).list_models()
        assert all(m.capabilities.tools is True for m in models)

    def test_context_window_set(self) -> None:
        models = AnthropicApiKeyProvider(_API_KEY).list_models()
        assert all(m.context_window > 0 for m in models)


# ---------------------------------------------------------------------------
# call() — streaming events
# ---------------------------------------------------------------------------


class TestCallStreamingEvents:
    async def test_yields_message_start(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_text_events("Hi", model="claude-sonnet-4-6"))
        events = [e async for e in provider.call(_make_opts())]
        assert isinstance(events[0], MessageStartEvent)
        assert events[0].model == "claude-sonnet-4-6"
        assert events[0].provider == "test-anthropic"

    async def test_message_start_carries_input_tokens(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_text_events(input_tokens=42))
        events = [e async for e in provider.call(_make_opts())]
        start = next(e for e in events if isinstance(e, MessageStartEvent))
        assert start.input_tokens == 42

    async def test_yields_text_delta(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_text_events("Hello world"))
        events = [e async for e in provider.call(_make_opts())]
        text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "Hello world"

    async def test_yields_message_stop_with_tokens(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_text_events(input_tokens=7, output_tokens=3))
        events = [e async for e in provider.call(_make_opts())]
        stop = next(e for e in events if isinstance(e, MessageStopEvent))
        assert stop.input_tokens == 7
        assert stop.output_tokens == 3
        assert stop.stop_reason == "end_turn"

    async def test_multiple_text_deltas(self, mock_span: MockSpan) -> None:
        raw = [
            _msg_start(),
            _block_start_text(),
            _text_delta("A"),
            _text_delta("B"),
            _block_stop(),
            _msg_delta(),
            _msg_stop(),
        ]
        provider = _make_provider(events=raw)
        events = [e async for e in provider.call(_make_opts())]
        texts = [e.text for e in events if isinstance(e, TextDeltaEvent)]
        assert texts == ["A", "B"]

    async def test_ping_events_ignored(self, mock_span: MockSpan) -> None:
        raw = [
            _msg_start(),
            _ping(),
            _block_start_text(),
            _text_delta(),
            _block_stop(),
            _msg_delta(),
            _msg_stop(),
        ]
        provider = _make_provider(events=raw)
        events = [e async for e in provider.call(_make_opts())]
        types = {type(e).__name__ for e in events}
        assert "PingEvent" not in types

    async def test_system_prompt_forwarded(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> Any:
                for e in _text_events():
                    yield e

            return _gen()

        mock_messages: Any = MagicMock()
        mock_messages.create = _create
        mock_client: Any = MagicMock()
        mock_client.messages = mock_messages

        provider = AnthropicApiKeyProvider(_API_KEY, name="test-anthropic", _client=mock_client)
        [e async for e in provider.call(_make_opts(system="Be concise."))]
        # System prompt is wrapped with cache_control for per-call prompt caching.
        assert captured[0]["system"] == [
            {"type": "text", "text": "Be concise.", "cache_control": {"type": "ephemeral"}}
        ]

    async def test_temperature_forwarded(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> Any:
                for e in _text_events():
                    yield e

            return _gen()

        mock_messages: Any = MagicMock()
        mock_messages.create = _create
        mock_client: Any = MagicMock()
        mock_client.messages = mock_messages

        provider = AnthropicApiKeyProvider(_API_KEY, name="test-anthropic", _client=mock_client)
        [e async for e in provider.call(_make_opts(temperature=0.7))]
        assert captured[0]["temperature"] == 0.7

    async def test_max_tokens_forwarded(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> Any:
                for e in _text_events():
                    yield e

            return _gen()

        mock_messages: Any = MagicMock()
        mock_messages.create = _create
        mock_client: Any = MagicMock()
        mock_client.messages = mock_messages

        provider = AnthropicApiKeyProvider(_API_KEY, name="test-anthropic", _client=mock_client)
        [e async for e in provider.call(_make_opts(max_tokens=512))]
        assert captured[0]["max_tokens"] == 512

    async def test_tools_forwarded_when_present(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> Any:
                for e in _text_events():
                    yield e

            return _gen()

        mock_messages: Any = MagicMock()
        mock_messages.create = _create
        mock_client: Any = MagicMock()
        mock_client.messages = mock_messages

        provider = AnthropicApiKeyProvider(_API_KEY, name="test-anthropic", _client=mock_client)
        opts = _make_opts(
            tools=[{"name": "fn", "description": "does stuff", "input_schema": {"type": "object"}}]
        )
        [e async for e in provider.call(opts)]
        assert "tools" in captured[0]
        assert captured[0].get("tool_choice") == {"type": "auto"}


# ---------------------------------------------------------------------------
# call() — tool-use events
# ---------------------------------------------------------------------------


class TestCallToolEvents:
    async def test_yields_tool_use_start(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_tool_events(tool_name="search", tool_id="toolu_1"))
        events = [e async for e in provider.call(_make_opts())]
        start = next(e for e in events if isinstance(e, ToolUseStartEvent))
        assert start.name == "search"
        assert start.id == "toolu_1"

    async def test_yields_tool_input_delta(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_tool_events(partial_json='{"q":"python"}'))
        events = [e async for e in provider.call(_make_opts())]
        delta = next(e for e in events if isinstance(e, ToolInputDeltaEvent))
        assert delta.partial_json == '{"q":"python"}'

    async def test_tool_input_delta_id_matches_start_id(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_tool_events(tool_id="toolu_99"))
        events = [e async for e in provider.call(_make_opts())]
        start = next(e for e in events if isinstance(e, ToolUseStartEvent))
        delta = next(e for e in events if isinstance(e, ToolInputDeltaEvent))
        assert delta.id == start.id == "toolu_99"

    async def test_stop_reason_tool_use(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_tool_events())
        events = [e async for e in provider.call(_make_opts())]
        stop = next(e for e in events if isinstance(e, MessageStopEvent))
        assert stop.stop_reason == "tool_use"

    async def test_multiple_tool_calls(self, mock_span: MockSpan) -> None:
        raw = [
            _msg_start(),
            _block_start_tool(index=0, id="toolu_a", name="fn_a"),
            _tool_delta(partial_json="{}", index=0),
            _block_stop(index=0),
            _block_start_tool(index=1, id="toolu_b", name="fn_b"),
            _tool_delta(partial_json='{"x":2}', index=1),
            _block_stop(index=1),
            _msg_delta(stop_reason="tool_use"),
            _msg_stop(),
        ]
        provider = _make_provider(events=raw)
        events = [e async for e in provider.call(_make_opts())]
        starts = [e for e in events if isinstance(e, ToolUseStartEvent)]
        deltas = [e for e in events if isinstance(e, ToolInputDeltaEvent)]
        assert [s.id for s in starts] == ["toolu_a", "toolu_b"]
        assert deltas[0].id == "toolu_a"
        assert deltas[1].id == "toolu_b"


# ---------------------------------------------------------------------------
# call() — thinking events
# ---------------------------------------------------------------------------


class TestCallThinkingEvents:
    async def test_yields_thinking_delta(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_thinking_events("My reasoning here"))
        events = [e async for e in provider.call(_make_opts())]
        td = next(e for e in events if isinstance(e, ThinkingDeltaEvent))
        assert td.thinking == "My reasoning here"

    async def test_thinking_followed_by_text(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_thinking_events())
        events = [e async for e in provider.call(_make_opts())]
        assert any(isinstance(e, ThinkingDeltaEvent) for e in events)
        assert any(isinstance(e, TextDeltaEvent) for e in events)

    async def test_thinking_budget_forwarded(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> Any:
                for e in _text_events():
                    yield e

            return _gen()

        mock_messages: Any = MagicMock()
        mock_messages.create = _create
        mock_client: Any = MagicMock()
        mock_client.messages = mock_messages

        provider = AnthropicApiKeyProvider(_API_KEY, name="test-anthropic", _client=mock_client)
        [
            e
            async for e in provider.call(
                _make_opts(enable_thinking=True, thinking_budget_tokens=4000)
            )
        ]
        assert captured[0]["thinking"] == {"type": "enabled", "budget_tokens": 4000}

    async def test_no_thinking_key_without_enable(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> Any:
                for e in _text_events():
                    yield e

            return _gen()

        mock_messages: Any = MagicMock()
        mock_messages.create = _create
        mock_client: Any = MagicMock()
        mock_client.messages = mock_messages

        provider = AnthropicApiKeyProvider(_API_KEY, name="test-anthropic", _client=mock_client)
        [e async for e in provider.call(_make_opts())]
        assert "thinking" not in captured[0]


# ---------------------------------------------------------------------------
# call() — error handling
# ---------------------------------------------------------------------------


class TestCallErrors:
    async def test_429_raises_rate_limit_error(self, mock_span: MockSpan) -> None:
        provider = _make_provider(error=_rate_limit_error())
        with pytest.raises(ProviderRateLimitError) as exc_info:
            [e async for e in provider.call(_make_opts())]
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider_name == "test-anthropic"

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
        assert exc_info.value.provider_name == "test-anthropic"

    async def test_rate_limit_provider_name(self, mock_span: MockSpan) -> None:
        provider = _make_provider(error=_rate_limit_error())
        with pytest.raises(ProviderRateLimitError) as exc_info:
            [e async for e in provider.call(_make_opts())]
        assert exc_info.value.provider_name == "test-anthropic"


# ---------------------------------------------------------------------------
# call() — audit log
# ---------------------------------------------------------------------------


class TestCallAuditLog:
    async def test_audit_written_on_rate_limit_error(self, mock_span: MockSpan) -> None:
        audit = CollectingAuditLog()
        provider = _make_provider(error=_rate_limit_error(), audit_log=audit)
        with pytest.raises(ProviderRateLimitError):
            [e async for e in provider.call(_make_opts(model="claude-sonnet-4-6", session_id="s1"))]

        assert len(audit.entries) == 1
        entry = audit.entries[0]
        assert entry.event == "anthropic.call.failed"
        assert entry.level == "error"
        assert entry.provider_name == "test-anthropic"
        assert entry.provider_kind == "anthropic"
        assert entry.model == "claude-sonnet-4-6"
        assert entry.session_id == "s1"
        assert "error_type" in entry.detail

    async def test_audit_written_on_server_error(self, mock_span: MockSpan) -> None:
        audit = CollectingAuditLog()
        provider = _make_provider(error=_server_error(), audit_log=audit)
        with pytest.raises(ProviderServerError):
            [e async for e in provider.call(_make_opts())]
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "anthropic.call.failed"

    async def test_audit_written_on_timeout(self, mock_span: MockSpan) -> None:
        audit = CollectingAuditLog()
        provider = _make_provider(error=_timeout_error(), audit_log=audit)
        with pytest.raises(ProviderTimeoutError):
            [e async for e in provider.call(_make_opts())]
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "anthropic.call.failed"

    async def test_no_audit_on_success(self, mock_span: MockSpan) -> None:
        audit = CollectingAuditLog()
        provider = _make_provider(events=_text_events(), audit_log=audit)
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
        provider = _make_provider(events=_text_events())
        [e async for e in provider.call(_make_opts())]
        assert mock_span.name == "anthropic.model.call"

    async def test_span_attribute_provider_name(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_text_events())
        [e async for e in provider.call(_make_opts())]
        assert mock_span.attributes["provider.name"] == "test-anthropic"

    async def test_span_attribute_model(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_text_events())
        [e async for e in provider.call(_make_opts(model="claude-opus-4-7"))]
        assert mock_span.attributes["model"] == "claude-opus-4-7"

    async def test_invocation_event_attached(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_text_events())
        [e async for e in provider.call(_make_opts())]
        event_names = [e[0] for e in mock_span.events]
        assert "provider.invocation" in event_names

    async def test_invocation_event_provider_kind(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_text_events())
        [e async for e in provider.call(_make_opts())]
        inv = next(e for e in mock_span.events if e[0] == "provider.invocation")
        assert inv[1]["provider.kind"] == "anthropic"

    async def test_invocation_event_session_id(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_text_events())
        [e async for e in provider.call(_make_opts(session_id="sess-xyz"))]
        inv = next(e for e in mock_span.events if e[0] == "provider.invocation")
        assert inv[1]["session.id"] == "sess-xyz"

    async def test_span_ended_on_success(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_text_events())
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
        assert error_events[0][1]["provider.name"] == "test-anthropic"

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
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    async def test_returns_token_count(self) -> None:
        provider = _make_provider(count_tokens_result=SimpleNamespace(input_tokens=77))
        result = await provider.count_tokens(ModelCountReq(model="claude-sonnet-4-6", messages=[]))
        assert result.input_tokens == 77

    async def test_count_tokens_with_system_forwarded(self) -> None:
        captured: list[dict[str, Any]] = []

        async def _count(**kwargs: Any) -> Any:
            captured.append(kwargs)
            return SimpleNamespace(input_tokens=50)

        mock_messages: Any = MagicMock()
        mock_messages.count_tokens = _count
        mock_client: Any = MagicMock()
        mock_client.messages = mock_messages

        provider = AnthropicApiKeyProvider(_API_KEY, name="test-anthropic", _client=mock_client)
        await provider.count_tokens(
            ModelCountReq(model="claude-sonnet-4-6", messages=[], system="Be helpful.")
        )
        assert captured[0]["system"] == "Be helpful."

    async def test_count_tokens_with_tools_forwarded(self) -> None:
        captured: list[dict[str, Any]] = []

        async def _count(**kwargs: Any) -> Any:
            captured.append(kwargs)
            return SimpleNamespace(input_tokens=50)

        mock_messages: Any = MagicMock()
        mock_messages.count_tokens = _count
        mock_client: Any = MagicMock()
        mock_client.messages = mock_messages

        provider = AnthropicApiKeyProvider(_API_KEY, name="test-anthropic", _client=mock_client)
        await provider.count_tokens(
            ModelCountReq(
                model="claude-sonnet-4-6",
                messages=[],
                tools=[{"name": "fn", "description": "does X", "input_schema": {"type": "object"}}],
            )
        )
        assert "tools" in captured[0]


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_delegates_to_client(self) -> None:
        mock_client: Any = MagicMock()
        mock_client.close = AsyncMock(return_value=None)
        provider = AnthropicApiKeyProvider(_API_KEY, _client=mock_client)
        await provider.close()
        mock_client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# call() — per-call prompt-cache header injection
# ---------------------------------------------------------------------------


def _msg_start_with_cache(
    cache_creation: int = 0,
    cache_read: int = 0,
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 10,
) -> Any:
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            model=model,
            usage=SimpleNamespace(
                input_tokens=input_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
            ),
        ),
    )


class TestCacheControlInjection:
    async def test_system_wrapped_with_cache_control_list(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> Any:
                for e in _text_events():
                    yield e

            return _gen()

        mock_messages: Any = MagicMock()
        mock_messages.create = _create
        mock_client: Any = MagicMock()
        mock_client.messages = mock_messages

        provider = AnthropicApiKeyProvider(_API_KEY, name="test-anthropic", _client=mock_client)
        [e async for e in provider.call(_make_opts(system="You are helpful."))]
        sys = captured[0]["system"]
        assert isinstance(sys, list)
        assert len(sys) == 1
        assert sys[0]["type"] == "text"
        assert sys[0]["text"] == "You are helpful."
        assert sys[0]["cache_control"] == {"type": "ephemeral"}

    async def test_no_system_key_without_system_prompt(self, mock_span: MockSpan) -> None:
        captured: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> Any:
                for e in _text_events():
                    yield e

            return _gen()

        mock_messages: Any = MagicMock()
        mock_messages.create = _create
        mock_client: Any = MagicMock()
        mock_client.messages = mock_messages

        provider = AnthropicApiKeyProvider(_API_KEY, name="test-anthropic", _client=mock_client)
        [e async for e in provider.call(_make_opts())]
        assert "system" not in captured[0]

    async def test_count_tokens_system_not_wrapped(self) -> None:
        captured: list[dict[str, Any]] = []

        async def _count(**kwargs: Any) -> Any:
            captured.append(kwargs)
            return SimpleNamespace(input_tokens=50)

        mock_messages: Any = MagicMock()
        mock_messages.count_tokens = _count
        mock_client: Any = MagicMock()
        mock_client.messages = mock_messages

        provider = AnthropicApiKeyProvider(_API_KEY, name="test-anthropic", _client=mock_client)
        await provider.count_tokens(
            ModelCountReq(model="claude-sonnet-4-6", messages=[], system="Count tokens.")
        )
        assert captured[0]["system"] == "Count tokens."


# ---------------------------------------------------------------------------
# call() — cache hit/miss token extraction
# ---------------------------------------------------------------------------


class TestCacheMetrics:
    async def test_cache_creation_tokens_in_message_stop(self, mock_span: MockSpan) -> None:
        raw = [
            _msg_start_with_cache(cache_creation=150, cache_read=0),
            _block_start_text(),
            _text_delta("hi"),
            _block_stop(),
            _msg_delta(),
            _msg_stop(),
        ]
        provider = _make_provider(events=raw)
        events = [e async for e in provider.call(_make_opts())]
        stop = next(e for e in events if isinstance(e, MessageStopEvent))
        assert stop.cache_creation_input_tokens == 150
        assert stop.cache_read_input_tokens == 0

    async def test_cache_read_tokens_in_message_stop(self, mock_span: MockSpan) -> None:
        raw = [
            _msg_start_with_cache(cache_creation=0, cache_read=200),
            _block_start_text(),
            _text_delta("hi"),
            _block_stop(),
            _msg_delta(),
            _msg_stop(),
        ]
        provider = _make_provider(events=raw)
        events = [e async for e in provider.call(_make_opts())]
        stop = next(e for e in events if isinstance(e, MessageStopEvent))
        assert stop.cache_read_input_tokens == 200
        assert stop.cache_creation_input_tokens == 0

    async def test_cache_metrics_event_on_span(self, mock_span: MockSpan) -> None:
        raw = [
            _msg_start_with_cache(cache_creation=50, cache_read=100),
            _block_start_text(),
            _text_delta("hi"),
            _block_stop(),
            _msg_delta(),
            _msg_stop(),
        ]
        provider = _make_provider(events=raw)
        [e async for e in provider.call(_make_opts())]
        event_names = [e[0] for e in mock_span.events]
        assert "provider.cache_metrics" in event_names

    async def test_cache_metrics_hit_true_when_read_tokens_nonzero(
        self, mock_span: MockSpan
    ) -> None:
        raw = [
            _msg_start_with_cache(cache_creation=0, cache_read=300),
            _block_start_text(),
            _text_delta("hi"),
            _block_stop(),
            _msg_delta(),
            _msg_stop(),
        ]
        provider = _make_provider(events=raw)
        [e async for e in provider.call(_make_opts())]
        cm = next(e for e in mock_span.events if e[0] == "provider.cache_metrics")
        assert cm[1]["cache.hit"] is True
        assert cm[1]["cache.read_tokens"] == 300

    async def test_cache_metrics_hit_false_when_no_read_tokens(self, mock_span: MockSpan) -> None:
        raw = [
            _msg_start_with_cache(cache_creation=80, cache_read=0),
            _block_start_text(),
            _text_delta("hi"),
            _block_stop(),
            _msg_delta(),
            _msg_stop(),
        ]
        provider = _make_provider(events=raw)
        [e async for e in provider.call(_make_opts())]
        cm = next(e for e in mock_span.events if e[0] == "provider.cache_metrics")
        assert cm[1]["cache.hit"] is False
        assert cm[1]["cache.creation_tokens"] == 80

    async def test_zero_cache_tokens_when_absent_from_api(self, mock_span: MockSpan) -> None:
        provider = _make_provider(events=_text_events())
        events = [e async for e in provider.call(_make_opts())]
        stop = next(e for e in events if isinstance(e, MessageStopEvent))
        assert stop.cache_creation_input_tokens == 0
        assert stop.cache_read_input_tokens == 0


# ---------------------------------------------------------------------------
# _convert_message — block conversion
# ---------------------------------------------------------------------------


def _blk(**kw: Any) -> Any:
    return SimpleNamespace(**kw)


class TestConvertMessageBlocks:
    def test_string_content_passthrough(self) -> None:
        msg = SimpleNamespace(role="user", content="plain")
        assert _convert_message(msg) == {"role": "user", "content": "plain"}

    def test_text_block_without_cache_control(self) -> None:
        msg = SimpleNamespace(
            role="assistant",
            content=[_blk(type="text", text="hi", cache_control=None)],
        )
        out = _convert_message(msg)
        assert out["content"] == [{"type": "text", "text": "hi"}]

    def test_text_block_with_cache_control(self) -> None:
        msg = SimpleNamespace(
            role="assistant",
            content=[_blk(type="text", text="hi", cache_control=object())],
        )
        out = _convert_message(msg)
        assert out["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_tool_use_block(self) -> None:
        msg = SimpleNamespace(
            role="assistant",
            content=[_blk(type="tool_use", id="t1", name="fn", input={"a": 1})],
        )
        out = _convert_message(msg)
        assert out["content"] == [{"type": "tool_use", "id": "t1", "name": "fn", "input": {"a": 1}}]

    def test_tool_result_string_content(self) -> None:
        msg = SimpleNamespace(
            role="user",
            content=[_blk(type="tool_result", tool_use_id="t1", content="done")],
        )
        out = _convert_message(msg)
        assert out["content"] == [{"type": "tool_result", "tool_use_id": "t1", "content": "done"}]

    def test_tool_result_list_content(self) -> None:
        msg = SimpleNamespace(
            role="user",
            content=[_blk(type="tool_result", tool_use_id="t2", content=["a", "b"])],
        )
        out = _convert_message(msg)
        assert out["content"][0]["content"] == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_thinking_block(self) -> None:
        msg = SimpleNamespace(
            role="assistant",
            content=[_blk(type="thinking", thinking="reason", signature="sig")],
        )
        out = _convert_message(msg)
        assert out["content"] == [{"type": "thinking", "thinking": "reason", "signature": "sig"}]

    def test_unknown_block_type_skipped(self) -> None:
        msg = SimpleNamespace(
            role="user",
            content=[_blk(type="image", source="x"), _blk(type="text", text="ok")],
        )
        out = _convert_message(msg)
        assert out["content"] == [{"type": "text", "text": "ok"}]


# ---------------------------------------------------------------------------
# call() — stream-internal branches and generic-exception wrapping
# ---------------------------------------------------------------------------


class TestCallStreamBranches:
    async def test_generic_exception_wrapped_as_provider_call_error(
        self, mock_span: MockSpan
    ) -> None:
        audit = CollectingAuditLog()
        provider = _make_provider(error=RuntimeError("kaboom"), audit_log=audit)
        with pytest.raises(ProviderCallError) as exc_info:
            [e async for e in provider.call(_make_opts())]
        assert "kaboom" in str(exc_info.value)
        assert exc_info.value.provider_name == "test-anthropic"
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "anthropic.call.failed"

    async def test_provider_call_error_from_stream_reraised(self, mock_span: MockSpan) -> None:
        err = ProviderCallError("inner", provider_name="test-anthropic")
        provider = _make_provider(error=err)
        with pytest.raises(ProviderCallError) as exc_info:
            [e async for e in provider.call(_make_opts())]
        assert exc_info.value is err

    async def test_input_json_delta_without_known_tool_id_skipped(
        self, mock_span: MockSpan
    ) -> None:
        raw = [
            _msg_start(),
            _tool_delta(partial_json='{"x":1}', index=7),  # no block_start at index 7
            _msg_delta(),
            _msg_stop(),
        ]
        provider = _make_provider(events=raw)
        events = [e async for e in provider.call(_make_opts())]
        assert not any(isinstance(e, ToolInputDeltaEvent) for e in events)

    async def test_unknown_content_block_delta_type_ignored(self, mock_span: MockSpan) -> None:
        raw = [
            _msg_start(),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="signature_delta", signature="sig"),
            ),
            _msg_delta(),
            _msg_stop(),
        ]
        provider = _make_provider(events=raw)
        events = [e async for e in provider.call(_make_opts())]
        assert any(isinstance(e, MessageStopEvent) for e in events)


# ---------------------------------------------------------------------------
# count_tokens — error handling
# ---------------------------------------------------------------------------


def _make_count_provider(error: Exception) -> AnthropicApiKeyProvider:
    async def _count(**_kwargs: Any) -> Any:
        raise error

    mock_messages: Any = MagicMock()
    mock_messages.count_tokens = _count
    mock_client: Any = MagicMock()
    mock_client.messages = mock_messages
    return AnthropicApiKeyProvider(_API_KEY, name="test-anthropic", _client=mock_client)


class TestCountTokensErrors:
    async def test_timeout_raises_provider_timeout_error(self) -> None:
        provider = _make_count_provider(_timeout_error())
        with pytest.raises(ProviderTimeoutError):
            await provider.count_tokens(ModelCountReq(model="claude-sonnet-4-6", messages=[]))

    async def test_connection_error_raises_provider_call_error(self) -> None:
        provider = _make_count_provider(_connection_error())
        with pytest.raises(ProviderCallError):
            await provider.count_tokens(ModelCountReq(model="claude-sonnet-4-6", messages=[]))

    async def test_status_error_raises_rate_limit_error(self) -> None:
        provider = _make_count_provider(_rate_limit_error())
        with pytest.raises(ProviderRateLimitError):
            await provider.count_tokens(ModelCountReq(model="claude-sonnet-4-6", messages=[]))
