"""Unit coverage for sdk-provider message conversion, error wrap/re-raise
branches, routing-condition leaf comparisons, and list/close fan-out paths."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from meridian_sdk_provider import (
    ModelCallOpts,
    ModelCapabilities,
    ModelEntry,
    ModelEvent,
    ModelRouter,
    ModelRoutingPolicy,
    ModelRoutingRule,
    NoProviderFoundError,
    ProviderCallError,
    ProviderCapabilities,
    ProviderRegistry,
    RoutingCondition,
    RoutingError,
    TokenRange,
)
from meridian_sdk_provider import ollama as ollama_mod
from meridian_sdk_provider import openai as openai_mod
from meridian_sdk_provider import openrouter as openrouter_mod
from meridian_sdk_provider.fake import FakeModelAdapter
from meridian_sdk_provider.ollama import OllamaProvider
from meridian_sdk_provider.openai import OpenAIProvider
from meridian_sdk_provider.openrouter import OpenRouterProvider
from meridian_sdk_provider.router import _condition_matches, _parse_model_ref
from meridian_sdk_provider.telemetry import record_cache_metrics
from meridian_sdk_provider.types import (
    CacheControl,
    MessageStartEvent,
    MessageStopEvent,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from tests.conftest import FakeProvider, make_opts


class _Msg:
    def __init__(self, role: str, content: Any) -> None:
        self.role = role
        self.content = content


def _list_content_messages() -> list[_Msg]:
    return [
        _Msg("assistant", [TextBlock(text="hi"), ToolUseBlock(id="t1", name="f", input={"a": 1})]),
        _Msg("tool", [ToolResultBlock(tool_use_id="t1", content="done")]),
    ]


def _tool() -> ToolDefinition:
    return ToolDefinition(name="f", description="d", input_schema={"type": "object"})


# ---------------------------------------------------------------------------
# Telemetry + fake adapter leaf paths
# ---------------------------------------------------------------------------


def test_record_cache_metrics() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    class _Span:
        def add_event(self, name: str, attrs: dict[str, Any]) -> None:
            events.append((name, attrs))

    record_cache_metrics(_Span(), cache_creation_input_tokens=3, cache_read_input_tokens=5)
    assert events[0][0] == "provider.cache_metrics"
    assert events[0][1]["cache.hit"] is True


def test_fake_adapter_list_models() -> None:
    entry = ModelEntry(
        provider="fake", model="m", context_window=100, capabilities=ModelCapabilities()
    )
    assert FakeModelAdapter(models=[entry]).list_models() == [entry]


# ---------------------------------------------------------------------------
# Message conversion: list-content branches across all three adapters
# ---------------------------------------------------------------------------


def test_ollama_convert_message_and_tools() -> None:
    converted = [ollama_mod._convert_message(m) for m in _list_content_messages()]
    assert converted[0]["tool_calls"][0]["function"]["name"] == "f"
    assert converted[1] == {"role": "tool", "content": "done"}
    assert ollama_mod._convert_tools([_tool()])[0]["function"]["name"] == "f"


def test_openai_convert_message_list_content() -> None:
    converted = [openai_mod._convert_message(m) for m in _list_content_messages()]
    assert converted[0]["role"] == "assistant"
    assert converted[0]["tool_calls"][0]["id"] == "t1"
    assert converted[1] == {"role": "tool", "tool_call_id": "t1", "content": "done"}
    text_only = openai_mod._convert_message(_Msg("user", [TextBlock(text="just text")]))
    assert text_only == {"role": "user", "content": "just text"}


def test_openrouter_convert_message_with_cache_control() -> None:
    msg = _Msg("user", [TextBlock(text="cached", cache_control=CacheControl())])
    out = openrouter_mod._convert_message(msg)
    assert out["content"][0]["cache_control"] == {"type": "ephemeral"}
    tool_msgs = [openrouter_mod._convert_message(m) for m in _list_content_messages()]
    assert tool_msgs[0]["tool_calls"][0]["id"] == "t1"
    assert openrouter_mod._convert_tools([_tool()])[0]["function"]["name"] == "f"
    text_only = openrouter_mod._convert_message(_Msg("user", [TextBlock(text="plain")]))
    assert text_only == {"role": "user", "content": "plain"}


# ---------------------------------------------------------------------------
# HTTP transport helper (ollama / openrouter)
# ---------------------------------------------------------------------------


class _Transport(httpx.AsyncBaseTransport):
    def __init__(self, path: str, response: httpx.Response) -> None:
        self._path = path
        self._response = response

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == self._path:
            return self._response
        return httpx.Response(404, text="not found")


def _ollama(body: bytes, status: int = 200) -> OllamaProvider:
    transport = _Transport("/api/chat", httpx.Response(status, content=body))
    client = httpx.AsyncClient(transport=transport, base_url="http://localhost:11434")
    return OllamaProvider(name="t", _http=client)


def _openrouter(body: bytes, status: int = 200) -> OpenRouterProvider:
    transport = _Transport("/api/v1/chat/completions", httpx.Response(status, content=body))
    client = httpx.AsyncClient(transport=transport, base_url="https://openrouter.ai/api/v1")
    return OpenRouterProvider("key", name="t", _http=client)


async def test_ollama_stream_with_tools_and_blank_line() -> None:
    body = (
        b'{"message":{"content":"hi"},"done":false}\n'
        b"\n"
        b'{"message":{"content":""},"done":true,"done_reason":"stop"}\n'
    )
    provider = _ollama(body)
    opts = ModelCallOpts(
        model="llama3.2",
        messages=[{"role": "user", "content": "hi"}],
        tools=[_tool()],
    )
    events = [e async for e in provider.call(opts)]
    assert any(isinstance(e, MessageStopEvent) for e in events)


async def test_ollama_generic_exception_wrapped() -> None:
    provider = _ollama(b"not json\n")
    with pytest.raises(ProviderCallError):
        _ = [e async for e in provider.call(make_opts(model="llama3.2"))]


async def test_openrouter_stream_with_tools() -> None:
    body = (
        b'data: {"model":"m","choices":[{"delta":{"content":"hi"},"index":0}]}\n\ndata: [DONE]\n\n'
    )
    provider = _openrouter(body)
    opts = ModelCallOpts(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[_tool()],
    )
    events = [e async for e in provider.call(opts)]
    assert any(isinstance(e, MessageStartEvent) for e in events)


async def test_openrouter_generic_exception_wrapped() -> None:
    provider = _openrouter(b"data: not-json\n\n")
    with pytest.raises(ProviderCallError):
        _ = [e async for e in provider.call(make_opts(model="openai/gpt-4o"))]


# ---------------------------------------------------------------------------
# OpenAI: generic wrap + ProviderCallError re-raise
# ---------------------------------------------------------------------------


def _openai_with_error(error: Exception) -> OpenAIProvider:
    from unittest.mock import AsyncMock, MagicMock

    async def _create(**_kwargs: Any) -> Any:
        raise error

    client: Any = MagicMock()
    client.chat.completions.create = _create
    client.close = AsyncMock(return_value=None)
    return OpenAIProvider("key", name="t", _client=client)


async def test_openai_generic_exception_wrapped() -> None:
    provider = _openai_with_error(ValueError("boom"))
    with pytest.raises(ProviderCallError):
        _ = [e async for e in provider.call(make_opts(model="gpt-4o"))]


async def test_openai_provider_call_error_reraised() -> None:
    provider = _openai_with_error(ProviderCallError("nope", provider_name="t"))
    with pytest.raises(ProviderCallError):
        _ = [e async for e in provider.call(make_opts(model="gpt-4o"))]


# ---------------------------------------------------------------------------
# Router: parse + routing-condition leaf comparisons
# ---------------------------------------------------------------------------


def test_parse_model_ref_requires_colon() -> None:
    with pytest.raises(RoutingError):
        _parse_model_ref("noColon")


@pytest.mark.parametrize(
    ("token_range", "tokens"),
    [
        (TokenRange(gt=5), None),  # estimated tokens None
        (TokenRange(gte=10), 5),  # gte not met
        (TokenRange(lt=10), 20),  # lt not met
        (TokenRange(lte=10), 20),  # lte not met
    ],
)
def test_condition_token_range_mismatches(token_range: TokenRange, tokens: int | None) -> None:
    cond = RoutingCondition(estimated_input_tokens=token_range)
    opts = make_opts(estimated_input_tokens=tokens)
    assert _condition_matches(cond, opts) is False


# ---------------------------------------------------------------------------
# Router: resolve/registry/list/close fan-out
# ---------------------------------------------------------------------------


def _policy(model: str) -> ModelRoutingPolicy:
    return ModelRoutingPolicy(rules=[ModelRoutingRule(model=model)], fallbacks=[])


def test_router_set_policy_and_registry_property() -> None:
    registry = ProviderRegistry()
    router = ModelRouter(policy=_policy("p:m"), registry=registry)
    assert router.registry is registry
    router.set_policy(_policy("q:n"))


async def test_router_registry_missing_slot_raises() -> None:
    router = ModelRouter(policy=_policy("missing:m"), registry=ProviderRegistry())
    with pytest.raises(NoProviderFoundError):
        _ = [e async for e in router.call(make_opts(model="missing:m"))]


async def test_router_plain_dict_missing_provider_raises() -> None:
    router = ModelRouter(policy=_policy("missing:m"), providers={})
    with pytest.raises(NoProviderFoundError):
        _ = [e async for e in router.call(make_opts(model="missing:m"))]


class _EmptyProvider:
    name = "p"
    kind = "fake"

    def __init__(self) -> None:
        self.capabilities = ProviderCapabilities()

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        for _ in ():
            yield MessageStartEvent(type="message_start", model="m", provider="p")


async def test_router_empty_stream_returns_nothing() -> None:
    router = ModelRouter(policy=_policy("p:m"), providers={"p": _EmptyProvider()})
    events = [e async for e in router.call(make_opts(model="p:m"))]
    assert events == []


class _FailAfterFirstProvider:
    name = "p"
    kind = "fake"

    def __init__(self) -> None:
        self.capabilities = ProviderCapabilities()

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        yield MessageStartEvent(type="message_start", model="m", provider="p")
        raise RuntimeError("mid-stream boom")


async def test_router_error_after_first_event_reraises() -> None:
    router = ModelRouter(policy=_policy("p:m"), providers={"p": _FailAfterFirstProvider()})
    with pytest.raises(RuntimeError):
        _ = [e async for e in router.call(make_opts(model="p:m"))]


def test_router_list_models_registry_and_plain() -> None:
    registry = ProviderRegistry(providers={"p": FakeProvider(name="p")})
    assert ModelRouter(policy=_policy("p:m"), registry=registry).list_models() == []
    assert (
        ModelRouter(policy=_policy("p:m"), providers={"p": FakeProvider(name="p")}).list_models()
        == []
    )


async def test_router_close_plain_providers() -> None:
    provider = FakeProvider(name="p")
    await ModelRouter(policy=_policy("p:m"), providers={"p": provider}).close()
    assert provider.closed is True


# ---------------------------------------------------------------------------
# Registry: swap_all surfaces failure
# ---------------------------------------------------------------------------


async def test_registry_swap_all_records_failure() -> None:
    from unittest.mock import AsyncMock

    registry = ProviderRegistry()
    registry._atomic_replace_all = AsyncMock(side_effect=RuntimeError("swap boom"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await registry.swap_all({"p": FakeProvider(name="p")})
