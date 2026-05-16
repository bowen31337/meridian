"""Tests for OTel span creation, invocation event, and failure recording."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from meridian_sdk_provider import (
    FallbackRule,
    ModelRouter,
    ModelRoutingPolicy,
    ModelRoutingRule,
    ProviderRateLimitError,
)
from meridian_sdk_provider._version import SDK_PROVIDER_VERSION
from meridian_sdk_provider.telemetry import TRACER_NAME
from tests.conftest import FakeProvider, make_opts


@pytest.fixture
def span_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(TRACER_NAME, SDK_PROVIDER_VERSION)
    # Patch get_tracer so each test gets a fresh tracer backed by its own exporter
    # rather than fighting the OTel global singleton.
    with patch("meridian_sdk_provider.telemetry.get_tracer", return_value=tracer), \
         patch("meridian_sdk_provider.router.get_tracer", return_value=tracer):
        yield exporter
    exporter.clear()


def _get_spans(exporter: InMemorySpanExporter) -> list:
    return list(exporter.get_finished_spans())


def _get_provider_call_spans(exporter: InMemorySpanExporter) -> list:
    return [s for s in _get_spans(exporter) if s.name == "provider.call"]


# ─── Span lifecycle ───────────────────────────────────────────────────────────


async def test_span_created_and_ended_on_success(span_exporter: InMemorySpanExporter) -> None:
    provider = FakeProvider(name="p")
    router = ModelRouter(
        policy=ModelRoutingPolicy(rules=[ModelRoutingRule(model="p:m")]),
        providers={"p": provider},
    )
    async for _ in router.call(make_opts()):
        pass

    spans = _get_provider_call_spans(span_exporter)
    assert len(spans) == 1
    assert spans[0].status.status_code.name == "UNSET"


async def test_span_ends_on_pre_stream_failure(span_exporter: InMemorySpanExporter) -> None:
    provider = FakeProvider(
        name="p", raise_on_call=ProviderRateLimitError("rate limited", "p")
    )
    router = ModelRouter(
        policy=ModelRoutingPolicy(rules=[ModelRoutingRule(model="p:m")]),
        providers={"p": provider},
    )
    with pytest.raises(ProviderRateLimitError):
        async for _ in router.call(make_opts()):
            pass

    spans = _get_provider_call_spans(span_exporter)
    assert len(spans) == 1
    assert spans[0].status.status_code.name == "ERROR"


# ─── Invocation event ─────────────────────────────────────────────────────────


async def test_invocation_event_recorded(span_exporter: InMemorySpanExporter) -> None:
    provider = FakeProvider(name="p", kind="fake")
    router = ModelRouter(
        policy=ModelRoutingPolicy(rules=[ModelRoutingRule(model="p:m")]),
        providers={"p": provider},
    )
    async for _ in router.call(make_opts(session_id="sess-abc")):
        pass

    spans = _get_provider_call_spans(span_exporter)
    events = [e for e in spans[0].events if e.name == "provider.invocation"]
    assert len(events) == 1
    attrs = events[0].attributes
    assert attrs["provider.name"] == "p"
    assert attrs["provider.kind"] == "fake"
    assert attrs["model"] == "m"
    assert attrs["session.id"] == "sess-abc"


async def test_fallback_invocation_event_recorded(span_exporter: InMemorySpanExporter) -> None:
    primary = FakeProvider(
        name="primary", raise_on_call=ProviderRateLimitError("rate limited", "primary")
    )
    fallback = FakeProvider(name="fallback")
    router = ModelRouter(
        policy=ModelRoutingPolicy(
            rules=[ModelRoutingRule(model="primary:m")],
            fallbacks=[FallbackRule(on="rate_limit", model="fallback:m")],
        ),
        providers={"primary": primary, "fallback": fallback},
    )
    async for _ in router.call(make_opts()):
        pass

    spans = _get_provider_call_spans(span_exporter)
    invocation_events = [e for e in spans[0].events if e.name == "provider.invocation"]
    assert len(invocation_events) == 2

    primary_ev = next(e for e in invocation_events if e.attributes["provider.name"] == "primary")
    fallback_ev = next(e for e in invocation_events if e.attributes["provider.name"] == "fallback")
    assert "fallback" in fallback_ev.attributes["routing.rule"]
    assert primary_ev.attributes["routing.rule"] == "primary:m"


# ─── Error event ──────────────────────────────────────────────────────────────


async def test_error_event_recorded_on_failure(span_exporter: InMemorySpanExporter) -> None:
    provider = FakeProvider(
        name="p", raise_on_call=ProviderRateLimitError("rate limited", "p")
    )
    router = ModelRouter(
        policy=ModelRoutingPolicy(rules=[ModelRoutingRule(model="p:m")]),
        providers={"p": provider},
    )
    with pytest.raises(ProviderRateLimitError):
        async for _ in router.call(make_opts()):
            pass

    spans = _get_provider_call_spans(span_exporter)
    error_events = [e for e in spans[0].events if e.name == "provider.error"]
    assert len(error_events) >= 1
    attrs = error_events[0].attributes
    assert attrs["provider.name"] == "p"
    assert "ProviderRateLimitError" in attrs["error.type"]


# ─── Span attributes ──────────────────────────────────────────────────────────


async def test_span_carries_routing_rule(span_exporter: InMemorySpanExporter) -> None:
    provider = FakeProvider(name="p")
    router = ModelRouter(
        policy=ModelRoutingPolicy(rules=[ModelRoutingRule(model="p:specific")]),
        providers={"p": provider},
    )
    async for _ in router.call(make_opts()):
        pass

    spans = _get_provider_call_spans(span_exporter)
    assert spans[0].attributes["routing.rule"] == "p:specific"


async def test_span_carries_session_id(span_exporter: InMemorySpanExporter) -> None:
    provider = FakeProvider(name="p")
    router = ModelRouter(
        policy=ModelRoutingPolicy(rules=[ModelRoutingRule(model="p:m")]),
        providers={"p": provider},
    )
    async for _ in router.call(make_opts(session_id="test-session")):
        pass

    spans = _get_provider_call_spans(span_exporter)
    assert spans[0].attributes.get("session.id") == "test-session"
