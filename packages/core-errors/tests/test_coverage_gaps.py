"""Unit coverage for core-errors telemetry and base error http_status."""

from __future__ import annotations

from opentelemetry import trace

from core_errors._telemetry import get_tracer, record_invocation_event
from core_errors._types import MeridianError, StructuredEvent

from .conftest import MockSpan


def test_get_tracer_returns_tracer() -> None:
    assert isinstance(get_tracer(), trace.Tracer)


def test_meridian_error_default_http_status_is_500() -> None:
    err = MeridianError(code="x", message="m", timestamp="2026-01-01T00:00:00Z")
    assert err.http_status() == 500


def test_record_invocation_event_skips_non_scalar() -> None:
    span = MockSpan()
    # timestamp=None is non-scalar, exercising the isinstance-False branch that
    # filters it out of the emitted span attributes.
    event = StructuredEvent(name="meridian.error.invocation", code="X", timestamp=None)  # type: ignore[arg-type]
    record_invocation_event(span, event)  # type: ignore[arg-type]

    assert span.events
    _name, attrs = span.events[0]
    assert "timestamp" not in attrs
    assert attrs["code"] == "X"
