"""Unit tests for the api-models telemetry helpers."""

from __future__ import annotations

from opentelemetry import trace

from api_models._telemetry import get_tracer


def test_get_tracer_returns_tracer() -> None:
    tracer = get_tracer()
    assert isinstance(tracer, trace.Tracer)
