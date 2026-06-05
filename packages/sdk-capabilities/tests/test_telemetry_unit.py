"""Unit coverage for the sdk-capabilities telemetry helper."""

from __future__ import annotations

from opentelemetry import trace

from sdk_capabilities._telemetry import get_tracer


def test_get_tracer_returns_tracer() -> None:
    assert isinstance(get_tracer(), trace.Tracer)
