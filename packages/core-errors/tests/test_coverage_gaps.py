"""Unit coverage for core-errors telemetry and base error http_status."""

from __future__ import annotations

from opentelemetry import trace

from core_errors._telemetry import get_tracer
from core_errors._types import MeridianError


def test_get_tracer_returns_tracer() -> None:
    assert isinstance(get_tracer(), trace.Tracer)


def test_meridian_error_default_http_status_is_500() -> None:
    err = MeridianError(code="x", message="m", timestamp="2026-01-01T00:00:00Z")
    assert err.http_status() == 500
