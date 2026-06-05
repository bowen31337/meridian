"""Unit coverage for system-ulid telemetry, noop audit, and overflow guard."""

from __future__ import annotations

import pytest
from opentelemetry import trace

from system_ulid._audit import NoopAuditLog
from system_ulid._generator import MonotonicUlidGenerator, _time_ms
from system_ulid._telemetry import get_tracer
from system_ulid._types import AuditLogEntry


def test_get_tracer_returns_tracer() -> None:
    assert isinstance(get_tracer(), trace.Tracer)


def test_noop_audit_log_write_is_silent() -> None:
    entry = AuditLogEntry(
        level="error",
        event="test.event",
        prefix="ulid",
        timestamp="2026-01-01T00:00:00Z",
    )
    assert NoopAuditLog().write(entry) is None


def test_generate_raises_on_random_overflow() -> None:
    gen = MonotonicUlidGenerator()
    # Force the same-millisecond path with the random component at its ceiling.
    gen._last_ms = _time_ms() + 10_000
    gen._last_rand = (1 << 80) - 1
    with pytest.raises(OverflowError):
        gen.generate()
