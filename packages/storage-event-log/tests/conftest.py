"""
Shared test fixtures for the event-log conformance suite.

OTel is mocked with a lightweight MockSpan / MockTracer pair that captures
span names, attributes, events, status, and recorded exceptions without
requiring any SDK configuration.
"""
from __future__ import annotations

from typing import Any

import pytest

from storage_event_log._audit import AuditLog
from storage_event_log._types import AuditLogEntry


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

    def __enter__(self) -> "MockSpan":
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
    monkeypatch.setattr("storage_event_log._runtime.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_span(mock_tracer: MockTracer) -> MockSpan:
    return mock_tracer.span


# ---------------------------------------------------------------------------
# Audit log capture
# ---------------------------------------------------------------------------

class CapturingAuditLog(AuditLog):
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


@pytest.fixture()
def audit_log() -> CapturingAuditLog:
    return CapturingAuditLog()
