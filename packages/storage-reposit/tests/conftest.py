"""
Shared test fixtures for the storage-reposit conformance suite.

OTel is mocked with a lightweight MockSpan / MockTracer pair.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import sqlite3
from typing import Any

import pytest
from storage_event_log import SessionEvent
from storage_reposit._audit import AuditLog
from storage_reposit._types import AuditLogEntry

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
    monkeypatch.setattr("storage_reposit._runtime.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_span(mock_tracer: MockTracer) -> MockSpan:
    return mock_tracer.span


@pytest.fixture()
def mock_reader_tracer(monkeypatch: pytest.MonkeyPatch) -> MockTracer:
    tracer = MockTracer()
    monkeypatch.setattr("storage_reposit._reader_runtime.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_reader_span(mock_reader_tracer: MockTracer) -> MockSpan:
    return mock_reader_tracer.span


@pytest.fixture()
def mock_migration_tracer(monkeypatch: pytest.MonkeyPatch) -> MockTracer:
    tracer = MockTracer()
    monkeypatch.setattr("storage_reposit._migration_runtime.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_migration_span(mock_migration_tracer: MockTracer) -> MockSpan:
    return mock_migration_tracer.span


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


# ---------------------------------------------------------------------------
# Event handler stubs
# ---------------------------------------------------------------------------


class CapturingEventHandler:
    """Records every (session_id, event) pair passed to handle()."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def handle(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        event: Any,
    ) -> None:
        self.calls.append({"session_id": session_id, "event": event})


class FailingEventHandler:
    """Raises on the Nth call (1-indexed)."""

    def __init__(self, fail_on: int = 1, exc: Exception | None = None) -> None:
        self._fail_on = fail_on
        self._exc = exc or RuntimeError("handler error")
        self._count = 0

    async def handle(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        event: Any,
    ) -> None:
        self._count += 1
        if self._count == self._fail_on:
            raise self._exc


# ---------------------------------------------------------------------------
# Reader stubs
# ---------------------------------------------------------------------------


class StubReader:
    """LocalEventLogReader substitute for ReaderRuntime tests."""

    def __init__(
        self,
        *,
        raises: Exception | None = None,
        events: list[SessionEvent] | None = None,
    ) -> None:
        self._raises = raises
        self._events = events or []

    async def read_events(
        self,
        session_id: str,
        since: int = -1,
        *,
        follow: bool = False,
        **_: Any,
    ) -> AsyncIterator[SessionEvent]:
        if self._raises:
            raise self._raises
        for event in self._events:
            yield event


# ---------------------------------------------------------------------------
# Store stubs
# ---------------------------------------------------------------------------


class StubStore:
    """SQLiteProjectionStore substitute for MigrationRuntime tests."""

    def __init__(
        self,
        *,
        raises: Exception | None = None,
        returns: int = 0,
    ) -> None:
        self._raises = raises
        self._returns = returns

    def migrate(self) -> int:
        if self._raises:
            raise self._raises
        return self._returns


# ---------------------------------------------------------------------------
# Phase projection stubs
# ---------------------------------------------------------------------------


class StubPhaseProjection:
    """PhaseProjection substitute for PhaseProjectionRuntime tests."""

    def __init__(
        self,
        *,
        raises: Exception | None = None,
        returns: str = "created",
    ) -> None:
        self._raises = raises
        self._returns = returns

    def current_phase(self, session_id: str) -> str:
        if self._raises:
            raise self._raises
        return self._returns


@pytest.fixture()
def mock_phase_tracer(monkeypatch: pytest.MonkeyPatch) -> MockTracer:
    tracer = MockTracer()
    monkeypatch.setattr("storage_reposit._phase.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_phase_span(mock_phase_tracer: MockTracer) -> MockSpan:
    return mock_phase_tracer.span


# ---------------------------------------------------------------------------
# Phase state machine stubs
# ---------------------------------------------------------------------------


class StubPhaseStateMachine:
    """PhaseStateMachine substitute for PhaseStateMachineRuntime tests."""

    def __init__(
        self,
        *,
        raises: Exception | None = None,
        returns: str = "running",
    ) -> None:
        self._raises = raises
        self._returns = returns

    def next_phase(self, current: str, event_type: str) -> str:
        if self._raises:
            raise self._raises
        return self._returns


@pytest.fixture()
def mock_state_machine_tracer(monkeypatch: pytest.MonkeyPatch) -> MockTracer:
    tracer = MockTracer()
    monkeypatch.setattr("storage_reposit._state_machine.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_state_machine_span(mock_state_machine_tracer: MockTracer) -> MockSpan:
    return mock_state_machine_tracer.span
