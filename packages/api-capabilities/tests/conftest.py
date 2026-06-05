"""Shared fixtures: MockTracer/MockSpan for OTel, CapturingAuditLog, and app factory."""

from __future__ import annotations

from typing import Any

from api_capabilities._audit import AuditLog
from api_capabilities._registry import CapabilityRegistry
from api_capabilities._routes import make_router
from api_capabilities._types import AuditLogEntry
from core_errors import install_error_handler
from fastapi import FastAPI
import pytest

# ---------------------------------------------------------------------------
# OTel mock
# ---------------------------------------------------------------------------


class MockSpan:
    def __init__(self, name: str = "", attributes: dict[str, Any] | None = None) -> None:
        self.name = name
        self.attributes: dict[str, Any] = dict(attributes or {})
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
        self.spans: list[MockSpan] = []

    def start_as_current_span(
        self, name: str, *, attributes: dict[str, Any] | None = None, **_: Any
    ) -> MockSpan:
        span = MockSpan(name=name, attributes=attributes)
        self.spans.append(span)
        return span

    @property
    def span(self) -> MockSpan:
        """Return the most recently created span."""
        return self.spans[-1]


@pytest.fixture()
def mock_tracer(monkeypatch: pytest.MonkeyPatch) -> MockTracer:
    tracer = MockTracer()
    monkeypatch.setattr("api_capabilities._routes.get_tracer", lambda: tracer)
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


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def make_app(
    audit_log: AuditLog | None = None,
    registry: CapabilityRegistry | None = None,
) -> FastAPI:
    app = FastAPI()
    install_error_handler(app)
    router = make_router(registry=registry, audit_log=audit_log)
    app.include_router(router)
    return app
