"""Shared fixtures: MockTracer/MockSpan for OTel, CapturingAuditLog, and app factory."""
from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI

from core_errors import install_error_handler
from meridian_sdk_provider import (
    FakeModelAdapter,
    ModelRouter,
    ModelRoutingPolicy,
    ModelRoutingRule,
)
from meridian_sdk_provider.protocol import ModelCapabilities, ModelEntry

from api_models._audit import AuditLog
from api_models._routes import make_router
from api_models._types import AuditLogEntry


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

    def __enter__(self) -> "MockSpan":
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
        return self.spans[-1]


@pytest.fixture()
def mock_tracer(monkeypatch: pytest.MonkeyPatch) -> MockTracer:
    tracer = MockTracer()
    monkeypatch.setattr("api_models._routes.get_tracer", lambda: tracer)
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

_TEST_MODELS = [
    ModelEntry(
        provider="anthropic",
        model="claude-opus-4-7",
        context_window=200000,
        capabilities=ModelCapabilities(
            streaming=True, thinking=True, vision=True, tools=True, cache=True
        ),
    ),
    ModelEntry(
        provider="anthropic",
        model="claude-haiku-4-5",
        context_window=200000,
        capabilities=ModelCapabilities(
            streaming=True, thinking=False, vision=True, tools=True, cache=True
        ),
    ),
]


def make_test_model_router(models: list[ModelEntry] | None = None) -> ModelRouter:
    adapter = FakeModelAdapter(
        name="anthropic",
        models=models if models is not None else _TEST_MODELS,
    )
    policy = ModelRoutingPolicy(rules=[ModelRoutingRule(model="anthropic:claude-opus-4-7")])
    router = ModelRouter(policy=policy)
    router.register_provider(adapter)
    return router


def make_app(
    model_router: ModelRouter | None = None,
    audit_log: AuditLog | None = None,
) -> FastAPI:
    app = FastAPI()
    install_error_handler(app)
    router = make_router(
        model_router=model_router or make_test_model_router(),
        audit_log=audit_log,
    )
    app.include_router(router)
    return app
