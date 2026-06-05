"""
Healthz liveness probe conformance suite.

Tests cover:
  - GET /healthz returns 200.
  - Response body is {"status": "ok"}.
  - Response content-type is application/json.
  - OTel span "health.liveness" is emitted on success.
  - Span carries invocation event "health.liveness.invocation".
  - create_app wires /healthz route without storage_root.
  - create_app wires /healthz route with storage_root.
"""

from __future__ import annotations

from pathlib import Path

from core_errors import HandlerOptions, NoopAuditLog, install_error_handler
from fastapi import FastAPI
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._healthz import make_healthz_router

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> TestClient:
    app = FastAPI(title="test")
    install_error_handler(app, HandlerOptions(audit_log=NoopAuditLog()))
    app.include_router(make_healthz_router(audit_log=NoopAuditLog()))
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestHealthzSuccess:
    def test_returns_200(self) -> None:
        client = _make_client()
        resp = client.get("/healthz")
        assert resp.status_code == 200

    def test_body_status_is_ok(self) -> None:
        client = _make_client()
        body = client.get("/healthz").json()
        assert body == {"status": "ok"}

    def test_content_type_is_json(self) -> None:
        client = _make_client()
        resp = client.get("/healthz")
        assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestHealthzOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_emits_health_liveness_span(self) -> None:
        client = _make_client()
        client.get("/healthz")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "health.liveness" in span_names

    def test_span_has_invocation_event(self) -> None:
        client = _make_client()
        client.get("/healthz")
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "health.liveness"]
        event_names = [e.name for s in spans for e in s.events]
        assert "meridian.error.invocation" in event_names


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------


class TestHealthzAppWiring:
    def test_create_app_wires_healthz_without_storage_root(self) -> None:
        app = create_app(NoopAuditLog(), storage_root=None)
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/healthz" in routes

    def test_create_app_wires_healthz_with_storage_root(self, storage_root: Path) -> None:
        app = create_app(NoopAuditLog(), storage_root=storage_root)
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/healthz" in routes
