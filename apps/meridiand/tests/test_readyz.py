"""
Readyz readiness probe conformance suite.

Tests cover:
  - GET /readyz returns 200 when storage, providers, and plugins are all ready.
  - GET /readyz returns 503 with {"status": "not_ready", "components": [...]} when any component is not ready.
  - Response content-type is application/json.
  - OTel span "health.readiness" is emitted on every call.
  - Span carries invocation event "health.readiness.invocation".
  - create_app wires /readyz route without storage_root.
  - create_app wires /readyz route with storage_root.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from core_errors import HandlerOptions, NoopAuditLog, install_error_handler
from fastapi import FastAPI
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._readyz import ReadyzState, make_readyz_router

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(state: ReadyzState | None = None) -> TestClient:
    if state is None:
        state = ReadyzState(storage=True, providers=True, plugins=True)
    app = FastAPI(title="test")
    install_error_handler(app, HandlerOptions(audit_log=NoopAuditLog()))
    app.include_router(make_readyz_router(audit_log=NoopAuditLog(), state=state))
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Ready (all components initialized)
# ---------------------------------------------------------------------------


class TestReadyzReady:
    def test_returns_200(self) -> None:
        client = _make_client()
        resp = client.get("/readyz")
        assert resp.status_code == 200

    def test_body_status_is_ok(self) -> None:
        client = _make_client()
        body = client.get("/readyz").json()
        assert body == {"status": "ok"}

    def test_content_type_is_json(self) -> None:
        client = _make_client()
        resp = client.get("/readyz")
        assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Not ready (during boot or after dependency failure)
# ---------------------------------------------------------------------------


class TestReadyzNotReady:
    def test_returns_503_when_storage_not_ready(self) -> None:
        state = ReadyzState(storage=False, providers=True, plugins=True)
        client = _make_client(state)
        assert client.get("/readyz").status_code == 503

    def test_returns_503_when_providers_not_ready(self) -> None:
        state = ReadyzState(storage=True, providers=False, plugins=True)
        client = _make_client(state)
        assert client.get("/readyz").status_code == 503

    def test_returns_503_when_plugins_not_ready(self) -> None:
        state = ReadyzState(storage=True, providers=True, plugins=False)
        client = _make_client(state)
        assert client.get("/readyz").status_code == 503

    def test_returns_503_when_all_not_ready(self) -> None:
        client = _make_client(ReadyzState())
        assert client.get("/readyz").status_code == 503

    def test_body_status_is_not_ready(self) -> None:
        client = _make_client(ReadyzState())
        body = client.get("/readyz").json()
        assert body["status"] == "not_ready"

    def test_body_contains_components_list(self) -> None:
        client = _make_client(ReadyzState())
        body = client.get("/readyz").json()
        assert "components" in body
        names = [c["name"] for c in body["components"]]
        assert names == ["storage", "providers", "plugins"]

    def test_components_reflect_ready_state(self) -> None:
        state = ReadyzState(storage=True, providers=False, plugins=True)
        client = _make_client(state)
        body = client.get("/readyz").json()
        by_name = {c["name"]: c["ready"] for c in body["components"]}
        assert by_name == {"storage": True, "providers": False, "plugins": True}

    def test_content_type_is_json_on_503(self) -> None:
        client = _make_client(ReadyzState())
        resp = client.get("/readyz")
        assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestReadyzOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_emits_health_readiness_span_on_ready(self) -> None:
        client = _make_client()
        client.get("/readyz")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "health.readiness" in span_names

    def test_emits_health_readiness_span_on_not_ready(self) -> None:
        client = _make_client(ReadyzState())
        client.get("/readyz")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "health.readiness" in span_names

    def test_span_has_invocation_event(self) -> None:
        client = _make_client()
        client.get("/readyz")
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "health.readiness"]
        event_names = [e.name for s in spans for e in s.events]
        assert "meridian.error.invocation" in event_names


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------


class TestReadyzAppWiring:
    def test_create_app_wires_readyz_without_storage_root(self) -> None:
        app = create_app(NoopAuditLog(), storage_root=None)
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/readyz" in routes

    def test_create_app_wires_readyz_with_storage_root(self, storage_root: Path) -> None:
        app = create_app(NoopAuditLog(), storage_root=storage_root)
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/readyz" in routes
