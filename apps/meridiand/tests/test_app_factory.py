"""
Application factory conformance suite.

Tests cover:
  - create_app returns a FastAPI instance.
  - GZipMiddleware is always registered.
  - CORSMiddleware is not registered when cors is None or allow_origins is empty.
  - CORSMiddleware is registered when cors.allow_origins is non-empty.
  - CORS response headers are present for matching origins.
  - CORS response headers are absent when not configured.
  - OTel span "app.factory.create" emitted on factory invocation.
  - Span carries cors.enabled and gzip.enabled attributes.
  - Span carries "app.factory.create" structured event.
  - Factory failure writes audit log entry with event "app.factory.create.failed".
  - Factory failure re-raises the original exception to the caller.
  - Routes under /v1/ prefix exist when storage_root is supplied.
  - /v1/x/ routes absent (404) when storage_root is None.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.testclient import TestClient

from core_errors import AuditLog, AuditLogEntry, NoopAuditLog
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._config import CorsConfig

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CapturingAuditLog(AuditLog):
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


# ---------------------------------------------------------------------------
# TestCreateApp: basic factory behaviour
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_returns_fastapi_instance(self) -> None:
        app = create_app(NoopAuditLog())
        assert isinstance(app, FastAPI)

    def test_no_storage_root_returns_app(self) -> None:
        app = create_app(NoopAuditLog())
        assert app is not None

    def test_with_storage_root_returns_app(self, storage_root: Path) -> None:
        app = create_app(NoopAuditLog(), storage_root=storage_root)
        assert isinstance(app, FastAPI)


# ---------------------------------------------------------------------------
# TestGzipMiddleware
# ---------------------------------------------------------------------------


class TestGzipMiddleware:
    def test_gzip_middleware_always_registered(self) -> None:
        app = create_app(NoopAuditLog())
        assert any(m.cls is GZipMiddleware for m in app.user_middleware)

    def test_gzip_registered_even_without_cors(self) -> None:
        app = create_app(NoopAuditLog(), cors=None)
        assert any(m.cls is GZipMiddleware for m in app.user_middleware)

    def test_gzip_registered_with_cors(self) -> None:
        cors = CorsConfig(allow_origins=["http://localhost:3000"])
        app = create_app(NoopAuditLog(), cors=cors)
        assert any(m.cls is GZipMiddleware for m in app.user_middleware)


# ---------------------------------------------------------------------------
# TestCorsMiddleware
# ---------------------------------------------------------------------------


class TestCorsMiddleware:
    def test_cors_not_registered_by_default(self) -> None:
        app = create_app(NoopAuditLog())
        assert not any(m.cls is CORSMiddleware for m in app.user_middleware)

    def test_cors_not_registered_when_cors_is_none(self) -> None:
        app = create_app(NoopAuditLog(), cors=None)
        assert not any(m.cls is CORSMiddleware for m in app.user_middleware)

    def test_cors_not_registered_for_empty_origins(self) -> None:
        cors = CorsConfig(allow_origins=[])
        app = create_app(NoopAuditLog(), cors=cors)
        assert not any(m.cls is CORSMiddleware for m in app.user_middleware)

    def test_cors_registered_when_origins_configured(self) -> None:
        cors = CorsConfig(allow_origins=["http://localhost:3000"])
        app = create_app(NoopAuditLog(), cors=cors)
        assert any(m.cls is CORSMiddleware for m in app.user_middleware)

    def test_cors_headers_present_for_matching_origin(self) -> None:
        cors = CorsConfig(allow_origins=["http://localhost:3000"])
        app = create_app(NoopAuditLog(), cors=cors)

        @app.get("/v1/test-cors")
        def _endpoint():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/v1/test-cors", headers={"Origin": "http://localhost:3000"})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_no_cors_headers_when_not_configured(self) -> None:
        app = create_app(NoopAuditLog())

        @app.get("/v1/test-no-cors")
        def _endpoint():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/v1/test-no-cors", headers={"Origin": "http://localhost:3000"})
        assert "access-control-allow-origin" not in resp.headers

    def test_cors_preflight_returns_200(self) -> None:
        cors = CorsConfig(allow_origins=["http://localhost:3000"])
        app = create_app(NoopAuditLog(), cors=cors)
        client = TestClient(app)
        resp = client.options(
            "/v1/any",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_cors_non_matching_origin_gets_no_header(self) -> None:
        cors = CorsConfig(allow_origins=["http://allowed.example.com"])
        app = create_app(NoopAuditLog(), cors=cors)

        @app.get("/v1/test-mismatch")
        def _endpoint():
            return {}

        client = TestClient(app)
        resp = client.get("/v1/test-mismatch", headers={"Origin": "http://evil.example.com"})
        assert resp.headers.get("access-control-allow-origin") != "http://evil.example.com"


# ---------------------------------------------------------------------------
# TestOtelSpan
# ---------------------------------------------------------------------------


class TestOtelSpan:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_factory_emits_create_span(self) -> None:
        create_app(NoopAuditLog())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "app.factory.create" in span_names

    def test_span_has_cors_enabled_false_by_default(self) -> None:
        create_app(NoopAuditLog())
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "app.factory.create"
        )
        assert span.attributes["cors.enabled"] is False

    def test_span_has_cors_enabled_true_when_configured(self) -> None:
        cors = CorsConfig(allow_origins=["http://localhost:3000"])
        create_app(NoopAuditLog(), cors=cors)
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "app.factory.create"
        )
        assert span.attributes["cors.enabled"] is True

    def test_span_has_gzip_enabled_attribute(self) -> None:
        create_app(NoopAuditLog())
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "app.factory.create"
        )
        assert span.attributes["gzip.enabled"] is True

    def test_span_has_app_factory_create_event(self) -> None:
        create_app(NoopAuditLog())
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "app.factory.create"
        )
        event_names = [e.name for e in span.events]
        assert "app.factory.create" in event_names

    def test_create_event_includes_router_count(self) -> None:
        create_app(NoopAuditLog())
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "app.factory.create"
        )
        evt = next(e for e in span.events if e.name == "app.factory.create")
        assert "router_count" in evt.attributes

    def test_create_event_includes_cors_enabled(self) -> None:
        create_app(NoopAuditLog())
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "app.factory.create"
        )
        evt = next(e for e in span.events if e.name == "app.factory.create")
        assert "cors_enabled" in evt.attributes

    def test_create_event_includes_gzip_enabled(self) -> None:
        create_app(NoopAuditLog())
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "app.factory.create"
        )
        evt = next(e for e in span.events if e.name == "app.factory.create")
        assert evt.attributes["gzip_enabled"] is True


# ---------------------------------------------------------------------------
# TestFactoryFailure
# ---------------------------------------------------------------------------


class TestFactoryFailure:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_failure_reraises_exception(self, monkeypatch) -> None:
        from meridiand import _app

        def _explode(*args, **kwargs):
            raise RuntimeError("injected failure")

        monkeypatch.setattr(_app, "install_error_handler", _explode)
        with pytest.raises(RuntimeError, match="injected failure"):
            create_app(NoopAuditLog())

    def test_failure_writes_audit_log_entry(self, monkeypatch) -> None:
        from meridiand import _app

        def _explode(*args, **kwargs):
            raise RuntimeError("injected failure")

        monkeypatch.setattr(_app, "install_error_handler", _explode)
        audit = _CapturingAuditLog()
        with pytest.raises(RuntimeError):
            create_app(audit)
        assert any(e.event == "app.factory.create.failed" for e in audit.entries)

    def test_failure_audit_level_is_error(self, monkeypatch) -> None:
        from meridiand import _app

        def _explode(*args, **kwargs):
            raise RuntimeError("injected failure")

        monkeypatch.setattr(_app, "install_error_handler", _explode)
        audit = _CapturingAuditLog()
        with pytest.raises(RuntimeError):
            create_app(audit)
        entry = next(e for e in audit.entries if e.event == "app.factory.create.failed")
        assert entry.level == "error"

    def test_failure_audit_code_is_create_app_failed(self, monkeypatch) -> None:
        from meridiand import _app

        def _explode(*args, **kwargs):
            raise RuntimeError("injected failure")

        monkeypatch.setattr(_app, "install_error_handler", _explode)
        audit = _CapturingAuditLog()
        with pytest.raises(RuntimeError):
            create_app(audit)
        entry = next(e for e in audit.entries if e.event == "app.factory.create.failed")
        assert entry.code == "create_app_failed"

    def test_failure_audit_detail_has_error_type(self, monkeypatch) -> None:
        from meridiand import _app

        def _explode(*args, **kwargs):
            raise RuntimeError("injected failure")

        monkeypatch.setattr(_app, "install_error_handler", _explode)
        audit = _CapturingAuditLog()
        with pytest.raises(RuntimeError):
            create_app(audit)
        entry = next(e for e in audit.entries if e.event == "app.factory.create.failed")
        assert entry.detail["error_type"] == "RuntimeError"

    def test_failure_emits_error_span(self, monkeypatch) -> None:
        from opentelemetry.trace import StatusCode

        from meridiand import _app

        def _explode(*args, **kwargs):
            raise RuntimeError("injected failure")

        monkeypatch.setattr(_app, "install_error_handler", _explode)
        with pytest.raises(RuntimeError):
            create_app(NoopAuditLog())
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "app.factory.create"
        )
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# TestRouteWiring
# ---------------------------------------------------------------------------


class TestRouteWiring:
    def test_spawn_route_present_with_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/x/sessions/wiring-test/spawn",
            json={"parent_capabilities": ["agent.spawn"], "child_capabilities": []},
        )
        assert resp.status_code != 404

    def test_spawn_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/x/sessions/wiring-test/spawn",
            json={"parent_capabilities": ["agent.spawn"], "child_capabilities": []},
        )
        assert resp.status_code == 404

    def test_phase_route_absent_without_event_log(self, storage_root: Path) -> None:
        app = create_app(
            FileAuditLog(storage_root), storage_root=storage_root, event_log=None
        )
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(f"/v1/x/sessions/wiring-phase/phase")
        assert resp.status_code == 404
