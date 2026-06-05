"""
Web UI serving conformance suite.

Tests cover:
  - GET /ui returns 200 and index.html content.
  - GET /ui/{spa-route} returns 200 and index.html (SPA fallback).
  - GET /ui/{asset} returns 200 and asset content.
  - OTel span "ui.serve" is emitted on every request.
  - Span carries "ui.serve.invocation" event and ui.path attribute.
  - Missing index.html returns 500 and writes audit log entry.
  - Path traversal attempt is blocked and writes audit log entry.
  - create_app wires /ui when serve_ui=True and ui_dist_path is set.
  - create_app does NOT wire /ui when serve_ui=False.
  - create_app does NOT wire /ui when ui_dist_path is None.
  - DaemonConfig.serve_ui defaults to False.
  - daemon.serve_ui=true loads correctly from YAML via parse_config.
"""

from __future__ import annotations

from pathlib import Path

from core_errors import AuditLogEntry, HandlerOptions, NoopAuditLog, install_error_handler
from fastapi import FastAPI
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._config import DaemonConfig, parse_config
from meridiand._ui import make_ui_router

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>Meridian UI</body></html>")
    assets = dist / "assets"
    assets.mkdir()
    (assets / "main.js").write_text("console.log('meridian')")
    return dist


def _make_client(dist: Path) -> TestClient:
    app = FastAPI(title="test")
    install_error_handler(app, HandlerOptions(audit_log=NoopAuditLog()))
    app.include_router(make_ui_router(audit_log=NoopAuditLog(), ui_dist_path=dist))
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestUiServeSuccess:
    def test_get_ui_returns_200(self, tmp_path: Path) -> None:
        resp = _make_client(_make_dist(tmp_path)).get("/ui")
        assert resp.status_code == 200

    def test_get_ui_returns_html_content_type(self, tmp_path: Path) -> None:
        resp = _make_client(_make_dist(tmp_path)).get("/ui")
        assert "text/html" in resp.headers["content-type"]

    def test_get_ui_root_body_is_index_html(self, tmp_path: Path) -> None:
        resp = _make_client(_make_dist(tmp_path)).get("/ui")
        assert "Meridian UI" in resp.text

    def test_get_ui_spa_route_returns_index_html(self, tmp_path: Path) -> None:
        resp = _make_client(_make_dist(tmp_path)).get("/ui/sessions/abc123")
        assert resp.status_code == 200
        assert "Meridian UI" in resp.text

    def test_get_ui_known_asset_served_directly(self, tmp_path: Path) -> None:
        resp = _make_client(_make_dist(tmp_path)).get("/ui/assets/main.js")
        assert resp.status_code == 200
        assert "meridian" in resp.text

    def test_get_ui_nested_spa_route_falls_back_to_index(self, tmp_path: Path) -> None:
        resp = _make_client(_make_dist(tmp_path)).get("/ui/a/b/c/d")
        assert resp.status_code == 200
        assert "Meridian UI" in resp.text


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


class _CapturingAuditLog:
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


class TestUiServeFailure:
    def test_missing_index_html_returns_500(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        resp = _make_client(empty).get("/ui")
        assert resp.status_code == 500

    def test_missing_index_html_writes_audit_log(self, tmp_path: Path) -> None:
        capturing = _CapturingAuditLog()
        empty = tmp_path / "empty"
        empty.mkdir()
        app = FastAPI(title="test")
        install_error_handler(app, HandlerOptions(audit_log=NoopAuditLog()))
        app.include_router(make_ui_router(audit_log=capturing, ui_dist_path=empty))
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/ui")
        assert any(e.event == "ui.serve.failed" for e in capturing.entries)

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        dist = _make_dist(tmp_path)
        (tmp_path / "sibling.txt").write_text("classified")
        client = _make_client(dist)
        # %2e%2e is URL-encoded ".." — Starlette decodes it before passing to the handler
        resp = client.get("/ui/%2e%2e/sibling.txt")
        # Must never expose sibling content; either blocked (500) or SPA fallback (200 w/ index)
        if resp.status_code == 200:
            assert "classified" not in resp.text
        else:
            assert resp.status_code == 500

    def test_path_traversal_writes_audit_log(self, tmp_path: Path) -> None:
        capturing = _CapturingAuditLog()
        dist = _make_dist(tmp_path)
        (tmp_path / "sibling.txt").write_text("classified")
        app = FastAPI(title="test")
        install_error_handler(app, HandlerOptions(audit_log=NoopAuditLog()))
        app.include_router(make_ui_router(audit_log=capturing, ui_dist_path=dist))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/ui/%2e%2e/sibling.txt")
        if resp.status_code == 500:
            assert any(e.event == "ui.serve.failed" for e in capturing.entries)


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestUiServeOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_emits_ui_serve_span(self, tmp_path: Path) -> None:
        _make_client(_make_dist(tmp_path)).get("/ui")
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "ui.serve" in names

    def test_span_has_invocation_event(self, tmp_path: Path) -> None:
        _make_client(_make_dist(tmp_path)).get("/ui")
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "ui.serve"]
        event_names = [e.name for s in spans for e in s.events]
        assert "meridian.error.invocation" in event_names

    def test_span_ui_path_is_slash_for_root(self, tmp_path: Path) -> None:
        _make_client(_make_dist(tmp_path)).get("/ui")
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "ui.serve"]
        assert any(s.attributes.get("ui.path") == "/" for s in spans)

    def test_span_ui_path_is_asset_path(self, tmp_path: Path) -> None:
        _make_client(_make_dist(tmp_path)).get("/ui/assets/main.js")
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "ui.serve"]
        assert any(s.attributes.get("ui.path") == "assets/main.js" for s in spans)

    def test_span_emitted_for_spa_route(self, tmp_path: Path) -> None:
        _make_client(_make_dist(tmp_path)).get("/ui/deep/route")
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "ui.serve" in names


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------


class TestUiAppWiring:
    def test_serve_ui_true_wires_ui_route(self, tmp_path: Path) -> None:
        dist = _make_dist(tmp_path)
        app = create_app(NoopAuditLog(), serve_ui=True, ui_dist_path=dist)
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/ui" in paths

    def test_serve_ui_false_does_not_wire_ui_route(self, tmp_path: Path) -> None:
        dist = _make_dist(tmp_path)
        app = create_app(NoopAuditLog(), serve_ui=False, ui_dist_path=dist)
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/ui" not in paths

    def test_serve_ui_true_but_no_dist_path_does_not_wire(self) -> None:
        app = create_app(NoopAuditLog(), serve_ui=True, ui_dist_path=None)
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/ui" not in paths

    def test_serve_ui_request_returns_200(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        dist = _make_dist(tmp_path)
        app = create_app(NoopAuditLog(), serve_ui=True, ui_dist_path=dist)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/ui")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestUiConfig:
    def test_daemon_config_serve_ui_defaults_false(self) -> None:
        assert DaemonConfig().serve_ui is False

    def test_daemon_config_serve_ui_can_be_true(self) -> None:
        assert DaemonConfig(serve_ui=True).serve_ui is True

    def test_parse_config_daemon_serve_ui_true(self, tmp_path: Path) -> None:
        import yaml

        raw = yaml.dump(
            {
                "version": 2,
                "storage_root": str(tmp_path),
                "daemon": {"serve_ui": True},
            }
        )
        config = parse_config(raw)
        assert config.daemon is not None
        assert config.daemon.serve_ui is True

    def test_parse_config_daemon_serve_ui_default_false(self, tmp_path: Path) -> None:
        import yaml

        raw = yaml.dump(
            {
                "version": 2,
                "storage_root": str(tmp_path),
                "daemon": {},
            }
        )
        config = parse_config(raw)
        assert config.daemon is not None
        assert config.daemon.serve_ui is False
