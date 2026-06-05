"""
OpenAPI export endpoint conformance suite.

Tests cover:
  - GET /v1/openapi returns 200 on success.
  - Response content-type is application/yaml.
  - Response body is valid YAML.
  - Parsed YAML contains 'openapi', 'info', and 'paths' keys.
  - When dest_path is configured, file is written on success.
  - Written file contains valid YAML with correct 'openapi' version field.
  - When dest_path parent does not exist, it is created automatically.
  - When dest_path is not configured, no file is written.
  - On write failure (unwritable dest), returns 500 with code "openapi_export_failed".
  - On failure, audit log entry written with event "openapi.export.failed".
  - Audit entry level is "error" on failure.
  - Audit entry code is "openapi_export_failed" on failure.
  - Audit detail includes "message" on failure.
  - Error response body has error.code and error.message on failure.
  - OTel span "openapi.export" emitted on success.
  - OTel span "openapi.export" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries openapi.dest attribute.
  - create_app always wires openapi_export route (no storage_root required).
"""

from __future__ import annotations

import json
from pathlib import Path

from core_errors import HandlerOptions, NoopAuditLog, install_error_handler
from fastapi import FastAPI
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._openapi_export import make_openapi_export_router
import yaml

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    audit_log: FileAuditLog | None = None,
    dest_path: Path | None = None,
    storage_root: Path | None = None,
) -> TestClient:
    _audit = audit_log or NoopAuditLog()
    if storage_root is not None:
        app = create_app(_audit, storage_root=storage_root)
    else:
        app = FastAPI(title="test")
        install_error_handler(app, HandlerOptions(audit_log=_audit))
        app.include_router(make_openapi_export_router(audit_log=_audit, dest_path=dest_path))
    return TestClient(app, raise_server_exceptions=False)


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestOpenApiExportSuccess:
    def test_returns_200(self, tmp_path: Path) -> None:
        client = _make_client()
        resp = client.get("/v1/openapi")
        assert resp.status_code == 200

    def test_content_type_is_yaml(self, tmp_path: Path) -> None:
        client = _make_client()
        resp = client.get("/v1/openapi")
        assert "application/yaml" in resp.headers["content-type"]

    def test_body_is_valid_yaml(self, tmp_path: Path) -> None:
        client = _make_client()
        resp = client.get("/v1/openapi")
        parsed = yaml.safe_load(resp.content)
        assert isinstance(parsed, dict)

    def test_yaml_has_openapi_key(self, tmp_path: Path) -> None:
        client = _make_client()
        parsed = yaml.safe_load(client.get("/v1/openapi").content)
        assert "openapi" in parsed

    def test_yaml_has_info_key(self, tmp_path: Path) -> None:
        client = _make_client()
        parsed = yaml.safe_load(client.get("/v1/openapi").content)
        assert "info" in parsed

    def test_yaml_has_paths_key(self, tmp_path: Path) -> None:
        client = _make_client()
        parsed = yaml.safe_load(client.get("/v1/openapi").content)
        assert "paths" in parsed


# ---------------------------------------------------------------------------
# Dest path writing
# ---------------------------------------------------------------------------


class TestOpenApiExportDestPath:
    def test_writes_file_when_dest_configured(self, tmp_path: Path) -> None:
        dest = tmp_path / "openapi.yaml"
        client = _make_client(dest_path=dest)
        client.get("/v1/openapi")
        assert dest.exists()

    def test_written_file_is_valid_yaml(self, tmp_path: Path) -> None:
        dest = tmp_path / "openapi.yaml"
        client = _make_client(dest_path=dest)
        client.get("/v1/openapi")
        parsed = yaml.safe_load(dest.read_text())
        assert isinstance(parsed, dict)

    def test_written_yaml_has_openapi_version(self, tmp_path: Path) -> None:
        dest = tmp_path / "openapi.yaml"
        client = _make_client(dest_path=dest)
        client.get("/v1/openapi")
        parsed = yaml.safe_load(dest.read_text())
        assert "openapi" in parsed
        assert isinstance(parsed["openapi"], str)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        dest = tmp_path / "nested" / "dir" / "openapi.yaml"
        client = _make_client(dest_path=dest)
        client.get("/v1/openapi")
        assert dest.exists()

    def test_no_file_written_without_dest(self, tmp_path: Path) -> None:
        client = _make_client()
        client.get("/v1/openapi")
        assert not any(tmp_path.iterdir()) if tmp_path.exists() else True


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


class TestOpenApiExportFailure:
    def test_unwritable_dest_returns_500(self, tmp_path: Path) -> None:
        # Point dest at an existing directory so write_text fails
        dest = tmp_path  # directories are not writable as files
        audit_log = FileAuditLog(tmp_path / "audit_root")
        client = _make_client(audit_log=audit_log, dest_path=dest)
        resp = client.get("/v1/openapi")
        assert resp.status_code == 500

    def test_failure_error_code(self, tmp_path: Path) -> None:
        dest = tmp_path
        audit_log = FileAuditLog(tmp_path / "audit_root")
        client = _make_client(audit_log=audit_log, dest_path=dest)
        body = client.get("/v1/openapi").json()
        assert body["error"]["code"] == "openapi_export_failed"

    def test_failure_error_has_message(self, tmp_path: Path) -> None:
        dest = tmp_path
        audit_log = FileAuditLog(tmp_path / "audit_root")
        client = _make_client(audit_log=audit_log, dest_path=dest)
        body = client.get("/v1/openapi").json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestOpenApiExportAuditLog:
    def test_failure_writes_audit_entry(self, tmp_path: Path) -> None:
        dest = tmp_path
        audit_root = tmp_path / "audit_root"
        audit_log = FileAuditLog(audit_root)
        client = _make_client(audit_log=audit_log, dest_path=dest)
        client.get("/v1/openapi")
        records = _audit_records(audit_root)
        assert any(r.get("event") == "openapi.export.failed" for r in records)

    def test_failure_audit_level_is_error(self, tmp_path: Path) -> None:
        dest = tmp_path
        audit_root = tmp_path / "audit_root"
        audit_log = FileAuditLog(audit_root)
        client = _make_client(audit_log=audit_log, dest_path=dest)
        client.get("/v1/openapi")
        record = next(
            r for r in _audit_records(audit_root) if r.get("event") == "openapi.export.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_code(self, tmp_path: Path) -> None:
        dest = tmp_path
        audit_root = tmp_path / "audit_root"
        audit_log = FileAuditLog(audit_root)
        client = _make_client(audit_log=audit_log, dest_path=dest)
        client.get("/v1/openapi")
        record = next(
            r for r in _audit_records(audit_root) if r.get("event") == "openapi.export.failed"
        )
        assert record["code"] == "openapi_export_failed"

    def test_failure_audit_detail_has_message(self, tmp_path: Path) -> None:
        dest = tmp_path
        audit_root = tmp_path / "audit_root"
        audit_log = FileAuditLog(audit_root)
        client = _make_client(audit_log=audit_log, dest_path=dest)
        client.get("/v1/openapi")
        record = next(
            r for r in _audit_records(audit_root) if r.get("event") == "openapi.export.failed"
        )
        assert "message" in record.get("detail", {})
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestOpenApiExportOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_openapi_export_span(self, tmp_path: Path) -> None:
        client = _make_client()
        client.get("/v1/openapi")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "openapi.export" in span_names

    def test_failure_emits_openapi_export_span(self, tmp_path: Path) -> None:
        dest = tmp_path
        audit_log = FileAuditLog(tmp_path / "audit_root")
        client = _make_client(audit_log=audit_log, dest_path=dest)
        client.get("/v1/openapi")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "openapi.export" in span_names

    def test_failure_span_has_error_status(self, tmp_path: Path) -> None:
        from opentelemetry.trace import StatusCode

        dest = tmp_path
        audit_log = FileAuditLog(tmp_path / "audit_root")
        client = _make_client(audit_log=audit_log, dest_path=dest)
        client.get("/v1/openapi")
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "openapi.export"]
        assert any(s.status.status_code == StatusCode.ERROR for s in spans)

    def test_success_span_has_dest_attribute(self, tmp_path: Path) -> None:
        dest = tmp_path / "openapi.yaml"
        client = _make_client(dest_path=dest)
        client.get("/v1/openapi")
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "openapi.export"]
        assert any("openapi.dest" in s.attributes for s in spans)


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------


class TestOpenApiExportAppWiring:
    def test_create_app_wires_openapi_route_without_storage_root(self) -> None:
        app = create_app(NoopAuditLog(), storage_root=None)
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/v1/openapi" in routes

    def test_create_app_wires_openapi_route_with_storage_root(self, storage_root: Path) -> None:
        app = create_app(NoopAuditLog(), storage_root=storage_root)
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/v1/openapi" in routes
