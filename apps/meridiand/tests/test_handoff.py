"""
Handoff endpoint conformance suite.

Tests cover:
  - POST /v1/x/sessions/{id}/handoff returns 200 with no output_schema (unconditional accept).
  - POST /v1/x/sessions/{id}/handoff returns 200 when terminal_message satisfies schema.
  - Response has session_id, parent_session_id, status="completed", validation="passed".
  - Manifest status updated to "completed" on success.
  - Returns 422 with code "handoff_schema_invalid" when schema doesn't match (first attempt).
  - error.retry_allowed=True on first schema failure.
  - error.validation_errors is a non-empty list on failure.
  - Manifest status updated to "waiting_for_user" on first schema failure.
  - Returns 422 with retry_allowed=False when schema fails on second attempt (retry exhausted).
  - Manifest stays "waiting_for_user" after retry is exhausted.
  - Second attempt with valid message → 200 (completed).
  - Returns 404 with code "handoff_session_not_found" when session manifest not found.
  - Audit log written on schema failure (event="handoff.schema_invalid").
  - Audit log written on session not found (event="handoff.validate.failed").
  - Audit detail includes session_id, parent_session_id, validation_errors, retry_allowed.
  - OTel span "handoff.validate" emitted on every invocation (success and failure).
  - Span has session.id attribute.
  - Span set to ERROR status on failure.
  - Route exists when storage_root is supplied.
  - Route omitted when storage_root is None.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA = {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"]}
_VALID_MSG = {"result": "done"}
_INVALID_MSG = {"result": 42}  # result must be string


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _write_manifest(
    storage_root: Path,
    session_id: str,
    *,
    output_schema: dict | None = None,
    status: str = "spawned",
    parent_session_id: str = "parent-1",
) -> None:
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "child_session_id": session_id,
        "parent_session_id": parent_session_id,
        "capabilities": ["exec.shell"],
        "output_schema": output_schema,
        "created_at": "2026-05-18T00:00:00+00:00",
        "status": status,
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest))


def _read_manifest(storage_root: Path, session_id: str) -> dict:
    path = storage_root / "sessions" / session_id / "manifest.json"
    return json.loads(path.read_text())


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Success: no output_schema
# ---------------------------------------------------------------------------


class TestHandoffNoSchema:
    def test_returns_200_without_output_schema(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "no-schema-1")
        client = _make_client(storage_root)
        resp = client.post("/v1/x/sessions/no-schema-1/handoff", json={"terminal_message": "anything"})
        assert resp.status_code == 200

    def test_response_status_completed(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "no-schema-2")
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/no-schema-2/handoff", json={"terminal_message": None}).json()
        assert body["status"] == "completed"

    def test_response_validation_passed(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "no-schema-3")
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/no-schema-3/handoff", json={"terminal_message": {}}).json()
        assert body["validation"] == "passed"

    def test_manifest_status_updated_to_completed(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "no-schema-4")
        client = _make_client(storage_root)
        client.post("/v1/x/sessions/no-schema-4/handoff", json={"terminal_message": "x"})
        assert _read_manifest(storage_root, "no-schema-4")["status"] == "completed"

    def test_response_has_session_id(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "no-schema-5")
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/no-schema-5/handoff", json={"terminal_message": 1}).json()
        assert body["session_id"] == "no-schema-5"

    def test_response_has_parent_session_id(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "no-schema-6", parent_session_id="p-42")
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/no-schema-6/handoff", json={"terminal_message": 1}).json()
        assert body["parent_session_id"] == "p-42"


# ---------------------------------------------------------------------------
# Success: output_schema matches
# ---------------------------------------------------------------------------


class TestHandoffSchemaPass:
    def test_returns_200_when_schema_matches(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "pass-1", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        resp = client.post("/v1/x/sessions/pass-1/handoff", json={"terminal_message": _VALID_MSG})
        assert resp.status_code == 200

    def test_manifest_status_completed_on_schema_pass(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "pass-2", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        client.post("/v1/x/sessions/pass-2/handoff", json={"terminal_message": _VALID_MSG})
        assert _read_manifest(storage_root, "pass-2")["status"] == "completed"

    def test_response_validation_passed_on_schema_match(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "pass-3", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/pass-3/handoff", json={"terminal_message": _VALID_MSG}).json()
        assert body["validation"] == "passed"


# ---------------------------------------------------------------------------
# First schema failure → waiting_for_user, retry_allowed=True
# ---------------------------------------------------------------------------


class TestHandoffFirstFailure:
    def test_returns_422_on_schema_mismatch(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "fail-1", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        resp = client.post("/v1/x/sessions/fail-1/handoff", json={"terminal_message": _INVALID_MSG})
        assert resp.status_code == 422

    def test_error_code_handoff_schema_invalid(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "fail-2", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/fail-2/handoff", json={"terminal_message": _INVALID_MSG}).json()
        assert body["error"]["code"] == "handoff_schema_invalid"

    def test_retry_allowed_true_on_first_failure(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "fail-3", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/fail-3/handoff", json={"terminal_message": _INVALID_MSG}).json()
        assert body["error"]["retry_allowed"] is True

    def test_validation_errors_present_on_failure(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "fail-4", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/fail-4/handoff", json={"terminal_message": _INVALID_MSG}).json()
        assert isinstance(body["error"]["validation_errors"], list)
        assert len(body["error"]["validation_errors"]) > 0

    def test_manifest_status_waiting_for_user_on_first_failure(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "fail-5", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        client.post("/v1/x/sessions/fail-5/handoff", json={"terminal_message": _INVALID_MSG})
        assert _read_manifest(storage_root, "fail-5")["status"] == "waiting_for_user"

    def test_error_message_mentions_session_id(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "fail-6", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/fail-6/handoff", json={"terminal_message": _INVALID_MSG}).json()
        assert "fail-6" in body["error"]["message"]


# ---------------------------------------------------------------------------
# Retry exhausted: second failure → retry_allowed=False
# ---------------------------------------------------------------------------


class TestHandoffRetryExhausted:
    def test_returns_422_on_second_failure(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "retry-1", output_schema=_SCHEMA, status="waiting_for_user")
        client = _make_client(storage_root)
        resp = client.post("/v1/x/sessions/retry-1/handoff", json={"terminal_message": _INVALID_MSG})
        assert resp.status_code == 422

    def test_retry_allowed_false_on_second_failure(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "retry-2", output_schema=_SCHEMA, status="waiting_for_user")
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/retry-2/handoff", json={"terminal_message": _INVALID_MSG}).json()
        assert body["error"]["retry_allowed"] is False

    def test_manifest_stays_waiting_for_user_after_exhausted_retry(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "retry-3", output_schema=_SCHEMA, status="waiting_for_user")
        client = _make_client(storage_root)
        client.post("/v1/x/sessions/retry-3/handoff", json={"terminal_message": _INVALID_MSG})
        assert _read_manifest(storage_root, "retry-3")["status"] == "waiting_for_user"

    def test_second_attempt_with_valid_message_succeeds(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "retry-4", output_schema=_SCHEMA, status="waiting_for_user")
        client = _make_client(storage_root)
        resp = client.post("/v1/x/sessions/retry-4/handoff", json={"terminal_message": _VALID_MSG})
        assert resp.status_code == 200

    def test_manifest_completed_after_successful_retry(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "retry-5", output_schema=_SCHEMA, status="waiting_for_user")
        client = _make_client(storage_root)
        client.post("/v1/x/sessions/retry-5/handoff", json={"terminal_message": _VALID_MSG})
        assert _read_manifest(storage_root, "retry-5")["status"] == "completed"


# ---------------------------------------------------------------------------
# Session not found
# ---------------------------------------------------------------------------


class TestHandoffSessionNotFound:
    def test_returns_404_when_no_manifest(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/sessions/ghost-1/handoff", json={"terminal_message": "x"})
        assert resp.status_code == 404

    def test_error_code_handoff_session_not_found(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/ghost-2/handoff", json={"terminal_message": "x"}).json()
        assert body["error"]["code"] == "handoff_session_not_found"

    def test_error_message_mentions_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/ghost-3/handoff", json={"terminal_message": "x"}).json()
        assert "ghost-3" in body["error"]["message"]


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestHandoffAuditLog:
    def test_schema_failure_writes_audit_log(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "audit-1", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        client.post("/v1/x/sessions/audit-1/handoff", json={"terminal_message": _INVALID_MSG})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "handoff.schema_invalid" for r in records)

    def test_schema_failure_audit_level_error(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "audit-2", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        client.post("/v1/x/sessions/audit-2/handoff", json={"terminal_message": _INVALID_MSG})
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "handoff.schema_invalid")
        assert record["level"] == "error"

    def test_schema_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "audit-3", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        client.post("/v1/x/sessions/audit-3/handoff", json={"terminal_message": _INVALID_MSG})
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "handoff.schema_invalid")
        assert record["detail"]["session_id"] == "audit-3"

    def test_schema_failure_audit_detail_has_parent_session_id(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "audit-4", output_schema=_SCHEMA, parent_session_id="p-audit")
        client = _make_client(storage_root)
        client.post("/v1/x/sessions/audit-4/handoff", json={"terminal_message": _INVALID_MSG})
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "handoff.schema_invalid")
        assert record["detail"]["parent_session_id"] == "p-audit"

    def test_schema_failure_audit_detail_has_validation_errors(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "audit-5", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        client.post("/v1/x/sessions/audit-5/handoff", json={"terminal_message": _INVALID_MSG})
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "handoff.schema_invalid")
        assert isinstance(record["detail"]["validation_errors"], list)
        assert len(record["detail"]["validation_errors"]) > 0

    def test_schema_failure_audit_detail_retry_allowed_true(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "audit-6", output_schema=_SCHEMA)
        client = _make_client(storage_root)
        client.post("/v1/x/sessions/audit-6/handoff", json={"terminal_message": _INVALID_MSG})
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "handoff.schema_invalid")
        assert record["detail"]["retry_allowed"] is True

    def test_schema_failure_audit_detail_retry_allowed_false_on_retry(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "audit-7", output_schema=_SCHEMA, status="waiting_for_user")
        client = _make_client(storage_root)
        client.post("/v1/x/sessions/audit-7/handoff", json={"terminal_message": _INVALID_MSG})
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "handoff.schema_invalid")
        assert record["detail"]["retry_allowed"] is False

    def test_session_not_found_writes_audit_log(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/sessions/audit-ghost/handoff", json={"terminal_message": "x"})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "handoff.validate.failed" for r in records)


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestHandoffOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _make_client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_handoff_validate_span(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "otel-1")
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/otel-1/handoff", json={"terminal_message": "ok"})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "handoff.validate" in span_names

    def test_failure_emits_handoff_validate_span(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "otel-2", output_schema=_SCHEMA)
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/otel-2/handoff", json={"terminal_message": _INVALID_MSG})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "handoff.validate" in span_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "otel-3")
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/otel-3/handoff", json={"terminal_message": "ok"})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        assert spans["handoff.validate"].attributes["session.id"] == "otel-3"

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        _write_manifest(storage_root, "otel-4", output_schema=_SCHEMA)
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/otel-4/handoff", json={"terminal_message": _INVALID_MSG})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        handoff_spans = [s for name, s in spans.items() if name == "handoff.validate"]
        assert any(s.status.status_code == StatusCode.ERROR for s in handoff_spans)

    def test_session_not_found_emits_span(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/otel-5/handoff", json={"terminal_message": "x"})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "handoff.validate" in span_names


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestHandoffRouteWiring:
    def test_route_exists_with_storage_root(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "wire-1")
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/wire-1/handoff", json={"terminal_message": "x"})
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/any/handoff", json={"terminal_message": "x"})
        assert resp.status_code == 404


# FastAPI TestClient must be importable here
from fastapi.testclient import TestClient  # noqa: E402
