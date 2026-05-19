"""
Lifecycle hooks endpoint conformance suite.

Tests cover:
  - POST /v1/x/hooks returns 201 on success.
  - Response fields: id, event, name, handler, match, timeout_ms, failure_mode, status, created_at.
  - id has "hook_" prefix.
  - IDs are unique across calls.
  - status is always "active" on creation.
  - All five handler types accepted: in_process, subprocess, mcp, http, container.
  - All three failure_mode values accepted: ignore, warn, abort.
  - match is null when omitted; stored when provided.
  - metadata is null when omitted; stored when provided.
  - secret_reads is null when omitted; stored when provided.
  - secret_reads list is persisted in the hook resource JSON.
  - Empty event returns 422 with code "hook_invalid_request".
  - Empty name returns 422 with code "hook_invalid_request".
  - Zero timeout_ms returns 422 with code "hook_invalid_request".
  - Negative timeout_ms returns 422 with code "hook_invalid_request".
  - Missing required fields (event, name, handler, timeout_ms, failure_mode) return 422.
  - Invalid handler value returns 422.
  - Invalid failure_mode value returns 422.
  - Hook resource JSON written to storage_root/hooks/{id}.json.
  - Persisted resource has correct event, name, handler, timeout_ms, failure_mode, status.
  - Not written to disk on validation failure.
  - On validation failure, audit log entry written with event "hook.create.failed".
  - Audit entry level is "error" on failure.
  - Audit entry code is "hook_invalid_request" on validation failure.
  - Audit detail includes hook_id, event, name, message on failure.
  - Error response body has error.code and error.message on failure.
  - OTel span "hook.create" emitted on success.
  - OTel span "hook.create" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries hook.id, hook.event, hook.name, hook.handler attributes.
  - create_app wires hooks router when storage_root is supplied.
  - create_app omits hooks route when storage_root is None.
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


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _body(**overrides) -> dict:
    base: dict = {
        "event": "tool_call.requested",
        "name": "my-hook",
        "handler": "in_process",
        "timeout_ms": 5000,
        "failure_mode": "ignore",
    }
    base.update(overrides)
    return base


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _hook_resource(storage_root: Path, hook_id: str) -> dict:
    path = storage_root / "hooks" / f"{hook_id}.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestHookCreateSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body())
        assert resp.status_code == 201

    def test_in_process_handler_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(handler="in_process"))
        assert resp.status_code == 201

    def test_subprocess_handler_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(handler="subprocess"))
        assert resp.status_code == 201

    def test_mcp_handler_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(handler="mcp"))
        assert resp.status_code == 201

    def test_http_handler_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(handler="http"))
        assert resp.status_code == 201

    def test_container_handler_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(handler="container"))
        assert resp.status_code == 201

    def test_failure_mode_ignore_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(failure_mode="ignore"))
        assert resp.status_code == 201

    def test_failure_mode_warn_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(failure_mode="warn"))
        assert resp.status_code == 201

    def test_failure_mode_abort_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(failure_mode="abort"))
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Response fields
# ---------------------------------------------------------------------------


class TestHookCreateResponse:
    def test_response_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body()).json()
        assert "id" in body
        assert isinstance(body["id"], str)

    def test_id_has_hook_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body()).json()
        assert body["id"].startswith("hook_")

    def test_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = client.post("/v1/x/hooks", json=_body()).json()["id"]
        id2 = client.post("/v1/x/hooks", json=_body()).json()["id"]
        assert id1 != id2

    def test_response_has_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body(event="session.created")).json()
        assert body["event"] == "session.created"

    def test_response_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body(name="pre-tool-hook")).json()
        assert body["name"] == "pre-tool-hook"

    def test_response_has_handler(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body(handler="subprocess")).json()
        assert body["handler"] == "subprocess"

    def test_response_has_timeout_ms(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body(timeout_ms=3000)).json()
        assert body["timeout_ms"] == 3000

    def test_response_has_failure_mode(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body(failure_mode="abort")).json()
        assert body["failure_mode"] == "abort"

    def test_response_status_is_active(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body()).json()
        assert body["status"] == "active"

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body()).json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0

    def test_match_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body()).json()
        assert body["match"] is None

    def test_match_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/hooks",
            json=_body(match={"session_id": "sess-abc", "agent_id": "agent-xyz"}),
        ).json()
        assert body["match"]["session_id"] == "sess-abc"
        assert body["match"]["agent_id"] == "agent-xyz"

    def test_metadata_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body()).json()
        assert body["metadata"] is None

    def test_metadata_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        meta = {"owner": "platform", "version": "1"}
        body = client.post("/v1/x/hooks", json=_body(metadata=meta)).json()
        assert body["metadata"] == meta


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestHookCreateValidation:
    def test_missing_event_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["event"]
        resp = client.post("/v1/x/hooks", json=payload)
        assert resp.status_code == 422

    def test_missing_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["name"]
        resp = client.post("/v1/x/hooks", json=payload)
        assert resp.status_code == 422

    def test_missing_handler_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["handler"]
        resp = client.post("/v1/x/hooks", json=payload)
        assert resp.status_code == 422

    def test_missing_timeout_ms_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["timeout_ms"]
        resp = client.post("/v1/x/hooks", json=payload)
        assert resp.status_code == 422

    def test_missing_failure_mode_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["failure_mode"]
        resp = client.post("/v1/x/hooks", json=payload)
        assert resp.status_code == 422

    def test_invalid_handler_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(handler="grpc"))
        assert resp.status_code == 422

    def test_invalid_failure_mode_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(failure_mode="crash"))
        assert resp.status_code == 422

    def test_empty_event_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(event="   "))
        assert resp.status_code == 422

    def test_empty_event_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body(event="   ")).json()
        assert body["error"]["code"] == "hook_invalid_request"

    def test_empty_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(name="   "))
        assert resp.status_code == 422

    def test_empty_name_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body(name="   ")).json()
        assert body["error"]["code"] == "hook_invalid_request"

    def test_zero_timeout_ms_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(timeout_ms=0))
        assert resp.status_code == 422

    def test_zero_timeout_ms_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body(timeout_ms=0)).json()
        assert body["error"]["code"] == "hook_invalid_request"

    def test_negative_timeout_ms_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body(timeout_ms=-1))
        assert resp.status_code == 422

    def test_negative_timeout_ms_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body(timeout_ms=-1)).json()
        assert body["error"]["code"] == "hook_invalid_request"

    def test_validation_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body(event="")).json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestHookPersistence:
    def test_hook_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        hook_id = client.post("/v1/x/hooks", json=_body()).json()["id"]
        assert (storage_root / "hooks" / f"{hook_id}.json").exists()

    def test_persisted_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        hook_id = client.post("/v1/x/hooks", json=_body(event="session.created")).json()["id"]
        resource = _hook_resource(storage_root, hook_id)
        assert resource["event"] == "session.created"

    def test_persisted_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        hook_id = client.post("/v1/x/hooks", json=_body(name="persist-hook")).json()["id"]
        resource = _hook_resource(storage_root, hook_id)
        assert resource["name"] == "persist-hook"

    def test_persisted_handler(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        hook_id = client.post("/v1/x/hooks", json=_body(handler="http")).json()["id"]
        resource = _hook_resource(storage_root, hook_id)
        assert resource["handler"] == "http"

    def test_persisted_timeout_ms(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        hook_id = client.post("/v1/x/hooks", json=_body(timeout_ms=12000)).json()["id"]
        resource = _hook_resource(storage_root, hook_id)
        assert resource["timeout_ms"] == 12000

    def test_persisted_failure_mode(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        hook_id = client.post("/v1/x/hooks", json=_body(failure_mode="warn")).json()["id"]
        resource = _hook_resource(storage_root, hook_id)
        assert resource["failure_mode"] == "warn"

    def test_persisted_status_is_active(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        hook_id = client.post("/v1/x/hooks", json=_body()).json()["id"]
        resource = _hook_resource(storage_root, hook_id)
        assert resource["status"] == "active"

    def test_not_written_on_validation_failure(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/hooks", json=_body(event="   "))
        hooks_dir = storage_root / "hooks"
        files = list(hooks_dir.glob("*.json")) if hooks_dir.exists() else []
        assert files == []


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestHookAuditLog:
    def test_validation_failure_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/hooks", json=_body(event=""))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "hook.create.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/hooks", json=_body(event=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "hook.create.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_code_is_hook_invalid_request(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/hooks", json=_body(timeout_ms=0))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "hook.create.failed"
        )
        assert record["code"] == "hook_invalid_request"

    def test_failure_audit_detail_has_hook_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/hooks", json=_body(event=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "hook.create.failed"
        )
        assert record["detail"]["hook_id"].startswith("hook_")

    def test_failure_audit_detail_has_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/hooks", json=_body(name="   "))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "hook.create.failed"
        )
        assert "event" in record["detail"]

    def test_failure_audit_detail_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/hooks", json=_body(name="audit-hook", event=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "hook.create.failed"
        )
        assert record["detail"]["name"] == "audit-hook"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/hooks", json=_body(timeout_ms=-1))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "hook.create.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestHookOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_hook_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/x/hooks", json=_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "hook.create" in span_names

    def test_failure_emits_hook_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/x/hooks", json=_body(event=""))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "hook.create" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.post("/v1/x/hooks", json=_body(timeout_ms=0))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("hook.create")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_hook_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/x/hooks", json=_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("hook.create")
        assert span is not None
        assert span.attributes["hook.id"].startswith("hook_")

    def test_success_span_has_hook_event_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/x/hooks", json=_body(event="model_call.started"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("hook.create")
        assert span is not None
        assert span.attributes["hook.event"] == "model_call.started"

    def test_success_span_has_hook_name_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/x/hooks", json=_body(name="otel-hook"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("hook.create")
        assert span is not None
        assert span.attributes["hook.name"] == "otel-hook"

    def test_success_span_has_hook_handler_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/x/hooks", json=_body(handler="mcp"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("hook.create")
        assert span is not None
        assert span.attributes["hook.handler"] == "mcp"


# ---------------------------------------------------------------------------
# secret_reads field
# ---------------------------------------------------------------------------


class TestHookSecretReads:
    def test_secret_reads_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body()).json()
        assert body["secret_reads"] is None

    def test_secret_reads_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/hooks",
            json=_body(secret_reads=["api_key", "db_pass"]),
        ).json()
        assert body["secret_reads"] == ["api_key", "db_pass"]

    def test_secret_reads_persisted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        hook_id = client.post(
            "/v1/x/hooks",
            json=_body(secret_reads=["my_secret"]),
        ).json()["id"]
        resource = _hook_resource(storage_root, hook_id)
        assert resource["secret_reads"] == ["my_secret"]

    def test_secret_reads_null_persisted_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        hook_id = client.post("/v1/x/hooks", json=_body()).json()["id"]
        resource = _hook_resource(storage_root, hook_id)
        assert resource["secret_reads"] is None

    def test_secret_reads_empty_list_stored(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/hooks", json=_body(secret_reads=[])).json()
        assert body["secret_reads"] == []


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestHookRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/hooks", json=_body())
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/hooks", json=_body())
        assert resp.status_code == 404
