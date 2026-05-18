"""
Cron endpoint conformance suite.

Tests cover:
  - POST /v1/x/cron returns 201 for all six trigger types.
  - Response fields: id, trigger_type, session_id, status, created_at.
  - status is always "active" on creation.
  - id has "cron_" prefix.
  - IDs are unique across calls.
  - name is optional; stored as null when omitted.
  - name is stored when provided.
  - metadata is optional; stored as null when omitted.
  - metadata is stored when provided.
  - Missing trigger-specific field returns 422 with code "cron_invalid_request".
  - Missing session_id returns 422 (Pydantic validation).
  - Missing trigger_type returns 422 (Pydantic validation).
  - Invalid trigger_type returns 422 (Pydantic validation).
  - Cron resource JSON written to storage_root/cron/{id}.json.
  - Persisted resource has correct trigger_type, session_id, status.
  - Persisted resource has the trigger-specific field set.
  - On validation failure, audit log entry written with event "cron.create.failed".
  - Audit entry level is "error" on failure.
  - Audit entry code is "cron_invalid_request" on validation failure.
  - Audit detail includes cron_id, trigger_type, session_id, message on failure.
  - Error response body has error.code and error.message on failure.
  - OTel span "cron.create" emitted on success.
  - OTel span "cron.create" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries cron.trigger_type and cron.session_id attributes.
  - create_app wires cron router when storage_root is supplied.
  - create_app omits cron route when storage_root is None.
  - DELETE /v1/x/cron/{id} returns 204 and removes the resource file.
  - DELETE returns 404 with code "cron_not_found" for unknown id.
  - Second DELETE on the same id returns 404.
  - Not-found audit entry written with event "cron.delete.failed".
  - OTel span "cron.delete" emitted on success and failure.
  - create_app omits delete route when storage_root is None.
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


def _body(
    trigger_type: str = "interval",
    session_id: str = "sess-abc",
    **extras,
) -> dict:
    base: dict = {"trigger_type": trigger_type, "session_id": session_id}
    # Provide the required trigger-specific field by default
    defaults = {
        "timestamp": {"timestamp": "2026-06-01T00:00:00Z"},
        "interval": {"interval": "5m"},
        "channel_event": {"channel_id": "chan-1"},
        "file_change": {"path": "/workspace/data.csv"},
        "webhook": {"webhook_id": "wh-1"},
        "memory_anniversary": {"memory_key": "user.birthday", "days_before": 3},
    }
    base.update(defaults.get(trigger_type, {}))
    base.update(extras)
    return base


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _cron_resource(storage_root: Path, cron_id: str) -> dict:
    path = storage_root / "cron" / f"{cron_id}.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Success: all six trigger types
# ---------------------------------------------------------------------------


class TestCronCreateSuccess:
    def test_interval_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/cron", json=_body("interval"))
        assert resp.status_code == 201

    def test_timestamp_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/cron", json=_body("timestamp"))
        assert resp.status_code == 201

    def test_channel_event_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/cron", json=_body("channel_event"))
        assert resp.status_code == 201

    def test_file_change_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/cron", json=_body("file_change"))
        assert resp.status_code == 201

    def test_webhook_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/cron", json=_body("webhook"))
        assert resp.status_code == 201

    def test_memory_anniversary_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/cron", json=_body("memory_anniversary"))
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Response fields
# ---------------------------------------------------------------------------


class TestCronCreateResponse:
    def test_response_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body()).json()
        assert "id" in body
        assert isinstance(body["id"], str)

    def test_id_has_cron_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body()).json()
        assert body["id"].startswith("cron_")

    def test_response_has_trigger_type(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("channel_event")).json()
        assert body["trigger_type"] == "channel_event"

    def test_response_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body(session_id="sess-xyz")).json()
        assert body["session_id"] == "sess-xyz"

    def test_response_status_is_active(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body()).json()
        assert body["status"] == "active"

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body()).json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0

    def test_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = client.post("/v1/x/cron", json=_body()).json()["id"]
        id2 = client.post("/v1/x/cron", json=_body()).json()["id"]
        assert id1 != id2

    def test_name_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body()).json()
        assert body["name"] is None

    def test_name_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body(name="nightly-sync")).json()
        assert body["name"] == "nightly-sync"

    def test_metadata_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body()).json()
        assert body["metadata"] is None

    def test_metadata_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        meta = {"owner": "alice", "priority": 1}
        body = client.post("/v1/x/cron", json=_body(metadata=meta)).json()
        assert body["metadata"] == meta

    def test_interval_field_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("interval", interval="1h")).json()
        assert body["interval"] == "1h"

    def test_timestamp_field_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        ts = "2026-07-04T12:00:00Z"
        body = client.post("/v1/x/cron", json=_body("timestamp", timestamp=ts)).json()
        assert body["timestamp"] == ts

    def test_channel_id_field_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("channel_event", channel_id="chan-99")).json()
        assert body["channel_id"] == "chan-99"

    def test_path_field_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("file_change", path="/tmp/log")).json()
        assert body["path"] == "/tmp/log"

    def test_webhook_id_field_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("webhook", webhook_id="wh-42")).json()
        assert body["webhook_id"] == "wh-42"

    def test_memory_key_field_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/cron", json=_body("memory_anniversary", memory_key="join.date")
        ).json()
        assert body["memory_key"] == "join.date"

    def test_days_before_field_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/cron", json=_body("memory_anniversary", days_before=7)
        ).json()
        assert body["days_before"] == 7


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestCronCreateValidation:
    def test_missing_session_id_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/cron", json={"trigger_type": "interval", "interval": "5m"})
        assert resp.status_code == 422

    def test_missing_trigger_type_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/cron", json={"session_id": "s1", "interval": "5m"})
        assert resp.status_code == 422

    def test_invalid_trigger_type_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/cron", json={"trigger_type": "crontab", "session_id": "s1"}
        )
        assert resp.status_code == 422

    def test_interval_without_interval_field_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/cron", json={"trigger_type": "interval", "session_id": "s1"}
        )
        assert resp.status_code == 422

    def test_timestamp_without_timestamp_field_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/cron", json={"trigger_type": "timestamp", "session_id": "s1"}
        )
        assert resp.status_code == 422

    def test_channel_event_without_channel_id_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/cron", json={"trigger_type": "channel_event", "session_id": "s1"}
        )
        assert resp.status_code == 422

    def test_file_change_without_path_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/cron", json={"trigger_type": "file_change", "session_id": "s1"}
        )
        assert resp.status_code == 422

    def test_webhook_without_webhook_id_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/cron", json={"trigger_type": "webhook", "session_id": "s1"}
        )
        assert resp.status_code == 422

    def test_memory_anniversary_without_memory_key_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/cron", json={"trigger_type": "memory_anniversary", "session_id": "s1"}
        )
        assert resp.status_code == 422

    def test_memory_anniversary_without_days_before_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/cron",
            json={"trigger_type": "memory_anniversary", "session_id": "s1", "memory_key": "user.birthday"},
        )
        assert resp.status_code == 422

    def test_memory_anniversary_days_before_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/cron",
            json={"trigger_type": "memory_anniversary", "session_id": "s1", "memory_key": "user.birthday"},
        ).json()
        assert body["error"]["code"] == "cron_invalid_request"

    def test_memory_anniversary_days_before_error_names_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/cron",
            json={"trigger_type": "memory_anniversary", "session_id": "s1", "memory_key": "user.birthday"},
        ).json()
        assert "days_before" in body["error"]["message"]

    def test_validation_error_code_is_cron_invalid_request(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/cron", json={"trigger_type": "interval", "session_id": "s1"}
        ).json()
        assert body["error"]["code"] == "cron_invalid_request"

    def test_validation_error_message_names_missing_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/cron", json={"trigger_type": "interval", "session_id": "s1"}
        ).json()
        assert "interval" in body["error"]["message"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestCronPersistence:
    def test_cron_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body()).json()["id"]
        assert (storage_root / "cron" / f"{cron_id}.json").exists()

    def test_persisted_trigger_type(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body("webhook")).json()["id"]
        resource = _cron_resource(storage_root, cron_id)
        assert resource["trigger_type"] == "webhook"

    def test_persisted_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body(session_id="sess-persist")).json()["id"]
        resource = _cron_resource(storage_root, cron_id)
        assert resource["session_id"] == "sess-persist"

    def test_persisted_status_is_active(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body()).json()["id"]
        resource = _cron_resource(storage_root, cron_id)
        assert resource["status"] == "active"

    def test_persisted_interval_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body("interval", interval="30m")).json()["id"]
        resource = _cron_resource(storage_root, cron_id)
        assert resource["interval"] == "30m"

    def test_persisted_webhook_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post(
            "/v1/x/cron", json=_body("webhook", webhook_id="wh-persist")
        ).json()["id"]
        resource = _cron_resource(storage_root, cron_id)
        assert resource["webhook_id"] == "wh-persist"

    def test_persisted_memory_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post(
            "/v1/x/cron", json=_body("memory_anniversary", memory_key="user.birthday")
        ).json()["id"]
        resource = _cron_resource(storage_root, cron_id)
        assert resource["memory_key"] == "user.birthday"

    def test_persisted_days_before(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post(
            "/v1/x/cron", json=_body("memory_anniversary", days_before=14)
        ).json()["id"]
        resource = _cron_resource(storage_root, cron_id)
        assert resource["days_before"] == 14

    def test_not_written_on_validation_failure(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/cron", json={"trigger_type": "interval", "session_id": "s1"})
        cron_dir = storage_root / "cron"
        files = list(cron_dir.glob("*.json")) if cron_dir.exists() else []
        assert files == []


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestCronAuditLog:
    def test_validation_failure_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/cron", json={"trigger_type": "interval", "session_id": "s1"})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "cron.create.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/cron", json={"trigger_type": "webhook", "session_id": "s1"})
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "cron.create.failed")
        assert record["level"] == "error"

    def test_failure_audit_code_is_cron_invalid_request(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/cron", json={"trigger_type": "file_change", "session_id": "s1"})
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "cron.create.failed")
        assert record["code"] == "cron_invalid_request"

    def test_failure_audit_detail_has_trigger_type(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/cron", json={"trigger_type": "channel_event", "session_id": "s1"})
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "cron.create.failed")
        assert record["detail"]["trigger_type"] == "channel_event"

    def test_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/cron", json={"trigger_type": "interval", "session_id": "audit-sess"})
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "cron.create.failed")
        assert record["detail"]["session_id"] == "audit-sess"

    def test_failure_audit_detail_has_cron_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/cron", json={"trigger_type": "timestamp", "session_id": "s1"})
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "cron.create.failed")
        assert record["detail"]["cron_id"].startswith("cron_")

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/cron", json={"trigger_type": "memory_anniversary", "session_id": "s1"})
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "cron.create.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestCronOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_cron_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/x/cron", json=_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "cron.create" in span_names

    def test_failure_emits_cron_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/x/cron", json={"trigger_type": "interval", "session_id": "s1"})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "cron.create" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.post("/v1/x/cron", json={"trigger_type": "webhook", "session_id": "s1"})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("cron.create")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_trigger_type_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/x/cron", json=_body("channel_event"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("cron.create")
        assert span is not None
        assert span.attributes["cron.trigger_type"] == "channel_event"

    def test_success_span_has_session_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/x/cron", json=_body(session_id="otel-sess"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("cron.create")
        assert span is not None
        assert span.attributes["cron.session_id"] == "otel-sess"

    def test_success_span_has_cron_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/x/cron", json=_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("cron.create")
        assert span is not None
        assert span.attributes["cron.id"].startswith("cron_")


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestCronRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/cron", json=_body())
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/cron", json=_body())
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE success
# ---------------------------------------------------------------------------


class TestCronDeleteSuccess:
    def test_delete_returns_204(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body()).json()["id"]
        resp = client.delete(f"/v1/x/cron/{cron_id}")
        assert resp.status_code == 204

    def test_delete_response_has_no_body(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body()).json()["id"]
        resp = client.delete(f"/v1/x/cron/{cron_id}")
        assert resp.content == b""


# ---------------------------------------------------------------------------
# DELETE not found
# ---------------------------------------------------------------------------


class TestCronDeleteNotFound:
    def test_unknown_id_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.delete("/v1/x/cron/cron_nonexistent")
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.delete("/v1/x/cron/cron_nonexistent").json()
        assert body["error"]["code"] == "cron_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.delete("/v1/x/cron/cron_nonexistent").json()
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# DELETE persistence
# ---------------------------------------------------------------------------


class TestCronDeletePersistence:
    def test_file_removed_after_delete(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body()).json()["id"]
        client.delete(f"/v1/x/cron/{cron_id}")
        assert not (storage_root / "cron" / f"{cron_id}.json").exists()

    def test_second_delete_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body()).json()["id"]
        client.delete(f"/v1/x/cron/{cron_id}")
        resp = client.delete(f"/v1/x/cron/{cron_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE audit log
# ---------------------------------------------------------------------------


class TestCronDeleteAuditLog:
    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/x/cron/cron_missing")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "cron.delete.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/x/cron/cron_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "cron.delete.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/x/cron/cron_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "cron.delete.failed"
        )
        assert record["code"] == "cron_not_found"

    def test_not_found_audit_detail_has_cron_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/x/cron/cron_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "cron.delete.failed"
        )
        assert record["detail"]["cron_id"] == "cron_missing"

    def test_not_found_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/x/cron/cron_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "cron.delete.failed"
        )
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# DELETE OTel spans
# ---------------------------------------------------------------------------


class TestCronDeleteOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_cron_delete_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body()).json()["id"]
        _otel_exporter.clear()
        client.delete(f"/v1/x/cron/{cron_id}")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "cron.delete" in span_names

    def test_not_found_emits_cron_delete_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.delete("/v1/x/cron/cron_missing")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "cron.delete" in span_names

    def test_not_found_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.delete("/v1/x/cron/cron_missing")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("cron.delete")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_cron_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body()).json()["id"]
        _otel_exporter.clear()
        client.delete(f"/v1/x/cron/{cron_id}")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("cron.delete")
        assert span is not None
        assert span.attributes["cron.id"] == cron_id


# ---------------------------------------------------------------------------
# DELETE route wiring
# ---------------------------------------------------------------------------


class TestCronDeleteRouteWiring:
    def test_delete_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body()).json()["id"]
        resp = client.delete(f"/v1/x/cron/{cron_id}")
        assert resp.status_code != 404

    def test_delete_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.delete("/v1/x/cron/cron_any")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# next_fire_at field
# ---------------------------------------------------------------------------


class TestCronNextFireAt:
    def test_interval_response_has_next_fire_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("interval", interval="5m")).json()
        assert "next_fire_at" in body
        assert body["next_fire_at"] is not None

    def test_interval_next_fire_at_is_iso8601(self, storage_root: Path) -> None:
        from datetime import datetime
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("interval", interval="5m")).json()
        # Should parse as ISO-8601 without raising.
        dt = datetime.fromisoformat(body["next_fire_at"])
        assert dt is not None

    def test_interval_next_fire_at_is_in_future(self, storage_root: Path) -> None:
        from datetime import UTC, datetime
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("interval", interval="5m")).json()
        dt = datetime.fromisoformat(body["next_fire_at"])
        assert dt > datetime.now(UTC)

    def test_timestamp_next_fire_at_equals_timestamp(self, storage_root: Path) -> None:
        ts = "2030-01-01T00:00:00Z"
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("timestamp", timestamp=ts)).json()
        assert body["next_fire_at"] == ts

    def test_channel_event_next_fire_at_is_null(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("channel_event")).json()
        assert body["next_fire_at"] is None

    def test_file_change_next_fire_at_is_null(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("file_change")).json()
        assert body["next_fire_at"] is None

    def test_webhook_next_fire_at_is_null(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("webhook")).json()
        assert body["next_fire_at"] is None

    def test_memory_anniversary_next_fire_at_is_null(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("memory_anniversary")).json()
        assert body["next_fire_at"] is None

    def test_interval_next_fire_at_persisted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body("interval", interval="5m")).json()["id"]
        resource = _cron_resource(storage_root, cron_id)
        assert resource["next_fire_at"] is not None

    def test_invalid_interval_duration_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/cron", json=_body("interval", interval="notaduration"))
        assert resp.status_code == 422

    def test_invalid_interval_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body("interval", interval="notaduration")).json()
        assert body["error"]["code"] == "cron_invalid_request"


# ---------------------------------------------------------------------------
# missed_fires_policy field
# ---------------------------------------------------------------------------


class TestCronMissedFiresPolicy:
    def test_default_policy_is_skip(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body()).json()
        assert body["missed_fires_policy"] == "skip"

    def test_catch_up_policy_stored(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/cron", json=_body(missed_fires_policy="catch_up")
        ).json()
        assert body["missed_fires_policy"] == "catch_up"

    def test_skip_policy_stored(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/cron", json=_body(missed_fires_policy="skip")
        ).json()
        assert body["missed_fires_policy"] == "skip"

    def test_missed_fires_policy_persisted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cron_id = client.post(
            "/v1/x/cron", json=_body(missed_fires_policy="catch_up")
        ).json()["id"]
        resource = _cron_resource(storage_root, cron_id)
        assert resource["missed_fires_policy"] == "catch_up"


# ---------------------------------------------------------------------------
# capabilities field
# ---------------------------------------------------------------------------


class TestCronCapabilities:
    def test_default_capabilities_is_empty_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body()).json()
        assert body["capabilities"] == []

    def test_capabilities_stored_when_provided(self, storage_root: Path) -> None:
        caps = ["agent.read", "agent.write"]
        client = _make_client(storage_root)
        body = client.post("/v1/x/cron", json=_body(capabilities=caps)).json()
        assert body["capabilities"] == caps

    def test_capabilities_persisted(self, storage_root: Path) -> None:
        caps = ["agent.network"]
        client = _make_client(storage_root)
        cron_id = client.post("/v1/x/cron", json=_body(capabilities=caps)).json()["id"]
        resource = _cron_resource(storage_root, cron_id)
        assert resource["capabilities"] == caps
