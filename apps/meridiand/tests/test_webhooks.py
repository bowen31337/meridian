"""
Webhook endpoint conformance suite.

Tests cover:
  - POST /v1/webhooks returns 201 on success.
  - Response fields: id, name, url, secret_ref, event_filter, max_retries, backoff,
    status, created_at.
  - id has "webhook_" prefix.
  - IDs are unique across calls.
  - status is always "active" on creation.
  - secret_ref is null when omitted.
  - metadata is null when omitted; stored when provided.
  - event_filter.session_id is null when omitted; stored when provided.
  - backoff "exponential" and "linear" both accepted.
  - Empty url returns 422 with code "webhook_invalid_request".
  - Empty event_filter.types returns 422 with code "webhook_invalid_request".
  - Negative max_retries returns 422 with code "webhook_invalid_request".
  - Missing required fields (name, url, event_filter, max_retries, backoff) return 422.
  - Webhook resource JSON written to storage_root/webhooks/{id}.json.
  - Persisted resource has correct name, url, backoff, max_retries, status.
  - Not written to disk on validation failure.
  - On validation failure, audit log entry written with event "webhook.create.failed".
  - Audit entry level is "error" on failure.
  - Audit entry code is "webhook_invalid_request" on validation failure.
  - Audit detail includes webhook_id, url, name, message on failure.
  - Error response body has error.code and error.message on failure.
  - OTel span "webhook.create" emitted on success.
  - OTel span "webhook.create" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries webhook.id, webhook.url, webhook.name attributes.
  - create_app wires webhook router when storage_root is supplied.
  - create_app omits webhook route when storage_root is None.
  - DELETE /v1/webhooks/{id} returns 204 on success.
  - DELETE response has no body.
  - DELETE unknown id returns 404 with code "webhook_not_found".
  - DELETE removes the webhook JSON file.
  - Second DELETE on same id returns 404.
  - DELETE not-found writes audit log entry with event "webhook.delete.failed".
  - Audit entry level is "error", code is "webhook_not_found".
  - Audit detail includes webhook_id and message.
  - OTel span "webhook.delete" emitted on success and failure.
  - Not-found span has ERROR status and webhook.id attribute.
  - DELETE route present with storage_root, absent without.
"""

from __future__ import annotations

import json
from pathlib import Path

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
        "name": "my-webhook",
        "url": "https://example.com/hook",
        "event_filter": {"types": ["session.completed", "session.failed"]},
        "max_retries": 3,
        "backoff": "exponential",
    }
    base.update(overrides)
    return base


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _webhook_resource(storage_root: Path, webhook_id: str) -> dict:
    path = storage_root / "webhooks" / f"{webhook_id}.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestWebhookCreateSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/webhooks", json=_body())
        assert resp.status_code == 201

    def test_exponential_backoff_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/webhooks", json=_body(backoff="exponential"))
        assert resp.status_code == 201

    def test_linear_backoff_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/webhooks", json=_body(backoff="linear"))
        assert resp.status_code == 201

    def test_zero_max_retries_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/webhooks", json=_body(max_retries=0))
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Response fields
# ---------------------------------------------------------------------------


class TestWebhookCreateResponse:
    def test_response_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body()).json()
        assert "id" in body
        assert isinstance(body["id"], str)

    def test_id_has_webhook_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body()).json()
        assert body["id"].startswith("webhook_")

    def test_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = client.post("/v1/webhooks", json=_body()).json()["id"]
        id2 = client.post("/v1/webhooks", json=_body()).json()["id"]
        assert id1 != id2

    def test_response_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body(name="deploy-hook")).json()
        assert body["name"] == "deploy-hook"

    def test_response_has_url(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/webhooks", json=_body(url="https://hooks.example.com/deploy")
        ).json()
        assert body["url"] == "https://hooks.example.com/deploy"

    def test_response_status_is_active(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body()).json()
        assert body["status"] == "active"

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body()).json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0

    def test_response_has_max_retries(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body(max_retries=5)).json()
        assert body["max_retries"] == 5

    def test_response_has_backoff(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body(backoff="linear")).json()
        assert body["backoff"] == "linear"

    def test_response_has_event_filter_types(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        types = ["session.completed"]
        body = client.post("/v1/webhooks", json=_body(event_filter={"types": types})).json()
        assert body["event_filter"]["types"] == types

    def test_event_filter_session_id_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body()).json()
        assert body["event_filter"]["session_id"] is None

    def test_event_filter_session_id_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        ef = {"types": ["session.completed"], "session_id": "sess-abc"}
        body = client.post("/v1/webhooks", json=_body(event_filter=ef)).json()
        assert body["event_filter"]["session_id"] == "sess-abc"

    def test_secret_ref_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body()).json()
        assert body["secret_ref"] is None

    def test_secret_ref_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/webhooks", json=_body(secret_ref="vault://default/webhook-secret")
        ).json()
        assert body["secret_ref"] == "vault://default/webhook-secret"

    def test_metadata_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body()).json()
        assert body["metadata"] is None

    def test_metadata_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        meta = {"team": "platform", "env": "prod"}
        body = client.post("/v1/webhooks", json=_body(metadata=meta)).json()
        assert body["metadata"] == meta


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestWebhookCreateValidation:
    def test_missing_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["name"]
        resp = client.post("/v1/webhooks", json=payload)
        assert resp.status_code == 422

    def test_missing_url_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["url"]
        resp = client.post("/v1/webhooks", json=payload)
        assert resp.status_code == 422

    def test_missing_event_filter_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["event_filter"]
        resp = client.post("/v1/webhooks", json=payload)
        assert resp.status_code == 422

    def test_missing_max_retries_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["max_retries"]
        resp = client.post("/v1/webhooks", json=payload)
        assert resp.status_code == 422

    def test_missing_backoff_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["backoff"]
        resp = client.post("/v1/webhooks", json=payload)
        assert resp.status_code == 422

    def test_invalid_backoff_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/webhooks", json=_body(backoff="fibonacci"))
        assert resp.status_code == 422

    def test_empty_url_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/webhooks", json=_body(url="   "))
        assert resp.status_code == 422

    def test_empty_url_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body(url="   ")).json()
        assert body["error"]["code"] == "webhook_invalid_request"

    def test_empty_event_types_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/webhooks", json=_body(event_filter={"types": []}))
        assert resp.status_code == 422

    def test_empty_event_types_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body(event_filter={"types": []})).json()
        assert body["error"]["code"] == "webhook_invalid_request"

    def test_negative_max_retries_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/webhooks", json=_body(max_retries=-1))
        assert resp.status_code == 422

    def test_negative_max_retries_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body(max_retries=-1)).json()
        assert body["error"]["code"] == "webhook_invalid_request"

    def test_validation_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/webhooks", json=_body(url="")).json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestWebhookPersistence:
    def test_webhook_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        webhook_id = client.post("/v1/webhooks", json=_body()).json()["id"]
        assert (storage_root / "webhooks" / f"{webhook_id}.json").exists()

    def test_persisted_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        webhook_id = client.post("/v1/webhooks", json=_body(name="persist-hook")).json()["id"]
        resource = _webhook_resource(storage_root, webhook_id)
        assert resource["name"] == "persist-hook"

    def test_persisted_url(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        webhook_id = client.post(
            "/v1/webhooks", json=_body(url="https://persist.example.com/hook")
        ).json()["id"]
        resource = _webhook_resource(storage_root, webhook_id)
        assert resource["url"] == "https://persist.example.com/hook"

    def test_persisted_status_is_active(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        webhook_id = client.post("/v1/webhooks", json=_body()).json()["id"]
        resource = _webhook_resource(storage_root, webhook_id)
        assert resource["status"] == "active"

    def test_persisted_backoff(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        webhook_id = client.post("/v1/webhooks", json=_body(backoff="linear")).json()["id"]
        resource = _webhook_resource(storage_root, webhook_id)
        assert resource["backoff"] == "linear"

    def test_persisted_max_retries(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        webhook_id = client.post("/v1/webhooks", json=_body(max_retries=7)).json()["id"]
        resource = _webhook_resource(storage_root, webhook_id)
        assert resource["max_retries"] == 7

    def test_persisted_secret_ref(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        webhook_id = client.post(
            "/v1/webhooks", json=_body(secret_ref="vault://default/my-secret")
        ).json()["id"]
        resource = _webhook_resource(storage_root, webhook_id)
        assert resource["secret_ref"] == "vault://default/my-secret"

    def test_not_written_on_validation_failure(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/webhooks", json=_body(event_filter={"types": []}))
        webhooks_dir = storage_root / "webhooks"
        files = list(webhooks_dir.glob("*.json")) if webhooks_dir.exists() else []
        assert files == []


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestWebhookAuditLog:
    def test_validation_failure_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/webhooks", json=_body(url=""))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "webhook.create.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/webhooks", json=_body(url=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.create.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_code_is_webhook_invalid_request(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/webhooks", json=_body(event_filter={"types": []}))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.create.failed"
        )
        assert record["code"] == "webhook_invalid_request"

    def test_failure_audit_detail_has_webhook_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/webhooks", json=_body(url=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.create.failed"
        )
        assert record["detail"]["webhook_id"].startswith("webhook_")

    def test_failure_audit_detail_has_url(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/webhooks", json=_body(url=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.create.failed"
        )
        assert "url" in record["detail"]

    def test_failure_audit_detail_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/webhooks", json=_body(name="audit-hook", url=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.create.failed"
        )
        assert record["detail"]["name"] == "audit-hook"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/webhooks", json=_body(max_retries=-1))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.create.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestWebhookOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_webhook_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/webhooks", json=_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "webhook.create" in span_names

    def test_failure_emits_webhook_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/webhooks", json=_body(url=""))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "webhook.create" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.post("/v1/webhooks", json=_body(event_filter={"types": []}))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("webhook.create")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_webhook_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/webhooks", json=_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("webhook.create")
        assert span is not None
        assert span.attributes["webhook.id"].startswith("webhook_")

    def test_success_span_has_webhook_url_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/webhooks", json=_body(url="https://otel.example.com/hook"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("webhook.create")
        assert span is not None
        assert span.attributes["webhook.url"] == "https://otel.example.com/hook"

    def test_success_span_has_webhook_name_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/webhooks", json=_body(name="otel-hook"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("webhook.create")
        assert span is not None
        assert span.attributes["webhook.name"] == "otel-hook"


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestWebhookRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/webhooks", json=_body())
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/webhooks", json=_body())
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE success
# ---------------------------------------------------------------------------


class TestWebhookDeleteSuccess:
    def test_delete_returns_204(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        webhook_id = client.post("/v1/webhooks", json=_body()).json()["id"]
        resp = client.delete(f"/v1/webhooks/{webhook_id}")
        assert resp.status_code == 204

    def test_delete_response_has_no_body(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        webhook_id = client.post("/v1/webhooks", json=_body()).json()["id"]
        resp = client.delete(f"/v1/webhooks/{webhook_id}")
        assert resp.content == b""


# ---------------------------------------------------------------------------
# DELETE not found
# ---------------------------------------------------------------------------


class TestWebhookDeleteNotFound:
    def test_unknown_id_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.delete("/v1/webhooks/webhook_nonexistent")
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.delete("/v1/webhooks/webhook_nonexistent").json()
        assert body["error"]["code"] == "webhook_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.delete("/v1/webhooks/webhook_nonexistent").json()
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# DELETE persistence
# ---------------------------------------------------------------------------


class TestWebhookDeletePersistence:
    def test_file_removed_after_delete(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        webhook_id = client.post("/v1/webhooks", json=_body()).json()["id"]
        client.delete(f"/v1/webhooks/{webhook_id}")
        assert not (storage_root / "webhooks" / f"{webhook_id}.json").exists()

    def test_second_delete_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        webhook_id = client.post("/v1/webhooks", json=_body()).json()["id"]
        client.delete(f"/v1/webhooks/{webhook_id}")
        resp = client.delete(f"/v1/webhooks/{webhook_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE audit log
# ---------------------------------------------------------------------------


class TestWebhookDeleteAuditLog:
    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/webhooks/webhook_missing")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "webhook.delete.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/webhooks/webhook_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.delete.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/webhooks/webhook_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.delete.failed"
        )
        assert record["code"] == "webhook_not_found"

    def test_not_found_audit_detail_has_webhook_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/webhooks/webhook_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.delete.failed"
        )
        assert record["detail"]["webhook_id"] == "webhook_missing"

    def test_not_found_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/webhooks/webhook_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.delete.failed"
        )
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# DELETE OTel spans
# ---------------------------------------------------------------------------


class TestWebhookDeleteOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_webhook_delete_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        webhook_id = client.post("/v1/webhooks", json=_body()).json()["id"]
        _otel_exporter.clear()
        client.delete(f"/v1/webhooks/{webhook_id}")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "webhook.delete" in span_names

    def test_not_found_emits_webhook_delete_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.delete("/v1/webhooks/webhook_missing")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "webhook.delete" in span_names

    def test_not_found_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.delete("/v1/webhooks/webhook_missing")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("webhook.delete")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_webhook_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        webhook_id = client.post("/v1/webhooks", json=_body()).json()["id"]
        _otel_exporter.clear()
        client.delete(f"/v1/webhooks/{webhook_id}")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("webhook.delete")
        assert span is not None
        assert span.attributes["webhook.id"] == webhook_id


# ---------------------------------------------------------------------------
# DELETE route wiring
# ---------------------------------------------------------------------------


class TestWebhookDeleteRouteWiring:
    def test_delete_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        webhook_id = client.post("/v1/webhooks", json=_body()).json()["id"]
        resp = client.delete(f"/v1/webhooks/{webhook_id}")
        assert resp.status_code != 404

    def test_delete_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.delete("/v1/webhooks/webhook_any")
        assert resp.status_code == 404
