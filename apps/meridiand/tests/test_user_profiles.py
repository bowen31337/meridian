"""
User profiles endpoint conformance suite.

Tests cover:
  - POST /v1/user_profiles returns 201 on success.
  - Response has id with "user_" prefix.
  - IDs are unique across calls.
  - Response has username, display_name, email, metadata, is_primary, created_at, updated_at.
  - display_name is null when omitted; stored when provided.
  - email is null when omitted; stored when provided.
  - metadata is null when omitted; stored when provided.
  - is_primary is true for the first profile created.
  - is_primary is false for subsequent profiles.
  - Profile JSON written to storage_root/user_profiles/{id}.json.
  - Persisted profile has correct username.
  - Not written to disk on validation failure.
  - Empty username returns 422 with code "user_profile_invalid_request".
  - Missing required fields return 422.
  - On validation failure, audit log entry written with event "user_profile.create.failed".
  - Audit entry level is "error" on failure.
  - Audit entry code is "user_profile_invalid_request" on validation failure.
  - Audit detail includes user_profile_id, username, message on failure.
  - Error response body has error.code and error.message on failure.
  - OTel span "user_profile.create" emitted on success.
  - OTel span "user_profile.create" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries user_profile.id and user_profile.username attributes.
  - create_app wires user_profiles router when storage_root is supplied.
  - create_app omits user_profiles route when storage_root is None.
  - DELETE /v1/user_profiles/{id} returns 204 on success.
  - DELETE response has no body.
  - DELETE returns 404 with code "user_profile_not_found" for unknown id.
  - DELETE returns 409 with code "user_profile_is_primary" for primary profile.
  - DELETE returns 409 with code "user_profile_has_active_sessions" when active sessions exist.
  - DELETE removes profile JSON from storage.
  - Second DELETE on same id returns 404.
  - DELETE failure writes audit log entry with event "user_profile.delete.failed".
  - DELETE audit entry level is "error" on failure.
  - DELETE audit detail includes user_profile_id and message.
  - OTel span "user_profile.delete" emitted on success.
  - OTel span "user_profile.delete" emitted on failure.
  - OTel span set to ERROR status on DELETE failure.
  - Span carries user_profile.id attribute on DELETE.
  - DELETE route present with storage_root; absent without.
  - PATCH /v1/user_profiles/{id} returns 200 with updated record.
  - PATCH updates display_name when provided.
  - PATCH updates capabilities when provided.
  - PATCH updates memories when provided.
  - PATCH only updates fields present in the request body.
  - PATCH updates updated_at but not created_at.
  - PATCH returns 404 with code "user_profile_not_found" for unknown id.
  - PATCH returns 422 with code "user_profile_invalid_request" for invalid capability.
  - PATCH persists changes to storage.
  - PATCH with empty capabilities clears the capabilities list.
  - PATCH with empty memories clears the memories list.
  - PATCH failure writes audit log entry with event "user_profile.update.failed".
  - PATCH audit entry level is "error" on failure.
  - PATCH audit detail includes user_profile_id and message.
  - OTel span "user_profile.update" emitted on success.
  - OTel span "user_profile.update" emitted on failure.
  - OTel span set to ERROR status on PATCH failure.
  - Span carries user_profile.id attribute on PATCH.
  - PATCH route present with storage_root; absent without.
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
    base: dict = {"username": "alice"}
    base.update(overrides)
    return base


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _profile_resource(storage_root: Path, user_id: str) -> dict:
    return json.loads((storage_root / "user_profiles" / f"{user_id}.json").read_text())


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestUserProfileCreateSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/user_profiles", json=_body())
        assert resp.status_code == 201

    def test_with_display_name_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/user_profiles", json=_body(display_name="Alice Smith"))
        assert resp.status_code == 201

    def test_with_email_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/user_profiles", json=_body(email="alice@example.com"))
        assert resp.status_code == 201

    def test_with_metadata_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/user_profiles", json=_body(metadata={"role": "admin"}))
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Response fields
# ---------------------------------------------------------------------------


class TestUserProfileCreateResponse:
    def test_response_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/user_profiles", json=_body()).json()
        assert "id" in body
        assert isinstance(body["id"], str)

    def test_id_has_user_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/user_profiles", json=_body()).json()
        assert body["id"].startswith("user_")

    def test_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = client.post("/v1/user_profiles", json=_body(username="alice")).json()["id"]
        id2 = client.post("/v1/user_profiles", json=_body(username="bob")).json()["id"]
        assert id1 != id2

    def test_response_has_username(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/user_profiles", json=_body(username="charlie")).json()
        assert body["username"] == "charlie"

    def test_display_name_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/user_profiles", json=_body()).json()
        assert body["display_name"] is None

    def test_display_name_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/user_profiles", json=_body(display_name="Alice Smith")).json()
        assert body["display_name"] == "Alice Smith"

    def test_email_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/user_profiles", json=_body()).json()
        assert body["email"] is None

    def test_email_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/user_profiles", json=_body(email="alice@example.com")).json()
        assert body["email"] == "alice@example.com"

    def test_metadata_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/user_profiles", json=_body()).json()
        assert body["metadata"] is None

    def test_metadata_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        meta = {"role": "admin", "tier": "paid"}
        body = client.post("/v1/user_profiles", json=_body(metadata=meta)).json()
        assert body["metadata"] == meta

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/user_profiles", json=_body()).json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0

    def test_response_has_updated_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/user_profiles", json=_body()).json()
        assert "updated_at" in body
        assert isinstance(body["updated_at"], str)
        assert len(body["updated_at"]) > 0


# ---------------------------------------------------------------------------
# is_primary flag
# ---------------------------------------------------------------------------


class TestUserProfileIsPrimary:
    def test_first_profile_is_primary(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/user_profiles", json=_body(username="alice")).json()
        assert body["is_primary"] is True

    def test_second_profile_is_not_primary(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="alice"))
        body = client.post("/v1/user_profiles", json=_body(username="bob")).json()
        assert body["is_primary"] is False

    def test_third_profile_is_not_primary(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="alice"))
        client.post("/v1/user_profiles", json=_body(username="bob"))
        body = client.post("/v1/user_profiles", json=_body(username="carol")).json()
        assert body["is_primary"] is False


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestUserProfilePersistence:
    def test_profile_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        assert (storage_root / "user_profiles" / f"{user_id}.json").exists()

    def test_persisted_username(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body(username="persist-user")).json()["id"]
        resource = _profile_resource(storage_root, user_id)
        assert resource["username"] == "persist-user"

    def test_persisted_is_primary_true_for_first(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body(username="first")).json()["id"]
        resource = _profile_resource(storage_root, user_id)
        assert resource["is_primary"] is True

    def test_persisted_is_primary_false_for_second(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="first"))
        user_id = client.post("/v1/user_profiles", json=_body(username="second")).json()["id"]
        resource = _profile_resource(storage_root, user_id)
        assert resource["is_primary"] is False

    def test_not_written_on_validation_failure(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username=""))
        profiles_dir = storage_root / "user_profiles"
        files = list(profiles_dir.glob("*.json")) if profiles_dir.exists() else []
        assert files == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestUserProfileCreateValidation:
    def test_missing_username_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/user_profiles", json={})
        assert resp.status_code == 422

    def test_empty_username_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/user_profiles", json=_body(username="   "))
        assert resp.status_code == 422

    def test_empty_username_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/user_profiles", json=_body(username="")).json()
        assert body["error"]["code"] == "user_profile_invalid_request"

    def test_validation_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/user_profiles", json=_body(username="")).json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestUserProfileAuditLog:
    def test_validation_failure_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username=""))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "user_profile.create.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.create.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_code_is_user_profile_invalid_request(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.create.failed"
        )
        assert record["code"] == "user_profile_invalid_request"

    def test_failure_audit_detail_has_user_profile_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.create.failed"
        )
        assert record["detail"]["user_profile_id"].startswith("user_")

    def test_failure_audit_detail_has_username(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="audit-user", email="x@x.com"))
        # Trigger a validation error with empty username but show name context via a different field
        client.post("/v1/user_profiles", json={"username": ""})
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "user_profile.create.failed"
        )
        assert "username" in record["detail"]

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.create.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestUserProfileOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_user_profile_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/user_profiles", json=_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "user_profile.create" in span_names

    def test_failure_emits_user_profile_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/user_profiles", json=_body(username=""))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "user_profile.create" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.post("/v1/user_profiles", json=_body(username=""))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("user_profile.create")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_user_profile_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/user_profiles", json=_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("user_profile.create")
        assert span is not None
        assert span.attributes["user_profile.id"].startswith("user_")

    def test_success_span_has_username_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="otel-user"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("user_profile.create")
        assert span is not None
        assert span.attributes["user_profile.username"] == "otel-user"


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestUserProfileRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/user_profiles", json=_body())
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/user_profiles", json=_body())
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE success
# ---------------------------------------------------------------------------


class TestUserProfileDeleteSuccess:
    def test_delete_returns_204(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="alice"))
        user_id = client.post("/v1/user_profiles", json=_body(username="bob")).json()["id"]
        resp = client.delete(f"/v1/user_profiles/{user_id}")
        assert resp.status_code == 204

    def test_delete_response_has_no_body(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="alice"))
        user_id = client.post("/v1/user_profiles", json=_body(username="bob")).json()["id"]
        resp = client.delete(f"/v1/user_profiles/{user_id}")
        assert resp.content == b""


# ---------------------------------------------------------------------------
# DELETE not found
# ---------------------------------------------------------------------------


class TestUserProfileDeleteNotFound:
    def test_unknown_id_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.delete("/v1/user_profiles/user_nonexistent")
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.delete("/v1/user_profiles/user_nonexistent").json()
        assert body["error"]["code"] == "user_profile_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.delete("/v1/user_profiles/user_nonexistent").json()
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# DELETE conflict: is_primary
# ---------------------------------------------------------------------------


class TestUserProfileDeleteIsPrimary:
    def test_primary_profile_returns_409(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body(username="alice")).json()["id"]
        resp = client.delete(f"/v1/user_profiles/{user_id}")
        assert resp.status_code == 409

    def test_primary_profile_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body(username="alice")).json()["id"]
        body = client.delete(f"/v1/user_profiles/{user_id}").json()
        assert body["error"]["code"] == "user_profile_is_primary"

    def test_primary_profile_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body(username="alice")).json()["id"]
        body = client.delete(f"/v1/user_profiles/{user_id}").json()
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# DELETE conflict: active sessions
# ---------------------------------------------------------------------------


class TestUserProfileDeleteActiveSessions:
    def test_active_session_returns_409(self, storage_root: Path, tmp_path: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="alice"))
        user_id = client.post("/v1/user_profiles", json=_body(username="bob")).json()["id"]
        session_dir = storage_root / "sessions" / "sess_active"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text(
            json.dumps({"user_profile_id": user_id, "status": "active"})
        )
        resp = client.delete(f"/v1/user_profiles/{user_id}")
        assert resp.status_code == 409

    def test_active_session_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="alice"))
        user_id = client.post("/v1/user_profiles", json=_body(username="bob")).json()["id"]
        session_dir = storage_root / "sessions" / "sess_active2"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text(
            json.dumps({"user_profile_id": user_id, "status": "active"})
        )
        body = client.delete(f"/v1/user_profiles/{user_id}").json()
        assert body["error"]["code"] == "user_profile_has_active_sessions"

    def test_closed_session_allows_delete(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="alice"))
        user_id = client.post("/v1/user_profiles", json=_body(username="bob")).json()["id"]
        session_dir = storage_root / "sessions" / "sess_closed"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text(
            json.dumps({"user_profile_id": user_id, "status": "closed"})
        )
        resp = client.delete(f"/v1/user_profiles/{user_id}")
        assert resp.status_code == 204

    def test_session_for_other_profile_allows_delete(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="alice"))
        user_id = client.post("/v1/user_profiles", json=_body(username="bob")).json()["id"]
        session_dir = storage_root / "sessions" / "sess_other"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text(
            json.dumps({"user_profile_id": "user_other", "status": "active"})
        )
        resp = client.delete(f"/v1/user_profiles/{user_id}")
        assert resp.status_code == 204


# ---------------------------------------------------------------------------
# DELETE persistence
# ---------------------------------------------------------------------------


class TestUserProfileDeletePersistence:
    def test_file_removed_after_delete(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="alice"))
        user_id = client.post("/v1/user_profiles", json=_body(username="bob")).json()["id"]
        client.delete(f"/v1/user_profiles/{user_id}")
        assert not (storage_root / "user_profiles" / f"{user_id}.json").exists()

    def test_second_delete_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="alice"))
        user_id = client.post("/v1/user_profiles", json=_body(username="bob")).json()["id"]
        client.delete(f"/v1/user_profiles/{user_id}")
        resp = client.delete(f"/v1/user_profiles/{user_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE audit log
# ---------------------------------------------------------------------------


class TestUserProfileDeleteAuditLog:
    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/user_profiles/user_missing")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "user_profile.delete.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/user_profiles/user_missing")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.delete.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/user_profiles/user_missing")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.delete.failed"
        )
        assert record["code"] == "user_profile_not_found"

    def test_not_found_audit_detail_has_user_profile_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/user_profiles/user_missing")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.delete.failed"
        )
        assert record["detail"]["user_profile_id"] == "user_missing"

    def test_not_found_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/user_profiles/user_missing")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.delete.failed"
        )
        assert len(record["detail"]["message"]) > 0

    def test_is_primary_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body(username="alice")).json()["id"]
        client.delete(f"/v1/user_profiles/{user_id}")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "user_profile.delete.failed" for r in records)

    def test_is_primary_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body(username="alice")).json()["id"]
        client.delete(f"/v1/user_profiles/{user_id}")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.delete.failed"
        )
        assert record["code"] == "user_profile_is_primary"


# ---------------------------------------------------------------------------
# DELETE OTel spans
# ---------------------------------------------------------------------------


class TestUserProfileDeleteOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_delete_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="alice"))
        user_id = client.post("/v1/user_profiles", json=_body(username="bob")).json()["id"]
        _otel_exporter.clear()
        client.delete(f"/v1/user_profiles/{user_id}")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "user_profile.delete" in span_names

    def test_not_found_emits_delete_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.delete("/v1/user_profiles/user_missing")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "user_profile.delete" in span_names

    def test_not_found_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.delete("/v1/user_profiles/user_missing")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("user_profile.delete")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_user_profile_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="alice"))
        user_id = client.post("/v1/user_profiles", json=_body(username="bob")).json()["id"]
        _otel_exporter.clear()
        client.delete(f"/v1/user_profiles/{user_id}")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("user_profile.delete")
        assert span is not None
        assert span.attributes["user_profile.id"] == user_id


# ---------------------------------------------------------------------------
# DELETE route wiring
# ---------------------------------------------------------------------------


class TestUserProfileDeleteRouteWiring:
    def test_delete_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/user_profiles", json=_body(username="alice"))
        user_id = client.post("/v1/user_profiles", json=_body(username="bob")).json()["id"]
        resp = client.delete(f"/v1/user_profiles/{user_id}")
        assert resp.status_code != 404

    def test_delete_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.delete("/v1/user_profiles/user_any")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH success
# ---------------------------------------------------------------------------


class TestUserProfileUpdateSuccess:
    def test_patch_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        resp = client.patch(f"/v1/user_profiles/{user_id}", json={"display_name": "Alice"})
        assert resp.status_code == 200

    def test_patch_updates_display_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        body = client.patch(f"/v1/user_profiles/{user_id}", json={"display_name": "Alice"}).json()
        assert body["display_name"] == "Alice"

    def test_patch_clears_display_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body(display_name="Alice")).json()["id"]
        body = client.patch(f"/v1/user_profiles/{user_id}", json={"display_name": None}).json()
        assert body["display_name"] is None

    def test_patch_updates_capabilities(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        caps = ["fs.read[/workspace/**]", "net.fetch[api.example.com]"]
        body = client.patch(f"/v1/user_profiles/{user_id}", json={"capabilities": caps}).json()
        assert body["capabilities"] == caps

    def test_patch_updates_memories(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        mems = ["Prefers dark mode", "Works in UTC+9"]
        body = client.patch(f"/v1/user_profiles/{user_id}", json={"memories": mems}).json()
        assert body["memories"] == mems

    def test_patch_omitted_field_unchanged(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body(display_name="Alice")).json()["id"]
        body = client.patch(f"/v1/user_profiles/{user_id}", json={"memories": ["hello"]}).json()
        assert body["display_name"] == "Alice"

    def test_patch_updates_updated_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        created = client.post("/v1/user_profiles", json=_body()).json()
        body = client.patch(
            f"/v1/user_profiles/{created['id']}", json={"display_name": "Bob"}
        ).json()
        assert body["updated_at"] >= created["updated_at"]

    def test_patch_does_not_change_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        created = client.post("/v1/user_profiles", json=_body()).json()
        body = client.patch(
            f"/v1/user_profiles/{created['id']}", json={"display_name": "Bob"}
        ).json()
        assert body["created_at"] == created["created_at"]

    def test_patch_response_includes_all_fields(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        body = client.patch(f"/v1/user_profiles/{user_id}", json={"display_name": "X"}).json()
        for field in ("id", "username", "display_name", "capabilities", "memories",
                      "is_primary", "created_at", "updated_at"):
            assert field in body

    def test_patch_empty_capabilities_clears_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        client.patch(f"/v1/user_profiles/{user_id}", json={"capabilities": ["exec.shell"]})
        body = client.patch(f"/v1/user_profiles/{user_id}", json={"capabilities": []}).json()
        assert body["capabilities"] == []

    def test_patch_empty_memories_clears_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        client.patch(f"/v1/user_profiles/{user_id}", json={"memories": ["foo"]})
        body = client.patch(f"/v1/user_profiles/{user_id}", json={"memories": []}).json()
        assert body["memories"] == []


# ---------------------------------------------------------------------------
# PATCH persistence
# ---------------------------------------------------------------------------


class TestUserProfileUpdatePersistence:
    def test_patch_writes_updated_profile_to_disk(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        client.patch(f"/v1/user_profiles/{user_id}", json={"display_name": "Persisted"})
        resource = _profile_resource(storage_root, user_id)
        assert resource["display_name"] == "Persisted"

    def test_patch_persists_capabilities(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        caps = ["kb.read[docs]", "memory.write[user]"]
        client.patch(f"/v1/user_profiles/{user_id}", json={"capabilities": caps})
        resource = _profile_resource(storage_root, user_id)
        assert resource["capabilities"] == caps

    def test_patch_persists_memories(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        mems = ["Uses vim", "Prefers Python"]
        client.patch(f"/v1/user_profiles/{user_id}", json={"memories": mems})
        resource = _profile_resource(storage_root, user_id)
        assert resource["memories"] == mems


# ---------------------------------------------------------------------------
# PATCH validation
# ---------------------------------------------------------------------------


class TestUserProfileUpdateValidation:
    def test_invalid_capability_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        resp = client.patch(
            f"/v1/user_profiles/{user_id}", json={"capabilities": ["NOT VALID"]}
        )
        assert resp.status_code == 422

    def test_invalid_capability_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        body = client.patch(
            f"/v1/user_profiles/{user_id}", json={"capabilities": ["bad cap!"]}
        ).json()
        assert body["error"]["code"] == "user_profile_invalid_request"

    def test_invalid_capability_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        body = client.patch(
            f"/v1/user_profiles/{user_id}", json={"capabilities": ["bad!"]}
        ).json()
        assert len(body["error"]["message"]) > 0

    def test_one_invalid_capability_in_list_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        resp = client.patch(
            f"/v1/user_profiles/{user_id}",
            json={"capabilities": ["fs.read[*]", "INVALID ENTRY"]},
        )
        assert resp.status_code == 422

    def test_invalid_capability_does_not_update_profile(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        client.patch(
            f"/v1/user_profiles/{user_id}", json={"capabilities": ["INVALID"]}
        )
        resource = _profile_resource(storage_root, user_id)
        assert resource["capabilities"] == []


# ---------------------------------------------------------------------------
# PATCH not found
# ---------------------------------------------------------------------------


class TestUserProfileUpdateNotFound:
    def test_unknown_id_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.patch("/v1/user_profiles/user_nonexistent", json={"display_name": "X"})
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.patch(
            "/v1/user_profiles/user_nonexistent", json={"display_name": "X"}
        ).json()
        assert body["error"]["code"] == "user_profile_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.patch(
            "/v1/user_profiles/user_nonexistent", json={"display_name": "X"}
        ).json()
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# PATCH audit log
# ---------------------------------------------------------------------------


class TestUserProfileUpdateAuditLog:
    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.patch("/v1/user_profiles/user_missing", json={"display_name": "X"})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "user_profile.update.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.patch("/v1/user_profiles/user_missing", json={"display_name": "X"})
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.update.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.patch("/v1/user_profiles/user_missing", json={"display_name": "X"})
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.update.failed"
        )
        assert record["code"] == "user_profile_not_found"

    def test_audit_detail_has_user_profile_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.patch("/v1/user_profiles/user_missing", json={"display_name": "X"})
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.update.failed"
        )
        assert record["detail"]["user_profile_id"] == "user_missing"

    def test_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.patch("/v1/user_profiles/user_missing", json={"display_name": "X"})
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.update.failed"
        )
        assert len(record["detail"]["message"]) > 0

    def test_invalid_capability_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        client.patch(f"/v1/user_profiles/{user_id}", json={"capabilities": ["BAD CAP"]})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "user_profile.update.failed" for r in records)

    def test_invalid_capability_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        client.patch(f"/v1/user_profiles/{user_id}", json={"capabilities": ["BAD CAP"]})
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "user_profile.update.failed"
        )
        assert record["code"] == "user_profile_invalid_request"


# ---------------------------------------------------------------------------
# PATCH OTel spans
# ---------------------------------------------------------------------------


class TestUserProfileUpdateOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_update_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        _otel_exporter.clear()
        client.patch(f"/v1/user_profiles/{user_id}", json={"display_name": "Alice"})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "user_profile.update" in span_names

    def test_failure_emits_update_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.patch("/v1/user_profiles/user_missing", json={"display_name": "X"})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "user_profile.update" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.patch("/v1/user_profiles/user_missing", json={"display_name": "X"})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("user_profile.update")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_user_profile_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        _otel_exporter.clear()
        client.patch(f"/v1/user_profiles/{user_id}", json={"display_name": "Alice"})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("user_profile.update")
        assert span is not None
        assert span.attributes["user_profile.id"] == user_id


# ---------------------------------------------------------------------------
# PATCH route wiring
# ---------------------------------------------------------------------------


class TestUserProfileUpdateRouteWiring:
    def test_patch_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        user_id = client.post("/v1/user_profiles", json=_body()).json()["id"]
        resp = client.patch(f"/v1/user_profiles/{user_id}", json={"display_name": "X"})
        assert resp.status_code != 404

    def test_patch_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.patch("/v1/user_profiles/user_any", json={"display_name": "X"})
        assert resp.status_code == 404
