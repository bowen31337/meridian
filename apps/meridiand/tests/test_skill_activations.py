"""
Skill activation conformance suite (PRD D2).

Tests cover:
  - Installing a skill does NOT auto-bind it to any agent.
  - POST /v1/agents/{agent_id}/skills returns 201 on success.
  - Response has id with "skillact_" prefix.
  - IDs are unique across calls.
  - Response has agent_id, skill_id, skill_version_id, status "pending".
  - Response has requested_at set; approved_at and revoked_at are null.
  - skill_version_id defaults to the skill's latest version when omitted.
  - skill_version_id is used when explicitly provided.
  - Activation record written to storage_root/skill_activations/{id}.json.
  - Returns 404 with code "skill_not_found" when skill_id does not exist.
  - Returns 422 with code "skill_activation_invalid_request" when skill_id is empty.
  - Returns 409 with code "skill_activation_conflict" when a pending activation exists.
  - Returns 409 with code "skill_activation_conflict" when an active activation exists.
  - Re-requesting after revocation returns 201 (revoked is not a blocker).
  - On success, audit log entry written with event "skill.activation.requested".
  - Success audit entry level is "info".
  - On failure, audit log entry written with event "skill.activation.request.failed".
  - Failure audit entry level is "error".
  - Failure audit detail includes agent_id, skill_id, message.
  - OTel span "skill.activation.request" emitted on success.
  - OTel span "skill.activation.request" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries agent.id, skill.id, skill.activation.id attributes.
  - POST /v1/agents/{agent_id}/skills/{skill_id}/approve returns 200 on success.
  - Approve transitions status from "pending" to "active".
  - Approve sets approved_at timestamp.
  - Returns 404 with code "skill_activation_not_found" when no activation exists.
  - Returns 409 with code "skill_activation_conflict" when activation is already active.
  - Returns 409 with code "skill_activation_conflict" when activation is revoked.
  - Approve audit event "skill.activation.approved" with level "info".
  - Approve failure audit event "skill.activation.approve.failed" with level "error".
  - OTel span "skill.activation.approve" emitted on success and failure.
  - DELETE /v1/agents/{agent_id}/skills/{skill_id} returns 200 on success.
  - Revoke transitions status from "active" to "revoked".
  - Revoke transitions status from "pending" to "revoked".
  - Revoke sets revoked_at timestamp.
  - Returns 404 with code "skill_activation_not_found" when no activation exists.
  - Returns 409 with code "skill_activation_conflict" when activation is already revoked.
  - Revoke audit event "skill.activation.revoked" with level "info".
  - Revoke failure audit event "skill.activation.revoke.failed" with level "error".
  - OTel span "skill.activation.revoke" emitted on success and failure.
  - GET /v1/agents/{agent_id}/skills returns 200 with paginated list.
  - Returns empty items when no activations exist for the agent.
  - Items filtered by agent_id (other agents' activations not included).
  - Items have id, agent_id, skill_id, skill_version_id, status, requested_at,
    approved_at, revoked_at.
  - Default limit is 20 and default offset is 0.
  - limit and offset query params work correctly.
  - total reflects full count independent of pagination.
  - OTel span "skill.activation.list" emitted on success.
  - Span carries agent.id attribute.
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

_AGENT_ID = "agent_test01"
_OTHER_AGENT_ID = "agent_other99"


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _skill_body(**overrides) -> dict:
    base: dict = {
        "name": "my-skill",
        "description": "A test skill",
        "instructions": "Do the thing.",
        "tools": [{"name": "bash", "description": "Run shell commands"}],
    }
    base.update(overrides)
    return base


def _create_skill(client: TestClient, **overrides) -> dict:
    resp = client.post("/v1/skills", json=_skill_body(**overrides))
    assert resp.status_code == 201
    return resp.json()


def _request_activation(
    client: TestClient,
    agent_id: str,
    skill_id: str,
    *,
    skill_version_id: str | None = None,
) -> dict:
    body: dict = {"skill_id": skill_id}
    if skill_version_id is not None:
        body["skill_version_id"] = skill_version_id
    resp = client.post(f"/v1/agents/{agent_id}/skills", json=body)
    assert resp.status_code == 201
    return resp.json()


def _approve_activation(client: TestClient, agent_id: str, skill_id: str) -> dict:
    resp = client.post(f"/v1/agents/{agent_id}/skills/{skill_id}/approve")
    assert resp.status_code == 200
    return resp.json()


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Key invariant: skill install never auto-activates
# ---------------------------------------------------------------------------


class TestNoAutoActivation:
    def test_install_does_not_create_activation_for_agent(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client)
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert body["items"] == []

    def test_install_does_not_create_activation_for_any_agent(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client)
        _create_skill(client, name="skill-b")
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert body["total"] == 0

    def test_activation_dir_absent_after_install(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client)
        assert not (storage_root / "skill_activations").exists()


# ---------------------------------------------------------------------------
# POST /v1/agents/{agent_id}/skills — request activation
# ---------------------------------------------------------------------------


class TestRequestActivationSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        resp = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        )
        assert resp.status_code == 201

    def test_response_id_has_skillact_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        ).json()
        assert body["id"].startswith("skillact_")

    def test_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill_a = _create_skill(client, name="skill-a")
        skill_b = _create_skill(client, name="skill-b")
        id1 = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill_a["id"]}
        ).json()["id"]
        id2 = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill_b["id"]}
        ).json()["id"]
        assert id1 != id2

    def test_response_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        ).json()
        assert body["agent_id"] == _AGENT_ID

    def test_response_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        ).json()
        assert body["skill_id"] == skill["id"]

    def test_status_is_pending(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        ).json()
        assert body["status"] == "pending"

    def test_approved_at_is_null(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        ).json()
        assert body["approved_at"] is None

    def test_revoked_at_is_null(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        ).json()
        assert body["revoked_at"] is None

    def test_requested_at_is_set(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        ).json()
        assert isinstance(body["requested_at"], str)
        assert len(body["requested_at"]) > 0

    def test_skill_version_id_defaults_to_latest(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        ).json()
        assert body["skill_version_id"] == skill["version"]["id"]

    def test_explicit_skill_version_id_stored(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        explicit_ver = skill["version"]["id"]
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills",
            json={"skill_id": skill["id"], "skill_version_id": explicit_ver},
        ).json()
        assert body["skill_version_id"] == explicit_ver

    def test_activation_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        activation = _request_activation(client, _AGENT_ID, skill["id"])
        path = storage_root / "skill_activations" / f"{activation['id']}.json"
        assert path.exists()

    def test_persisted_record_matches_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        activation = _request_activation(client, _AGENT_ID, skill["id"])
        path = storage_root / "skill_activations" / f"{activation['id']}.json"
        persisted = json.loads(path.read_text())
        assert persisted["agent_id"] == _AGENT_ID
        assert persisted["skill_id"] == skill["id"]
        assert persisted["status"] == "pending"

    def test_re_request_allowed_after_revocation(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}")
        resp = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "pending"


# ---------------------------------------------------------------------------
# POST /v1/agents/{agent_id}/skills — validation & conflict errors
# ---------------------------------------------------------------------------


class TestRequestActivationErrors:
    def test_unknown_skill_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": "skill_doesnotexist"}
        )
        assert resp.status_code == 404

    def test_unknown_skill_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": "skill_doesnotexist"}
        ).json()
        assert body["error"]["code"] == "skill_not_found"

    def test_empty_skill_id_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": "   "})
        assert resp.status_code == 422

    def test_empty_skill_id_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": ""}
        ).json()
        assert body["error"]["code"] == "skill_activation_invalid_request"

    def test_conflict_when_pending_activation_exists(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        resp = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        )
        assert resp.status_code == 409

    def test_conflict_error_code_when_pending(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        ).json()
        assert body["error"]["code"] == "skill_activation_conflict"

    def test_conflict_when_active_activation_exists(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        _approve_activation(client, _AGENT_ID, skill["id"])
        resp = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        )
        assert resp.status_code == 409

    def test_conflict_error_code_when_active(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        _approve_activation(client, _AGENT_ID, skill["id"])
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]}
        ).json()
        assert body["error"]["code"] == "skill_activation_conflict"

    def test_error_response_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": "skill_missing"}
        ).json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# POST /v1/agents/{agent_id}/skills — audit log
# ---------------------------------------------------------------------------


class TestRequestActivationAudit:
    def test_success_writes_requested_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill.activation.requested" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.requested"
        )
        assert record["level"] == "info"

    def test_success_audit_detail_has_activation_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        activation = _request_activation(client, _AGENT_ID, skill["id"])
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.requested"
        )
        assert record["detail"]["activation_id"] == activation["id"]

    def test_success_audit_detail_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.requested"
        )
        assert record["detail"]["agent_id"] == _AGENT_ID

    def test_success_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.requested"
        )
        assert record["detail"]["skill_id"] == skill["id"]

    def test_failure_writes_request_failed_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": "skill_missing"})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill.activation.request.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": "skill_missing"})
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.request.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_detail_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": "skill_missing"})
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.request.failed"
        )
        assert record["detail"]["agent_id"] == _AGENT_ID

    def test_failure_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": "skill_missing"})
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.request.failed"
        )
        assert record["detail"]["skill_id"] == "skill_missing"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": "skill_missing"})
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.request.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# POST /v1/agents/{agent_id}/skills — OTel
# ---------------------------------------------------------------------------


class TestRequestActivationOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        skill = _create_skill(client)
        _otel_exporter.clear()
        client.post(f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]})
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.activation.request" in names

    def test_failure_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        _otel_exporter.clear()
        client.post(f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": "skill_missing"})
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.activation.request" in names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        _otel_exporter.clear()
        client.post(f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": "skill_missing"})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.activation.request")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_span_has_agent_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        skill = _create_skill(client)
        _otel_exporter.clear()
        client.post(f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.activation.request")
        assert span is not None
        assert span.attributes["agent.id"] == _AGENT_ID

    def test_span_has_skill_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        skill = _create_skill(client)
        _otel_exporter.clear()
        client.post(f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.activation.request")
        assert span is not None
        assert span.attributes["skill.id"] == skill["id"]

    def test_span_has_activation_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        skill = _create_skill(client)
        _otel_exporter.clear()
        resp = client.post(f"/v1/agents/{_AGENT_ID}/skills", json={"skill_id": skill["id"]})
        activation_id = resp.json()["id"]
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.activation.request")
        assert span is not None
        assert span.attributes["skill.activation.id"] == activation_id


# ---------------------------------------------------------------------------
# POST /v1/agents/{agent_id}/skills/{skill_id}/approve
# ---------------------------------------------------------------------------


class TestApproveActivationSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        resp = client.post(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}/approve")
        assert resp.status_code == 200

    def test_status_becomes_active(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}/approve"
        ).json()
        assert body["status"] == "active"

    def test_approved_at_is_set(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}/approve"
        ).json()
        assert isinstance(body["approved_at"], str)
        assert len(body["approved_at"]) > 0

    def test_revoked_at_remains_null(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}/approve"
        ).json()
        assert body["revoked_at"] is None

    def test_persisted_record_is_active(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        activation = _request_activation(client, _AGENT_ID, skill["id"])
        client.post(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}/approve")
        path = storage_root / "skill_activations" / f"{activation['id']}.json"
        persisted = json.loads(path.read_text())
        assert persisted["status"] == "active"
        assert persisted["approved_at"] is not None


class TestApproveActivationErrors:
    def test_no_activation_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(f"/v1/agents/{_AGENT_ID}/skills/skill_missing/approve")
        assert resp.status_code == 404

    def test_no_activation_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills/skill_missing/approve"
        ).json()
        assert body["error"]["code"] == "skill_activation_not_found"

    def test_already_active_returns_409(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        _approve_activation(client, _AGENT_ID, skill["id"])
        resp = client.post(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}/approve")
        assert resp.status_code == 409

    def test_already_active_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        _approve_activation(client, _AGENT_ID, skill["id"])
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}/approve"
        ).json()
        assert body["error"]["code"] == "skill_activation_conflict"

    def test_revoked_activation_returns_409(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}")
        resp = client.post(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}/approve")
        assert resp.status_code == 409


class TestApproveActivationAudit:
    def test_success_writes_approved_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        client.post(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}/approve")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill.activation.approved" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        client.post(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}/approve")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.approved"
        )
        assert record["level"] == "info"

    def test_success_audit_detail_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        client.post(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}/approve")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.approved"
        )
        assert record["detail"]["agent_id"] == _AGENT_ID

    def test_success_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        client.post(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}/approve")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.approved"
        )
        assert record["detail"]["skill_id"] == skill["id"]

    def test_failure_writes_approve_failed_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(f"/v1/agents/{_AGENT_ID}/skills/skill_missing/approve")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill.activation.approve.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(f"/v1/agents/{_AGENT_ID}/skills/skill_missing/approve")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.approve.failed"
        )
        assert record["level"] == "error"


class TestApproveActivationOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        _otel_exporter.clear()
        client.post(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}/approve")
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.activation.approve" in names

    def test_failure_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        _otel_exporter.clear()
        client.post(f"/v1/agents/{_AGENT_ID}/skills/skill_missing/approve")
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.activation.approve" in names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        _otel_exporter.clear()
        client.post(f"/v1/agents/{_AGENT_ID}/skills/skill_missing/approve")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.activation.approve")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# DELETE /v1/agents/{agent_id}/skills/{skill_id} — revoke activation
# ---------------------------------------------------------------------------


class TestRevokeActivationSuccess:
    def test_revoke_active_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        _approve_activation(client, _AGENT_ID, skill["id"])
        resp = client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}")
        assert resp.status_code == 200

    def test_revoke_pending_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        resp = client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}")
        assert resp.status_code == 200

    def test_status_becomes_revoked_from_active(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        _approve_activation(client, _AGENT_ID, skill["id"])
        body = client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}").json()
        assert body["status"] == "revoked"

    def test_status_becomes_revoked_from_pending(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}").json()
        assert body["status"] == "revoked"

    def test_revoked_at_is_set(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}").json()
        assert isinstance(body["revoked_at"], str)
        assert len(body["revoked_at"]) > 0

    def test_persisted_record_is_revoked(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        activation = _request_activation(client, _AGENT_ID, skill["id"])
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}")
        path = storage_root / "skill_activations" / f"{activation['id']}.json"
        persisted = json.loads(path.read_text())
        assert persisted["status"] == "revoked"
        assert persisted["revoked_at"] is not None


class TestRevokeActivationErrors:
    def test_no_activation_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.delete(f"/v1/agents/{_AGENT_ID}/skills/skill_missing")
        assert resp.status_code == 404

    def test_no_activation_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.delete(f"/v1/agents/{_AGENT_ID}/skills/skill_missing").json()
        assert body["error"]["code"] == "skill_activation_not_found"

    def test_already_revoked_returns_409(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}")
        resp = client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}")
        assert resp.status_code == 409

    def test_already_revoked_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}")
        body = client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}").json()
        assert body["error"]["code"] == "skill_activation_conflict"


class TestRevokeActivationAudit:
    def test_success_writes_revoked_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill.activation.revoked" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.revoked"
        )
        assert record["level"] == "info"

    def test_success_audit_detail_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.revoked"
        )
        assert record["detail"]["agent_id"] == _AGENT_ID

    def test_success_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.revoked"
        )
        assert record["detail"]["skill_id"] == skill["id"]

    def test_failure_writes_revoke_failed_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/skill_missing")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill.activation.revoke.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/skill_missing")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.activation.revoke.failed"
        )
        assert record["level"] == "error"


class TestRevokeActivationOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        _otel_exporter.clear()
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/{skill['id']}")
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.activation.revoke" in names

    def test_failure_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        _otel_exporter.clear()
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/skill_missing")
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.activation.revoke" in names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        _otel_exporter.clear()
        client.delete(f"/v1/agents/{_AGENT_ID}/skills/skill_missing")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.activation.revoke")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# GET /v1/agents/{agent_id}/skills — list activations
# ---------------------------------------------------------------------------


class TestListActivationsSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get(f"/v1/agents/{_AGENT_ID}/skills")
        assert resp.status_code == 200

    def test_empty_when_no_activations(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert body["items"] == []
        assert body["total"] == 0

    def test_response_has_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert "items" in body

    def test_response_has_total(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert "total" in body

    def test_response_has_limit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert "limit" in body

    def test_response_has_offset(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert "offset" in body

    def test_default_limit_is_20(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert body["limit"] == 20

    def test_default_offset_is_0(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert body["offset"] == 0

    def test_one_activation_returned(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert body["total"] == 1
        assert len(body["items"]) == 1

    def test_item_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        activation = _request_activation(client, _AGENT_ID, skill["id"])
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert body["items"][0]["id"] == activation["id"]

    def test_item_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert body["items"][0]["agent_id"] == _AGENT_ID

    def test_item_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert body["items"][0]["skill_id"] == skill["id"]

    def test_item_has_skill_version_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert body["items"][0]["skill_version_id"] == skill["version"]["id"]

    def test_item_has_status(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert body["items"][0]["status"] == "pending"

    def test_item_has_requested_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert isinstance(body["items"][0]["requested_at"], str)

    def test_item_has_approved_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert "approved_at" in body["items"][0]

    def test_item_has_revoked_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert "revoked_at" in body["items"][0]

    def test_filtered_by_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill_a = _create_skill(client, name="skill-a")
        skill_b = _create_skill(client, name="skill-b")
        _request_activation(client, _AGENT_ID, skill_a["id"])
        _request_activation(client, _OTHER_AGENT_ID, skill_b["id"])
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert body["total"] == 1
        assert body["items"][0]["agent_id"] == _AGENT_ID

    def test_multiple_activations_same_agent(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill_a = _create_skill(client, name="skill-a")
        skill_b = _create_skill(client, name="skill-b")
        _request_activation(client, _AGENT_ID, skill_a["id"])
        _request_activation(client, _AGENT_ID, skill_b["id"])
        body = client.get(f"/v1/agents/{_AGENT_ID}/skills").json()
        assert body["total"] == 2
        assert len(body["items"]) == 2


class TestListActivationsPagination:
    def test_limit_restricts_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill_a = _create_skill(client, name="skill-a")
        skill_b = _create_skill(client, name="skill-b")
        _request_activation(client, _AGENT_ID, skill_a["id"])
        _request_activation(client, _AGENT_ID, skill_b["id"])
        body = client.get(
            f"/v1/agents/{_AGENT_ID}/skills", params={"limit": 1}
        ).json()
        assert len(body["items"]) == 1

    def test_limit_reflected_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get(
            f"/v1/agents/{_AGENT_ID}/skills", params={"limit": 5}
        ).json()
        assert body["limit"] == 5

    def test_offset_reflected_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get(
            f"/v1/agents/{_AGENT_ID}/skills", params={"offset": 3}
        ).json()
        assert body["offset"] == 3

    def test_total_reflects_full_count_not_page(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill_a = _create_skill(client, name="skill-a")
        skill_b = _create_skill(client, name="skill-b")
        skill_c = _create_skill(client, name="skill-c")
        _request_activation(client, _AGENT_ID, skill_a["id"])
        _request_activation(client, _AGENT_ID, skill_b["id"])
        _request_activation(client, _AGENT_ID, skill_c["id"])
        body = client.get(
            f"/v1/agents/{_AGENT_ID}/skills", params={"limit": 2}
        ).json()
        assert body["total"] == 3
        assert len(body["items"]) == 2

    def test_offset_beyond_total_returns_empty(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        _request_activation(client, _AGENT_ID, skill["id"])
        body = client.get(
            f"/v1/agents/{_AGENT_ID}/skills", params={"offset": 100}
        ).json()
        assert body["items"] == []
        assert body["total"] == 1


class TestListActivationsOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.get(f"/v1/agents/{_AGENT_ID}/skills")
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.activation.list" in names

    def test_span_has_agent_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.get(f"/v1/agents/{_AGENT_ID}/skills")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.activation.list")
        assert span is not None
        assert span.attributes["agent.id"] == _AGENT_ID
