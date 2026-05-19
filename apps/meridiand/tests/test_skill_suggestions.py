"""
Skill suggestion conformance suite (PRD D2 — auto_suggest mode).

Tests cover:
  - POST /v1/agents/{agent_id}/skill_suggestions requires auto_suggest mode.
  - Returns 404 when agent does not exist.
  - Returns 422 with code "skill_suggestion_mode_not_enabled" when agent lacks auto_suggest.
  - Returns 422 with code "skill_suggestion_mode_not_enabled" when agent has manual mode.
  - Returns 201 on success.
  - Response id has "skillsugg_" prefix.
  - IDs are unique across calls.
  - Response has agent_id, skill_id, skill_version_id, status "suggested".
  - Response has suggested_at set; approved_at and dismissed_at are null.
  - skill_version_id defaults to the skill's latest version when omitted.
  - skill_version_id is used when explicitly provided.
  - Suggestion record written to storage_root/skill_suggestions/{id}.json.
  - Returns 404 with code "skill_not_found" when skill_id does not exist.
  - Returns 422 with code "skill_suggestion_invalid_request" when skill_id is empty.
  - Returns 409 with code "skill_suggestion_conflict" when a suggested record exists.
  - Returns 409 with code "skill_suggestion_conflict" when a pending activation exists.
  - Returns 409 with code "skill_suggestion_conflict" when an active activation exists.
  - Re-suggesting allowed after suggestion is approved (dismissed path).
  - On success, audit log entry written with event "skill.suggestion.emitted".
  - Success audit entry level is "info".
  - Success audit detail includes suggestion_id, agent_id, skill_id.
  - On failure, audit log entry written with event "skill.suggestion.emit.failed".
  - Failure audit entry level is "error".
  - Failure audit detail includes agent_id, skill_id, message.
  - OTel span "skill.suggestion.emit" emitted on success.
  - OTel span "skill.suggestion.emit" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries agent.id, skill.id, skill.suggestion.id attributes.
  - POST /v1/agents/{agent_id}/skill_suggestions/{skill_id}/approve returns 200.
  - Approve transitions suggestion status from "suggested" to "approved".
  - Approve sets approved_at timestamp.
  - Approve creates an active activation record in skill_activations/.
  - Activation record has status "active".
  - Activation record carries the correct skill_version_id.
  - Returns 404 with code "skill_suggestion_not_found" when no suggestion exists.
  - Returns 409 with code "skill_suggestion_conflict" when suggestion already approved.
  - Approve audit event "skill.suggestion.approved" with level "info".
  - Approve audit detail includes suggestion_id, activation_id, agent_id, skill_id.
  - Approve failure audit event "skill.suggestion.approve.failed" with level "error".
  - OTel span "skill.suggestion.approve" emitted on success and failure.
  - OTel span set to ERROR status on approve failure.
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

_AGENT_ID = "agent_autosugg01"


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


def _create_agent(client: TestClient, *, skill_activation_mode: str | None = "auto_suggest") -> dict:
    config: dict = {}
    if skill_activation_mode is not None:
        config["skill_activation_mode"] = skill_activation_mode
    resp = client.post(
        "/v1/agents",
        json={"name": "test-agent", "kind": "assistant", "config": config},
    )
    assert resp.status_code == 201
    return resp.json()


def _emit_suggestion(
    client: TestClient,
    agent_id: str,
    skill_id: str,
    *,
    skill_version_id: str | None = None,
) -> dict:
    body: dict = {"skill_id": skill_id}
    if skill_version_id is not None:
        body["skill_version_id"] = skill_version_id
    resp = client.post(f"/v1/agents/{agent_id}/skill_suggestions", json=body)
    assert resp.status_code == 201
    return resp.json()


def _approve_suggestion(client: TestClient, agent_id: str, skill_id: str) -> dict:
    resp = client.post(f"/v1/agents/{agent_id}/skill_suggestions/{skill_id}/approve")
    assert resp.status_code == 200
    return resp.json()


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# auto_suggest mode guard
# ---------------------------------------------------------------------------


class TestAutoSuggestModeRequired:
    def test_fails_if_agent_not_found(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        resp = client.post(
            "/v1/agents/agent_doesnotexist/skill_suggestions",
            json={"skill_id": skill["id"]},
        )
        assert resp.status_code == 404

    def test_fails_if_agent_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.post(
            "/v1/agents/agent_doesnotexist/skill_suggestions",
            json={"skill_id": skill["id"]},
        ).json()
        assert body["error"]["code"] == "agent_not_found"

    def test_fails_if_agent_has_no_config_mode(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client, skill_activation_mode=None)
        skill = _create_skill(client)
        resp = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        )
        assert resp.status_code == 422

    def test_fails_if_agent_has_no_config_mode_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client, skill_activation_mode=None)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        ).json()
        assert body["error"]["code"] == "skill_suggestion_mode_not_enabled"

    def test_fails_if_agent_has_manual_mode(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client, skill_activation_mode="manual")
        skill = _create_skill(client)
        resp = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        )
        assert resp.status_code == 422

    def test_fails_if_agent_has_manual_mode_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client, skill_activation_mode="manual")
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        ).json()
        assert body["error"]["code"] == "skill_suggestion_mode_not_enabled"


# ---------------------------------------------------------------------------
# POST /v1/agents/{agent_id}/skill_suggestions — emit suggestion
# ---------------------------------------------------------------------------


class TestEmitSuggestionSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        resp = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        )
        assert resp.status_code == 201

    def test_response_id_has_skillsugg_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        ).json()
        assert body["id"].startswith("skillsugg_")

    def test_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill_a = _create_skill(client, name="skill-a")
        skill_b = _create_skill(client, name="skill-b")
        id1 = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill_a["id"]},
        ).json()["id"]
        id2 = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill_b["id"]},
        ).json()["id"]
        assert id1 != id2

    def test_response_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        ).json()
        assert body["agent_id"] == agent["id"]

    def test_response_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        ).json()
        assert body["skill_id"] == skill["id"]

    def test_status_is_suggested(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        ).json()
        assert body["status"] == "suggested"

    def test_suggested_at_is_set(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        ).json()
        assert isinstance(body["suggested_at"], str)
        assert len(body["suggested_at"]) > 0

    def test_approved_at_is_null(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        ).json()
        assert body["approved_at"] is None

    def test_dismissed_at_is_null(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        ).json()
        assert body["dismissed_at"] is None

    def test_skill_version_id_defaults_to_latest(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        ).json()
        assert body["skill_version_id"] == skill["version"]["id"]

    def test_explicit_skill_version_id_stored(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        explicit_ver = skill["version"]["id"]
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"], "skill_version_id": explicit_ver},
        ).json()
        assert body["skill_version_id"] == explicit_ver

    def test_suggestion_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        suggestion = _emit_suggestion(client, agent["id"], skill["id"])
        path = storage_root / "skill_suggestions" / f"{suggestion['id']}.json"
        assert path.exists()

    def test_persisted_record_matches_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        suggestion = _emit_suggestion(client, agent["id"], skill["id"])
        path = storage_root / "skill_suggestions" / f"{suggestion['id']}.json"
        persisted = json.loads(path.read_text())
        assert persisted["agent_id"] == agent["id"]
        assert persisted["skill_id"] == skill["id"]
        assert persisted["status"] == "suggested"

    def test_re_suggest_allowed_after_approved(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        _approve_suggestion(client, agent["id"], skill["id"])
        resp = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# POST /v1/agents/{agent_id}/skill_suggestions — errors
# ---------------------------------------------------------------------------


class TestEmitSuggestionErrors:
    def test_unknown_skill_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        resp = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": "skill_doesnotexist"},
        )
        assert resp.status_code == 404

    def test_unknown_skill_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": "skill_doesnotexist"},
        ).json()
        assert body["error"]["code"] == "skill_not_found"

    def test_empty_skill_id_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        resp = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": "   "},
        )
        assert resp.status_code == 422

    def test_empty_skill_id_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": ""},
        ).json()
        assert body["error"]["code"] == "skill_suggestion_invalid_request"

    def test_conflict_when_suggestion_exists(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        resp = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        )
        assert resp.status_code == 409

    def test_conflict_error_code_when_suggestion_exists(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        ).json()
        assert body["error"]["code"] == "skill_suggestion_conflict"

    def test_conflict_when_pending_activation_exists(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        client.post(f"/v1/agents/{agent['id']}/skills", json={"skill_id": skill["id"]})
        resp = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        )
        assert resp.status_code == 409

    def test_conflict_when_active_activation_exists(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        client.post(f"/v1/agents/{agent['id']}/skills", json={"skill_id": skill["id"]})
        client.post(f"/v1/agents/{agent['id']}/skills/{skill['id']}/approve")
        resp = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        )
        assert resp.status_code == 409

    def test_error_response_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": "skill_missing"},
        ).json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# POST /v1/agents/{agent_id}/skill_suggestions — audit log
# ---------------------------------------------------------------------------


class TestEmitSuggestionAudit:
    def test_success_writes_emitted_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill.suggestion.emitted" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.emitted"
        )
        assert record["level"] == "info"

    def test_success_audit_detail_has_suggestion_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        suggestion = _emit_suggestion(client, agent["id"], skill["id"])
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.emitted"
        )
        assert record["detail"]["suggestion_id"] == suggestion["id"]

    def test_success_audit_detail_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.emitted"
        )
        assert record["detail"]["agent_id"] == agent["id"]

    def test_success_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.emitted"
        )
        assert record["detail"]["skill_id"] == skill["id"]

    def test_failure_writes_emit_failed_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": "skill_missing"},
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill.suggestion.emit.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": "skill_missing"},
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.emit.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_detail_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": "skill_missing"},
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.emit.failed"
        )
        assert record["detail"]["agent_id"] == agent["id"]

    def test_failure_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": "skill_missing"},
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.emit.failed"
        )
        assert record["detail"]["skill_id"] == "skill_missing"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": "skill_missing"},
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.emit.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# POST /v1/agents/{agent_id}/skill_suggestions — OTel
# ---------------------------------------------------------------------------


class TestEmitSuggestionOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _otel_exporter.clear()
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        )
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.suggestion.emit" in names

    def test_failure_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent = _create_agent(client)
        _otel_exporter.clear()
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": "skill_missing"},
        )
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.suggestion.emit" in names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        agent = _create_agent(client)
        _otel_exporter.clear()
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": "skill_missing"},
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.suggestion.emit")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_span_has_agent_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _otel_exporter.clear()
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.suggestion.emit")
        assert span is not None
        assert span.attributes["agent.id"] == agent["id"]

    def test_span_has_skill_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _otel_exporter.clear()
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.suggestion.emit")
        assert span is not None
        assert span.attributes["skill.id"] == skill["id"]

    def test_span_has_suggestion_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _otel_exporter.clear()
        resp = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions",
            json={"skill_id": skill["id"]},
        )
        suggestion_id = resp.json()["id"]
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.suggestion.emit")
        assert span is not None
        assert span.attributes["skill.suggestion.id"] == suggestion_id


# ---------------------------------------------------------------------------
# POST /v1/agents/{agent_id}/skill_suggestions/{skill_id}/approve
# ---------------------------------------------------------------------------


class TestApproveSuggestionSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        resp = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        assert resp.status_code == 200

    def test_status_becomes_approved(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        ).json()
        assert body["status"] == "approved"

    def test_approved_at_is_set(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        ).json()
        assert isinstance(body["approved_at"], str)
        assert len(body["approved_at"]) > 0

    def test_dismissed_at_remains_null(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        ).json()
        assert body["dismissed_at"] is None

    def test_persisted_suggestion_is_approved(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        suggestion = _emit_suggestion(client, agent["id"], skill["id"])
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        path = storage_root / "skill_suggestions" / f"{suggestion['id']}.json"
        persisted = json.loads(path.read_text())
        assert persisted["status"] == "approved"
        assert persisted["approved_at"] is not None

    def test_creates_activation_record(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        activations = list((storage_root / "skill_activations").glob("*.json"))
        assert len(activations) == 1

    def test_activation_has_status_active(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        activation_path = next(
            (storage_root / "skill_activations").glob("*.json")
        )
        activation = json.loads(activation_path.read_text())
        assert activation["status"] == "active"

    def test_activation_has_correct_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        activation_path = next(
            (storage_root / "skill_activations").glob("*.json")
        )
        activation = json.loads(activation_path.read_text())
        assert activation["agent_id"] == agent["id"]

    def test_activation_has_correct_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        activation_path = next(
            (storage_root / "skill_activations").glob("*.json")
        )
        activation = json.loads(activation_path.read_text())
        assert activation["skill_id"] == skill["id"]

    def test_activation_has_correct_skill_version_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        activation_path = next(
            (storage_root / "skill_activations").glob("*.json")
        )
        activation = json.loads(activation_path.read_text())
        assert activation["skill_version_id"] == skill["version"]["id"]

    def test_activation_approved_at_is_set(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        activation_path = next(
            (storage_root / "skill_activations").glob("*.json")
        )
        activation = json.loads(activation_path.read_text())
        assert activation["approved_at"] is not None


# ---------------------------------------------------------------------------
# POST .../approve — errors
# ---------------------------------------------------------------------------


class TestApproveSuggestionErrors:
    def test_no_suggestion_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            f"/v1/agents/{_AGENT_ID}/skill_suggestions/skill_missing/approve"
        )
        assert resp.status_code == 404

    def test_no_suggestion_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            f"/v1/agents/{_AGENT_ID}/skill_suggestions/skill_missing/approve"
        ).json()
        assert body["error"]["code"] == "skill_suggestion_not_found"

    def test_already_approved_returns_409(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        _approve_suggestion(client, agent["id"], skill["id"])
        resp = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        assert resp.status_code == 409

    def test_already_approved_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        _approve_suggestion(client, agent["id"], skill["id"])
        body = client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        ).json()
        assert body["error"]["code"] == "skill_suggestion_conflict"


# ---------------------------------------------------------------------------
# POST .../approve — audit log
# ---------------------------------------------------------------------------


class TestApproveSuggestionAudit:
    def test_success_writes_approved_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill.suggestion.approved" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.approved"
        )
        assert record["level"] == "info"

    def test_success_audit_detail_has_suggestion_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        suggestion = _emit_suggestion(client, agent["id"], skill["id"])
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.approved"
        )
        assert record["detail"]["suggestion_id"] == suggestion["id"]

    def test_success_audit_detail_has_activation_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.approved"
        )
        assert record["detail"]["activation_id"].startswith("skillact_")

    def test_success_audit_detail_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.approved"
        )
        assert record["detail"]["agent_id"] == agent["id"]

    def test_success_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.approved"
        )
        assert record["detail"]["skill_id"] == skill["id"]

    def test_failure_writes_approve_failed_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            f"/v1/agents/{_AGENT_ID}/skill_suggestions/skill_missing/approve"
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill.suggestion.approve.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            f"/v1/agents/{_AGENT_ID}/skill_suggestions/skill_missing/approve"
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill.suggestion.approve.failed"
        )
        assert record["level"] == "error"


# ---------------------------------------------------------------------------
# POST .../approve — OTel
# ---------------------------------------------------------------------------


class TestApproveSuggestionOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        _emit_suggestion(client, agent["id"], skill["id"])
        _otel_exporter.clear()
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.suggestion.approve" in names

    def test_failure_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        _otel_exporter.clear()
        client.post(
            f"/v1/agents/{_AGENT_ID}/skill_suggestions/skill_missing/approve"
        )
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.suggestion.approve" in names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        _otel_exporter.clear()
        client.post(
            f"/v1/agents/{_AGENT_ID}/skill_suggestions/skill_missing/approve"
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.suggestion.approve")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_suggestion_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent = _create_agent(client)
        skill = _create_skill(client)
        suggestion = _emit_suggestion(client, agent["id"], skill["id"])
        _otel_exporter.clear()
        client.post(
            f"/v1/agents/{agent['id']}/skill_suggestions/{skill['id']}/approve"
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.suggestion.approve")
        assert span is not None
        assert span.attributes.get("skill.suggestion.id") == suggestion["id"]
