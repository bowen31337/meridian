"""
Agents endpoint conformance suite.

Tests cover:
  - GET /v1/agents returns 200 with items list.
  - Items list is empty when no agents exist.
  - Items list contains all non-deleted agents.
  - Items are sorted newest first (descending created_at, id).
  - Soft-deleted agents are excluded from the list.
  - name query parameter filters by name prefix.
  - created_after query parameter filters by created_at lower bound (exclusive).
  - created_before query parameter filters by created_at upper bound (exclusive).
  - Response has next_cursor null when no more pages.
  - Response has next_cursor token when more pages exist.
  - Link response header set when next_cursor is present.
  - Cursor-based pagination returns the correct next page.
  - limit query parameter controls page size.
  - Invalid cursor returns 400 with code "cursor_invalid".
  - Invalid cursor writes audit entry with event "agent.list.failed".
  - Audit entry level is "error" on cursor failure.
  - Audit entry detail includes message.
  - OTel span "agent.list" emitted on success and failure.
  - Failure span has ERROR status.
  - GET /v1/agents/{id}/versions returns 200 with items list.
  - Items list is empty for unknown agent id.
  - Items list contains all versions for the agent.
  - Items are sorted newest first (descending created_at, id).
  - Items only include versions belonging to the requested agent.
  - Response has next_cursor null when no more pages.
  - Response has next_cursor token when more pages exist.
  - Link response header set when next_cursor is present (middleware converts X-Next-Cursor).
  - Cursor-based pagination returns the correct next page.
  - limit query parameter controls page size.
  - Invalid cursor returns 400 with code "cursor_invalid".
  - Invalid cursor writes audit entry with event "agent.versions.list.failed".
  - Audit entry level is "error" on cursor failure.
  - Audit entry detail includes agent_id and message.
  - OTel span "agent.versions.list" emitted on success and failure.
  - Failure span has ERROR status.
  - Span carries agent.id attribute.
  - GET /v1/agents/{id}/versions/{ver} returns 200 with version record on success.
  - Response fields match the stored AgentVersionRecord.
  - Unknown agent id returns 404 with code "agent_not_found".
  - Unknown version id returns 404 with code "agent_version_not_found".
  - Version belonging to a different agent returns 404 with code "agent_version_not_found".
  - On failure, audit log entry written with event "agent.version.get.failed".
  - Audit entry level is "error" on failure.
  - Audit entry detail includes agent_id, version_id, message.
  - OTel span "agent.version.get" emitted on success and failure.
  - Failure span has ERROR status.
  - Span carries agent.id and agent.version.id attributes.
  - POST /v1/agents returns 201 on success.
  - Response has id with "agent_" prefix.
  - IDs are unique across calls.
  - Response has name, kind, created_at, version.
  - version object present in response.
  - version has id with "agentver_" prefix.
  - POST /v1/agents/{id}/versions returns 201 for a new version.
  - POST /v1/agents/{id}/versions returns 200 when hash matches existing version.
  - Version id has "agentver_" prefix.
  - Version id is SHA-256 of canonical JSON prefixed "agentver_".
  - Same content returns the same version id (idempotent).
  - Different content produces a different version id.
  - version_number increments for new content.
  - Existing version returned verbatim when hash matches.
  - Version JSON written to storage_root/agent_versions/{version_id}.json.
  - Unknown agent id returns 404 with code "agent_not_found".
  - Empty name returns 422 with code "agent_invalid_request".
  - Empty kind returns 422 with code "agent_invalid_request".
  - On failure, audit log entry written with event "agent.version.create.failed".
  - Audit entry level is "error" on failure.
  - Audit entry detail includes agent_id and message.
  - OTel span "agent.version.create" emitted on success and failure.
  - Failure span has ERROR status.
  - Span carries agent.id attribute.
  - version id is SHA-256 of canonical JSON body prefixed "agentver_".
  - Content-addressed id is stable: same agent_id+content -> same version id.
  - version has agent_id matching the agent id.
  - version has version_number 1 on first creation.
  - version has name, kind, config, capabilities, created_at.
  - config defaults to empty dict when omitted.
  - capabilities defaults to empty list when omitted.
  - config stored when provided.
  - capabilities stored when provided.
  - Agent JSON written to storage_root/agents/{id}.json.
  - AgentVersionRecord JSON written to storage_root/agent_versions/{version_id}.json.
  - Persisted agent has correct name and kind.
  - Persisted version has correct name, kind, config, capabilities.
  - Not written to disk on validation failure.
  - Empty name returns 422 with code "agent_invalid_request".
  - Empty kind returns 422 with code "agent_invalid_request".
  - Missing required fields return 422.
  - On validation failure, audit log entry written with event "agent.create.failed".
  - Audit entry level is "error" on failure.
  - Audit entry code is "agent_invalid_request" on validation failure.
  - Audit detail includes agent_id, name, message on failure.
  - Error response body has error.code and error.message on failure.
  - OTel span "agent.create" emitted on success.
  - OTel span "agent.create" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries agent.id and agent.name attributes.
  - create_app wires agents router when storage_root is supplied.
  - create_app omits agents route when storage_root is None.
  - DELETE /v1/agents/{id} returns 204 on success.
  - Soft-delete sets deleted_at on the persisted agent JSON.
  - Agent file is retained after soft-delete (not removed).
  - Agent version files are retained after soft-delete.
  - Unknown agent id returns 404 with code "agent_not_found".
  - 404 writes audit entry with event "agent.delete.failed".
  - Audit entry on failure has level "error" and detail.agent_id.
  - OTel span "agent.delete" emitted on success and failure.
  - Failure span has ERROR status.
  - Span carries agent.id attribute.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._agents import _content_version_id, make_agents_router
from core_errors import HandlerOptions, install_error_handler

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _body(**overrides) -> dict:
    base: dict = {
        "name": "my-agent",
        "kind": "claude",
    }
    base.update(overrides)
    return base


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _agent_resource(storage_root: Path, agent_id: str) -> dict:
    return json.loads((storage_root / "agents" / f"{agent_id}.json").read_text())


def _version_resource(storage_root: Path, version_id: str) -> dict:
    return json.loads((storage_root / "agent_versions" / f"{version_id}.json").read_text())


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestAgentCreateSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/agents", json=_body())
        assert resp.status_code == 201

    def test_with_config_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/agents", json=_body(config={"model": "claude-3-opus"}))
        assert resp.status_code == 201

    def test_with_capabilities_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/agents", json=_body(capabilities=["read", "write"]))
        assert resp.status_code == 201

    def test_minimal_body_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/agents", json={"name": "min-agent", "kind": "openai"})
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Response fields
# ---------------------------------------------------------------------------


class TestAgentCreateResponse:
    def test_response_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body()).json()
        assert "id" in body
        assert isinstance(body["id"], str)

    def test_id_has_agent_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body()).json()
        assert body["id"].startswith("agent_")

    def test_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = client.post("/v1/agents", json=_body()).json()["id"]
        id2 = client.post("/v1/agents", json=_body()).json()["id"]
        assert id1 != id2

    def test_response_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body(name="deploy-agent")).json()
        assert body["name"] == "deploy-agent"

    def test_response_has_kind(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body(kind="openai")).json()
        assert body["kind"] == "openai"

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body()).json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0

    def test_response_has_version(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body()).json()
        assert "version" in body
        assert isinstance(body["version"], dict)

    def test_version_id_has_agentver_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body()).json()
        assert body["version"]["id"].startswith("agentver_")

    def test_version_id_is_content_addressed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body(config={"k": "v"}, capabilities=["read"])).json()
        agent_id = body["id"]
        expected = _content_version_id(
            agent_id=agent_id,
            name="my-agent",
            kind="claude",
            config={"k": "v"},
            capabilities=["read"],
        )
        assert body["version"]["id"] == expected

    def test_version_id_stable_for_same_content(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body1 = client.post("/v1/agents", json=_body()).json()
        body2 = client.post("/v1/agents", json=_body()).json()
        agent_id1 = body1["id"]
        agent_id2 = body2["id"]
        ver1 = _content_version_id(
            agent_id=agent_id1, name="my-agent", kind="claude", config={}, capabilities=[]
        )
        ver2 = _content_version_id(
            agent_id=agent_id2, name="my-agent", kind="claude", config={}, capabilities=[]
        )
        assert body1["version"]["id"] == ver1
        assert body2["version"]["id"] == ver2

    def test_version_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body()).json()
        assert body["version"]["agent_id"] == body["id"]

    def test_version_has_version_number_1(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body()).json()
        assert body["version"]["version_number"] == 1

    def test_version_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body(name="named-agent")).json()
        assert body["version"]["name"] == "named-agent"

    def test_version_has_kind(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body(kind="openai")).json()
        assert body["version"]["kind"] == "openai"

    def test_version_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body()).json()
        assert "created_at" in body["version"]
        assert isinstance(body["version"]["created_at"], str)

    def test_version_config_defaults_to_empty_dict(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body()).json()
        assert body["version"]["config"] == {}

    def test_version_config_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cfg = {"model": "claude-3-opus", "temperature": 0.7}
        body = client.post("/v1/agents", json=_body(config=cfg)).json()
        assert body["version"]["config"] == cfg

    def test_version_capabilities_defaults_to_empty_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body()).json()
        assert body["version"]["capabilities"] == []

    def test_version_capabilities_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body(capabilities=["read", "write"])).json()
        assert body["version"]["capabilities"] == ["read", "write"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestAgentPersistence:
    def test_agent_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        assert (storage_root / "agents" / f"{agent_id}.json").exists()

    def test_version_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body()).json()
        version_id = body["version"]["id"]
        assert (storage_root / "agent_versions" / f"{version_id}.json").exists()

    def test_persisted_agent_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = client.post("/v1/agents", json=_body(name="stored-agent")).json()["id"]
        resource = _agent_resource(storage_root, agent_id)
        assert resource["name"] == "stored-agent"

    def test_persisted_agent_kind(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = client.post("/v1/agents", json=_body(kind="openai")).json()["id"]
        resource = _agent_resource(storage_root, agent_id)
        assert resource["kind"] == "openai"

    def test_persisted_version_config(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cfg = {"model": "gpt-4"}
        resp_body = client.post("/v1/agents", json=_body(config=cfg)).json()
        version_id = resp_body["version"]["id"]
        resource = _version_resource(storage_root, version_id)
        assert resource["config"] == cfg

    def test_persisted_version_capabilities(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp_body = client.post("/v1/agents", json=_body(capabilities=["search"])).json()
        version_id = resp_body["version"]["id"]
        resource = _version_resource(storage_root, version_id)
        assert resource["capabilities"] == ["search"]

    def test_persisted_version_agent_id_matches(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp_body = client.post("/v1/agents", json=_body()).json()
        agent_id = resp_body["id"]
        version_id = resp_body["version"]["id"]
        resource = _version_resource(storage_root, version_id)
        assert resource["agent_id"] == agent_id

    def test_not_written_on_validation_failure(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name=""))
        agents_dir = storage_root / "agents"
        files = list(agents_dir.glob("*.json")) if agents_dir.exists() else []
        assert files == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestAgentCreateValidation:
    def test_missing_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/agents", json={"kind": "claude"})
        assert resp.status_code == 422

    def test_empty_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/agents", json=_body(name="   "))
        assert resp.status_code == 422

    def test_empty_name_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body(name="")).json()
        assert body["error"]["code"] == "agent_invalid_request"

    def test_missing_kind_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/agents", json={"name": "my-agent"})
        assert resp.status_code == 422

    def test_empty_kind_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/agents", json=_body(kind="   "))
        assert resp.status_code == 422

    def test_empty_kind_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body(kind="")).json()
        assert body["error"]["code"] == "agent_invalid_request"

    def test_validation_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents", json=_body(name="")).json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAgentAuditLog:
    def test_validation_failure_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name=""))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "agent.create.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.create.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_code_is_agent_invalid_request(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.create.failed"
        )
        assert record["code"] == "agent_invalid_request"

    def test_failure_audit_detail_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.create.failed"
        )
        assert record["detail"]["agent_id"].startswith("agent_")

    def test_failure_audit_detail_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.create.failed"
        )
        assert "name" in record["detail"]

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.create.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestAgentOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_agent_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/agents", json=_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "agent.create" in span_names

    def test_failure_emits_agent_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/agents", json=_body(name=""))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "agent.create" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode
        client = self._client(storage_root)
        client.post("/v1/agents", json=_body(name=""))
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "agent.create"]
        assert any(s.status.status_code == StatusCode.ERROR for s in spans)

    def test_success_span_has_agent_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/agents", json=_body())
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "agent.create"]
        assert any("agent.id" in s.attributes for s in spans)

    def test_success_span_has_agent_name_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/agents", json=_body(name="span-test-agent"))
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "agent.create"]
        assert any(s.attributes.get("agent.name") == "span-test-agent" for s in spans)


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------


class TestAgentAppWiring:
    def test_create_app_wires_agents_route(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/v1/agents" in routes

    def test_create_app_omits_agents_route_without_storage_root(self) -> None:
        from meridiand._audit import FileAuditLog
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            audit_log = FileAuditLog(Path(tmp))
            app = create_app(audit_log, storage_root=None)
            routes = [r.path for r in app.routes if hasattr(r, "path")]
            assert "/v1/agents" not in routes


# ---------------------------------------------------------------------------
# Delete – success
# ---------------------------------------------------------------------------


class TestAgentDeleteSuccess:
    def test_returns_204(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        resp = client.delete(f"/v1/agents/{agent_id}")
        assert resp.status_code == 204

    def test_agent_file_still_exists(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        client.delete(f"/v1/agents/{agent_id}")
        assert (storage_root / "agents" / f"{agent_id}.json").exists()

    def test_deleted_at_set_in_agent_file(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        client.delete(f"/v1/agents/{agent_id}")
        record = _agent_resource(storage_root, agent_id)
        assert "deleted_at" in record
        assert isinstance(record["deleted_at"], str)
        assert len(record["deleted_at"]) > 0

    def test_version_file_retained_after_delete(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp_body = client.post("/v1/agents", json=_body()).json()
        agent_id = resp_body["id"]
        version_id = resp_body["version"]["id"]
        client.delete(f"/v1/agents/{agent_id}")
        assert (storage_root / "agent_versions" / f"{version_id}.json").exists()

    def test_no_audit_entry_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        client.delete(f"/v1/agents/{agent_id}")
        records = _audit_records(storage_root)
        assert not any(r.get("event") == "agent.delete.failed" for r in records)


# ---------------------------------------------------------------------------
# Delete – not found
# ---------------------------------------------------------------------------


class TestAgentDeleteNotFound:
    def test_unknown_id_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.delete("/v1/agents/agent_unknown")
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.delete("/v1/agents/agent_unknown").json()
        assert body["error"]["code"] == "agent_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.delete("/v1/agents/agent_unknown").json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0

    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/agents/agent_unknown")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "agent.delete.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/agents/agent_unknown")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "agent.delete.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/agents/agent_unknown")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "agent.delete.failed"
        )
        assert record["code"] == "agent_not_found"

    def test_not_found_audit_detail_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/agents/agent_unknown")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "agent.delete.failed"
        )
        assert record["detail"]["agent_id"] == "agent_unknown"

    def test_not_found_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/agents/agent_unknown")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "agent.delete.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# Delete – OTel spans
# ---------------------------------------------------------------------------


class TestAgentDeleteOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_agent_delete_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        _otel_exporter.clear()
        client.delete(f"/v1/agents/{agent_id}")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "agent.delete" in span_names

    def test_failure_emits_agent_delete_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.delete("/v1/agents/agent_unknown")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "agent.delete" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode
        client = self._client(storage_root)
        client.delete("/v1/agents/agent_unknown")
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "agent.delete"]
        assert any(s.status.status_code == StatusCode.ERROR for s in spans)

    def test_span_has_agent_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        _otel_exporter.clear()
        client.delete(f"/v1/agents/{agent_id}")
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "agent.delete"]
        assert any(s.attributes.get("agent.id") == agent_id for s in spans)


# ---------------------------------------------------------------------------
# Version create – helpers
# ---------------------------------------------------------------------------


def _ver_body(**overrides) -> dict:
    base: dict = {"name": "my-agent", "kind": "claude"}
    base.update(overrides)
    return base


def _create_agent(client: TestClient) -> str:
    return client.post("/v1/agents", json=_body()).json()["id"]


# ---------------------------------------------------------------------------
# Version create – success (new version)
# ---------------------------------------------------------------------------


class TestAgentVersionCreateSuccess:
    def test_returns_201_for_new_version(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        resp = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2"))
        assert resp.status_code == 201

    def test_with_config_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        resp = client.post(
            f"/v1/agents/{agent_id}/versions",
            json=_ver_body(config={"model": "claude-3-opus"}),
        )
        assert resp.status_code == 201

    def test_with_capabilities_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        resp = client.post(
            f"/v1/agents/{agent_id}/versions",
            json=_ver_body(capabilities=["read", "write"]),
        )
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Version create – idempotency (existing version)
# ---------------------------------------------------------------------------


class TestAgentVersionCreateIdempotency:
    def test_returns_200_when_hash_matches(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        payload = _ver_body(name="v2", config={"k": "v"})
        client.post(f"/v1/agents/{agent_id}/versions", json=payload)
        resp = client.post(f"/v1/agents/{agent_id}/versions", json=payload)
        assert resp.status_code == 200

    def test_same_content_returns_same_version_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        payload = _ver_body(name="stable", capabilities=["search"])
        id1 = client.post(f"/v1/agents/{agent_id}/versions", json=payload).json()["id"]
        id2 = client.post(f"/v1/agents/{agent_id}/versions", json=payload).json()["id"]
        assert id1 == id2

    def test_existing_version_returned_verbatim(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        payload = _ver_body(name="stable")
        first = client.post(f"/v1/agents/{agent_id}/versions", json=payload).json()
        second = client.post(f"/v1/agents/{agent_id}/versions", json=payload).json()
        assert first == second

    def test_different_content_produces_different_version_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        id1 = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2")).json()["id"]
        id2 = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v3")).json()["id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# Version create – response fields
# ---------------------------------------------------------------------------


class TestAgentVersionCreateResponse:
    def test_version_id_has_agentver_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2")).json()
        assert body["id"].startswith("agentver_")

    def test_version_id_is_content_addressed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        payload = _ver_body(name="v2", config={"k": "v"}, capabilities=["read"])
        body = client.post(f"/v1/agents/{agent_id}/versions", json=payload).json()
        expected = _content_version_id(
            agent_id=agent_id,
            name="v2",
            kind="claude",
            config={"k": "v"},
            capabilities=["read"],
        )
        assert body["id"] == expected

    def test_response_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2")).json()
        assert body["agent_id"] == agent_id

    def test_response_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="my-name")).json()
        assert body["name"] == "my-name"

    def test_response_has_kind(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(kind="openai")).json()
        assert body["kind"] == "openai"

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2")).json()
        assert isinstance(body.get("created_at"), str)
        assert len(body["created_at"]) > 0

    def test_response_config_defaults_to_empty_dict(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2")).json()
        assert body["config"] == {}

    def test_response_config_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        cfg = {"model": "gpt-4", "temperature": 0.5}
        body = client.post(
            f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2", config=cfg)
        ).json()
        assert body["config"] == cfg

    def test_response_capabilities_defaults_to_empty_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2")).json()
        assert body["capabilities"] == []

    def test_response_capabilities_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.post(
            f"/v1/agents/{agent_id}/versions",
            json=_ver_body(name="v2", capabilities=["search", "write"]),
        ).json()
        assert body["capabilities"] == ["search", "write"]

    def test_version_number_increments_for_new_content(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        v2 = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2")).json()
        v3 = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v3")).json()
        assert v2["version_number"] > 1
        assert v3["version_number"] > v2["version_number"]


# ---------------------------------------------------------------------------
# Version create – persistence
# ---------------------------------------------------------------------------


class TestAgentVersionCreatePersistence:
    def test_version_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        version_id = client.post(
            f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2")
        ).json()["id"]
        assert (storage_root / "agent_versions" / f"{version_id}.json").exists()

    def test_persisted_version_has_correct_fields(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        cfg = {"model": "gpt-4"}
        resp = client.post(
            f"/v1/agents/{agent_id}/versions",
            json=_ver_body(name="v2", kind="openai", config=cfg, capabilities=["search"]),
        ).json()
        resource = _version_resource(storage_root, resp["id"])
        assert resource["name"] == "v2"
        assert resource["kind"] == "openai"
        assert resource["config"] == cfg
        assert resource["capabilities"] == ["search"]
        assert resource["agent_id"] == agent_id


# ---------------------------------------------------------------------------
# Version create – not found
# ---------------------------------------------------------------------------


class TestAgentVersionCreateNotFound:
    def test_unknown_agent_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/agents/agent_unknown/versions", json=_ver_body())
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents/agent_unknown/versions", json=_ver_body()).json()
        assert body["error"]["code"] == "agent_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/agents/agent_unknown/versions", json=_ver_body()).json()
        assert len(body["error"]["message"]) > 0

    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents/agent_unknown/versions", json=_ver_body())
        records = _audit_records(storage_root)
        assert any(r.get("event") == "agent.version.create.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents/agent_unknown/versions", json=_ver_body())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.version.create.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents/agent_unknown/versions", json=_ver_body())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.version.create.failed"
        )
        assert record["code"] == "agent_not_found"

    def test_not_found_audit_detail_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents/agent_unknown/versions", json=_ver_body())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.version.create.failed"
        )
        assert record["detail"]["agent_id"] == "agent_unknown"


# ---------------------------------------------------------------------------
# Version create – validation errors
# ---------------------------------------------------------------------------


class TestAgentVersionCreateValidation:
    def test_empty_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        resp = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name=""))
        assert resp.status_code == 422

    def test_empty_name_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="")).json()
        assert body["error"]["code"] == "agent_invalid_request"

    def test_empty_kind_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        resp = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(kind=""))
        assert resp.status_code == 422

    def test_empty_kind_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(kind="")).json()
        assert body["error"]["code"] == "agent_invalid_request"

    def test_validation_failure_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name=""))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "agent.version.create.failed" for r in records)

    def test_validation_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.version.create.failed"
        )
        assert record["level"] == "error"

    def test_validation_failure_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.version.create.failed"
        )
        assert record["code"] == "agent_invalid_request"

    def test_validation_failure_audit_detail_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.version.create.failed"
        )
        assert record["detail"]["agent_id"] == agent_id

    def test_validation_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.version.create.failed"
        )
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# Version create – OTel spans
# ---------------------------------------------------------------------------


class TestAgentVersionCreateOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_version_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        _otel_exporter.clear()
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2"))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "agent.version.create" in span_names

    def test_failure_emits_version_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/agents/agent_unknown/versions", json=_ver_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "agent.version.create" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode
        client = self._client(storage_root)
        client.post("/v1/agents/agent_unknown/versions", json=_ver_body())
        spans = [
            s for s in _otel_exporter.get_finished_spans()
            if s.name == "agent.version.create"
        ]
        assert any(s.status.status_code == StatusCode.ERROR for s in spans)

    def test_span_has_agent_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        _otel_exporter.clear()
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2"))
        spans = [
            s for s in _otel_exporter.get_finished_spans()
            if s.name == "agent.version.create"
        ]
        assert any(s.attributes.get("agent.id") == agent_id for s in spans)


# ---------------------------------------------------------------------------
# Version get – helpers
# ---------------------------------------------------------------------------


def _create_agent_with_version(client: TestClient) -> tuple[str, str]:
    """Returns (agent_id, version_id)."""
    resp = client.post("/v1/agents", json=_body()).json()
    return resp["id"], resp["version"]["id"]


# ---------------------------------------------------------------------------
# Version get – success
# ---------------------------------------------------------------------------


class TestAgentVersionGetSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id, version_id = _create_agent_with_version(client)
        resp = client.get(f"/v1/agents/{agent_id}/versions/{version_id}")
        assert resp.status_code == 200

    def test_response_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id, version_id = _create_agent_with_version(client)
        body = client.get(f"/v1/agents/{agent_id}/versions/{version_id}").json()
        assert body["id"] == version_id

    def test_response_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id, version_id = _create_agent_with_version(client)
        body = client.get(f"/v1/agents/{agent_id}/versions/{version_id}").json()
        assert body["agent_id"] == agent_id

    def test_response_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/agents", json=_body(name="named-agent")).json()
        agent_id, version_id = resp["id"], resp["version"]["id"]
        body = client.get(f"/v1/agents/{agent_id}/versions/{version_id}").json()
        assert body["name"] == "named-agent"

    def test_response_has_kind(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/agents", json=_body(kind="openai")).json()
        agent_id, version_id = resp["id"], resp["version"]["id"]
        body = client.get(f"/v1/agents/{agent_id}/versions/{version_id}").json()
        assert body["kind"] == "openai"

    def test_response_has_version_number(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id, version_id = _create_agent_with_version(client)
        body = client.get(f"/v1/agents/{agent_id}/versions/{version_id}").json()
        assert isinstance(body.get("version_number"), int)

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id, version_id = _create_agent_with_version(client)
        body = client.get(f"/v1/agents/{agent_id}/versions/{version_id}").json()
        assert isinstance(body.get("created_at"), str)
        assert len(body["created_at"]) > 0

    def test_response_has_config(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        cfg = {"model": "claude-3-opus"}
        resp = client.post("/v1/agents", json=_body(config=cfg)).json()
        agent_id, version_id = resp["id"], resp["version"]["id"]
        body = client.get(f"/v1/agents/{agent_id}/versions/{version_id}").json()
        assert body["config"] == cfg

    def test_response_has_capabilities(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        caps = ["read", "write"]
        resp = client.post("/v1/agents", json=_body(capabilities=caps)).json()
        agent_id, version_id = resp["id"], resp["version"]["id"]
        body = client.get(f"/v1/agents/{agent_id}/versions/{version_id}").json()
        assert body["capabilities"] == caps

    def test_response_matches_stored_record(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id, version_id = _create_agent_with_version(client)
        body = client.get(f"/v1/agents/{agent_id}/versions/{version_id}").json()
        stored = _version_resource(storage_root, version_id)
        assert body == stored

    def test_returns_version_created_via_version_endpoint(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        create_resp = client.post(
            f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2")
        ).json()
        version_id = create_resp["id"]
        get_resp = client.get(f"/v1/agents/{agent_id}/versions/{version_id}").json()
        assert get_resp == create_resp


# ---------------------------------------------------------------------------
# Version get – not found
# ---------------------------------------------------------------------------


class TestAgentVersionGetNotFound:
    def test_unknown_agent_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/agents/agent_unknown/versions/agentver_abc123")
        assert resp.status_code == 404

    def test_unknown_agent_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/agents/agent_unknown/versions/agentver_abc123").json()
        assert body["error"]["code"] == "agent_not_found"

    def test_unknown_agent_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/agents/agent_unknown/versions/agentver_abc123").json()
        assert len(body["error"]["message"]) > 0

    def test_unknown_version_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        resp = client.get(f"/v1/agents/{agent_id}/versions/agentver_doesnotexist")
        assert resp.status_code == 404

    def test_unknown_version_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.get(f"/v1/agents/{agent_id}/versions/agentver_doesnotexist").json()
        assert body["error"]["code"] == "agent_version_not_found"

    def test_unknown_version_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.get(f"/v1/agents/{agent_id}/versions/agentver_doesnotexist").json()
        assert len(body["error"]["message"]) > 0

    def test_version_from_other_agent_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id_a, version_id_a = _create_agent_with_version(client)
        agent_id_b = _create_agent(client)
        resp = client.get(f"/v1/agents/{agent_id_b}/versions/{version_id_a}")
        assert resp.status_code == 404

    def test_version_from_other_agent_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id_a, version_id_a = _create_agent_with_version(client)
        agent_id_b = _create_agent(client)
        body = client.get(f"/v1/agents/{agent_id_b}/versions/{version_id_a}").json()
        assert body["error"]["code"] == "agent_version_not_found"


# ---------------------------------------------------------------------------
# Version get – audit log
# ---------------------------------------------------------------------------


class TestAgentVersionGetAuditLog:
    def test_unknown_agent_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/agents/agent_unknown/versions/agentver_abc123")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "agent.version.get.failed" for r in records)

    def test_unknown_version_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.get(f"/v1/agents/{agent_id}/versions/agentver_doesnotexist")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "agent.version.get.failed" for r in records)

    def test_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/agents/agent_unknown/versions/agentver_abc123")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.version.get.failed"
        )
        assert record["level"] == "error"

    def test_audit_code_agent_not_found(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/agents/agent_unknown/versions/agentver_abc123")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.version.get.failed"
        )
        assert record["code"] == "agent_not_found"

    def test_audit_code_version_not_found(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.get(f"/v1/agents/{agent_id}/versions/agentver_doesnotexist")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.version.get.failed"
        )
        assert record["code"] == "agent_version_not_found"

    def test_audit_detail_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/agents/agent_unknown/versions/agentver_abc123")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.version.get.failed"
        )
        assert record["detail"]["agent_id"] == "agent_unknown"

    def test_audit_detail_has_version_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/agents/agent_unknown/versions/agentver_abc123")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.version.get.failed"
        )
        assert record["detail"]["version_id"] == "agentver_abc123"

    def test_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/agents/agent_unknown/versions/agentver_abc123")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.version.get.failed"
        )
        assert len(record["detail"]["message"]) > 0

    def test_no_audit_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id, version_id = _create_agent_with_version(client)
        client.get(f"/v1/agents/{agent_id}/versions/{version_id}")
        records = _audit_records(storage_root)
        assert not any(r.get("event") == "agent.version.get.failed" for r in records)


# ---------------------------------------------------------------------------
# Version get – OTel spans
# ---------------------------------------------------------------------------


class TestAgentVersionGetOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_version_get_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent_id, version_id = _create_agent_with_version(client)
        _otel_exporter.clear()
        client.get(f"/v1/agents/{agent_id}/versions/{version_id}")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "agent.version.get" in span_names

    def test_failure_emits_version_get_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.get("/v1/agents/agent_unknown/versions/agentver_abc123")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "agent.version.get" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode
        client = self._client(storage_root)
        client.get("/v1/agents/agent_unknown/versions/agentver_abc123")
        spans = [
            s for s in _otel_exporter.get_finished_spans()
            if s.name == "agent.version.get"
        ]
        assert any(s.status.status_code == StatusCode.ERROR for s in spans)

    def test_span_has_agent_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent_id, version_id = _create_agent_with_version(client)
        _otel_exporter.clear()
        client.get(f"/v1/agents/{agent_id}/versions/{version_id}")
        spans = [
            s for s in _otel_exporter.get_finished_spans()
            if s.name == "agent.version.get"
        ]
        assert any(s.attributes.get("agent.id") == agent_id for s in spans)

    def test_span_has_version_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent_id, version_id = _create_agent_with_version(client)
        _otel_exporter.clear()
        client.get(f"/v1/agents/{agent_id}/versions/{version_id}")
        spans = [
            s for s in _otel_exporter.get_finished_spans()
            if s.name == "agent.version.get"
        ]
        assert any(s.attributes.get("agent.version.id") == version_id for s in spans)


# ---------------------------------------------------------------------------
# Agent list – success
# ---------------------------------------------------------------------------


class TestAgentListSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/agents")
        assert resp.status_code == 200

    def test_empty_items_when_no_agents(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/agents").json()
        assert body["items"] == []

    def test_response_has_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/agents").json()
        assert "items" in body
        assert isinstance(body["items"], list)

    def test_response_has_next_cursor(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/agents").json()
        assert "next_cursor" in body

    def test_response_has_limit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/agents").json()
        assert "limit" in body
        assert isinstance(body["limit"], int)

    def test_next_cursor_null_when_no_more_pages(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body())
        body = client.get("/v1/agents").json()
        assert body["next_cursor"] is None

    def test_items_contain_created_agent(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        body = client.get("/v1/agents").json()
        ids = [item["id"] for item in body["items"]]
        assert agent_id in ids

    def test_items_contain_all_agents(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = client.post("/v1/agents", json=_body(name="agent-a")).json()["id"]
        id2 = client.post("/v1/agents", json=_body(name="agent-b")).json()["id"]
        body = client.get("/v1/agents").json()
        ids = {item["id"] for item in body["items"]}
        assert id1 in ids
        assert id2 in ids

    def test_items_sorted_newest_first(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name="agent-a"))
        client.post("/v1/agents", json=_body(name="agent-b"))
        client.post("/v1/agents", json=_body(name="agent-c"))
        body = client.get("/v1/agents").json()
        created_ats = [item["created_at"] for item in body["items"]]
        assert created_ats == sorted(created_ats, reverse=True)

    def test_excludes_soft_deleted_agents(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = client.post("/v1/agents", json=_body(name="to-delete")).json()["id"]
        client.delete(f"/v1/agents/{agent_id}")
        body = client.get("/v1/agents").json()
        ids = [item["id"] for item in body["items"]]
        assert agent_id not in ids

    def test_non_deleted_agents_still_appear_after_other_deletion(
        self, storage_root: Path
    ) -> None:
        client = _make_client(storage_root)
        keep_id = client.post("/v1/agents", json=_body(name="keeper")).json()["id"]
        del_id = client.post("/v1/agents", json=_body(name="gone")).json()["id"]
        client.delete(f"/v1/agents/{del_id}")
        body = client.get("/v1/agents").json()
        ids = [item["id"] for item in body["items"]]
        assert keep_id in ids
        assert del_id not in ids

    def test_item_fields_match_stored_record(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        body = client.get("/v1/agents").json()
        item = next(i for i in body["items"] if i["id"] == agent_id)
        stored = _agent_resource(storage_root, agent_id)
        assert item == stored


# ---------------------------------------------------------------------------
# Agent list – filters
# ---------------------------------------------------------------------------


class TestAgentListFilters:
    def test_name_prefix_returns_matching_agents(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name="alpha-one"))
        client.post("/v1/agents", json=_body(name="alpha-two"))
        client.post("/v1/agents", json=_body(name="beta-one"))
        body = client.get("/v1/agents?name=alpha").json()
        assert len(body["items"]) == 2
        assert all(item["name"].startswith("alpha") for item in body["items"])

    def test_name_prefix_excludes_non_matching(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name="alpha-agent"))
        client.post("/v1/agents", json=_body(name="beta-agent"))
        body = client.get("/v1/agents?name=alpha").json()
        assert not any(item["name"].startswith("beta") for item in body["items"])

    def test_name_prefix_exact_match_included(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name="exact"))
        body = client.get("/v1/agents?name=exact").json()
        assert len(body["items"]) == 1

    def test_name_prefix_no_match_returns_empty(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name="alpha-agent"))
        body = client.get("/v1/agents?name=zzz").json()
        assert body["items"] == []

    def test_name_prefix_omitted_returns_all(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name="alpha-agent"))
        client.post("/v1/agents", json=_body(name="beta-agent"))
        body = client.get("/v1/agents").json()
        assert len(body["items"]) == 2

    def test_created_after_far_past_returns_all(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body())
        body = client.get("/v1/agents?created_after=1970-01-01T00:00:00%2B00:00").json()
        assert len(body["items"]) == 1

    def test_created_after_far_future_returns_empty(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body())
        body = client.get("/v1/agents?created_after=9999-12-31T23:59:59%2B00:00").json()
        assert body["items"] == []

    def test_created_before_far_future_returns_all(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body())
        body = client.get("/v1/agents?created_before=9999-12-31T23:59:59%2B00:00").json()
        assert len(body["items"]) == 1

    def test_created_before_far_past_returns_empty(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body())
        body = client.get("/v1/agents?created_before=1970-01-01T00:00:00%2B00:00").json()
        assert body["items"] == []

    def test_created_after_and_before_both_pass_returns_agents(
        self, storage_root: Path
    ) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body())
        url = (
            "/v1/agents"
            "?created_after=1970-01-01T00:00:00%2B00:00"
            "&created_before=9999-12-31T23:59:59%2B00:00"
        )
        body = client.get(url).json()
        assert len(body["items"]) == 1

    def test_name_prefix_and_created_after_combined(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name="alpha-agent"))
        client.post("/v1/agents", json=_body(name="beta-agent"))
        url = "/v1/agents?name=alpha&created_after=1970-01-01T00:00:00%2B00:00"
        body = client.get(url).json()
        assert len(body["items"]) == 1
        assert body["items"][0]["name"] == "alpha-agent"


# ---------------------------------------------------------------------------
# Agent list – pagination
# ---------------------------------------------------------------------------


class TestAgentListPagination:
    def test_next_cursor_set_when_more_pages(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name="agent-a"))
        client.post("/v1/agents", json=_body(name="agent-b"))
        client.post("/v1/agents", json=_body(name="agent-c"))
        body = client.get("/v1/agents?limit=2").json()
        assert body["next_cursor"] is not None

    def test_link_header_set_when_next_cursor_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name="agent-a"))
        client.post("/v1/agents", json=_body(name="agent-b"))
        client.post("/v1/agents", json=_body(name="agent-c"))
        resp = client.get("/v1/agents?limit=2")
        assert "link" in resp.headers

    def test_limit_controls_page_size(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name="agent-a"))
        client.post("/v1/agents", json=_body(name="agent-b"))
        client.post("/v1/agents", json=_body(name="agent-c"))
        body = client.get("/v1/agents?limit=2").json()
        assert len(body["items"]) == 2

    def test_cursor_returns_next_page(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/agents", json=_body(name="agent-a"))
        client.post("/v1/agents", json=_body(name="agent-b"))
        client.post("/v1/agents", json=_body(name="agent-c"))
        first = client.get("/v1/agents?limit=2").json()
        cursor = first["next_cursor"]
        second = client.get(f"/v1/agents?limit=2&cursor={cursor}").json()
        assert len(second["items"]) >= 1
        first_ids = {i["id"] for i in first["items"]}
        second_ids = {i["id"] for i in second["items"]}
        assert first_ids.isdisjoint(second_ids)

    def test_pagination_traverses_all_agents(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        for i in range(5):
            client.post("/v1/agents", json=_body(name=f"agent-{i}"))
        all_ids: set[str] = set()
        cursor = None
        while True:
            url = "/v1/agents?limit=2"
            if cursor:
                url += f"&cursor={cursor}"
            body = client.get(url).json()
            for item in body["items"]:
                all_ids.add(item["id"])
            cursor = body["next_cursor"]
            if cursor is None:
                break
        assert len(all_ids) == 5

    def test_invalid_cursor_returns_400(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/agents?cursor=not-valid-base64!!!")
        assert resp.status_code == 400

    def test_invalid_cursor_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/agents?cursor=not-valid-base64!!!").json()
        assert body["error"]["code"] == "cursor_invalid"


# ---------------------------------------------------------------------------
# Agent list – audit log
# ---------------------------------------------------------------------------


class TestAgentListAuditLog:
    def test_invalid_cursor_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/agents?cursor=bad!!!")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "agent.list.failed" for r in records)

    def test_invalid_cursor_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/agents?cursor=bad!!!")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.list.failed"
        )
        assert record["level"] == "error"

    def test_invalid_cursor_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/agents?cursor=bad!!!")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.list.failed"
        )
        assert record["code"] == "cursor_invalid"

    def test_invalid_cursor_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/agents?cursor=bad!!!")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.list.failed"
        )
        assert len(record["detail"]["message"]) > 0

    def test_no_audit_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/agents")
        records = _audit_records(storage_root)
        assert not any(r.get("event") == "agent.list.failed" for r in records)


# ---------------------------------------------------------------------------
# Agent list – OTel spans
# ---------------------------------------------------------------------------


class TestAgentListOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_agent_list_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.get("/v1/agents")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "agent.list" in span_names

    def test_failure_emits_agent_list_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.get("/v1/agents?cursor=bad!!!")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "agent.list" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode
        client = self._client(storage_root)
        client.get("/v1/agents?cursor=bad!!!")
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "agent.list"]
        assert any(s.status.status_code == StatusCode.ERROR for s in spans)


# ---------------------------------------------------------------------------
# Version list – success
# ---------------------------------------------------------------------------


class TestAgentVersionListSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        resp = client.get(f"/v1/agents/{agent_id}/versions")
        assert resp.status_code == 200

    def test_returns_200_for_unknown_agent(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/agents/agent_unknown/versions")
        assert resp.status_code == 200

    def test_response_has_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.get(f"/v1/agents/{agent_id}/versions").json()
        assert "items" in body
        assert isinstance(body["items"], list)

    def test_empty_items_for_unknown_agent(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/agents/agent_unknown/versions").json()
        assert body["items"] == []

    def test_response_has_next_cursor(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.get(f"/v1/agents/{agent_id}/versions").json()
        assert "next_cursor" in body

    def test_response_has_limit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.get(f"/v1/agents/{agent_id}/versions").json()
        assert "limit" in body
        assert isinstance(body["limit"], int)

    def test_next_cursor_null_when_no_more_pages(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.get(f"/v1/agents/{agent_id}/versions").json()
        assert body["next_cursor"] is None

    def test_items_contain_initial_version(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id, version_id = _create_agent_with_version(client)
        body = client.get(f"/v1/agents/{agent_id}/versions").json()
        ids = [item["id"] for item in body["items"]]
        assert version_id in ids

    def test_items_contain_all_versions_for_agent(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        v1_id = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v2")).json()["id"]
        v2_id = client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="v3")).json()["id"]
        body = client.get(f"/v1/agents/{agent_id}/versions").json()
        ids = {item["id"] for item in body["items"]}
        assert v1_id in ids
        assert v2_id in ids

    def test_items_only_include_versions_for_requested_agent(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id_a, version_id_a = _create_agent_with_version(client)
        agent_id_b, version_id_b = _create_agent_with_version(client)
        body = client.get(f"/v1/agents/{agent_id_a}/versions").json()
        ids = {item["id"] for item in body["items"]}
        assert version_id_a in ids
        assert version_id_b not in ids

    def test_items_sorted_newest_first(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="va"))
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="vb"))
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="vc"))
        body = client.get(f"/v1/agents/{agent_id}/versions").json()
        created_ats = [item["created_at"] for item in body["items"]]
        assert created_ats == sorted(created_ats, reverse=True)

    def test_item_fields_match_stored_record(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id, version_id = _create_agent_with_version(client)
        body = client.get(f"/v1/agents/{agent_id}/versions").json()
        item = next(i for i in body["items"] if i["id"] == version_id)
        stored = _version_resource(storage_root, version_id)
        assert item == stored


# ---------------------------------------------------------------------------
# Version list – pagination
# ---------------------------------------------------------------------------


class TestAgentVersionListPagination:
    def test_next_cursor_set_when_more_pages(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="va"))
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="vb"))
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="vc"))
        body = client.get(f"/v1/agents/{agent_id}/versions?limit=2").json()
        assert body["next_cursor"] is not None

    def test_link_header_set_when_next_cursor_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="va"))
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="vb"))
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="vc"))
        resp = client.get(f"/v1/agents/{agent_id}/versions?limit=2")
        assert "link" in resp.headers

    def test_limit_controls_page_size(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="va"))
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="vb"))
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="vc"))
        body = client.get(f"/v1/agents/{agent_id}/versions?limit=2").json()
        assert len(body["items"]) == 2

    def test_cursor_returns_next_page(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="va"))
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="vb"))
        client.post(f"/v1/agents/{agent_id}/versions", json=_ver_body(name="vc"))
        first = client.get(f"/v1/agents/{agent_id}/versions?limit=2").json()
        cursor = first["next_cursor"]
        second = client.get(f"/v1/agents/{agent_id}/versions?limit=2&cursor={cursor}").json()
        assert len(second["items"]) >= 1
        first_ids = {i["id"] for i in first["items"]}
        second_ids = {i["id"] for i in second["items"]}
        assert first_ids.isdisjoint(second_ids)

    def test_invalid_cursor_returns_400(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        resp = client.get(f"/v1/agents/{agent_id}/versions?cursor=not-valid-base64!!!")
        assert resp.status_code == 400

    def test_invalid_cursor_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        body = client.get(f"/v1/agents/{agent_id}/versions?cursor=not-valid-base64!!!").json()
        assert body["error"]["code"] == "cursor_invalid"


# ---------------------------------------------------------------------------
# Version list – audit log
# ---------------------------------------------------------------------------


class TestAgentVersionListAuditLog:
    def test_invalid_cursor_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.get(f"/v1/agents/{agent_id}/versions?cursor=bad!!!")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "agent.versions.list.failed" for r in records)

    def test_invalid_cursor_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.get(f"/v1/agents/{agent_id}/versions?cursor=bad!!!")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.versions.list.failed"
        )
        assert record["level"] == "error"

    def test_invalid_cursor_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.get(f"/v1/agents/{agent_id}/versions?cursor=bad!!!")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.versions.list.failed"
        )
        assert record["code"] == "cursor_invalid"

    def test_invalid_cursor_audit_detail_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.get(f"/v1/agents/{agent_id}/versions?cursor=bad!!!")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.versions.list.failed"
        )
        assert record["detail"]["agent_id"] == agent_id

    def test_invalid_cursor_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.get(f"/v1/agents/{agent_id}/versions?cursor=bad!!!")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "agent.versions.list.failed"
        )
        assert len(record["detail"]["message"]) > 0

    def test_no_audit_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        agent_id = _create_agent(client)
        client.get(f"/v1/agents/{agent_id}/versions")
        records = _audit_records(storage_root)
        assert not any(r.get("event") == "agent.versions.list.failed" for r in records)


# ---------------------------------------------------------------------------
# Version list – OTel spans
# ---------------------------------------------------------------------------


class TestAgentVersionListOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_versions_list_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        _otel_exporter.clear()
        client.get(f"/v1/agents/{agent_id}/versions")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "agent.versions.list" in span_names

    def test_failure_emits_versions_list_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        client.get(f"/v1/agents/{agent_id}/versions?cursor=bad!!!")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "agent.versions.list" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode
        client = self._client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        client.get(f"/v1/agents/{agent_id}/versions?cursor=bad!!!")
        spans = [
            s for s in _otel_exporter.get_finished_spans()
            if s.name == "agent.versions.list"
        ]
        assert any(s.status.status_code == StatusCode.ERROR for s in spans)

    def test_span_has_agent_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        agent_id = client.post("/v1/agents", json=_body()).json()["id"]
        _otel_exporter.clear()
        client.get(f"/v1/agents/{agent_id}/versions")
        spans = [
            s for s in _otel_exporter.get_finished_spans()
            if s.name == "agent.versions.list"
        ]
        assert any(s.attributes.get("agent.id") == agent_id for s in spans)
