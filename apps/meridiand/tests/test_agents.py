"""
Agents endpoint conformance suite.

Tests cover:
  - POST /v1/agents returns 201 on success.
  - Response has id with "agent_" prefix.
  - IDs are unique across calls.
  - Response has name, kind, created_at, version.
  - version object present in response.
  - version has id with "agentver_" prefix.
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
