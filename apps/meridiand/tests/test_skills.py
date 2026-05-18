"""
Skills endpoint conformance suite.

Tests cover:
  - POST /v1/skills returns 201 on success.
  - Response has id with "skill_" prefix.
  - IDs are unique across calls.
  - Response has name, description, created_at, metadata, version.
  - metadata is null when omitted; stored when provided.
  - tests is empty list when omitted; stored when provided.
  - version has id with "skillver_" prefix.
  - version has skill_id matching the skill id.
  - version has version_number 1 on first creation.
  - version has instructions, tools, tests, created_at.
  - Skill JSON written to storage_root/skills/{id}.json.
  - SkillVersionRecord JSON written to storage_root/skill_versions/{version_id}.json.
  - Persisted skill has correct name, description.
  - Persisted version has correct instructions and tools.
  - Not written to disk on validation failure.
  - Empty name returns 422 with code "skill_invalid_request".
  - Empty instructions returns 422 with code "skill_invalid_request".
  - Empty tools list returns 422 with code "skill_invalid_request".
  - Missing required fields return 422.
  - On validation failure, audit log entry written with event "skill.create.failed".
  - Audit entry level is "error" on failure.
  - Audit entry code is "skill_invalid_request" on validation failure.
  - Audit detail includes skill_id, name, message on failure.
  - Error response body has error.code and error.message on failure.
  - OTel span "skill.create" emitted on success.
  - OTel span "skill.create" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries skill.id and skill.name attributes.
  - create_app wires skills router when storage_root is supplied.
  - create_app omits skills route when storage_root is None.
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
        "name": "my-skill",
        "description": "A skill that does something useful",
        "instructions": "Step 1: do the thing. Step 2: verify it worked.",
        "tools": [{"name": "bash", "description": "Run shell commands"}],
    }
    base.update(overrides)
    return base


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _skill_resource(storage_root: Path, skill_id: str) -> dict:
    return json.loads((storage_root / "skills" / f"{skill_id}.json").read_text())


def _version_resource(storage_root: Path, version_id: str) -> dict:
    return json.loads((storage_root / "skill_versions" / f"{version_id}.json").read_text())


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestSkillCreateSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/skills", json=_body())
        assert resp.status_code == 201

    def test_with_optional_tests_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        tests = [{"name": "basic", "input": {"cmd": "echo hi"}, "expected_output": "hi"}]
        resp = client.post("/v1/skills", json=_body(tests=tests))
        assert resp.status_code == 201

    def test_with_metadata_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/skills", json=_body(metadata={"team": "platform"}))
        assert resp.status_code == 201

    def test_multiple_tools_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        tools = [
            {"name": "bash", "description": "Run shell commands"},
            {"name": "read_file", "description": "Read a file"},
        ]
        resp = client.post("/v1/skills", json=_body(tools=tools))
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Response fields
# ---------------------------------------------------------------------------


class TestSkillCreateResponse:
    def test_response_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body()).json()
        assert "id" in body
        assert isinstance(body["id"], str)

    def test_id_has_skill_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body()).json()
        assert body["id"].startswith("skill_")

    def test_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = client.post("/v1/skills", json=_body()).json()["id"]
        id2 = client.post("/v1/skills", json=_body()).json()["id"]
        assert id1 != id2

    def test_response_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body(name="deploy-skill")).json()
        assert body["name"] == "deploy-skill"

    def test_response_has_description(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body(description="Does deploys")).json()
        assert body["description"] == "Does deploys"

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body()).json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0

    def test_metadata_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body()).json()
        assert body["metadata"] is None

    def test_metadata_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        meta = {"team": "platform", "env": "prod"}
        body = client.post("/v1/skills", json=_body(metadata=meta)).json()
        assert body["metadata"] == meta

    def test_response_has_version(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body()).json()
        assert "version" in body
        assert isinstance(body["version"], dict)

    def test_version_id_has_skillver_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body()).json()
        assert body["version"]["id"].startswith("skillver_")

    def test_version_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vid1 = client.post("/v1/skills", json=_body()).json()["version"]["id"]
        vid2 = client.post("/v1/skills", json=_body()).json()["version"]["id"]
        assert vid1 != vid2

    def test_version_skill_id_matches_skill(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body()).json()
        assert body["version"]["skill_id"] == body["id"]

    def test_version_number_is_1(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body()).json()
        assert body["version"]["version_number"] == 1

    def test_version_has_instructions(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/skills", json=_body(instructions="Do the thing carefully")
        ).json()
        assert body["version"]["instructions"] == "Do the thing carefully"

    def test_version_has_tools(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        tools = [{"name": "bash", "description": "Run shell commands"}]
        body = client.post("/v1/skills", json=_body(tools=tools)).json()
        assert body["version"]["tools"][0]["name"] == "bash"

    def test_version_tests_empty_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body()).json()
        assert body["version"]["tests"] == []

    def test_version_tests_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        tests = [{"name": "basic", "input": {"cmd": "echo hi"}, "expected_output": "hi"}]
        body = client.post("/v1/skills", json=_body(tests=tests)).json()
        assert body["version"]["tests"][0]["name"] == "basic"

    def test_version_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body()).json()
        assert "created_at" in body["version"]
        assert isinstance(body["version"]["created_at"], str)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestSkillCreateValidation:
    def test_missing_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["name"]
        resp = client.post("/v1/skills", json=payload)
        assert resp.status_code == 422

    def test_missing_description_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["description"]
        resp = client.post("/v1/skills", json=payload)
        assert resp.status_code == 422

    def test_missing_instructions_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["instructions"]
        resp = client.post("/v1/skills", json=payload)
        assert resp.status_code == 422

    def test_missing_tools_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = _body()
        del payload["tools"]
        resp = client.post("/v1/skills", json=payload)
        assert resp.status_code == 422

    def test_empty_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/skills", json=_body(name="   "))
        assert resp.status_code == 422

    def test_empty_name_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body(name="")).json()
        assert body["error"]["code"] == "skill_invalid_request"

    def test_empty_instructions_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/skills", json=_body(instructions="  "))
        assert resp.status_code == 422

    def test_empty_instructions_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body(instructions="")).json()
        assert body["error"]["code"] == "skill_invalid_request"

    def test_empty_tools_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/skills", json=_body(tools=[]))
        assert resp.status_code == 422

    def test_empty_tools_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body(tools=[])).json()
        assert body["error"]["code"] == "skill_invalid_request"

    def test_validation_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body(name="")).json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestSkillPersistence:
    def test_skill_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill_id = client.post("/v1/skills", json=_body()).json()["id"]
        assert (storage_root / "skills" / f"{skill_id}.json").exists()

    def test_version_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body()).json()
        version_id = body["version"]["id"]
        assert (storage_root / "skill_versions" / f"{version_id}.json").exists()

    def test_persisted_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill_id = client.post("/v1/skills", json=_body(name="persist-skill")).json()["id"]
        resource = _skill_resource(storage_root, skill_id)
        assert resource["name"] == "persist-skill"

    def test_persisted_description(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill_id = client.post(
            "/v1/skills", json=_body(description="Persisted description")
        ).json()["id"]
        resource = _skill_resource(storage_root, skill_id)
        assert resource["description"] == "Persisted description"

    def test_persisted_version_instructions(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/skills", json=_body(instructions="Persisted instructions")
        ).json()
        version = _version_resource(storage_root, body["version"]["id"])
        assert version["instructions"] == "Persisted instructions"

    def test_persisted_version_tools(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        tools = [{"name": "grep", "description": "Search files"}]
        body = client.post("/v1/skills", json=_body(tools=tools)).json()
        version = _version_resource(storage_root, body["version"]["id"])
        assert version["tools"][0]["name"] == "grep"

    def test_persisted_version_number(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/skills", json=_body()).json()
        version = _version_resource(storage_root, body["version"]["id"])
        assert version["version_number"] == 1

    def test_not_written_on_validation_failure(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/skills", json=_body(tools=[]))
        skills_dir = storage_root / "skills"
        files = list(skills_dir.glob("*.json")) if skills_dir.exists() else []
        assert files == []


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestSkillAuditLog:
    def test_validation_failure_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/skills", json=_body(name=""))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill.create.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/skills", json=_body(instructions=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill.create.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_code_is_skill_invalid_request(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/skills", json=_body(tools=[]))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill.create.failed"
        )
        assert record["code"] == "skill_invalid_request"

    def test_failure_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/skills", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill.create.failed"
        )
        assert record["detail"]["skill_id"].startswith("skill_")

    def test_failure_audit_detail_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/skills", json=_body(name="audit-skill", instructions=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill.create.failed"
        )
        assert record["detail"]["name"] == "audit-skill"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/skills", json=_body(tools=[]))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill.create.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestSkillOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_skill_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/skills", json=_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.create" in span_names

    def test_failure_emits_skill_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/skills", json=_body(name=""))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.create" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.post("/v1/skills", json=_body(tools=[]))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.create")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_skill_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/skills", json=_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.create")
        assert span is not None
        assert span.attributes["skill.id"].startswith("skill_")

    def test_success_span_has_skill_name_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/skills", json=_body(name="otel-skill"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.create")
        assert span is not None
        assert span.attributes["skill.name"] == "otel-skill"


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestSkillRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/skills", json=_body())
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/skills", json=_body())
        assert resp.status_code == 404
