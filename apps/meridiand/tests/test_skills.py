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
  - GET /v1/skills/{id}/versions/{ver} returns 200 with SkillVersionRecord.
  - GET /v1/skills/{id}/versions/{ver} returns 404 if version not found.
  - GET /v1/skills/{id}/versions/{ver} returns 404 if version belongs to different skill.
  - Response fields: id, skill_id, version_number, instructions, tools, tests, created_at.
  - On not-found, audit log entry written with event "skill.version.get.failed".
  - On not-found, audit entry code is "skill_version_not_found".
  - OTel span "skill.version.get" emitted on success.
  - OTel span "skill.version.get" emitted on failure.
  - OTel span set to ERROR on not-found.
  - Span carries skill.id and skill.version.id attributes.
  - GET /v1/skills returns 200 with paginated result page.
  - GET /v1/skills returns empty items list when no skills exist.
  - GET /v1/skills response has items, total, limit, offset fields.
  - GET /v1/skills items have id, name, description, created_at, metadata, version.
  - GET /v1/skills returns all installed skills.
  - GET /v1/skills default limit is 20, default offset is 0.
  - GET /v1/skills limit query param restricts items returned.
  - GET /v1/skills offset query param skips items.
  - GET /v1/skills total reflects full count independent of pagination.
  - GET /v1/skills OTel span "skill.list" emitted on success.
  - GET /v1/skills/{id}/versions returns 200 with paginated result page.
  - GET /v1/skills/{id}/versions response has items, total, limit, offset fields.
  - GET /v1/skills/{id}/versions items are SkillVersionRecord objects.
  - GET /v1/skills/{id}/versions returns only versions for the requested skill.
  - GET /v1/skills/{id}/versions returns empty items for unknown skill_id.
  - GET /v1/skills/{id}/versions orders versions newest first.
  - GET /v1/skills/{id}/versions limit query param restricts items returned.
  - GET /v1/skills/{id}/versions offset query param skips items.
  - GET /v1/skills/{id}/versions total reflects full count independent of pagination.
  - GET /v1/skills/{id}/versions OTel span "skill.versions.list" emitted on success.
  - GET /v1/skills/{id}/versions span carries skill.id attribute.
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


# ---------------------------------------------------------------------------
# GET /v1/skills/{id}/versions/{ver} — success
# ---------------------------------------------------------------------------


def _create_skill(client: TestClient, **overrides) -> dict:
    resp = client.post("/v1/skills", json=_body(**overrides))
    assert resp.status_code == 201
    return resp.json()


class TestSkillVersionGetSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        resp = client.get(f"/v1/skills/{skill['id']}/versions/{skill['version']['id']}")
        assert resp.status_code == 200

    def test_response_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions/{skill['version']['id']}").json()
        assert body["id"] == skill["version"]["id"]

    def test_response_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions/{skill['version']['id']}").json()
        assert body["skill_id"] == skill["id"]

    def test_response_has_version_number(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions/{skill['version']['id']}").json()
        assert body["version_number"] == 1

    def test_response_has_instructions(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client, instructions="Do the thing carefully")
        body = client.get(f"/v1/skills/{skill['id']}/versions/{skill['version']['id']}").json()
        assert body["instructions"] == "Do the thing carefully"

    def test_response_has_tools(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client, tools=[{"name": "grep", "description": "Search files"}])
        body = client.get(f"/v1/skills/{skill['id']}/versions/{skill['version']['id']}").json()
        assert body["tools"][0]["name"] == "grep"

    def test_response_has_tests(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        tests = [{"name": "basic", "input": {"cmd": "echo hi"}, "expected_output": "hi"}]
        skill = _create_skill(client, tests=tests)
        body = client.get(f"/v1/skills/{skill['id']}/versions/{skill['version']['id']}").json()
        assert body["tests"][0]["name"] == "basic"

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions/{skill['version']['id']}").json()
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0


# ---------------------------------------------------------------------------
# GET /v1/skills/{id}/versions/{ver} — not found
# ---------------------------------------------------------------------------


class TestSkillVersionGetNotFound:
    def test_unknown_version_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        resp = client.get(f"/v1/skills/{skill['id']}/versions/skillver_doesnotexist")
        assert resp.status_code == 404

    def test_unknown_version_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions/skillver_doesnotexist").json()
        assert body["error"]["code"] == "skill_version_not_found"

    def test_unknown_version_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions/skillver_doesnotexist").json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0

    def test_wrong_skill_id_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        ver_id = skill["version"]["id"]
        resp = client.get(f"/v1/skills/skill_wrongid/versions/{ver_id}")
        assert resp.status_code == 404

    def test_wrong_skill_id_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        ver_id = skill["version"]["id"]
        body = client.get(f"/v1/skills/skill_wrongid/versions/{ver_id}").json()
        assert body["error"]["code"] == "skill_version_not_found"


# ---------------------------------------------------------------------------
# GET /v1/skills/{id}/versions/{ver} — audit log
# ---------------------------------------------------------------------------


class TestSkillVersionGetAuditLog:
    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        client.get(f"/v1/skills/{skill['id']}/versions/skillver_doesnotexist")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill.version.get.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        client.get(f"/v1/skills/{skill['id']}/versions/skillver_doesnotexist")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill.version.get.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        client.get(f"/v1/skills/{skill['id']}/versions/skillver_doesnotexist")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill.version.get.failed"
        )
        assert record["code"] == "skill_version_not_found"

    def test_not_found_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        client.get(f"/v1/skills/{skill['id']}/versions/skillver_doesnotexist")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill.version.get.failed"
        )
        assert record["detail"]["skill_id"] == skill["id"]

    def test_not_found_audit_detail_has_version_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        client.get(f"/v1/skills/{skill['id']}/versions/skillver_doesnotexist")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill.version.get.failed"
        )
        assert record["detail"]["version_id"] == "skillver_doesnotexist"

    def test_not_found_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        client.get(f"/v1/skills/{skill['id']}/versions/skillver_doesnotexist")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill.version.get.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# GET /v1/skills/{id}/versions/{ver} — OTel
# ---------------------------------------------------------------------------


class TestSkillVersionGetOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        skill = _create_skill(client)
        _otel_exporter.clear()
        client.get(f"/v1/skills/{skill['id']}/versions/{skill['version']['id']}")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.version.get" in span_names

    def test_not_found_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        skill = _create_skill(client)
        _otel_exporter.clear()
        client.get(f"/v1/skills/{skill['id']}/versions/skillver_doesnotexist")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.version.get" in span_names

    def test_not_found_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        skill = _create_skill(client)
        _otel_exporter.clear()
        client.get(f"/v1/skills/{skill['id']}/versions/skillver_doesnotexist")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.version.get")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_span_has_skill_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        skill = _create_skill(client)
        _otel_exporter.clear()
        client.get(f"/v1/skills/{skill['id']}/versions/{skill['version']['id']}")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.version.get")
        assert span is not None
        assert span.attributes["skill.id"] == skill["id"]

    def test_span_has_version_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        skill = _create_skill(client)
        ver_id = skill["version"]["id"]
        _otel_exporter.clear()
        client.get(f"/v1/skills/{skill['id']}/versions/{ver_id}")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.version.get")
        assert span is not None
        assert span.attributes["skill.version.id"] == ver_id


# ---------------------------------------------------------------------------
# GET /v1/skills — list installed skills
# ---------------------------------------------------------------------------


class TestSkillListSuccess:
    def test_empty_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/skills")
        assert resp.status_code == 200

    def test_empty_items_when_no_skills(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills").json()
        assert body["items"] == []

    def test_empty_total_is_zero(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills").json()
        assert body["total"] == 0

    def test_response_has_items_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills").json()
        assert "items" in body

    def test_response_has_total_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills").json()
        assert "total" in body

    def test_response_has_limit_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills").json()
        assert "limit" in body

    def test_response_has_offset_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills").json()
        assert "offset" in body

    def test_default_limit_is_20(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills").json()
        assert body["limit"] == 20

    def test_default_offset_is_0(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills").json()
        assert body["offset"] == 0

    def test_with_one_skill_total_is_1(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client)
        body = client.get("/v1/skills").json()
        assert body["total"] == 1

    def test_with_one_skill_items_has_one_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client)
        body = client.get("/v1/skills").json()
        assert len(body["items"]) == 1

    def test_item_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get("/v1/skills").json()
        assert body["items"][0]["id"] == skill["id"]

    def test_item_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client, name="listed-skill")
        body = client.get("/v1/skills").json()
        assert body["items"][0]["name"] == "listed-skill"

    def test_item_has_description(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client, description="Listed skill description")
        body = client.get("/v1/skills").json()
        assert body["items"][0]["description"] == "Listed skill description"

    def test_item_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client)
        body = client.get("/v1/skills").json()
        assert isinstance(body["items"][0]["created_at"], str)
        assert len(body["items"][0]["created_at"]) > 0

    def test_item_has_metadata(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client, metadata={"env": "prod"})
        body = client.get("/v1/skills").json()
        assert body["items"][0]["metadata"] == {"env": "prod"}

    def test_item_has_version(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client)
        body = client.get("/v1/skills").json()
        assert isinstance(body["items"][0]["version"], dict)

    def test_item_version_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get("/v1/skills").json()
        assert body["items"][0]["version"]["id"] == skill["version"]["id"]

    def test_multiple_skills_all_returned(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client, name="skill-a")
        _create_skill(client, name="skill-b")
        _create_skill(client, name="skill-c")
        body = client.get("/v1/skills").json()
        assert body["total"] == 3
        assert len(body["items"]) == 3


class TestSkillListPagination:
    def test_limit_restricts_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client, name="skill-a")
        _create_skill(client, name="skill-b")
        body = client.get("/v1/skills", params={"limit": 1}).json()
        assert len(body["items"]) == 1

    def test_limit_reflected_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills", params={"limit": 5}).json()
        assert body["limit"] == 5

    def test_offset_reflected_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills", params={"offset": 3}).json()
        assert body["offset"] == 3

    def test_total_reflects_full_count_not_page(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client, name="skill-a")
        _create_skill(client, name="skill-b")
        _create_skill(client, name="skill-c")
        body = client.get("/v1/skills", params={"limit": 2}).json()
        assert body["total"] == 3
        assert len(body["items"]) == 2

    def test_offset_skips_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client, name="skill-a")
        _create_skill(client, name="skill-b")
        body_all = client.get("/v1/skills").json()
        first_id = body_all["items"][0]["id"]
        body_paged = client.get("/v1/skills", params={"offset": 1}).json()
        assert len(body_paged["items"]) == 1
        assert body_paged["items"][0]["id"] != first_id

    def test_offset_beyond_total_returns_empty(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client)
        body = client.get("/v1/skills", params={"offset": 100}).json()
        assert body["items"] == []
        assert body["total"] == 1


class TestSkillListOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_emits_skill_list_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.get("/v1/skills")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.list" in span_names


# ---------------------------------------------------------------------------
# GET /v1/skills/{id}/versions — list skill versions
# ---------------------------------------------------------------------------


def _write_version(storage_root: Path, skill_id: str, created_at: str, version_number: int) -> dict:
    versions_dir = storage_root / "skill_versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    import uuid

    version_id = f"skillver_{uuid.uuid4().hex}"
    record = {
        "id": version_id,
        "skill_id": skill_id,
        "version_number": version_number,
        "instructions": f"Instructions v{version_number}",
        "tools": [{"name": "bash", "description": "Run shell commands"}],
        "tests": [],
        "created_at": created_at,
    }
    (versions_dir / f"{version_id}.json").write_text(json.dumps(record))
    return record


class TestSkillVersionsListSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        resp = client.get(f"/v1/skills/{skill['id']}/versions")
        assert resp.status_code == 200

    def test_response_has_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert "items" in body

    def test_response_has_total(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert "total" in body

    def test_response_has_limit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert "limit" in body

    def test_response_has_offset(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert "offset" in body

    def test_default_limit_is_20(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert body["limit"] == 20

    def test_default_offset_is_0(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert body["offset"] == 0

    def test_one_version_after_create(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert body["total"] == 1
        assert len(body["items"]) == 1

    def test_item_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert body["items"][0]["id"] == skill["version"]["id"]

    def test_item_has_skill_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert body["items"][0]["skill_id"] == skill["id"]

    def test_item_has_version_number(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert body["items"][0]["version_number"] == 1

    def test_item_has_instructions(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client, instructions="Do the thing carefully")
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert body["items"][0]["instructions"] == "Do the thing carefully"

    def test_item_has_tools(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client, tools=[{"name": "grep", "description": "Search files"}])
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert body["items"][0]["tools"][0]["name"] == "grep"

    def test_item_has_tests(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        tests = [{"name": "basic", "input": {"cmd": "echo hi"}, "expected_output": "hi"}]
        skill = _create_skill(client, tests=tests)
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert body["items"][0]["tests"][0]["name"] == "basic"

    def test_item_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions").json()
        assert isinstance(body["items"][0]["created_at"], str)
        assert len(body["items"][0]["created_at"]) > 0

    def test_returns_empty_for_unknown_skill(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills/skill_doesnotexist/versions").json()
        assert body["items"] == []
        assert body["total"] == 0

    def test_only_returns_versions_for_requested_skill(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill_a = _create_skill(client, name="skill-a")
        skill_b = _create_skill(client, name="skill-b")
        body = client.get(f"/v1/skills/{skill_a['id']}/versions").json()
        for item in body["items"]:
            assert item["skill_id"] == skill_a["id"]
            assert item["skill_id"] != skill_b["id"]

    def test_newest_first_ordering(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        skill_id = skill["id"]
        older = _write_version(storage_root, skill_id, "2024-01-01T00:00:00+00:00", 2)
        newer = _write_version(storage_root, skill_id, "2025-01-01T00:00:00+00:00", 3)
        body = client.get(f"/v1/skills/{skill_id}/versions").json()
        ids = [item["id"] for item in body["items"]]
        assert ids.index(newer["id"]) < ids.index(older["id"])


class TestSkillVersionsListPagination:
    def test_limit_restricts_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        skill_id = skill["id"]
        _write_version(storage_root, skill_id, "2024-06-01T00:00:00+00:00", 2)
        body = client.get(f"/v1/skills/{skill_id}/versions", params={"limit": 1}).json()
        assert len(body["items"]) == 1

    def test_limit_reflected_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions", params={"limit": 5}).json()
        assert body["limit"] == 5

    def test_offset_reflected_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions", params={"offset": 2}).json()
        assert body["offset"] == 2

    def test_total_reflects_full_count_not_page(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        skill_id = skill["id"]
        _write_version(storage_root, skill_id, "2024-06-01T00:00:00+00:00", 2)
        _write_version(storage_root, skill_id, "2024-07-01T00:00:00+00:00", 3)
        body = client.get(f"/v1/skills/{skill_id}/versions", params={"limit": 2}).json()
        assert body["total"] == 3
        assert len(body["items"]) == 2

    def test_offset_skips_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        skill_id = skill["id"]
        _write_version(storage_root, skill_id, "2024-06-01T00:00:00+00:00", 2)
        body_all = client.get(f"/v1/skills/{skill_id}/versions").json()
        first_id = body_all["items"][0]["id"]
        body_paged = client.get(f"/v1/skills/{skill_id}/versions", params={"offset": 1}).json()
        assert len(body_paged["items"]) == 1
        assert body_paged["items"][0]["id"] != first_id

    def test_offset_beyond_total_returns_empty(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        skill = _create_skill(client)
        body = client.get(f"/v1/skills/{skill['id']}/versions", params={"offset": 100}).json()
        assert body["items"] == []
        assert body["total"] == 1


class TestSkillVersionsListOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_emits_skill_versions_list_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        skill = _create_skill(client)
        _otel_exporter.clear()
        client.get(f"/v1/skills/{skill['id']}/versions")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill.versions.list" in span_names

    def test_span_has_skill_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        skill = _create_skill(client)
        _otel_exporter.clear()
        client.get(f"/v1/skills/{skill['id']}/versions")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill.versions.list")
        assert span is not None
        assert span.attributes["skill.id"] == skill["id"]
