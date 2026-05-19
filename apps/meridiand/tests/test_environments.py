"""
Environments endpoint conformance suite.

Tests cover:
  - POST /v1/environments returns 201 on success.
  - Response has id with "env_" prefix.
  - IDs are unique across calls.
  - Response has name, backend, created_at, updated_at.
  - Response includes image, template, workspace_path, env_passthrough,
    network_policy, caps_envelope, default_timeout_ms fields.
  - Environment JSON written to storage_root/environments/{id}.json.
  - Persisted record has correct name and backend.
  - Not written to disk on validation failure.
  - Empty name returns 422 with code "environment_invalid_request".
  - Empty backend returns 422 with code "environment_invalid_request".
  - Missing required fields return 422.
  - On validation failure, audit log entry written with event "environment.create.failed".
  - Audit entry level is "error" on failure.
  - Audit entry code is "environment_invalid_request" on validation failure.
  - Audit detail includes environment_id, name, message on failure.
  - Error response body has error.code and error.message on failure.
  - OTel span "environment.create" emitted on success.
  - OTel span "environment.create" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries environment.id, environment.name, environment.backend attributes.
  - create_app wires environments router when storage_root is supplied.
  - create_app omits environments route when storage_root is None.
  - GET /v1/environments returns 200 with items list.
  - GET returns empty items when no environments exist.
  - GET returns all created environments.
  - Each item in GET response has id, name, backend, created_at.
  - OTel span "environment.list" emitted on GET.
  - GET /v1/environments/{id} returns 200 on success.
  - GET /v1/environments/{id} returns 404 with code "environment_not_found" for unknown id.
  - GET /v1/environments/{id} response has all fields.
  - OTel span "environment.get" emitted on success.
  - OTel span "environment.get" emitted on failure.
  - GET single failure writes audit log entry.
  - PATCH /v1/environments/{id} returns 200 on success.
  - PATCH updates the name field.
  - PATCH updates image, template, workspace_path, env_passthrough, network_policy,
    caps_envelope, default_timeout_ms fields.
  - PATCH updates updated_at but not created_at.
  - PATCH with empty name returns 422.
  - PATCH returns 404 for unknown id.
  - PATCH failure writes audit log entry.
  - OTel span "environment.update" emitted on success.
  - OTel span "environment.update" emitted on failure.
  - DELETE /v1/environments/{id} returns 204 on success.
  - DELETE response has no body.
  - DELETE returns 404 with code "environment_not_found" for unknown id.
  - DELETE removes environment JSON from storage.
  - Second DELETE on same id returns 404.
  - DELETE failure writes audit log entry with event "environment.delete.failed".
  - DELETE audit entry level is "error" on failure.
  - DELETE audit detail includes environment_id and message.
  - OTel span "environment.delete" emitted on success.
  - OTel span "environment.delete" emitted on failure.
  - OTel span set to ERROR status on DELETE failure.
  - Span carries environment.id attribute on DELETE.
  - DELETE route present with storage_root; absent without.
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
    base: dict = {"name": "my-env", "backend": "docker"}
    base.update(overrides)
    return base


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _env_resource(storage_root: Path, env_id: str) -> dict:
    return json.loads((storage_root / "environments" / f"{env_id}.json").read_text())


def _create_env(client: TestClient, **overrides) -> dict:
    return client.post("/v1/environments", json=_body(**overrides)).json()


# ---------------------------------------------------------------------------
# POST success
# ---------------------------------------------------------------------------


class TestEnvironmentCreateSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/environments", json=_body())
        assert resp.status_code == 201

    def test_various_backends_accepted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        for backend in ("docker", "firecracker", "process", "ssh"):
            resp = client.post(
                "/v1/environments", json=_body(name=f"env-{backend}", backend=backend)
            )
            assert resp.status_code == 201


# ---------------------------------------------------------------------------
# POST response fields
# ---------------------------------------------------------------------------


class TestEnvironmentCreateResponse:
    def test_response_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/environments", json=_body()).json()
        assert "id" in body
        assert isinstance(body["id"], str)

    def test_id_has_env_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/environments", json=_body()).json()
        assert body["id"].startswith("env_")

    def test_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = client.post("/v1/environments", json=_body(name="env-a")).json()["id"]
        id2 = client.post("/v1/environments", json=_body(name="env-b")).json()["id"]
        assert id1 != id2

    def test_response_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/environments", json=_body(name="my-env")).json()
        assert body["name"] == "my-env"

    def test_response_has_backend(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/environments", json=_body(backend="firecracker")).json()
        assert body["backend"] == "firecracker"

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/environments", json=_body()).json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0

    def test_response_has_updated_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/environments", json=_body()).json()
        assert "updated_at" in body
        assert isinstance(body["updated_at"], str)
        assert len(body["updated_at"]) > 0

    def test_response_has_image_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/environments", json=_body(image="ubuntu:22.04")
        ).json()
        assert body["image"] == "ubuntu:22.04"

    def test_response_has_template_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/environments", json=_body(template="python-3.11")
        ).json()
        assert body["template"] == "python-3.11"

    def test_response_has_workspace_path_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/environments", json=_body(workspace_path="/workspace")
        ).json()
        assert body["workspace_path"] == "/workspace"

    def test_response_has_env_passthrough_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/environments", json=_body(env_passthrough=["HOME", "PATH"])
        ).json()
        assert body["env_passthrough"] == ["HOME", "PATH"]

    def test_response_has_network_policy_field(self, storage_root: Path) -> None:
        policy = {"egress_allowed": True, "allowed_hosts": ["example.com"]}
        client = _make_client(storage_root)
        body = client.post(
            "/v1/environments", json=_body(network_policy=policy)
        ).json()
        assert body["network_policy"] == policy

    def test_response_has_caps_envelope_field(self, storage_root: Path) -> None:
        caps = {"cpu_millicores": 500, "memory_mb": 256}
        client = _make_client(storage_root)
        body = client.post(
            "/v1/environments", json=_body(caps_envelope=caps)
        ).json()
        assert body["caps_envelope"] == caps

    def test_response_has_default_timeout_ms_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/environments", json=_body(default_timeout_ms=30000)
        ).json()
        assert body["default_timeout_ms"] == 30000

    def test_optional_fields_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/environments", json=_body()).json()
        for field in ("image", "template", "workspace_path", "env_passthrough",
                      "network_policy", "caps_envelope", "default_timeout_ms"):
            assert body[field] is None


# ---------------------------------------------------------------------------
# POST persistence
# ---------------------------------------------------------------------------


class TestEnvironmentPersistence:
    def test_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = client.post("/v1/environments", json=_body()).json()["id"]
        assert (storage_root / "environments" / f"{env_id}.json").exists()

    def test_persisted_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = client.post(
            "/v1/environments", json=_body(name="persist-env")
        ).json()["id"]
        resource = _env_resource(storage_root, env_id)
        assert resource["name"] == "persist-env"

    def test_persisted_backend(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = client.post(
            "/v1/environments", json=_body(backend="firecracker")
        ).json()["id"]
        resource = _env_resource(storage_root, env_id)
        assert resource["backend"] == "firecracker"

    def test_not_written_on_validation_failure(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/environments", json=_body(name=""))
        envs_dir = storage_root / "environments"
        files = list(envs_dir.glob("*.json")) if envs_dir.exists() else []
        assert files == []


# ---------------------------------------------------------------------------
# POST validation errors
# ---------------------------------------------------------------------------


class TestEnvironmentCreateValidation:
    def test_missing_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/environments", json={"backend": "docker"})
        assert resp.status_code == 422

    def test_missing_backend_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/environments", json={"name": "my-env"})
        assert resp.status_code == 422

    def test_empty_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/environments", json=_body(name="   "))
        assert resp.status_code == 422

    def test_empty_name_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/environments", json=_body(name="")).json()
        assert body["error"]["code"] == "environment_invalid_request"

    def test_empty_name_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/environments", json=_body(name="")).json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0

    def test_empty_backend_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/environments", json=_body(backend="   "))
        assert resp.status_code == 422

    def test_empty_backend_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/environments", json=_body(backend="")).json()
        assert body["error"]["code"] == "environment_invalid_request"


# ---------------------------------------------------------------------------
# POST audit log
# ---------------------------------------------------------------------------


class TestEnvironmentCreateAuditLog:
    def test_validation_failure_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/environments", json=_body(name=""))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "environment.create.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/environments", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "environment.create.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_code_is_invalid_request(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/environments", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "environment.create.failed"
        )
        assert record["code"] == "environment_invalid_request"

    def test_failure_audit_detail_has_environment_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/environments", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "environment.create.failed"
        )
        assert record["detail"]["environment_id"].startswith("env_")

    def test_failure_audit_detail_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/environments", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "environment.create.failed"
        )
        assert "name" in record["detail"]

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/environments", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "environment.create.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# POST OTel spans
# ---------------------------------------------------------------------------


class TestEnvironmentCreateOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/environments", json=_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "environment.create" in span_names

    def test_failure_emits_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/environments", json=_body(name=""))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "environment.create" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.post("/v1/environments", json=_body(name=""))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("environment.create")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_environment_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/environments", json=_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("environment.create")
        assert span is not None
        assert span.attributes["environment.id"].startswith("env_")

    def test_success_span_has_name_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/environments", json=_body(name="otel-env"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("environment.create")
        assert span is not None
        assert span.attributes["environment.name"] == "otel-env"

    def test_success_span_has_backend_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/environments", json=_body(backend="firecracker"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("environment.create")
        assert span is not None
        assert span.attributes["environment.backend"] == "firecracker"


# ---------------------------------------------------------------------------
# POST route wiring
# ---------------------------------------------------------------------------


class TestEnvironmentCreateRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/environments", json=_body())
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/environments", json=_body())
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET list success
# ---------------------------------------------------------------------------


class TestEnvironmentListSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/environments")
        assert resp.status_code == 200

    def test_empty_items_when_no_environments(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/environments").json()
        assert body["items"] == []

    def test_returns_all_created_environments(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/environments", json=_body(name="env-a"))
        client.post("/v1/environments", json=_body(name="env-b"))
        body = client.get("/v1/environments").json()
        assert len(body["items"]) == 2

    def test_single_environment_in_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = client.post("/v1/environments", json=_body(name="solo")).json()["id"]
        body = client.get("/v1/environments").json()
        assert len(body["items"]) == 1
        assert body["items"][0]["id"] == env_id

    def test_item_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/environments", json=_body())
        item = client.get("/v1/environments").json()["items"][0]
        assert "id" in item
        assert item["id"].startswith("env_")

    def test_item_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/environments", json=_body(name="named-env"))
        item = client.get("/v1/environments").json()["items"][0]
        assert item["name"] == "named-env"

    def test_item_has_backend(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/environments", json=_body(backend="firecracker"))
        item = client.get("/v1/environments").json()["items"][0]
        assert item["backend"] == "firecracker"

    def test_item_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/environments", json=_body())
        item = client.get("/v1/environments").json()["items"][0]
        assert "created_at" in item
        assert isinstance(item["created_at"], str)
        assert len(item["created_at"]) > 0


# ---------------------------------------------------------------------------
# GET list OTel spans
# ---------------------------------------------------------------------------


class TestEnvironmentListOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_emits_list_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.get("/v1/environments")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "environment.list" in span_names


# ---------------------------------------------------------------------------
# GET list route wiring
# ---------------------------------------------------------------------------


class TestEnvironmentListRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/environments")
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/environments")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET single success
# ---------------------------------------------------------------------------


class TestEnvironmentGetSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        resp = client.get(f"/v1/environments/{env_id}")
        assert resp.status_code == 200

    def test_response_has_correct_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        body = client.get(f"/v1/environments/{env_id}").json()
        assert body["id"] == env_id

    def test_response_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client, name="get-test-env")["id"]
        body = client.get(f"/v1/environments/{env_id}").json()
        assert body["name"] == "get-test-env"

    def test_response_has_backend(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client, backend="firecracker")["id"]
        body = client.get(f"/v1/environments/{env_id}").json()
        assert body["backend"] == "firecracker"

    def test_response_has_all_optional_fields(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(
            client,
            image="ubuntu:22.04",
            workspace_path="/workspace",
            env_passthrough=["PATH"],
            default_timeout_ms=5000,
        )["id"]
        body = client.get(f"/v1/environments/{env_id}").json()
        assert body["image"] == "ubuntu:22.04"
        assert body["workspace_path"] == "/workspace"
        assert body["env_passthrough"] == ["PATH"]
        assert body["default_timeout_ms"] == 5000


# ---------------------------------------------------------------------------
# GET single not found
# ---------------------------------------------------------------------------


class TestEnvironmentGetNotFound:
    def test_unknown_id_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/environments/env_nonexistent")
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/environments/env_nonexistent").json()
        assert body["error"]["code"] == "environment_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/environments/env_nonexistent").json()
        assert len(body["error"]["message"]) > 0

    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/environments/env_missing")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "environment.get.failed" for r in records)

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/environments/env_missing")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "environment.get.failed"
        )
        assert record["code"] == "environment_not_found"


# ---------------------------------------------------------------------------
# GET single OTel spans
# ---------------------------------------------------------------------------


class TestEnvironmentGetOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_get_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        env_id = _create_env(client)["id"]
        _otel_exporter.clear()
        client.get(f"/v1/environments/{env_id}")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "environment.get" in span_names

    def test_not_found_emits_get_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.get("/v1/environments/env_missing")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "environment.get" in span_names

    def test_not_found_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.get("/v1/environments/env_missing")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("environment.get")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_environment_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        env_id = _create_env(client)["id"]
        _otel_exporter.clear()
        client.get(f"/v1/environments/{env_id}")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("environment.get")
        assert span is not None
        assert span.attributes["environment.id"] == env_id


# ---------------------------------------------------------------------------
# PATCH success
# ---------------------------------------------------------------------------


class TestEnvironmentUpdateSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        resp = client.patch(f"/v1/environments/{env_id}", json={"name": "updated"})
        assert resp.status_code == 200

    def test_updates_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        body = client.patch(
            f"/v1/environments/{env_id}", json={"name": "new-name"}
        ).json()
        assert body["name"] == "new-name"

    def test_updates_image(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        body = client.patch(
            f"/v1/environments/{env_id}", json={"image": "debian:12"}
        ).json()
        assert body["image"] == "debian:12"

    def test_updates_template(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        body = client.patch(
            f"/v1/environments/{env_id}", json={"template": "node-18"}
        ).json()
        assert body["template"] == "node-18"

    def test_updates_workspace_path(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        body = client.patch(
            f"/v1/environments/{env_id}", json={"workspace_path": "/repo"}
        ).json()
        assert body["workspace_path"] == "/repo"

    def test_updates_env_passthrough(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        body = client.patch(
            f"/v1/environments/{env_id}", json={"env_passthrough": ["HOME", "USER"]}
        ).json()
        assert body["env_passthrough"] == ["HOME", "USER"]

    def test_updates_network_policy(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        policy = {"egress_allowed": False}
        body = client.patch(
            f"/v1/environments/{env_id}", json={"network_policy": policy}
        ).json()
        assert body["network_policy"] == policy

    def test_updates_caps_envelope(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        caps = {"memory_mb": 1024}
        body = client.patch(
            f"/v1/environments/{env_id}", json={"caps_envelope": caps}
        ).json()
        assert body["caps_envelope"] == caps

    def test_updates_default_timeout_ms(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        body = client.patch(
            f"/v1/environments/{env_id}", json={"default_timeout_ms": 60000}
        ).json()
        assert body["default_timeout_ms"] == 60000

    def test_updates_updated_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        created = _create_env(client)
        env_id = created["id"]
        original_created_at = created["created_at"]
        body = client.patch(
            f"/v1/environments/{env_id}", json={"name": "patched"}
        ).json()
        assert "updated_at" in body
        assert body["created_at"] == original_created_at

    def test_preserves_backend_on_patch(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client, backend="firecracker")["id"]
        body = client.patch(
            f"/v1/environments/{env_id}", json={"name": "patched"}
        ).json()
        assert body["backend"] == "firecracker"

    def test_patch_persisted_to_disk(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        client.patch(f"/v1/environments/{env_id}", json={"name": "disk-check"})
        resource = _env_resource(storage_root, env_id)
        assert resource["name"] == "disk-check"


# ---------------------------------------------------------------------------
# PATCH validation / not found
# ---------------------------------------------------------------------------


class TestEnvironmentUpdateErrors:
    def test_empty_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        resp = client.patch(f"/v1/environments/{env_id}", json={"name": "   "})
        assert resp.status_code == 422

    def test_empty_name_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        body = client.patch(f"/v1/environments/{env_id}", json={"name": ""}).json()
        assert body["error"]["code"] == "environment_invalid_request"

    def test_unknown_id_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.patch("/v1/environments/env_nonexistent", json={"name": "x"})
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.patch(
            "/v1/environments/env_nonexistent", json={"name": "x"}
        ).json()
        assert body["error"]["code"] == "environment_not_found"

    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.patch("/v1/environments/env_missing", json={"name": "x"})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "environment.update.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.patch("/v1/environments/env_missing", json={"name": "x"})
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "environment.update.failed"
        )
        assert record["level"] == "error"


# ---------------------------------------------------------------------------
# PATCH OTel spans
# ---------------------------------------------------------------------------


class TestEnvironmentUpdateOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_update_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        env_id = _create_env(client)["id"]
        _otel_exporter.clear()
        client.patch(f"/v1/environments/{env_id}", json={"name": "updated"})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "environment.update" in span_names

    def test_not_found_emits_update_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.patch("/v1/environments/env_missing", json={"name": "x"})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "environment.update" in span_names

    def test_not_found_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.patch("/v1/environments/env_missing", json={"name": "x"})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("environment.update")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_environment_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        env_id = _create_env(client)["id"]
        _otel_exporter.clear()
        client.patch(f"/v1/environments/{env_id}", json={"name": "updated"})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("environment.update")
        assert span is not None
        assert span.attributes["environment.id"] == env_id


# ---------------------------------------------------------------------------
# DELETE success
# ---------------------------------------------------------------------------


class TestEnvironmentDeleteSuccess:
    def test_delete_returns_204(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        resp = client.delete(f"/v1/environments/{env_id}")
        assert resp.status_code == 204

    def test_delete_response_has_no_body(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        resp = client.delete(f"/v1/environments/{env_id}")
        assert resp.content == b""

    def test_delete_removes_file(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        client.delete(f"/v1/environments/{env_id}")
        assert not (storage_root / "environments" / f"{env_id}.json").exists()

    def test_deleted_environment_absent_from_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        client.delete(f"/v1/environments/{env_id}")
        items = client.get("/v1/environments").json()["items"]
        assert all(item["id"] != env_id for item in items)

    def test_second_delete_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        client.delete(f"/v1/environments/{env_id}")
        resp = client.delete(f"/v1/environments/{env_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE not found
# ---------------------------------------------------------------------------


class TestEnvironmentDeleteNotFound:
    def test_unknown_id_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.delete("/v1/environments/env_nonexistent")
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.delete("/v1/environments/env_nonexistent").json()
        assert body["error"]["code"] == "environment_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.delete("/v1/environments/env_nonexistent").json()
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# DELETE audit log
# ---------------------------------------------------------------------------


class TestEnvironmentDeleteAuditLog:
    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/environments/env_missing")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "environment.delete.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/environments/env_missing")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "environment.delete.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/environments/env_missing")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "environment.delete.failed"
        )
        assert record["code"] == "environment_not_found"

    def test_not_found_audit_detail_has_environment_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/environments/env_missing")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "environment.delete.failed"
        )
        assert record["detail"]["environment_id"] == "env_missing"

    def test_not_found_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/environments/env_missing")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "environment.delete.failed"
        )
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# DELETE OTel spans
# ---------------------------------------------------------------------------


class TestEnvironmentDeleteOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_delete_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        env_id = _create_env(client)["id"]
        _otel_exporter.clear()
        client.delete(f"/v1/environments/{env_id}")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "environment.delete" in span_names

    def test_not_found_emits_delete_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.delete("/v1/environments/env_missing")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "environment.delete" in span_names

    def test_not_found_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.delete("/v1/environments/env_missing")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("environment.delete")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_environment_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        env_id = _create_env(client)["id"]
        _otel_exporter.clear()
        client.delete(f"/v1/environments/{env_id}")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("environment.delete")
        assert span is not None
        assert span.attributes["environment.id"] == env_id


# ---------------------------------------------------------------------------
# DELETE route wiring
# ---------------------------------------------------------------------------


class TestEnvironmentDeleteRouteWiring:
    def test_delete_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        resp = client.delete(f"/v1/environments/{env_id}")
        assert resp.status_code != 404

    def test_delete_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.delete("/v1/environments/env_any")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE 409 conflict (agent reference)
# ---------------------------------------------------------------------------


def _write_agent_with_env(storage_root: Path, env_id: str, agent_id: str = "agent_ref") -> None:
    agents_dir = storage_root / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "id": agent_id,
        "name": "ref-agent",
        "kind": "claude",
        "default_environment_id": env_id,
        "created_at": "2024-01-01T00:00:00+00:00",
        "version": {},
    }
    (agents_dir / f"{agent_id}.json").write_text(json.dumps(record))


class TestEnvironmentDeleteConflict:
    def test_returns_409_when_agent_references_env(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        _write_agent_with_env(storage_root, env_id)
        resp = client.delete(f"/v1/environments/{env_id}")
        assert resp.status_code == 409

    def test_conflict_error_code_is_environment_in_use(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        _write_agent_with_env(storage_root, env_id)
        body = client.delete(f"/v1/environments/{env_id}").json()
        assert body["error"]["code"] == "environment_in_use"

    def test_conflict_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        _write_agent_with_env(storage_root, env_id)
        body = client.delete(f"/v1/environments/{env_id}").json()
        assert len(body["error"]["message"]) > 0

    def test_conflict_does_not_remove_environment_file(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        _write_agent_with_env(storage_root, env_id)
        client.delete(f"/v1/environments/{env_id}")
        assert (storage_root / "environments" / f"{env_id}.json").exists()

    def test_delete_succeeds_after_reference_removed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        agent_file = storage_root / "agents" / "agent_ref.json"
        _write_agent_with_env(storage_root, env_id)
        assert client.delete(f"/v1/environments/{env_id}").status_code == 409
        agent_file.unlink()
        assert client.delete(f"/v1/environments/{env_id}").status_code == 204

    def test_conflict_writes_audit_log(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        _write_agent_with_env(storage_root, env_id)
        client.delete(f"/v1/environments/{env_id}")
        records = _audit_records(storage_root)
        assert any(
            r.get("event") == "environment.delete.failed"
            and r.get("code") == "environment_in_use"
            for r in records
        )

    def test_conflict_audit_detail_has_environment_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        env_id = _create_env(client)["id"]
        _write_agent_with_env(storage_root, env_id)
        client.delete(f"/v1/environments/{env_id}")
        records = _audit_records(storage_root)
        record = next(
            r for r in records
            if r.get("event") == "environment.delete.failed"
            and r.get("code") == "environment_in_use"
        )
        assert record["detail"]["environment_id"] == env_id
