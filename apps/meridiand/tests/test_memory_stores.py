"""
Memory stores endpoint conformance suite.

Tests cover:
  - POST /v1/memory_stores returns 201 on success.
  - Response has id with "memstore_" prefix.
  - IDs are unique across calls.
  - Response has name, backend, scope, metadata, created_at.
  - metadata is null when omitted; stored when provided.
  - backend "sqlite-vec" accepted.
  - backend "pgvector" accepted.
  - backend "http" accepted.
  - scope "global" accepted.
  - scope "user" accepted.
  - scope "agent" accepted.
  - scope "project" accepted.
  - Store JSON written to storage_root/memory_stores/{id}.json.
  - Persisted record has correct name, backend, scope.
  - Not written to disk on validation failure.
  - Empty name returns 422 with code "memory_store_invalid_request".
  - Missing required fields return 422.
  - On validation failure, audit log entry written with event "memory_store.create.failed".
  - Audit entry level is "error" on failure.
  - Audit entry code is "memory_store_invalid_request" on validation failure.
  - Audit detail includes memory_store_id, name, message on failure.
  - Error response body has error.code and error.message on failure.
  - OTel span "memory_store.create" emitted on success.
  - OTel span "memory_store.create" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries memory_store.id, memory_store.name, memory_store.backend, memory_store.scope attributes.
  - create_app wires memory_stores router when storage_root is supplied.
  - create_app omits memory_stores route when storage_root is None.
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
    base: dict = {"name": "my-store", "backend": "sqlite-vec", "scope": "global"}
    base.update(overrides)
    return base


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _store_resource(storage_root: Path, store_id: str) -> dict:
    return json.loads((storage_root / "memory_stores" / f"{store_id}.json").read_text())


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestMemoryStoreCreateSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json=_body())
        assert resp.status_code == 201

    def test_sqlite_vec_backend_accepted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json=_body(backend="sqlite-vec"))
        assert resp.status_code == 201

    def test_pgvector_backend_accepted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json=_body(backend="pgvector"))
        assert resp.status_code == 201

    def test_http_backend_accepted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json=_body(backend="http"))
        assert resp.status_code == 201

    def test_global_scope_accepted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json=_body(scope="global"))
        assert resp.status_code == 201

    def test_user_scope_accepted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json=_body(scope="user"))
        assert resp.status_code == 201

    def test_agent_scope_accepted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json=_body(scope="agent"))
        assert resp.status_code == 201

    def test_project_scope_accepted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json=_body(scope="project"))
        assert resp.status_code == 201

    def test_with_metadata_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json=_body(metadata={"dim": 1536}))
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Response fields
# ---------------------------------------------------------------------------


class TestMemoryStoreCreateResponse:
    def test_response_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/memory_stores", json=_body()).json()
        assert "id" in body
        assert isinstance(body["id"], str)

    def test_id_has_memstore_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/memory_stores", json=_body()).json()
        assert body["id"].startswith("memstore_")

    def test_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = client.post("/v1/memory_stores", json=_body(name="store-a")).json()["id"]
        id2 = client.post("/v1/memory_stores", json=_body(name="store-b")).json()["id"]
        assert id1 != id2

    def test_response_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/memory_stores", json=_body(name="my-store")).json()
        assert body["name"] == "my-store"

    def test_response_has_backend(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/memory_stores", json=_body(backend="pgvector")).json()
        assert body["backend"] == "pgvector"

    def test_response_has_scope(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/memory_stores", json=_body(scope="agent")).json()
        assert body["scope"] == "agent"

    def test_metadata_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/memory_stores", json=_body()).json()
        assert body["metadata"] is None

    def test_metadata_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        meta = {"dim": 768, "model": "text-embedding-3-small"}
        body = client.post("/v1/memory_stores", json=_body(metadata=meta)).json()
        assert body["metadata"] == meta

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/memory_stores", json=_body()).json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestMemoryStorePersistence:
    def test_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = client.post("/v1/memory_stores", json=_body()).json()["id"]
        assert (storage_root / "memory_stores" / f"{store_id}.json").exists()

    def test_persisted_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = client.post("/v1/memory_stores", json=_body(name="persist-store")).json()["id"]
        resource = _store_resource(storage_root, store_id)
        assert resource["name"] == "persist-store"

    def test_persisted_backend(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = client.post("/v1/memory_stores", json=_body(backend="http")).json()["id"]
        resource = _store_resource(storage_root, store_id)
        assert resource["backend"] == "http"

    def test_persisted_scope(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = client.post("/v1/memory_stores", json=_body(scope="project")).json()["id"]
        resource = _store_resource(storage_root, store_id)
        assert resource["scope"] == "project"

    def test_not_written_on_validation_failure(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores", json=_body(name=""))
        stores_dir = storage_root / "memory_stores"
        files = list(stores_dir.glob("*.json")) if stores_dir.exists() else []
        assert files == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestMemoryStoreCreateValidation:
    def test_missing_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json={"backend": "sqlite-vec", "scope": "global"})
        assert resp.status_code == 422

    def test_missing_backend_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json={"name": "s", "scope": "global"})
        assert resp.status_code == 422

    def test_missing_scope_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json={"name": "s", "backend": "sqlite-vec"})
        assert resp.status_code == 422

    def test_empty_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json=_body(name="   "))
        assert resp.status_code == 422

    def test_empty_name_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/memory_stores", json=_body(name="")).json()
        assert body["error"]["code"] == "memory_store_invalid_request"

    def test_empty_name_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/memory_stores", json=_body(name="")).json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0

    def test_invalid_backend_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json=_body(backend="redis"))
        assert resp.status_code == 422

    def test_invalid_scope_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json=_body(scope="workspace"))
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestMemoryStoreAuditLog:
    def test_validation_failure_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores", json=_body(name=""))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "memory_store.create.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.create.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_code_is_invalid_request(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.create.failed"
        )
        assert record["code"] == "memory_store_invalid_request"

    def test_failure_audit_detail_has_memory_store_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.create.failed"
        )
        assert record["detail"]["memory_store_id"].startswith("memstore_")

    def test_failure_audit_detail_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.create.failed"
        )
        assert "name" in record["detail"]

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.create.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestMemoryStoreOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_memory_store_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/memory_stores", json=_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "memory_store.create" in span_names

    def test_failure_emits_memory_store_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/memory_stores", json=_body(name=""))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "memory_store.create" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.post("/v1/memory_stores", json=_body(name=""))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.create")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_memory_store_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/memory_stores", json=_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.create")
        assert span is not None
        assert span.attributes["memory_store.id"].startswith("memstore_")

    def test_success_span_has_name_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/memory_stores", json=_body(name="otel-store"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.create")
        assert span is not None
        assert span.attributes["memory_store.name"] == "otel-store"

    def test_success_span_has_backend_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/memory_stores", json=_body(backend="pgvector"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.create")
        assert span is not None
        assert span.attributes["memory_store.backend"] == "pgvector"

    def test_success_span_has_scope_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/memory_stores", json=_body(scope="user"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.create")
        assert span is not None
        assert span.attributes["memory_store.scope"] == "user"


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestMemoryStoreRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/memory_stores", json=_body())
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/memory_stores", json=_body())
        assert resp.status_code == 404
