"""
Memory store write endpoint conformance suite.

Tests cover:
  - POST /v1/memory_stores/{id}/write returns 201 on success.
  - Returns 404 for non-existent store_id.
  - Response has store_id, key, content, scope, embedder_id, action, created_at fields.
  - action is "inserted" for a new key.
  - action is "updated" for an existing key.
  - embedder_id defaults to "hash-128" when omitted.
  - Custom embedder_id accepted and echoed in response.
  - scope defaults to "global" when omitted.
  - Custom scope accepted and echoed in response.
  - Written memory is retrievable via query_runs.
  - 404 error response has error.code "memory_store_not_found".
  - 404 error response has error.message.
  - Audit log written on not-found failure.
  - Audit entry level is "error" on failure.
  - Audit entry event is "memory_store.write.failed" on failure.
  - Audit detail includes memory_store_id, key, message.
  - OTel span "memory_store.write" emitted on success.
  - OTel span "memory_store.write" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries memory_store.id and memory_store.write.key attributes.
  - create_app wires write route when storage_root is supplied.
  - create_app omits write route when storage_root is None.
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


def _create_store(client: TestClient, **overrides) -> str:
    body: dict = {"name": "test-store", "backend": "sqlite-vec", "scope": "global"}
    body.update(overrides)
    resp = client.post("/v1/memory_stores", json=body)
    assert resp.status_code == 201
    return resp.json()["id"]


def _write_body(**overrides) -> dict:
    base: dict = {"key": "mem_1", "content": "hello world"}
    base.update(overrides)
    return base


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestMemoryStoreWriteSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        resp = client.post(f"/v1/memory_stores/{store_id}/write", json=_write_body())
        assert resp.status_code == 201

    def test_action_is_inserted_for_new_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = client.post(f"/v1/memory_stores/{store_id}/write", json=_write_body()).json()
        assert body["action"] == "inserted"

    def test_action_is_updated_for_existing_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        client.post(f"/v1/memory_stores/{store_id}/write", json=_write_body(key="k1"))
        body = client.post(
            f"/v1/memory_stores/{store_id}/write", json=_write_body(key="k1")
        ).json()
        assert body["action"] == "updated"

    def test_default_embedder_id_is_hash_128(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = client.post(f"/v1/memory_stores/{store_id}/write", json=_write_body()).json()
        assert body["embedder_id"] == "hash-128"

    def test_custom_embedder_id_echoed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = client.post(
            f"/v1/memory_stores/{store_id}/write",
            json=_write_body(embedder_id="text-embedding-3-small"),
        ).json()
        assert body["embedder_id"] == "text-embedding-3-small"

    def test_default_scope_is_global(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = client.post(f"/v1/memory_stores/{store_id}/write", json=_write_body()).json()
        assert body["scope"] == "global"

    def test_custom_scope_echoed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = client.post(
            f"/v1/memory_stores/{store_id}/write", json=_write_body(scope="agent")
        ).json()
        assert body["scope"] == "agent"


# ---------------------------------------------------------------------------
# Response fields
# ---------------------------------------------------------------------------


class TestMemoryStoreWriteResponseFields:
    def test_response_has_store_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = client.post(f"/v1/memory_stores/{store_id}/write", json=_write_body()).json()
        assert body["store_id"] == store_id

    def test_response_has_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = client.post(
            f"/v1/memory_stores/{store_id}/write", json=_write_body(key="my_key")
        ).json()
        assert body["key"] == "my_key"

    def test_response_has_content(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = client.post(
            f"/v1/memory_stores/{store_id}/write",
            json=_write_body(content="some memory text"),
        ).json()
        assert body["content"] == "some memory text"

    def test_response_has_scope(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = client.post(f"/v1/memory_stores/{store_id}/write", json=_write_body()).json()
        assert "scope" in body

    def test_response_has_embedder_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = client.post(f"/v1/memory_stores/{store_id}/write", json=_write_body()).json()
        assert "embedder_id" in body

    def test_response_has_action(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = client.post(f"/v1/memory_stores/{store_id}/write", json=_write_body()).json()
        assert "action" in body

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = client.post(f"/v1/memory_stores/{store_id}/write", json=_write_body()).json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class TestMemoryStoreWriteRetrieval:
    def test_written_memory_retrievable_via_query(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        client.post(
            f"/v1/memory_stores/{store_id}/write",
            json=_write_body(key="entry_a", content="unique phrase xyzzy"),
        )
        result = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json={"query": "xyzzy"},
        ).json()
        assert result["count"] >= 1

    def test_update_replaces_previous_content(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        client.post(
            f"/v1/memory_stores/{store_id}/write",
            json=_write_body(key="k1", content="old content alpha"),
        )
        client.post(
            f"/v1/memory_stores/{store_id}/write",
            json=_write_body(key="k1", content="new content beta"),
        )
        result = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json={"query": "beta"},
        ).json()
        assert result["count"] >= 1
        contents = [r["content"] for r in result["results"]]
        assert any("new content beta" in c for c in contents)


# ---------------------------------------------------------------------------
# Not found
# ---------------------------------------------------------------------------


class TestMemoryStoreWriteNotFound:
    def test_unknown_store_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/memory_stores/memstore_doesnotexist/write", json=_write_body()
        )
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/memory_stores/memstore_doesnotexist/write", json=_write_body()
        ).json()
        assert body["error"]["code"] == "memory_store_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/memory_stores/memstore_doesnotexist/write", json=_write_body()
        ).json()
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestMemoryStoreWriteAuditLog:
    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/write", json=_write_body())
        records = _audit_records(storage_root)
        assert any(r.get("event") == "memory_store.write.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/write", json=_write_body())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.write.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_code_is_not_found(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/write", json=_write_body())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.write.failed"
        )
        assert record["code"] == "memory_store_not_found"

    def test_failure_audit_detail_has_memory_store_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/write", json=_write_body())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.write.failed"
        )
        assert record["detail"]["memory_store_id"] == "memstore_missing"

    def test_failure_audit_detail_has_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/memory_stores/memstore_missing/write", json=_write_body(key="my_key")
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.write.failed"
        )
        assert record["detail"]["key"] == "my_key"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/write", json=_write_body())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.write.failed"
        )
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestMemoryStoreWriteOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_memory_store_write_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        store_id = _create_store(client)
        _otel_exporter.clear()
        client.post(f"/v1/memory_stores/{store_id}/write", json=_write_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "memory_store.write" in span_names

    def test_failure_emits_memory_store_write_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/write", json=_write_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "memory_store.write" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/write", json=_write_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.write")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_store_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        store_id = _create_store(client)
        _otel_exporter.clear()
        client.post(f"/v1/memory_stores/{store_id}/write", json=_write_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.write")
        assert span is not None
        assert span.attributes["memory_store.id"] == store_id

    def test_success_span_has_key_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        store_id = _create_store(client)
        _otel_exporter.clear()
        client.post(
            f"/v1/memory_stores/{store_id}/write", json=_write_body(key="span_key")
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.write")
        assert span is not None
        assert span.attributes["memory_store.write.key"] == "span_key"


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestMemoryStoreWriteRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        resp = client.post(f"/v1/memory_stores/{store_id}/write", json=_write_body())
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/memory_stores/memstore_x/write", json=_write_body())
        assert resp.status_code == 404
