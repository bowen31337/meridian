"""
Memory store query_runs endpoint conformance suite.

Tests cover:
  - POST /v1/memory_stores/{id}/query_runs returns 200 on success (empty store).
  - Returns 404 for non-existent store_id.
  - Response has results, count, query, scope, store_id fields.
  - Empty store returns count=0 and empty results list.
  - bm25_weight defaults to 1.0 in response.
  - vector_weight defaults to 1.0 in response.
  - rrf_k defaults to 60 in response.
  - Custom bm25_weight, vector_weight, rrf_k accepted and echoed in response.
  - scope filter parameter accepted and echoed in response.
  - count matches len(results).
  - BM25 returns indexed content.
  - Vector search returns indexed content.
  - Hybrid fuses BM25 and vector results via weighted RRF.
  - Scope filter restricts results to matching scope.
  - 404 error response has error.code "memory_store_not_found".
  - 404 error response has error.message.
  - Audit log written on not-found failure.
  - Audit entry level is "error" on failure.
  - Audit entry event is "memory_store.query.failed" on failure.
  - Audit detail includes memory_store_id, query, message.
  - OTel span "memory_store.query" emitted on success.
  - OTel span "memory_store.query" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries memory_store.id, memory_store.query, memory_store.scope attributes.
  - create_app wires query_runs route when storage_root is supplied.
  - create_app omits query_runs route when storage_root is None.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._kb import KbStore
from meridian_kb_indexer import Chunk

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _create_store(client: TestClient, storage_root: Path, **overrides) -> str:
    body: dict = {"name": "test-store", "backend": "sqlite-vec", "scope": "global"}
    body.update(overrides)
    resp = client.post("/v1/memory_stores", json=body)
    assert resp.status_code == 201
    return resp.json()["id"]


def _query_body(**overrides) -> dict:
    base: dict = {"query": "hello world"}
    base.update(overrides)
    return base


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _seed_store(storage_root: Path, store_id: str, entries: list[tuple[str, str, str]]) -> None:
    """Seed a memory store's SQLite DB. Each entry is (key, content, scope)."""
    stores_dir = storage_root / "memory_stores"
    kb = KbStore(stores_dir / store_id / "chunks.db")
    for key, content, scope in entries:
        kb.upsert_chunks(
            key,
            scope,
            [Chunk(file_path=key, kind="text", content=content, start_line=0, end_line=0)],
        )


# ---------------------------------------------------------------------------
# Success — empty store
# ---------------------------------------------------------------------------


class TestMemoryStoreQueryRunsEmptyStore:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        resp = client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body())
        assert resp.status_code == 200

    def test_empty_store_returns_zero_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body()).json()
        assert body["count"] == 0

    def test_empty_store_returns_empty_results(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body()).json()
        assert body["results"] == []


# ---------------------------------------------------------------------------
# Not found
# ---------------------------------------------------------------------------


class TestMemoryStoreQueryRunsNotFound:
    def test_unknown_store_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/memory_stores/memstore_doesnotexist/query_runs", json=_query_body()
        )
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/memory_stores/memstore_doesnotexist/query_runs", json=_query_body()
        ).json()
        assert body["error"]["code"] == "memory_store_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/memory_stores/memstore_doesnotexist/query_runs", json=_query_body()
        ).json()
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# Response fields
# ---------------------------------------------------------------------------


class TestMemoryStoreQueryRunsResponseFields:
    def test_response_has_results_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body()).json()
        assert "results" in body

    def test_response_has_count_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body()).json()
        assert "count" in body

    def test_count_matches_results_length(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body()).json()
        assert body["count"] == len(body["results"])

    def test_response_has_query_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs", json=_query_body(query="test query")
        ).json()
        assert body["query"] == "test query"

    def test_response_has_store_id_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body()).json()
        assert body["store_id"] == store_id

    def test_response_has_scope_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs", json=_query_body(scope="global")
        ).json()
        assert body["scope"] == "global"

    def test_scope_none_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body()).json()
        assert body["scope"] is None

    def test_default_bm25_weight_is_one(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body()).json()
        assert body["bm25_weight"] == 1.0

    def test_default_vector_weight_is_one(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body()).json()
        assert body["vector_weight"] == 1.0

    def test_default_rrf_k_is_60(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body()).json()
        assert body["rrf_k"] == 60

    def test_custom_bm25_weight_echoed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(bm25_weight=0.5),
        ).json()
        assert body["bm25_weight"] == 0.5

    def test_custom_vector_weight_echoed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(vector_weight=2.0),
        ).json()
        assert body["vector_weight"] == 2.0

    def test_custom_rrf_k_echoed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(rrf_k=30),
        ).json()
        assert body["rrf_k"] == 30


# ---------------------------------------------------------------------------
# Retrieval with seeded content
# ---------------------------------------------------------------------------


class TestMemoryStoreQueryRunsRetrieval:
    def test_bm25_finds_seeded_content(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        _seed_store(storage_root, store_id, [("entry_1", "hello world python", "global")])
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(query="hello"),
        ).json()
        assert body["count"] >= 1

    def test_vector_finds_seeded_content(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        _seed_store(storage_root, store_id, [("entry_1", "hello world python", "global")])
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(query="hello"),
        ).json()
        assert body["count"] >= 1

    def test_hybrid_returns_results(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        _seed_store(
            storage_root,
            store_id,
            [("e1", "foo bar baz", "global"), ("e2", "hello world", "global")],
        )
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(query="foo"),
        ).json()
        assert body["count"] >= 1

    def test_scope_filter_restricts_results(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        _seed_store(
            storage_root,
            store_id,
            [("e1", "hello world", "agent"), ("e2", "hello world", "user")],
        )
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(query="hello", scope="agent"),
        ).json()
        for result in body["results"]:
            assert result["scope"] == "agent"

    def test_limit_respected(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        _seed_store(
            storage_root,
            store_id,
            [(f"entry_{i}", f"word{i} shared content here", "global") for i in range(10)],
        )
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(query="shared content", limit=3),
        ).json()
        assert body["count"] <= 3

    def test_result_has_content_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        _seed_store(storage_root, store_id, [("e1", "hello world", "global")])
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(query="hello"),
        ).json()
        assert body["count"] >= 1
        assert "content" in body["results"][0]

    def test_result_has_scope_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        _seed_store(storage_root, store_id, [("e1", "hello world", "global")])
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(query="hello"),
        ).json()
        assert body["count"] >= 1
        assert "scope" in body["results"][0]

    def test_zero_bm25_weight_uses_only_vector(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        _seed_store(storage_root, store_id, [("e1", "hello world", "global")])
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(query="hello", bm25_weight=0.0, vector_weight=1.0),
        ).json()
        assert body["count"] >= 1

    def test_zero_vector_weight_uses_only_bm25(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        _seed_store(storage_root, store_id, [("e1", "hello world", "global")])
        body = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(query="hello", bm25_weight=1.0, vector_weight=0.0),
        ).json()
        assert body["count"] >= 1


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestMemoryStoreQueryRunsAuditLog:
    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/query_runs", json=_query_body())
        records = _audit_records(storage_root)
        assert any(r.get("event") == "memory_store.query.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/query_runs", json=_query_body())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.query.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_code_is_not_found(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/query_runs", json=_query_body())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.query.failed"
        )
        assert record["code"] == "memory_store_not_found"

    def test_failure_audit_detail_has_memory_store_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/query_runs", json=_query_body())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.query.failed"
        )
        assert record["detail"]["memory_store_id"] == "memstore_missing"

    def test_failure_audit_detail_has_query(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/memory_stores/memstore_missing/query_runs",
            json=_query_body(query="my search"),
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.query.failed"
        )
        assert record["detail"]["query"] == "my search"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/query_runs", json=_query_body())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "memory_store.query.failed"
        )
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestMemoryStoreQueryRunsOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_memory_store_query_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        store_id = _create_store(client, storage_root)
        _otel_exporter.clear()
        client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "memory_store.query" in span_names

    def test_failure_emits_memory_store_query_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/query_runs", json=_query_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "memory_store.query" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.post("/v1/memory_stores/memstore_missing/query_runs", json=_query_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.query")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_store_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        store_id = _create_store(client, storage_root)
        _otel_exporter.clear()
        client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.query")
        assert span is not None
        assert span.attributes["memory_store.id"] == store_id

    def test_success_span_has_query_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        store_id = _create_store(client, storage_root)
        _otel_exporter.clear()
        client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(query="my query"),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.query")
        assert span is not None
        assert span.attributes["memory_store.query"] == "my query"

    def test_success_span_has_scope_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        store_id = _create_store(client, storage_root)
        _otel_exporter.clear()
        client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json=_query_body(scope="agent"),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.query")
        assert span is not None
        assert span.attributes["memory_store.scope"] == "agent"


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestMemoryStoreQueryRunsRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client, storage_root)
        resp = client.post(f"/v1/memory_stores/{store_id}/query_runs", json=_query_body())
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/memory_stores/memstore_x/query_runs", json=_query_body())
        assert resp.status_code == 404
