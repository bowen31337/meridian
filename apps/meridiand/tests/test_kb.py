"""
KB API conformance suite.

Tests cover:
  - POST /v1/x/kb/index returns 200 with scope, row_count, status fields.
  - POST /v1/x/kb/index with explicit path indexes that file under requested scope.
  - POST /v1/x/kb/index with explicit scope scans WORKSPACE directory.
  - POST /v1/x/kb/index with no path/scope uses WORKSPACE env or cwd under scope "global".
  - POST /v1/x/kb/index updates storage_root/kb/status.json with row_counts and last_updated.
  - POST /v1/x/kb/index accumulates row_counts across multiple scopes.
  - POST /v1/x/kb/index scope must be one of global/project/agent/session.
  - On index failure, returns 422 with code "kb_index_failed".
  - On index failure, writes audit log entry with event "kb.index.failed".
  - Audit log detail includes scope and path.
  - OTel span "kb.index" is emitted on success.
  - OTel span is set to ERROR status on index failure.
  - GET /v1/x/kb returns 200 with status, last_updated, row_counts fields.
  - GET /v1/x/kb returns default values when no status.json exists.
  - GET /v1/x/kb reflects counts written by POST /v1/x/kb/index.
  - On status read failure, returns 422 with code "kb_status_failed".
  - OTel span "kb.status" is emitted on success.
  - POST /v1/x/kb/query returns 200 with results, count, query, method fields.
  - POST /v1/x/kb/query bm25 method finds content matching query terms.
  - POST /v1/x/kb/query glob method matches file paths by pattern.
  - POST /v1/x/kb/query vector method returns semantically scored results.
  - POST /v1/x/kb/query hybrid method fuses glob + bm25 + vector via RRF.
  - POST /v1/x/kb/query scope filter narrows results to that scope.
  - POST /v1/x/kb/query limit is respected.
  - POST /v1/x/kb/query result items have required fields including content_hash.
  - On query failure, returns 422 with code "kb_query_failed".
  - On query failure, writes audit log entry with event "kb.query.failed".
  - Audit log detail includes query text.
  - OTel span "kb.query" is emitted on success.
  - OTel span is set to ERROR status on query failure.
  - create_app wires the kb router when storage_root is supplied.
  - create_app omits the kb route when storage_root is None.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
import pytest

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path) -> TestClient:
    audit = FileAuditLog(storage_root)
    app = create_app(audit, storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _make_py_file(directory: Path, name: str = "sample.py") -> Path:
    p = directory / name
    p.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")
    return p


# ---------------------------------------------------------------------------
# POST /v1/x/kb/index — success cases
# ---------------------------------------------------------------------------


class TestKbIndexEndpoint:
    def test_returns_200_on_success_with_path(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        resp = client.post("/v1/x/kb/index", json={"path": str(f)})
        assert resp.status_code == 200

    def test_response_has_scope_global_by_default(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        body = client.post("/v1/x/kb/index", json={"path": str(f)}).json()
        assert body["scope"] == "global"

    def test_response_has_row_count(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        body = client.post("/v1/x/kb/index", json={"path": str(f)}).json()
        assert "row_count" in body
        assert body["row_count"] >= 1

    def test_response_status_indexed(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        body = client.post("/v1/x/kb/index", json={"path": str(f)}).json()
        assert body["status"] == "indexed"

    def test_explicit_scope_used_in_response(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        body = client.post("/v1/x/kb/index", json={"path": str(f), "scope": "project"}).json()
        assert body["scope"] == "project"

    def test_default_scope_is_global(
        self, storage_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKSPACE", str(tmp_path))
        _make_py_file(tmp_path)
        client = _make_client(storage_root)
        body = client.post("/v1/x/kb/index", json={}).json()
        assert body["scope"] == "global"

    def test_invalid_scope_returns_422(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        resp = client.post("/v1/x/kb/index", json={"path": str(f), "scope": "workspace"})
        assert resp.status_code == 422

    def test_status_json_written_after_index(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        assert (storage_root / "kb" / "status.json").exists()

    def test_status_json_has_last_updated(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        data = json.loads((storage_root / "kb" / "status.json").read_text())
        assert data["last_updated"] is not None

    def test_status_json_row_count_keyed_by_scope(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        data = json.loads((storage_root / "kb" / "status.json").read_text())
        assert "global" in data["row_counts"]
        assert data["row_counts"]["global"] >= 1

    def test_status_json_row_count_keyed_by_explicit_scope(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post(
            "/v1/x/kb/index", json={"path": str(tmp_path / "sample.py"), "scope": "project"}
        )
        data = json.loads((storage_root / "kb" / "status.json").read_text())
        assert "project" in data["row_counts"]

    def test_row_counts_accumulate_across_scopes(self, storage_root: Path, tmp_path: Path) -> None:
        f1 = _make_py_file(tmp_path, "a.py")
        f2 = _make_py_file(tmp_path, "b.py")
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f1), "scope": "global"})
        client.post("/v1/x/kb/index", json={"path": str(f2), "scope": "project"})
        data = json.loads((storage_root / "kb" / "status.json").read_text())
        assert "global" in data["row_counts"]
        assert "project" in data["row_counts"]

    def test_scope_scan_indexes_all_files(
        self, storage_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKSPACE", str(tmp_path))
        _make_py_file(tmp_path, "x.py")
        _make_py_file(tmp_path, "y.py")
        client = _make_client(storage_root)
        resp = client.post("/v1/x/kb/index", json={"scope": "global"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["row_count"] >= 2


# ---------------------------------------------------------------------------
# POST /v1/x/kb/index — failure cases
# ---------------------------------------------------------------------------


class TestKbIndexFailure:
    def test_nonexistent_path_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/kb/index", json={"path": "/nonexistent/file.py"})
        assert resp.status_code == 422

    def test_nonexistent_path_returns_kb_index_failed_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/kb/index", json={"path": "/nonexistent/file.py"})
        assert resp.json()["error"]["code"] == "kb_index_failed"

    def test_write_failure_returns_422(self, storage_root: Path, tmp_path: Path) -> None:
        # Block status write by placing a file where kb/ dir would be
        (storage_root / "kb").write_text("block")
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        resp = client.post("/v1/x/kb/index", json={"path": str(f)})
        assert resp.status_code == 422

    def test_write_failure_returns_kb_index_failed_code(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        (storage_root / "kb").write_text("block")
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        resp = client.post("/v1/x/kb/index", json={"path": str(f)})
        assert resp.json()["error"]["code"] == "kb_index_failed"

    def test_failure_writes_audit_log_event(self, storage_root: Path, tmp_path: Path) -> None:
        (storage_root / "kb").write_text("block")
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        records = [
            json.loads(line)
            for line in (storage_root / "audit.ndjson").read_text().splitlines()
            if line.strip()
        ]
        assert any(r.get("event") == "kb.index.failed" for r in records)

    def test_failure_audit_detail_has_scope(self, storage_root: Path, tmp_path: Path) -> None:
        (storage_root / "kb").write_text("block")
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        records = [
            json.loads(line)
            for line in (storage_root / "audit.ndjson").read_text().splitlines()
            if line.strip()
        ]
        record = next(r for r in records if r.get("event") == "kb.index.failed")
        assert "scope" in record["detail"]

    def test_failure_audit_detail_has_path(self, storage_root: Path, tmp_path: Path) -> None:
        (storage_root / "kb").write_text("block")
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        records = [
            json.loads(line)
            for line in (storage_root / "audit.ndjson").read_text().splitlines()
            if line.strip()
        ]
        record = next(r for r in records if r.get("event") == "kb.index.failed")
        assert record["detail"]["path"] == str(f)


# ---------------------------------------------------------------------------
# GET /v1/x/kb — success cases
# ---------------------------------------------------------------------------


class TestKbStatusEndpoint:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/x/kb")
        assert resp.status_code == 200

    def test_default_status_is_idle(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/x/kb").json()
        assert body["status"] == "idle"

    def test_default_last_updated_is_null(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/x/kb").json()
        assert body["last_updated"] is None

    def test_default_row_counts_is_empty_dict(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/x/kb").json()
        assert body["row_counts"] == {}

    def test_reflects_row_counts_after_index(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.get("/v1/x/kb").json()
        assert "global" in body["row_counts"]
        assert body["row_counts"]["global"] >= 1

    def test_last_updated_set_after_index(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.get("/v1/x/kb").json()
        assert body["last_updated"] is not None

    def test_multiple_scopes_returned(self, storage_root: Path, tmp_path: Path) -> None:
        f1 = _make_py_file(tmp_path, "p.py")
        f2 = _make_py_file(tmp_path, "q.py")
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f1), "scope": "global"})
        client.post("/v1/x/kb/index", json={"path": str(f2), "scope": "project"})
        body = client.get("/v1/x/kb").json()
        assert "global" in body["row_counts"]
        assert "project" in body["row_counts"]


# ---------------------------------------------------------------------------
# GET /v1/x/kb — failure case
# ---------------------------------------------------------------------------


class TestKbStatusFailure:
    def test_corrupt_status_json_returns_422(self, storage_root: Path) -> None:
        kb_dir = storage_root / "kb"
        kb_dir.mkdir(parents=True, exist_ok=True)
        (kb_dir / "status.json").write_text("{not valid json{{")
        client = _make_client(storage_root)
        resp = client.get("/v1/x/kb")
        assert resp.status_code == 422

    def test_corrupt_status_json_returns_kb_status_failed_code(self, storage_root: Path) -> None:
        kb_dir = storage_root / "kb"
        kb_dir.mkdir(parents=True, exist_ok=True)
        (kb_dir / "status.json").write_text("{not valid json{{")
        client = _make_client(storage_root)
        resp = client.get("/v1/x/kb")
        assert resp.json()["error"]["code"] == "kb_status_failed"

    def test_corrupt_status_writes_audit_log(self, storage_root: Path) -> None:
        kb_dir = storage_root / "kb"
        kb_dir.mkdir(parents=True, exist_ok=True)
        (kb_dir / "status.json").write_text("{not valid json{{")
        client = _make_client(storage_root)
        client.get("/v1/x/kb")
        records = [
            json.loads(line)
            for line in (storage_root / "audit.ndjson").read_text().splitlines()
            if line.strip()
        ]
        assert any(r.get("event") == "kb.status.failed" for r in records)


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------


class TestKbAppWiring:
    def test_no_storage_root_no_index_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/kb/index", json={})
        assert resp.status_code == 404

    def test_no_storage_root_no_status_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/x/kb")
        assert resp.status_code == 404

    def test_with_storage_root_index_route_exists(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        resp = client.post("/v1/x/kb/index", json={"path": str(f)})
        assert resp.status_code != 404

    def test_with_storage_root_status_route_exists(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/x/kb")
        assert resp.status_code != 404


# ---------------------------------------------------------------------------
# OTel span tests
# ---------------------------------------------------------------------------


class TestKbOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _make_client(self, storage_root: Path) -> TestClient:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_index_success_emits_kb_index_span(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = self._make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "kb.index" in span_names

    def test_index_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": "/nonexistent/file.py"})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        kb_span = spans.get("kb.index")
        assert kb_span is not None
        assert kb_span.status.status_code == StatusCode.ERROR

    def test_status_success_emits_kb_status_span(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        client.get("/v1/x/kb")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "kb.status" in span_names

    def test_query_success_emits_kb_query_span(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = self._make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        _otel_exporter.clear()
        client.post("/v1/x/kb/query", json={"query": "foo", "method": "bm25"})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "kb.query" in span_names

    def test_query_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        db_path = storage_root / "kb" / "chunks.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_bytes(b"not a valid sqlite database")
        client = self._make_client(storage_root)
        _otel_exporter.clear()
        client.post("/v1/x/kb/query", json={"query": "foo"})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        kb_span = spans.get("kb.query")
        assert kb_span is not None
        assert kb_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# POST /v1/x/kb/query — success cases
# ---------------------------------------------------------------------------


class TestKbQueryEndpoint:
    def test_returns_200_with_empty_index(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/kb/query", json={"query": "foo"})
        assert resp.status_code == 200

    def test_response_has_results_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/kb/query", json={"query": "foo"}).json()
        assert "results" in body

    def test_response_has_count_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/kb/query", json={"query": "foo"}).json()
        assert "count" in body

    def test_count_matches_results_length(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/kb/query", json={"query": "foo"}).json()
        assert body["count"] == len(body["results"])

    def test_response_has_query_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/kb/query", json={"query": "foo"}).json()
        assert body["query"] == "foo"

    def test_response_has_method_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/kb/query", json={"query": "foo", "method": "bm25"}).json()
        assert body["method"] == "bm25"

    def test_default_method_is_hybrid(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/kb/query", json={"query": "foo"}).json()
        assert body["method"] == "hybrid"

    def test_empty_index_returns_zero_results(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/kb/query", json={"query": "foo"}).json()
        assert body["count"] == 0
        assert body["results"] == []

    def test_bm25_finds_indexed_content(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.post("/v1/x/kb/query", json={"query": "foo", "method": "bm25"}).json()
        assert body["count"] >= 1
        assert any(r["file_path"] == str(f) for r in body["results"])

    def test_glob_finds_matching_file_path(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.post("/v1/x/kb/query", json={"query": "*.py", "method": "glob"}).json()
        assert body["count"] >= 1
        assert any(r["file_path"] == str(f) for r in body["results"])

    def test_glob_no_match_returns_empty(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.post("/v1/x/kb/query", json={"query": "*.ts", "method": "glob"}).json()
        assert body["count"] == 0

    def test_vector_finds_indexed_content(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.post("/v1/x/kb/query", json={"query": "foo", "method": "vector"}).json()
        assert body["count"] >= 1

    def test_hybrid_returns_results_after_index(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.post("/v1/x/kb/query", json={"query": "foo", "method": "hybrid"}).json()
        assert body["count"] >= 1

    def test_result_has_file_path_field(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.post("/v1/x/kb/query", json={"query": "foo", "method": "bm25"}).json()
        assert body["count"] >= 1
        assert "file_path" in body["results"][0]

    def test_result_has_kind_field(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.post("/v1/x/kb/query", json={"query": "foo", "method": "bm25"}).json()
        assert body["count"] >= 1
        assert "kind" in body["results"][0]

    def test_result_has_content_field(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.post("/v1/x/kb/query", json={"query": "foo", "method": "bm25"}).json()
        assert body["count"] >= 1
        assert "content" in body["results"][0]

    def test_result_has_start_and_end_line(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.post("/v1/x/kb/query", json={"query": "foo", "method": "bm25"}).json()
        assert body["count"] >= 1
        r = body["results"][0]
        assert "start_line" in r
        assert "end_line" in r

    def test_result_has_content_hash_field(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.post("/v1/x/kb/query", json={"query": "foo", "method": "bm25"}).json()
        assert body["count"] >= 1
        r = body["results"][0]
        assert "content_hash" in r
        assert len(r["content_hash"]) == 64  # SHA-256 hex digest

    def test_result_has_scope_field(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.post("/v1/x/kb/query", json={"query": "foo", "method": "bm25"}).json()
        assert body["count"] >= 1
        assert body["results"][0]["scope"] == "global"

    def test_scope_filter_narrows_bm25_results(self, storage_root: Path, tmp_path: Path) -> None:
        f1 = _make_py_file(tmp_path, "a.py")
        f2 = _make_py_file(tmp_path, "b.py")
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f1), "scope": "global"})
        client.post("/v1/x/kb/index", json={"path": str(f2), "scope": "project"})
        body = client.post(
            "/v1/x/kb/query",
            json={"query": "foo", "method": "bm25", "scope": "global"},
        ).json()
        assert all(r["scope"] == "global" for r in body["results"])
        assert all(r["file_path"] == str(f1) for r in body["results"])

    def test_limit_respected(self, storage_root: Path, tmp_path: Path) -> None:
        for name in ["a.py", "b.py", "c.py"]:
            _make_py_file(tmp_path, name)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"scope": "global", "path": str(tmp_path / "a.py")})
        client.post("/v1/x/kb/index", json={"scope": "global", "path": str(tmp_path / "b.py")})
        client.post("/v1/x/kb/index", json={"scope": "global", "path": str(tmp_path / "c.py")})
        body = client.post(
            "/v1/x/kb/query", json={"query": "foo", "method": "bm25", "limit": 1}
        ).json()
        assert body["count"] <= 1
        assert len(body["results"]) <= 1

    def test_reindex_updates_stored_chunks(self, storage_root: Path, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/index", json={"path": str(f)})
        f.write_text("def unique_reindex_marker():\n    pass\n")
        client.post("/v1/x/kb/index", json={"path": str(f)})
        body = client.post(
            "/v1/x/kb/query",
            json={"query": "unique_reindex_marker", "method": "bm25"},
        ).json()
        assert body["count"] >= 1

    def test_invalid_query_scope_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/kb/query", json={"query": "foo", "scope": "workspace"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /v1/x/kb/query — failure cases
# ---------------------------------------------------------------------------


class TestKbQueryFailure:
    def _corrupt_db(self, storage_root: Path) -> None:
        db_path = storage_root / "kb" / "chunks.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_bytes(b"not a valid sqlite database")

    def test_corrupted_db_returns_422(self, storage_root: Path) -> None:
        self._corrupt_db(storage_root)
        client = _make_client(storage_root)
        resp = client.post("/v1/x/kb/query", json={"query": "foo"})
        assert resp.status_code == 422

    def test_corrupted_db_returns_kb_query_failed_code(self, storage_root: Path) -> None:
        self._corrupt_db(storage_root)
        client = _make_client(storage_root)
        resp = client.post("/v1/x/kb/query", json={"query": "foo"})
        assert resp.json()["error"]["code"] == "kb_query_failed"

    def test_failure_writes_audit_log_event(self, storage_root: Path) -> None:
        self._corrupt_db(storage_root)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/query", json={"query": "foo"})
        records = [
            json.loads(line)
            for line in (storage_root / "audit.ndjson").read_text().splitlines()
            if line.strip()
        ]
        assert any(r.get("event") == "kb.query.failed" for r in records)

    def test_failure_audit_detail_has_query(self, storage_root: Path) -> None:
        self._corrupt_db(storage_root)
        client = _make_client(storage_root)
        client.post("/v1/x/kb/query", json={"query": "test_query_value"})
        records = [
            json.loads(line)
            for line in (storage_root / "audit.ndjson").read_text().splitlines()
            if line.strip()
        ]
        record = next(r for r in records if r.get("event") == "kb.query.failed")
        assert record["detail"]["query"] == "test_query_value"


# ---------------------------------------------------------------------------
# App wiring — query route
# ---------------------------------------------------------------------------


class TestKbQueryWiring:
    def test_no_storage_root_no_query_route(self, storage_root: Path) -> None:
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(storage_root)
        from meridiand._app import create_app

        app = create_app(audit)
        from fastapi.testclient import TestClient

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/kb/query", json={"query": "foo"})
        assert resp.status_code == 404

    def test_with_storage_root_query_route_exists(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/kb/query", json={"query": "foo"})
        assert resp.status_code != 404
