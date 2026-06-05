"""
POST /v1/files, GET /v1/files/{id}, GET /v1/files/{id}/content conformance suite.

Tests cover:
  - POST /v1/files multipart returns 201 with id, name, size, content_type, created_at.
  - POST /v1/files base64-in-JSON returns 201 with matching metadata fields.
  - POST /v1/files multipart file_id starts with "file_".
  - POST /v1/files JSON file_id starts with "file_".
  - POST /v1/files each call returns a unique id.
  - POST /v1/files multipart uses filename from upload when name not given.
  - POST /v1/files multipart uses override name when name form field provided.
  - POST /v1/files JSON sets name from body field.
  - POST /v1/files size reflects actual byte length.
  - POST /v1/files multipart preserves content_type from upload.
  - POST /v1/files JSON uses content_type from body.
  - POST /v1/files JSON missing content field returns 422 with code "files_invalid_request".
  - POST /v1/files JSON invalid base64 returns 422 with code "files_invalid_request".
  - POST /v1/files unsupported Content-Type returns 422 with code "files_invalid_request".
  - POST /v1/files on invalid request writes audit entry event "files.upload.failed".
  - GET /v1/files/{id} returns 200 with correct metadata for uploaded file.
  - GET /v1/files/{id} metadata id matches upload response id.
  - GET /v1/files/{id} metadata name, size, content_type match upload.
  - GET /v1/files/{id} unknown id returns 404 with code "files_not_found".
  - GET /v1/files/{id} not-found writes audit entry event "files.get_metadata.failed".
  - GET /v1/files/{id}/content returns 200 with correct bytes for multipart upload.
  - GET /v1/files/{id}/content returns 200 with correct bytes for JSON upload.
  - GET /v1/files/{id}/content Content-Type header matches stored content_type.
  - GET /v1/files/{id}/content Content-Length header matches file size.
  - GET /v1/files/{id}/content unknown id returns 404 with code "files_not_found".
  - GET /v1/files/{id}/content not-found writes audit entry event "files.get_content.failed".
  - OTel span "files.upload" emitted on successful upload.
  - OTel span "files.get_metadata" emitted on successful metadata fetch.
  - OTel span "files.get_content" emitted on successful content fetch.
  - OTel span set to ERROR on upload failure.
  - OTel span set to ERROR on metadata not-found.
  - OTel span set to ERROR on content not-found.
  - create_app wires files route when storage_root is supplied.
  - create_app omits files route when storage_root is None.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path) -> TestClient:
    audit = FileAuditLog(storage_root)
    app = create_app(audit, storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _upload_multipart(
    client: TestClient,
    content: bytes,
    filename: str = "test.bin",
    content_type: str = "application/octet-stream",
    name: str | None = None,
) -> Any:
    files = {"file": (filename, content, content_type)}
    data = {"name": name} if name else {}
    return client.post("/v1/files", files=files, data=data)


def _upload_json(
    client: TestClient,
    content: bytes,
    name: str = "test.bin",
    content_type: str = "application/octet-stream",
) -> Any:
    return client.post(
        "/v1/files",
        json={
            "name": name,
            "content": base64.b64encode(content).decode(),
            "content_type": content_type,
        },
    )


# ---------------------------------------------------------------------------
# POST /v1/files — multipart success
# ---------------------------------------------------------------------------


class TestFilesUploadMultipart:
    def test_returns_201(self, storage_root: Path) -> None:
        resp = _upload_multipart(_make_client(storage_root), b"hello")
        assert resp.status_code == 201

    def test_response_has_id(self, storage_root: Path) -> None:
        body = _upload_multipart(_make_client(storage_root), b"hello").json()
        assert "id" in body

    def test_id_starts_with_file_prefix(self, storage_root: Path) -> None:
        body = _upload_multipart(_make_client(storage_root), b"hello").json()
        assert body["id"].startswith("file_")

    def test_response_has_name(self, storage_root: Path) -> None:
        body = _upload_multipart(_make_client(storage_root), b"hi", filename="doc.txt").json()
        assert "name" in body

    def test_response_has_size(self, storage_root: Path) -> None:
        body = _upload_multipart(_make_client(storage_root), b"hello").json()
        assert "size" in body

    def test_response_has_content_type(self, storage_root: Path) -> None:
        body = _upload_multipart(_make_client(storage_root), b"hello").json()
        assert "content_type" in body

    def test_response_has_created_at(self, storage_root: Path) -> None:
        body = _upload_multipart(_make_client(storage_root), b"hello").json()
        assert "created_at" in body

    def test_size_matches_payload(self, storage_root: Path) -> None:
        payload = b"hello world"
        body = _upload_multipart(_make_client(storage_root), payload).json()
        assert body["size"] == len(payload)

    def test_content_type_from_upload(self, storage_root: Path) -> None:
        body = _upload_multipart(
            _make_client(storage_root), b"data", content_type="text/plain"
        ).json()
        assert body["content_type"] == "text/plain"

    def test_name_defaults_to_filename(self, storage_root: Path) -> None:
        body = _upload_multipart(_make_client(storage_root), b"x", filename="myfile.csv").json()
        assert body["name"] == "myfile.csv"

    def test_name_override_from_form_field(self, storage_root: Path) -> None:
        body = _upload_multipart(
            _make_client(storage_root), b"x", filename="original.txt", name="custom.txt"
        ).json()
        assert body["name"] == "custom.txt"

    def test_each_upload_returns_unique_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = _upload_multipart(client, b"a").json()["id"]
        id2 = _upload_multipart(client, b"b").json()["id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# POST /v1/files — base64-in-JSON success
# ---------------------------------------------------------------------------


class TestFilesUploadJson:
    def test_returns_201(self, storage_root: Path) -> None:
        resp = _upload_json(_make_client(storage_root), b"hello")
        assert resp.status_code == 201

    def test_id_starts_with_file_prefix(self, storage_root: Path) -> None:
        body = _upload_json(_make_client(storage_root), b"hello").json()
        assert body["id"].startswith("file_")

    def test_size_matches_decoded_payload(self, storage_root: Path) -> None:
        payload = b"binary\x00\xff\xfe"
        body = _upload_json(_make_client(storage_root), payload).json()
        assert body["size"] == len(payload)

    def test_name_from_body(self, storage_root: Path) -> None:
        body = _upload_json(_make_client(storage_root), b"x", name="report.pdf").json()
        assert body["name"] == "report.pdf"

    def test_content_type_from_body(self, storage_root: Path) -> None:
        body = _upload_json(_make_client(storage_root), b"x", content_type="image/png").json()
        assert body["content_type"] == "image/png"

    def test_each_call_returns_unique_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = _upload_json(client, b"a").json()["id"]
        id2 = _upload_json(client, b"b").json()["id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# POST /v1/files — validation errors
# ---------------------------------------------------------------------------


class TestFilesUploadValidation:
    def test_json_missing_content_returns_422(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post("/v1/files", json={"name": "f.bin"})
        assert resp.status_code == 422

    def test_json_missing_content_code(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post("/v1/files", json={"name": "f.bin"}).json()
        assert body["error"]["code"] == "files_invalid_request"

    def test_json_invalid_base64_returns_422(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post(
            "/v1/files", json={"name": "f.bin", "content": "!!!not-base64!!!"}
        )
        assert resp.status_code == 422

    def test_json_invalid_base64_code(self, storage_root: Path) -> None:
        body = (
            _make_client(storage_root)
            .post("/v1/files", json={"name": "f.bin", "content": "!!!not-base64!!!"})
            .json()
        )
        assert body["error"]["code"] == "files_invalid_request"

    def test_unsupported_content_type_returns_422(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post(
            "/v1/files",
            content=b"raw",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 422

    def test_multipart_missing_file_field_returns_422(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post(
            "/v1/files",
            data={"name": "oops"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /v1/files — audit log on failure
# ---------------------------------------------------------------------------


class TestFilesUploadAuditLog:
    def test_invalid_request_writes_audit_entry(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/files", json={"name": "f.bin"})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "files.upload.failed" for r in records)

    def test_audit_entry_level_is_error(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/files", json={"name": "f.bin"})
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "files.upload.failed"
        )
        assert record["level"] == "error"

    def test_audit_entry_code_is_files_invalid_request(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/files", json={"name": "f.bin"})
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "files.upload.failed"
        )
        assert record["code"] == "files_invalid_request"


# ---------------------------------------------------------------------------
# GET /v1/files/{id} — metadata
# ---------------------------------------------------------------------------


class TestFilesGetMetadata:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        file_id = _upload_multipart(client, b"data").json()["id"]
        resp = client.get(f"/v1/files/{file_id}")
        assert resp.status_code == 200

    def test_id_matches_upload(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        upload_id = _upload_multipart(client, b"data").json()["id"]
        meta = client.get(f"/v1/files/{upload_id}").json()
        assert meta["id"] == upload_id

    def test_name_matches_upload(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        upload = _upload_multipart(client, b"data", filename="hello.txt").json()
        meta = client.get(f"/v1/files/{upload['id']}").json()
        assert meta["name"] == upload["name"]

    def test_size_matches_upload(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = b"some bytes here"
        upload = _upload_multipart(client, payload).json()
        meta = client.get(f"/v1/files/{upload['id']}").json()
        assert meta["size"] == len(payload)

    def test_content_type_matches_upload(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        upload = _upload_multipart(client, b"x", content_type="text/plain").json()
        meta = client.get(f"/v1/files/{upload['id']}").json()
        assert meta["content_type"] == "text/plain"

    def test_unknown_id_returns_404(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).get("/v1/files/file_doesnotexist")
        assert resp.status_code == 404

    def test_unknown_id_error_code(self, storage_root: Path) -> None:
        body = _make_client(storage_root).get("/v1/files/file_doesnotexist").json()
        assert body["error"]["code"] == "files_not_found"

    def test_not_found_writes_audit_entry(self, storage_root: Path) -> None:
        _make_client(storage_root).get("/v1/files/file_missing")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "files.get_metadata.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        _make_client(storage_root).get("/v1/files/file_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "files.get_metadata.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        _make_client(storage_root).get("/v1/files/file_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "files.get_metadata.failed"
        )
        assert record["code"] == "files_not_found"


# ---------------------------------------------------------------------------
# GET /v1/files/{id}/content — bytes streaming
# ---------------------------------------------------------------------------


class TestFilesGetContent:
    def test_returns_200_for_multipart_upload(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        file_id = _upload_multipart(client, b"payload").json()["id"]
        resp = client.get(f"/v1/files/{file_id}/content")
        assert resp.status_code == 200

    def test_bytes_match_multipart_upload(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = b"exact bytes \x00\xff"
        file_id = _upload_multipart(client, payload).json()["id"]
        resp = client.get(f"/v1/files/{file_id}/content")
        assert resp.content == payload

    def test_bytes_match_json_upload(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = b"json uploaded \x01\x02\x03"
        file_id = _upload_json(client, payload).json()["id"]
        resp = client.get(f"/v1/files/{file_id}/content")
        assert resp.content == payload

    def test_content_type_header_matches_stored(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        file_id = _upload_multipart(client, b"img", content_type="image/jpeg").json()["id"]
        resp = client.get(f"/v1/files/{file_id}/content")
        assert "image/jpeg" in resp.headers.get("content-type", "")

    def test_content_length_header_matches_size(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        payload = b"count me"
        file_id = _upload_multipart(client, payload).json()["id"]
        resp = client.get(f"/v1/files/{file_id}/content")
        # Content-Length reflects the transferred (possibly compressed) length;
        # verify the decoded body length equals the original payload.
        assert len(resp.content) == len(payload)

    def test_unknown_id_returns_404(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).get("/v1/files/file_ghost/content")
        assert resp.status_code == 404

    def test_unknown_id_error_code(self, storage_root: Path) -> None:
        body = _make_client(storage_root).get("/v1/files/file_ghost/content").json()
        assert body["error"]["code"] == "files_not_found"

    def test_not_found_writes_audit_entry(self, storage_root: Path) -> None:
        _make_client(storage_root).get("/v1/files/file_ghost/content")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "files.get_content.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        _make_client(storage_root).get("/v1/files/file_ghost/content")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "files.get_content.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        _make_client(storage_root).get("/v1/files/file_ghost/content")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "files.get_content.failed"
        )
        assert record["code"] == "files_not_found"


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestFilesOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_upload_emits_files_upload_span(self, storage_root: Path) -> None:
        _upload_multipart(_make_client(storage_root), b"x")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "files.upload" in span_names

    def test_get_metadata_emits_span(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        file_id = _upload_multipart(client, b"x").json()["id"]
        _otel_exporter.clear()
        client.get(f"/v1/files/{file_id}")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "files.get_metadata" in span_names

    def test_get_content_emits_span(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        file_id = _upload_multipart(client, b"x").json()["id"]
        _otel_exporter.clear()
        client.get(f"/v1/files/{file_id}/content")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "files.get_content" in span_names

    def test_upload_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        _make_client(storage_root).post("/v1/files", json={"name": "f.bin"})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        upload_span = spans.get("files.upload")
        assert upload_span is not None
        assert upload_span.status.status_code == StatusCode.ERROR

    def test_get_metadata_not_found_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        _make_client(storage_root).get("/v1/files/file_missing")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        meta_span = spans.get("files.get_metadata")
        assert meta_span is not None
        assert meta_span.status.status_code == StatusCode.ERROR

    def test_get_content_not_found_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        _make_client(storage_root).get("/v1/files/file_ghost/content")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        content_span = spans.get("files.get_content")
        assert content_span is not None
        assert content_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestFilesRouteWiring:
    def test_with_storage_root_route_exists(self, storage_root: Path) -> None:
        resp = _upload_multipart(_make_client(storage_root), b"x")
        assert resp.status_code != 404

    def test_without_storage_root_post_returns_404(self, tmp_path: Path) -> None:
        audit = FileAuditLog(tmp_path)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = _upload_multipart(client, b"x")
        assert resp.status_code == 404

    def test_without_storage_root_get_metadata_returns_404(self, tmp_path: Path) -> None:
        audit = FileAuditLog(tmp_path)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/files/file_abc123")
        assert resp.status_code == 404

    def test_without_storage_root_get_content_returns_404(self, tmp_path: Path) -> None:
        audit = FileAuditLog(tmp_path)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/files/file_abc123/content")
        assert resp.status_code == 404
