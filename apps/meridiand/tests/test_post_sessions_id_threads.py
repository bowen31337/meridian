"""
System integration test: POST /v1/sessions/{id}/threads creates a forked Thread.

Tests cover:
  - POST /v1/sessions/{id}/threads returns 201 on success.
  - Response includes thread_id, id, session_id, created_at, branch_of_event_seq, title.
  - id field equals thread_id.
  - branch_of_event_seq matches the request value.
  - title defaults to null when omitted from the request.
  - title is present in the response when supplied.
  - Created thread file is persisted to disk with branch_of_event_seq and title.
  - Created thread appears in GET /v1/sessions/{id}/threads listing.
  - Returns 404 with code "session_not_found" when session does not exist.
  - Returns 422 when branch_of_event_seq is missing from the request body.
  - On success, audit log entry is written with event "session.thread.created".
  - On success, audit detail includes session_id, thread_id, and branch_of_event_seq.
  - On failure (session not found), audit log entry is written with event
    "session.thread.create.failed".
  - On failure, error message is surfaced in response body.
  - OTel span "session.thread.create" is emitted on success.
  - OTel span has session.id attribute.
  - OTel span carries a structured invocation event on each call.
  - OTel span is set to ERROR status on failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from storage_event_log import EventLogWriter, LocalEventLogWriter

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    storage_root: Path,
    event_log: EventLogWriter | None = None,
) -> TestClient:
    audit = FileAuditLog(storage_root)
    writer = event_log or LocalEventLogWriter(storage_root)
    app = create_app(audit, storage_root=storage_root, event_log=writer)
    return TestClient(app, raise_server_exceptions=False)


def _post_session(client: TestClient) -> dict[str, Any]:
    resp = client.post("/v1/sessions", json={})
    return resp.json() | {"_status": resp.status_code}


def _post_thread(
    client: TestClient,
    session_id: str,
    *,
    branch_of_event_seq: int = 1,
    title: str | None = None,
) -> tuple[dict[str, Any], Any]:
    body: dict[str, Any] = {"branch_of_event_seq": branch_of_event_seq}
    if title is not None:
        body["title"] = title
    resp = client.post(f"/v1/sessions/{session_id}/threads", json=body)
    return resp.json() | {"_status": resp.status_code}, resp


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


class TestCreateThreadResponse:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"])
        assert result["_status"] == 201

    def test_has_thread_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"])
        assert "thread_id" in result
        assert result["thread_id"].startswith("thread_")

    def test_has_id_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"])
        assert "id" in result

    def test_id_equals_thread_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"])
        assert result["id"] == result["thread_id"]

    def test_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"])
        assert result["session_id"] == session["session_id"]

    def test_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"])
        assert "created_at" in result
        assert len(result["created_at"]) > 0

    def test_branch_of_event_seq_matches_request(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"], branch_of_event_seq=42)
        assert result["branch_of_event_seq"] == 42

    def test_title_null_when_not_supplied(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"])
        assert "title" in result
        assert result["title"] is None

    def test_title_present_when_supplied(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"], title="Fork at step 5")
        assert result["title"] == "Fork at step 5"

    def test_thread_id_is_unique_across_calls(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result1, _ = _post_thread(client, session["session_id"], branch_of_event_seq=1)
        result2, _ = _post_thread(client, session["session_id"], branch_of_event_seq=2)
        assert result1["thread_id"] != result2["thread_id"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestCreateThreadPersistence:
    def test_thread_file_written_to_disk(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"], branch_of_event_seq=7)
        thread_id = result["thread_id"]
        path = storage_root / "sessions" / session["session_id"] / "threads" / f"{thread_id}.json"
        assert path.exists()

    def test_thread_file_contains_branch_of_event_seq(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"], branch_of_event_seq=99)
        thread_id = result["thread_id"]
        path = storage_root / "sessions" / session["session_id"] / "threads" / f"{thread_id}.json"
        record = json.loads(path.read_text())
        assert record["branch_of_event_seq"] == 99

    def test_thread_file_contains_title_when_supplied(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(
            client, session["session_id"], branch_of_event_seq=3, title="My Fork"
        )
        thread_id = result["thread_id"]
        path = storage_root / "sessions" / session["session_id"] / "threads" / f"{thread_id}.json"
        record = json.loads(path.read_text())
        assert record["title"] == "My Fork"

    def test_created_thread_appears_in_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"], branch_of_event_seq=5)
        thread_id = result["thread_id"]
        list_resp = client.get(f"/v1/sessions/{session['session_id']}/threads")
        items = list_resp.json()["items"]
        assert any(t["thread_id"] == thread_id for t in items)

    def test_listed_thread_has_branch_of_event_seq(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"], branch_of_event_seq=13)
        thread_id = result["thread_id"]
        list_resp = client.get(f"/v1/sessions/{session['session_id']}/threads")
        items = list_resp.json()["items"]
        thread = next(t for t in items if t["thread_id"] == thread_id)
        assert thread["branch_of_event_seq"] == 13


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestCreateThreadErrors:
    def test_returns_404_for_missing_session(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _post_thread(client, "sess_doesnotexist", branch_of_event_seq=1)
        assert result["_status"] == 404

    def test_404_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _post_thread(client, "sess_doesnotexist", branch_of_event_seq=1)
        assert result["error"]["code"] == "session_not_found"

    def test_404_error_message_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _post_thread(client, "sess_doesnotexist", branch_of_event_seq=1)
        assert "message" in result["error"]
        assert len(result["error"]["message"]) > 0

    def test_missing_branch_of_event_seq_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        resp = client.post(f"/v1/sessions/{session['session_id']}/threads", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestCreateThreadAudit:
    def test_success_writes_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _post_thread(client, session["session_id"], branch_of_event_seq=1)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.thread.created" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _post_thread(client, session["session_id"], branch_of_event_seq=1)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.thread.created")
        assert record["level"] == "info"

    def test_success_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _post_thread(client, session["session_id"], branch_of_event_seq=1)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.thread.created")
        assert record["detail"]["session_id"] == session["session_id"]

    def test_success_audit_detail_has_thread_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_thread(client, session["session_id"], branch_of_event_seq=1)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.thread.created")
        assert record["detail"]["thread_id"] == result["thread_id"]

    def test_success_audit_detail_has_branch_of_event_seq(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _post_thread(client, session["session_id"], branch_of_event_seq=77)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.thread.created")
        assert record["detail"]["branch_of_event_seq"] == 77

    def test_failure_writes_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _post_thread(client, "sess_missing", branch_of_event_seq=1)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.thread.create.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _post_thread(client, "sess_missing", branch_of_event_seq=1)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.thread.create.failed")
        assert record["level"] == "error"

    def test_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _post_thread(client, "sess_missing", branch_of_event_seq=1)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.thread.create.failed")
        assert record["detail"]["session_id"] == "sess_missing"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _post_thread(client, "sess_missing", branch_of_event_seq=1)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.thread.create.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel instrumentation
# ---------------------------------------------------------------------------


class TestCreateThreadOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _otel_exporter.clear()
        _post_thread(client, session["session_id"], branch_of_event_seq=1)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.thread.create" in span_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _otel_exporter.clear()
        _post_thread(client, session["session_id"], branch_of_event_seq=1)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.thread.create")
        assert span is not None
        assert span.attributes["session.id"] == session["session_id"]

    def test_span_has_invocation_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _otel_exporter.clear()
        _post_thread(client, session["session_id"], branch_of_event_seq=1)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.thread.create")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = _make_client(storage_root)
        _otel_exporter.clear()
        _post_thread(client, "sess_missing", branch_of_event_seq=1)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.thread.create")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR
