"""
System integration test: POST /v1/sessions/{id}/messages appends a user or system message.

Tests cover:
  - POST /v1/sessions/{id}/messages returns 201 on success.
  - Response includes message_id, id, session_id, thread_id, role, content, created_at.
  - id field equals message_id.
  - message_id starts with "msg_".
  - message_id is unique across calls.
  - role is preserved in the response.
  - content is preserved in the response.
  - Uses session primary thread_id when thread_id omitted from request.
  - Uses provided thread_id when supplied in request.
  - Returns 422 with code "message_append_rejected" for role "assistant".
  - Returns 422 with code "message_append_rejected" for role "tool".
  - Returns 404 with code "session_not_found" when session does not exist.
  - Message record is written to messages.ndjson on disk.
  - Appended message appears in GET /v1/sessions/{id}/messages listing.
  - message.added event is written to the event log.
  - On success, audit log entry is written with event "session.message.appended".
  - On success, audit detail includes session_id, thread_id, message_id, and role.
  - On failure (role rejected), audit log entry is written with event "session.message.append.rejected".
  - On failure (session not found), audit log entry is written with event "session.message.append.failed".
  - On failure, error message is surfaced in response body.
  - OTel span "session.message.append" is emitted on success.
  - OTel span has session.id attribute.
  - OTel span carries a structured invocation event on each call.
  - OTel span is set to ERROR status on failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
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


def _post_message(
    client: TestClient,
    session_id: str,
    *,
    role: str = "user",
    content: Any = "Hello",
    thread_id: str | None = None,
) -> tuple[dict[str, Any], Any]:
    body: dict[str, Any] = {"role": role, "content": content}
    if thread_id is not None:
        body["thread_id"] = thread_id
    resp = client.post(f"/v1/sessions/{session_id}/messages", json=body)
    return resp.json() | {"_status": resp.status_code}, resp


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _event_records(storage_root: Path, session_id: str) -> list[dict[str, Any]]:
    events_dir = storage_root / "events"
    for path in events_dir.rglob(f"{session_id}.ndjson"):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return []


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


class TestAppendMessageResponse:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"])
        assert result["_status"] == 201

    def test_has_message_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"])
        assert "message_id" in result
        assert result["message_id"].startswith("msg_")

    def test_has_id_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"])
        assert "id" in result

    def test_id_equals_message_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"])
        assert result["id"] == result["message_id"]

    def test_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"])
        assert result["session_id"] == session["session_id"]

    def test_has_thread_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"])
        assert "thread_id" in result
        assert result["thread_id"]

    def test_has_role(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"], role="user")
        assert result["role"] == "user"

    def test_system_role_accepted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"], role="system")
        assert result["_status"] == 201
        assert result["role"] == "system"

    def test_has_content(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"], content="Hello world")
        assert result["content"] == "Hello world"

    def test_list_content_preserved(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        content = [{"type": "text", "text": "Hi"}]
        result, _ = _post_message(client, session["session_id"], content=content)
        assert result["content"] == content

    def test_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"])
        assert "created_at" in result
        assert len(result["created_at"]) > 0

    def test_message_id_is_unique_across_calls(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result1, _ = _post_message(client, session["session_id"])
        result2, _ = _post_message(client, session["session_id"])
        assert result1["message_id"] != result2["message_id"]

    def test_defaults_to_session_primary_thread(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"])
        assert result["thread_id"] == session["thread_id"]

    def test_uses_provided_thread_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        custom_thread = "thread_custom123"
        result, _ = _post_message(client, session_id, thread_id=custom_thread)
        assert result["thread_id"] == custom_thread


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestAppendMessagePersistence:
    def test_message_written_to_disk(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        result, _ = _post_message(client, session_id)
        thread_id = result["thread_id"]
        messages_path = storage_root / "threads" / session_id / thread_id / "messages.ndjson"
        assert messages_path.exists()

    def test_message_record_has_correct_role(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        result, _ = _post_message(client, session_id, role="system", content="You are helpful.")
        thread_id = result["thread_id"]
        messages_path = storage_root / "threads" / session_id / thread_id / "messages.ndjson"
        records = [json.loads(l) for l in messages_path.read_text().splitlines() if l.strip()]
        record = next(r for r in records if r.get("id") == result["message_id"])
        assert record["role"] == "system"

    def test_message_record_has_correct_content(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        result, _ = _post_message(client, session_id, content="Stored content")
        thread_id = result["thread_id"]
        messages_path = storage_root / "threads" / session_id / thread_id / "messages.ndjson"
        records = [json.loads(l) for l in messages_path.read_text().splitlines() if l.strip()]
        record = next(r for r in records if r.get("id") == result["message_id"])
        assert record["content"] == "Stored content"

    def test_appended_message_appears_in_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        result, _ = _post_message(client, session_id, content="Listed message")
        message_id = result["message_id"]
        list_resp = client.get(f"/v1/sessions/{session_id}/messages")
        items = list_resp.json()["items"]
        assert any(m["id"] == message_id for m in items)

    def test_multiple_messages_all_appear_in_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        r1, _ = _post_message(client, session_id, content="First")
        r2, _ = _post_message(client, session_id, content="Second")
        list_resp = client.get(f"/v1/sessions/{session_id}/messages")
        ids = {m["id"] for m in list_resp.json()["items"]}
        assert r1["message_id"] in ids
        assert r2["message_id"] in ids


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


class TestAppendMessageEventLog:
    def test_message_added_event_written(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        _post_message(client, session_id)
        records = _event_records(storage_root, session_id)
        assert any(r.get("type") == "message.added" for r in records)

    def test_event_data_has_message_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        result, _ = _post_message(client, session_id)
        records = _event_records(storage_root, session_id)
        event = next(r for r in records if r.get("type") == "message.added")
        assert event["data"]["message_id"] == result["message_id"]

    def test_event_data_has_role(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        _post_message(client, session_id, role="user")
        records = _event_records(storage_root, session_id)
        event = next(r for r in records if r.get("type") == "message.added")
        assert event["data"]["role"] == "user"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestAppendMessageErrors:
    def test_assistant_role_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"], role="assistant")
        assert result["_status"] == 422

    def test_assistant_role_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"], role="assistant")
        assert result["error"]["code"] == "message_append_rejected"

    def test_tool_role_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"], role="tool")
        assert result["_status"] == 422

    def test_tool_role_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"], role="tool")
        assert result["error"]["code"] == "message_append_rejected"

    def test_rejected_role_error_message_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"], role="assistant")
        assert "message" in result["error"]
        assert len(result["error"]["message"]) > 0

    def test_missing_session_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _post_message(client, "sess_doesnotexist")
        assert result["_status"] == 404

    def test_missing_session_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _post_message(client, "sess_doesnotexist")
        assert result["error"]["code"] == "session_not_found"

    def test_missing_session_error_message_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _post_message(client, "sess_doesnotexist")
        assert "message" in result["error"]
        assert len(result["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAppendMessageAudit:
    def test_success_writes_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _post_message(client, session["session_id"])
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.message.appended" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _post_message(client, session["session_id"])
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.message.appended")
        assert record["level"] == "info"

    def test_success_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _post_message(client, session["session_id"])
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.message.appended")
        assert record["detail"]["session_id"] == session["session_id"]

    def test_success_audit_detail_has_thread_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"])
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.message.appended")
        assert record["detail"]["thread_id"] == result["thread_id"]

    def test_success_audit_detail_has_message_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _post_message(client, session["session_id"])
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.message.appended")
        assert record["detail"]["message_id"] == result["message_id"]

    def test_success_audit_detail_has_role(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _post_message(client, session["session_id"], role="user")
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.message.appended")
        assert record["detail"]["role"] == "user"

    def test_role_rejection_writes_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _post_message(client, session["session_id"], role="assistant")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.message.append.rejected" for r in records)

    def test_role_rejection_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _post_message(client, session["session_id"], role="assistant")
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.message.append.rejected")
        assert record["level"] == "error"

    def test_role_rejection_audit_detail_has_role(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _post_message(client, session["session_id"], role="assistant")
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.message.append.rejected")
        assert record["detail"]["role"] == "assistant"

    def test_session_not_found_writes_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _post_message(client, "sess_missing")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.message.append.failed" for r in records)

    def test_session_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _post_message(client, "sess_missing")
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.message.append.failed")
        assert record["level"] == "error"

    def test_session_not_found_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _post_message(client, "sess_missing")
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.message.append.failed")
        assert record["detail"]["session_id"] == "sess_missing"

    def test_session_not_found_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _post_message(client, "sess_missing")
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.message.append.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel instrumentation
# ---------------------------------------------------------------------------


class TestAppendMessageOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _otel_exporter.clear()
        _post_message(client, session["session_id"])
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.message.append" in span_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _otel_exporter.clear()
        _post_message(client, session["session_id"])
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.message.append")
        assert span is not None
        assert span.attributes["session.id"] == session["session_id"]

    def test_span_has_invocation_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _otel_exporter.clear()
        _post_message(client, session["session_id"])
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.message.append")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = _make_client(storage_root)
        _otel_exporter.clear()
        _post_message(client, "sess_missing")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.message.append")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_role_rejection_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = _make_client(storage_root)
        session = _post_session(client)
        _otel_exporter.clear()
        _post_message(client, session["session_id"], role="tool")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.message.append")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR
