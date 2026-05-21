"""
System integration test: GET /v1/sessions/{id}/messages lists messages with cursor pagination.

Tests cover:
  - GET /v1/sessions/{id}/messages returns 200 with items, next_cursor, limit fields.
  - items is a list.
  - Each message item has id, thread_id, session_id, role, content, sequence, created_at.
  - Returns empty items list when the session has no threads directory.
  - items are sorted by created_at descending (most recent first).
  - Filtering by thread_id returns only messages from that thread.
  - Filtering by role returns only messages matching that role.
  - Filtering by both thread_id and role applies both filters.
  - cursor pagination: limit query param controls page size.
  - next_cursor is present in response body and Link header when more pages exist.
  - next_cursor is null when all items fit on one page.
  - cursor param advances to the next page.
  - Invalid cursor returns 400 with code "cursor_invalid".
  - Invalid cursor writes audit log entry with event "session.messages.list.failed".
  - On generic failure, returns 500 with code "session_messages_list_failed".
  - On failure, error message is surfaced in response body.
  - On failure, audit log entry is written with event "session.messages.list.failed".
  - Audit log entry written with event "session.messages.listed" on success.
  - Audit detail includes session_id and count on success.
  - OTel span "session.messages.list" is emitted on success.
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


def _get_messages(
    client: TestClient,
    session_id: str,
    *,
    cursor: str | None = None,
    limit: int | None = None,
    thread_id: str | None = None,
    role: str | None = None,
) -> tuple[dict[str, Any], Any]:
    params: dict[str, Any] = {}
    if cursor is not None:
        params["cursor"] = cursor
    if limit is not None:
        params["limit"] = limit
    if thread_id is not None:
        params["thread_id"] = thread_id
    if role is not None:
        params["role"] = role
    resp = client.get(f"/v1/sessions/{session_id}/messages", params=params)
    return resp.json() | {"_status": resp.status_code}, resp


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _seed_thread(
    storage_root: Path,
    session_id: str,
    thread_id: str,
    created_at: str,
) -> Path:
    thread_dir = storage_root / "threads" / session_id / thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)
    (thread_dir / "manifest.json").write_text(
        json.dumps({"id": thread_id, "session_id": session_id, "created_at": created_at})
    )
    return thread_dir


def _seed_message(
    thread_dir: Path,
    *,
    message_id: str,
    thread_id: str,
    session_id: str,
    role: str = "user",
    content: str = "[]",
    sequence: int = 0,
    created_at: str,
) -> None:
    messages_path = thread_dir / "messages.ndjson"
    record = {
        "id": message_id,
        "thread_id": thread_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "sequence": sequence,
        "created_at": created_at,
    }
    with messages_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


class TestListMessagesResponse:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_messages(client, session["session_id"])
        assert result["_status"] == 200

    def test_has_items_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_messages(client, session["session_id"])
        assert "items" in result

    def test_items_is_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_messages(client, session["session_id"])
        assert isinstance(result["items"], list)

    def test_has_next_cursor_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_messages(client, session["session_id"])
        assert "next_cursor" in result

    def test_has_limit_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_messages(client, session["session_id"])
        assert "limit" in result

    def test_empty_items_for_missing_session(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _get_messages(client, "sess_nonexistent")
        assert result["_status"] == 200
        assert result["items"] == []

    def test_empty_items_when_no_messages(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        _seed_thread(storage_root, session_id, "thread_empty", "2026-05-21T10:00:00+00:00")
        result, _ = _get_messages(client, session_id)
        assert result["items"] == []

    def test_message_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_a", "2026-05-21T10:00:00+00:00")
        _seed_message(
            thread_dir,
            message_id="msg_001",
            thread_id="thread_a",
            session_id=session_id,
            role="user",
            sequence=0,
            created_at="2026-05-21T10:00:00+00:00",
        )
        result, _ = _get_messages(client, session_id)
        assert result["items"][0]["id"] == "msg_001"

    def test_message_has_thread_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_a", "2026-05-21T10:00:00+00:00")
        _seed_message(
            thread_dir,
            message_id="msg_001",
            thread_id="thread_a",
            session_id=session_id,
            created_at="2026-05-21T10:00:00+00:00",
        )
        result, _ = _get_messages(client, session_id)
        assert result["items"][0]["thread_id"] == "thread_a"

    def test_message_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_a", "2026-05-21T10:00:00+00:00")
        _seed_message(
            thread_dir,
            message_id="msg_001",
            thread_id="thread_a",
            session_id=session_id,
            created_at="2026-05-21T10:00:00+00:00",
        )
        result, _ = _get_messages(client, session_id)
        assert result["items"][0]["session_id"] == session_id

    def test_message_has_role(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_a", "2026-05-21T10:00:00+00:00")
        _seed_message(
            thread_dir,
            message_id="msg_001",
            thread_id="thread_a",
            session_id=session_id,
            role="assistant",
            created_at="2026-05-21T10:00:00+00:00",
        )
        result, _ = _get_messages(client, session_id)
        assert result["items"][0]["role"] == "assistant"

    def test_message_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_a", "2026-05-21T10:00:00+00:00")
        _seed_message(
            thread_dir,
            message_id="msg_001",
            thread_id="thread_a",
            session_id=session_id,
            created_at="2026-05-21T10:00:00+00:00",
        )
        result, _ = _get_messages(client, session_id)
        assert result["items"][0]["created_at"] == "2026-05-21T10:00:00+00:00"

    def test_items_sorted_descending_by_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_a", "2026-05-21T10:00:00+00:00")
        _seed_message(
            thread_dir,
            message_id="msg_old",
            thread_id="thread_a",
            session_id=session_id,
            sequence=0,
            created_at="2026-01-01T00:00:00+00:00",
        )
        _seed_message(
            thread_dir,
            message_id="msg_new",
            thread_id="thread_a",
            session_id=session_id,
            sequence=1,
            created_at="2026-12-31T00:00:00+00:00",
        )
        result, _ = _get_messages(client, session_id)
        created_ats = [m["created_at"] for m in result["items"]]
        assert created_ats == sorted(created_ats, reverse=True)

    def test_messages_from_multiple_threads(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir_a = _seed_thread(storage_root, session_id, "thread_a", "2026-05-21T10:00:00+00:00")
        thread_dir_b = _seed_thread(storage_root, session_id, "thread_b", "2026-05-21T11:00:00+00:00")
        _seed_message(
            thread_dir_a,
            message_id="msg_a",
            thread_id="thread_a",
            session_id=session_id,
            created_at="2026-05-21T10:00:00+00:00",
        )
        _seed_message(
            thread_dir_b,
            message_id="msg_b",
            thread_id="thread_b",
            session_id=session_id,
            created_at="2026-05-21T11:00:00+00:00",
        )
        result, _ = _get_messages(client, session_id)
        ids = {m["id"] for m in result["items"]}
        assert ids == {"msg_a", "msg_b"}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestListMessagesFiltering:
    def test_filter_by_thread_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir_a = _seed_thread(storage_root, session_id, "thread_a", "2026-05-21T10:00:00+00:00")
        thread_dir_b = _seed_thread(storage_root, session_id, "thread_b", "2026-05-21T11:00:00+00:00")
        _seed_message(
            thread_dir_a,
            message_id="msg_a",
            thread_id="thread_a",
            session_id=session_id,
            created_at="2026-05-21T10:00:00+00:00",
        )
        _seed_message(
            thread_dir_b,
            message_id="msg_b",
            thread_id="thread_b",
            session_id=session_id,
            created_at="2026-05-21T11:00:00+00:00",
        )
        result, _ = _get_messages(client, session_id, thread_id="thread_a")
        assert all(m["thread_id"] == "thread_a" for m in result["items"])
        assert len(result["items"]) == 1
        assert result["items"][0]["id"] == "msg_a"

    def test_filter_by_role(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_a", "2026-05-21T10:00:00+00:00")
        _seed_message(
            thread_dir,
            message_id="msg_user",
            thread_id="thread_a",
            session_id=session_id,
            role="user",
            sequence=0,
            created_at="2026-05-21T10:00:00+00:00",
        )
        _seed_message(
            thread_dir,
            message_id="msg_assistant",
            thread_id="thread_a",
            session_id=session_id,
            role="assistant",
            sequence=1,
            created_at="2026-05-21T10:01:00+00:00",
        )
        result, _ = _get_messages(client, session_id, role="user")
        assert all(m["role"] == "user" for m in result["items"])
        assert len(result["items"]) == 1
        assert result["items"][0]["id"] == "msg_user"

    def test_filter_by_thread_id_and_role(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir_a = _seed_thread(storage_root, session_id, "thread_a", "2026-05-21T10:00:00+00:00")
        thread_dir_b = _seed_thread(storage_root, session_id, "thread_b", "2026-05-21T11:00:00+00:00")
        _seed_message(
            thread_dir_a,
            message_id="msg_a_user",
            thread_id="thread_a",
            session_id=session_id,
            role="user",
            created_at="2026-05-21T10:00:00+00:00",
        )
        _seed_message(
            thread_dir_a,
            message_id="msg_a_assistant",
            thread_id="thread_a",
            session_id=session_id,
            role="assistant",
            created_at="2026-05-21T10:01:00+00:00",
        )
        _seed_message(
            thread_dir_b,
            message_id="msg_b_user",
            thread_id="thread_b",
            session_id=session_id,
            role="user",
            created_at="2026-05-21T11:00:00+00:00",
        )
        result, _ = _get_messages(client, session_id, thread_id="thread_a", role="user")
        assert len(result["items"]) == 1
        assert result["items"][0]["id"] == "msg_a_user"

    def test_filter_by_nonexistent_thread_id_returns_empty(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_messages(client, session["session_id"], thread_id="thread_nonexistent")
        assert result["_status"] == 200
        assert result["items"] == []

    def test_filter_by_role_no_match_returns_empty(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_a", "2026-05-21T10:00:00+00:00")
        _seed_message(
            thread_dir,
            message_id="msg_user",
            thread_id="thread_a",
            session_id=session_id,
            role="user",
            created_at="2026-05-21T10:00:00+00:00",
        )
        result, _ = _get_messages(client, session_id, role="tool")
        assert result["items"] == []


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestListMessagesPagination:
    def _seed_messages(
        self, storage_root: Path, session_id: str, count: int
    ) -> None:
        thread_dir = _seed_thread(storage_root, session_id, "thread_pg", "2026-05-21T10:00:00+00:00")
        for i in range(count):
            _seed_message(
                thread_dir,
                message_id=f"msg_{i:03d}",
                thread_id="thread_pg",
                session_id=session_id,
                sequence=i,
                created_at=f"2026-05-{i + 1:02d}T00:00:00+00:00",
            )

    def test_limit_controls_page_size(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        self._seed_messages(storage_root, session_id, 5)
        result, _ = _get_messages(client, session_id, limit=2)
        assert len(result["items"]) == 2
        assert result["limit"] == 2

    def test_next_cursor_present_when_more_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        self._seed_messages(storage_root, session_id, 4)
        result, _ = _get_messages(client, session_id, limit=2)
        assert result["next_cursor"] is not None

    def test_link_header_when_more_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        self._seed_messages(storage_root, session_id, 4)
        _, resp = _get_messages(client, session_id, limit=2)
        assert "link" in {k.lower() for k in resp.headers}

    def test_next_cursor_null_when_all_fit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        self._seed_messages(storage_root, session_id, 3)
        result, _ = _get_messages(client, session_id, limit=50)
        assert result["next_cursor"] is None

    def test_cursor_advances_to_next_page(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        self._seed_messages(storage_root, session_id, 5)
        page1, _ = _get_messages(client, session_id, limit=2)
        cursor = page1["next_cursor"]
        assert cursor is not None
        page2, _ = _get_messages(client, session_id, cursor=cursor, limit=2)
        ids_page1 = {m["id"] for m in page1["items"]}
        ids_page2 = {m["id"] for m in page2["items"]}
        assert ids_page1.isdisjoint(ids_page2)

    def test_invalid_cursor_returns_400(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_messages(client, session["session_id"], cursor="not-valid-cursor!!!")
        assert result["_status"] == 400

    def test_invalid_cursor_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_messages(client, session["session_id"], cursor="not-valid-cursor!!!")
        assert result["error"]["code"] == "cursor_invalid"

    def test_invalid_cursor_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _get_messages(client, session["session_id"], cursor="not-valid-cursor!!!")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.messages.list.failed" for r in records)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestListMessagesAudit:
    def test_success_writes_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _get_messages(client, session["session_id"])
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.messages.listed" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _get_messages(client, session["session_id"])
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.messages.listed")
        assert record["level"] == "info"

    def test_success_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _get_messages(client, session["session_id"])
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.messages.listed")
        assert record["detail"]["session_id"] == session["session_id"]

    def test_success_audit_detail_has_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _get_messages(client, session["session_id"])
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.messages.listed")
        assert "count" in record["detail"]
        assert record["detail"]["count"] == 0

    def test_failure_writes_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_corrupt", "2026-05-21T10:00:00+00:00")
        (thread_dir / "messages.ndjson").write_text("not valid json {{{\n")
        _get_messages(client, session_id)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.messages.list.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_corrupt", "2026-05-21T10:00:00+00:00")
        (thread_dir / "messages.ndjson").write_text("not valid json {{{\n")
        _get_messages(client, session_id)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.messages.list.failed")
        assert record["level"] == "error"

    def test_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_corrupt", "2026-05-21T10:00:00+00:00")
        (thread_dir / "messages.ndjson").write_text("not valid json {{{\n")
        _get_messages(client, session_id)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.messages.list.failed")
        assert record["detail"]["session_id"] == session_id

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_corrupt", "2026-05-21T10:00:00+00:00")
        (thread_dir / "messages.ndjson").write_text("not valid json {{{\n")
        _get_messages(client, session_id)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.messages.list.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# Failure response
# ---------------------------------------------------------------------------


class TestListMessagesFailure:
    def test_failure_returns_500(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_corrupt", "2026-05-21T10:00:00+00:00")
        (thread_dir / "messages.ndjson").write_text("not valid json {{{\n")
        result, _ = _get_messages(client, session_id)
        assert result["_status"] == 500

    def test_failure_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_corrupt", "2026-05-21T10:00:00+00:00")
        (thread_dir / "messages.ndjson").write_text("not valid json {{{\n")
        result, _ = _get_messages(client, session_id)
        assert result["error"]["code"] == "session_messages_list_failed"

    def test_failure_error_message_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_corrupt", "2026-05-21T10:00:00+00:00")
        (thread_dir / "messages.ndjson").write_text("not valid json {{{\n")
        result, _ = _get_messages(client, session_id)
        assert "message" in result["error"]
        assert len(result["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel instrumentation
# ---------------------------------------------------------------------------


class TestListMessagesOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _otel_exporter.clear()
        _get_messages(client, session["session_id"])
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.messages.list" in span_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _otel_exporter.clear()
        _get_messages(client, session["session_id"])
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.messages.list")
        assert span is not None
        assert span.attributes["session.id"] == session["session_id"]

    def test_span_has_invocation_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _otel_exporter.clear()
        _get_messages(client, session["session_id"])
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.messages.list")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        thread_dir = _seed_thread(storage_root, session_id, "thread_corrupt", "2026-05-21T10:00:00+00:00")
        (thread_dir / "messages.ndjson").write_text("not valid json {{{\n")
        _otel_exporter.clear()
        _get_messages(client, session_id)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.messages.list")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR
