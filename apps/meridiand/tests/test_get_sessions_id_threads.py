"""
System integration test: GET /v1/sessions/{id}/threads lists Threads for a Session.

Tests cover:
  - GET /v1/sessions/{id}/threads returns 200 with items, next_cursor, limit fields.
  - items is a list.
  - Each thread item has thread_id, session_id, created_at, id, title, branch_of_event_seq.
  - title defaults to null for threads without an explicit title.
  - branch_of_event_seq defaults to null for the initial thread.
  - items are sorted by created_at descending (most recent first).
  - Returns empty items list when the session threads directory does not exist.
  - cursor pagination: limit query param controls page size.
  - next_cursor is present in response body and Link header is set when more pages
    exist (middleware converts X-Next-Cursor).
  - next_cursor is null when all items fit on one page.
  - cursor param advances to the next page.
  - Invalid cursor returns 400 with code "cursor_invalid".
  - Invalid cursor writes audit log entry with event "session.threads.list.failed".
  - On generic failure, returns 500 with code "session_threads_list_failed".
  - On failure, error message is surfaced in response body.
  - On failure, audit log entry is written with event "session.threads.list.failed".
  - Audit log entry written with event "session.threads.listed" on success.
  - Audit detail includes session_id and count on success.
  - OTel span "session.threads.list" is emitted on success.
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


def _get_threads(
    client: TestClient,
    session_id: str,
    *,
    cursor: str | None = None,
    limit: int | None = None,
) -> tuple[dict[str, Any], Any]:
    params: dict[str, Any] = {}
    if cursor is not None:
        params["cursor"] = cursor
    if limit is not None:
        params["limit"] = limit
    resp = client.get(f"/v1/sessions/{session_id}/threads", params=params)
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
    *,
    title: str | None = None,
    branch_of_event_seq: int | None = None,
) -> None:
    threads_dir = storage_root / "sessions" / session_id / "threads"
    threads_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "thread_id": thread_id,
        "session_id": session_id,
        "created_at": created_at,
    }
    if title is not None:
        record["title"] = title
    if branch_of_event_seq is not None:
        record["branch_of_event_seq"] = branch_of_event_seq
    (threads_dir / f"{thread_id}.json").write_text(json.dumps(record))


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


class TestListThreadsResponse:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"])
        assert result["_status"] == 200

    def test_has_items_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"])
        assert "items" in result

    def test_items_is_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"])
        assert isinstance(result["items"], list)

    def test_has_next_cursor_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"])
        assert "next_cursor" in result

    def test_has_limit_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"])
        assert "limit" in result

    def test_initial_thread_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"])
        assert len(result["items"]) == 1

    def test_thread_has_thread_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"])
        thread = result["items"][0]
        assert thread["thread_id"] == session["thread_id"]

    def test_thread_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"])
        thread = result["items"][0]
        assert thread["session_id"] == session["session_id"]

    def test_thread_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"])
        thread = result["items"][0]
        assert "created_at" in thread
        assert len(thread["created_at"]) > 0

    def test_thread_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"])
        thread = result["items"][0]
        assert "id" in thread
        assert thread["id"] == session["thread_id"]

    def test_thread_title_is_null_by_default(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"])
        thread = result["items"][0]
        assert "title" in thread
        assert thread["title"] is None

    def test_thread_branch_of_event_seq_is_null_by_default(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"])
        thread = result["items"][0]
        assert "branch_of_event_seq" in thread
        assert thread["branch_of_event_seq"] is None

    def test_thread_title_present_when_set(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        _seed_thread(
            storage_root,
            session_id,
            "thread_titled",
            "2026-05-21T10:00:00+00:00",
            title="My Thread",
        )
        result, _ = _get_threads(client, session_id)
        titled = next(t for t in result["items"] if t["thread_id"] == "thread_titled")
        assert titled["title"] == "My Thread"

    def test_thread_branch_of_event_seq_present_when_set(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        _seed_thread(
            storage_root,
            session_id,
            "thread_branched",
            "2026-05-21T11:00:00+00:00",
            branch_of_event_seq=42,
        )
        result, _ = _get_threads(client, session_id)
        branched = next(t for t in result["items"] if t["thread_id"] == "thread_branched")
        assert branched["branch_of_event_seq"] == 42

    def test_empty_items_for_missing_session(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _get_threads(client, "sess_nonexistent")
        assert result["_status"] == 200
        assert result["items"] == []

    def test_items_sorted_descending_by_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        _seed_thread(storage_root, session_id, "thread_old", "2026-01-01T00:00:00+00:00")
        _seed_thread(storage_root, session_id, "thread_new", "2026-12-31T00:00:00+00:00")
        result, _ = _get_threads(client, session_id)
        created_ats = [t["created_at"] for t in result["items"]]
        assert created_ats == sorted(created_ats, reverse=True)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestListThreadsPagination:
    def test_limit_controls_page_size(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        for i in range(4):
            _seed_thread(
                storage_root,
                session_id,
                f"thread_extra_{i}",
                f"2026-05-{i + 1:02d}T00:00:00+00:00",
            )
        result, _ = _get_threads(client, session_id, limit=2)
        assert len(result["items"]) == 2
        assert result["limit"] == 2

    def test_next_cursor_present_when_more_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        for i in range(3):
            _seed_thread(
                storage_root,
                session_id,
                f"thread_pg_{i}",
                f"2026-06-{i + 1:02d}T00:00:00+00:00",
            )
        result, _ = _get_threads(client, session_id, limit=2)
        assert result["next_cursor"] is not None

    def test_link_header_when_more_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        for i in range(3):
            _seed_thread(
                storage_root,
                session_id,
                f"thread_hdr_{i}",
                f"2026-07-{i + 1:02d}T00:00:00+00:00",
            )
        _, resp = _get_threads(client, session_id, limit=2)
        assert "link" in {k.lower() for k in resp.headers}

    def test_next_cursor_null_when_all_fit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"], limit=50)
        assert result["next_cursor"] is None

    def test_cursor_advances_to_next_page(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        for i in range(4):
            _seed_thread(
                storage_root,
                session_id,
                f"thread_page_{i}",
                f"2026-08-{i + 1:02d}T00:00:00+00:00",
            )
        page1, _ = _get_threads(client, session_id, limit=2)
        cursor = page1["next_cursor"]
        assert cursor is not None
        page2, _ = _get_threads(client, session_id, cursor=cursor, limit=2)
        ids_page1 = {t["id"] for t in page1["items"]}
        ids_page2 = {t["id"] for t in page2["items"]}
        assert ids_page1.isdisjoint(ids_page2)

    def test_invalid_cursor_returns_400(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"], cursor="not-valid-cursor!!!")
        assert result["_status"] == 400

    def test_invalid_cursor_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_threads(client, session["session_id"], cursor="not-valid-cursor!!!")
        assert result["error"]["code"] == "cursor_invalid"

    def test_invalid_cursor_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _get_threads(client, session["session_id"], cursor="not-valid-cursor!!!")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.threads.list.failed" for r in records)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestListThreadsAudit:
    def test_success_writes_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _get_threads(client, session["session_id"])
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.threads.listed" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _get_threads(client, session["session_id"])
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.threads.listed")
        assert record["level"] == "info"

    def test_success_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _get_threads(client, session["session_id"])
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.threads.listed")
        assert record["detail"]["session_id"] == session["session_id"]

    def test_success_audit_detail_has_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _get_threads(client, session["session_id"])
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.threads.listed")
        assert "count" in record["detail"]
        assert record["detail"]["count"] == 1

    def test_failure_writes_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        threads_dir = storage_root / "sessions" / session_id / "threads"
        (threads_dir / "corrupt.json").write_text("not valid json {{{")
        _get_threads(client, session_id)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.threads.list.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        threads_dir = storage_root / "sessions" / session_id / "threads"
        (threads_dir / "corrupt.json").write_text("not valid json {{{")
        _get_threads(client, session_id)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.threads.list.failed")
        assert record["level"] == "error"

    def test_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        threads_dir = storage_root / "sessions" / session_id / "threads"
        (threads_dir / "corrupt.json").write_text("not valid json {{{")
        _get_threads(client, session_id)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.threads.list.failed")
        assert record["detail"]["session_id"] == session_id

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        threads_dir = storage_root / "sessions" / session_id / "threads"
        (threads_dir / "corrupt.json").write_text("not valid json {{{")
        _get_threads(client, session_id)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.threads.list.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# Failure response
# ---------------------------------------------------------------------------


class TestListThreadsFailure:
    def test_failure_returns_500(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        threads_dir = storage_root / "sessions" / session_id / "threads"
        (threads_dir / "corrupt.json").write_text("not valid json {{{")
        result, _ = _get_threads(client, session_id)
        assert result["_status"] == 500

    def test_failure_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        threads_dir = storage_root / "sessions" / session_id / "threads"
        (threads_dir / "corrupt.json").write_text("not valid json {{{")
        result, _ = _get_threads(client, session_id)
        assert result["error"]["code"] == "session_threads_list_failed"

    def test_failure_error_message_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        threads_dir = storage_root / "sessions" / session_id / "threads"
        (threads_dir / "corrupt.json").write_text("not valid json {{{")
        result, _ = _get_threads(client, session_id)
        assert "message" in result["error"]
        assert len(result["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel instrumentation
# ---------------------------------------------------------------------------


class TestListThreadsOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _otel_exporter.clear()
        _get_threads(client, session["session_id"])
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.threads.list" in span_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _otel_exporter.clear()
        _get_threads(client, session["session_id"])
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.threads.list")
        assert span is not None
        assert span.attributes["session.id"] == session["session_id"]

    def test_span_has_invocation_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        _otel_exporter.clear()
        _get_threads(client, session["session_id"])
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.threads.list")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = _make_client(storage_root)
        session = _post_session(client)
        session_id = session["session_id"]
        threads_dir = storage_root / "sessions" / session_id / "threads"
        (threads_dir / "corrupt.json").write_text("not valid json {{{")
        _otel_exporter.clear()
        _get_threads(client, session_id)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.threads.list")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR
