"""
System integration test: GET /v1/sessions lists Sessions with cursor pagination.

Tests cover:
  - GET /v1/sessions returns 200 with items, next_cursor, limit fields.
  - items is a list.
  - Each session item has session_id, id, agent_id, created_at fields.
  - items are sorted by created_at descending (most recent first).
  - Returns empty items list when no sessions exist.
  - filter by phase matches sessions with matching phase field.
  - filter by agent_id matches sessions with matching agent_id.
  - filter by user_profile_id matches sessions with matching user_profile_id.
  - filter by channel_id matches sessions with matching channel_id.
  - filter by parent_session_id matches sessions with matching parent_session_id.
  - created_after excludes sessions at or before the boundary.
  - created_before excludes sessions at or after the boundary.
  - cursor pagination: limit query param controls page size.
  - next_cursor is present in response body and Link header is set when more pages exist.
  - next_cursor is null when all items fit on one page.
  - cursor param advances to the next page.
  - Invalid cursor returns 400 with code "cursor_invalid".
  - Invalid cursor writes audit log entry with event "session.list.failed".
  - On generic failure, returns 500 with code "session_list_failed".
  - On failure, error message is surfaced in response body.
  - On failure, audit log entry is written with event "session.list.failed".
  - Audit log entry written with event "session.listed" on success.
  - Audit detail includes count on success.
  - OTel span "session.list" is emitted on success.
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


def _post_session(client: TestClient, *, agent_id: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if agent_id is not None:
        body["agent_id"] = agent_id
    resp = client.post("/v1/sessions", json=body)
    return resp.json() | {"_status": resp.status_code}


def _get_sessions(
    client: TestClient,
    *,
    cursor: str | None = None,
    limit: int | None = None,
    phase: str | None = None,
    agent_id: str | None = None,
    user_profile_id: str | None = None,
    channel_id: str | None = None,
    parent_session_id: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
) -> tuple[dict[str, Any], Any]:
    params: dict[str, Any] = {}
    if cursor is not None:
        params["cursor"] = cursor
    if limit is not None:
        params["limit"] = limit
    if phase is not None:
        params["phase"] = phase
    if agent_id is not None:
        params["agent_id"] = agent_id
    if user_profile_id is not None:
        params["user_profile_id"] = user_profile_id
    if channel_id is not None:
        params["channel_id"] = channel_id
    if parent_session_id is not None:
        params["parent_session_id"] = parent_session_id
    if created_after is not None:
        params["created_after"] = created_after
    if created_before is not None:
        params["created_before"] = created_before
    resp = client.get("/v1/sessions", params=params)
    return resp.json() | {"_status": resp.status_code}, resp


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _seed_session(
    storage_root: Path,
    session_id: str,
    created_at: str,
    *,
    phase: str | None = None,
    agent_id: str | None = None,
    user_profile_id: str | None = None,
    channel_id: str | None = None,
    parent_session_id: str | None = None,
) -> None:
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "session_id": session_id,
        "id": session_id,
        "created_at": created_at,
    }
    if phase is not None:
        record["phase"] = phase
    if agent_id is not None:
        record["agent_id"] = agent_id
    if user_profile_id is not None:
        record["user_profile_id"] = user_profile_id
    if channel_id is not None:
        record["channel_id"] = channel_id
    if parent_session_id is not None:
        record["parent_session_id"] = parent_session_id
    (session_dir / "manifest.json").write_text(json.dumps(record))


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


class TestListSessionsResponse:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _get_sessions(client)
        assert result["_status"] == 200

    def test_has_items_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _get_sessions(client)
        assert "items" in result

    def test_items_is_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _get_sessions(client)
        assert isinstance(result["items"], list)

    def test_has_next_cursor_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _get_sessions(client)
        assert "next_cursor" in result

    def test_has_limit_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _get_sessions(client)
        assert "limit" in result

    def test_empty_items_when_no_sessions(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _get_sessions(client)
        assert result["items"] == []

    def test_created_session_appears_in_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session = _post_session(client)
        result, _ = _get_sessions(client)
        ids = [s["session_id"] for s in result["items"]]
        assert session["session_id"] in ids

    def test_session_item_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _post_session(client)
        result, _ = _get_sessions(client)
        item = result["items"][0]
        assert "session_id" in item

    def test_session_item_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _post_session(client)
        result, _ = _get_sessions(client)
        item = result["items"][0]
        assert "id" in item

    def test_session_item_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _post_session(client)
        result, _ = _get_sessions(client)
        item = result["items"][0]
        assert "created_at" in item
        assert len(item["created_at"]) > 0

    def test_session_item_has_agent_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _post_session(client)
        result, _ = _get_sessions(client)
        item = result["items"][0]
        assert "agent_id" in item

    def test_items_sorted_descending_by_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _seed_session(storage_root, "sess_old", "2026-01-01T00:00:00+00:00")
        _seed_session(storage_root, "sess_new", "2026-12-31T00:00:00+00:00")
        result, _ = _get_sessions(client)
        created_ats = [s["created_at"] for s in result["items"]]
        assert created_ats == sorted(created_ats, reverse=True)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestListSessionsFilters:
    def test_filter_by_phase_matches(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _seed_session(storage_root, "sess_running", "2026-05-01T00:00:00+00:00", phase="running")
        _seed_session(storage_root, "sess_idle", "2026-05-02T00:00:00+00:00", phase="idle")
        result, _ = _get_sessions(client, phase="running")
        ids = [s["session_id"] for s in result["items"]]
        assert "sess_running" in ids
        assert "sess_idle" not in ids

    def test_filter_by_phase_excludes_non_matching(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _seed_session(storage_root, "sess_a", "2026-05-01T00:00:00+00:00", phase="terminated")
        result, _ = _get_sessions(client, phase="running")
        assert result["items"] == []

    def test_filter_by_agent_id_matches(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _seed_session(storage_root, "sess_x", "2026-05-01T00:00:00+00:00", agent_id="agent_abc")
        _seed_session(storage_root, "sess_y", "2026-05-02T00:00:00+00:00", agent_id="agent_xyz")
        result, _ = _get_sessions(client, agent_id="agent_abc")
        ids = [s["session_id"] for s in result["items"]]
        assert "sess_x" in ids
        assert "sess_y" not in ids

    def test_filter_by_agent_id_excludes_non_matching(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _seed_session(storage_root, "sess_z", "2026-05-01T00:00:00+00:00", agent_id="agent_other")
        result, _ = _get_sessions(client, agent_id="agent_abc")
        assert result["items"] == []

    def test_filter_by_user_profile_id_matches(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _seed_session(
            storage_root, "sess_u1", "2026-05-01T00:00:00+00:00", user_profile_id="user_111"
        )
        _seed_session(
            storage_root, "sess_u2", "2026-05-02T00:00:00+00:00", user_profile_id="user_222"
        )
        result, _ = _get_sessions(client, user_profile_id="user_111")
        ids = [s["session_id"] for s in result["items"]]
        assert "sess_u1" in ids
        assert "sess_u2" not in ids

    def test_filter_by_channel_id_matches(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _seed_session(storage_root, "sess_c1", "2026-05-01T00:00:00+00:00", channel_id="chan_aaa")
        _seed_session(storage_root, "sess_c2", "2026-05-02T00:00:00+00:00", channel_id="chan_bbb")
        result, _ = _get_sessions(client, channel_id="chan_aaa")
        ids = [s["session_id"] for s in result["items"]]
        assert "sess_c1" in ids
        assert "sess_c2" not in ids

    def test_filter_by_parent_session_id_matches(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _seed_session(
            storage_root,
            "sess_child",
            "2026-05-01T00:00:00+00:00",
            parent_session_id="sess_parent",
        )
        _seed_session(storage_root, "sess_orphan", "2026-05-02T00:00:00+00:00")
        result, _ = _get_sessions(client, parent_session_id="sess_parent")
        ids = [s["session_id"] for s in result["items"]]
        assert "sess_child" in ids
        assert "sess_orphan" not in ids

    def test_created_after_excludes_sessions_at_boundary(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _seed_session(storage_root, "sess_before", "2026-01-01T00:00:00+00:00")
        _seed_session(storage_root, "sess_at", "2026-06-01T00:00:00+00:00")
        _seed_session(storage_root, "sess_after", "2026-12-01T00:00:00+00:00")
        result, _ = _get_sessions(client, created_after="2026-06-01T00:00:00+00:00")
        ids = [s["session_id"] for s in result["items"]]
        assert "sess_after" in ids
        assert "sess_at" not in ids
        assert "sess_before" not in ids

    def test_created_before_excludes_sessions_at_boundary(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _seed_session(storage_root, "sess_before", "2026-01-01T00:00:00+00:00")
        _seed_session(storage_root, "sess_at", "2026-06-01T00:00:00+00:00")
        _seed_session(storage_root, "sess_after", "2026-12-01T00:00:00+00:00")
        result, _ = _get_sessions(client, created_before="2026-06-01T00:00:00+00:00")
        ids = [s["session_id"] for s in result["items"]]
        assert "sess_before" in ids
        assert "sess_at" not in ids
        assert "sess_after" not in ids


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestListSessionsPagination:
    def test_limit_controls_page_size(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        for i in range(5):
            _seed_session(storage_root, f"sess_pg_{i}", f"2026-05-{i + 1:02d}T00:00:00+00:00")
        result, _ = _get_sessions(client, limit=2)
        assert len(result["items"]) == 2
        assert result["limit"] == 2

    def test_next_cursor_present_when_more_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        for i in range(4):
            _seed_session(storage_root, f"sess_cur_{i}", f"2026-06-{i + 1:02d}T00:00:00+00:00")
        result, _ = _get_sessions(client, limit=2)
        assert result["next_cursor"] is not None

    def test_link_header_when_more_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        for i in range(4):
            _seed_session(storage_root, f"sess_hdr_{i}", f"2026-07-{i + 1:02d}T00:00:00+00:00")
        _, resp = _get_sessions(client, limit=2)
        assert "link" in {k.lower() for k in resp.headers}

    def test_next_cursor_null_when_all_fit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _seed_session(storage_root, "sess_only", "2026-05-01T00:00:00+00:00")
        result, _ = _get_sessions(client, limit=50)
        assert result["next_cursor"] is None

    def test_cursor_advances_to_next_page(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        for i in range(5):
            _seed_session(storage_root, f"sess_adv_{i}", f"2026-08-{i + 1:02d}T00:00:00+00:00")
        page1, _ = _get_sessions(client, limit=2)
        cursor = page1["next_cursor"]
        assert cursor is not None
        page2, _ = _get_sessions(client, cursor=cursor, limit=2)
        ids_page1 = {s["id"] for s in page1["items"]}
        ids_page2 = {s["id"] for s in page2["items"]}
        assert ids_page1.isdisjoint(ids_page2)

    def test_invalid_cursor_returns_400(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _get_sessions(client, cursor="not-valid-cursor!!!")
        assert result["_status"] == 400

    def test_invalid_cursor_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result, _ = _get_sessions(client, cursor="not-valid-cursor!!!")
        assert result["error"]["code"] == "cursor_invalid"

    def test_invalid_cursor_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _get_sessions(client, cursor="not-valid-cursor!!!")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.list.failed" for r in records)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestListSessionsAudit:
    def test_success_writes_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _get_sessions(client)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.listed" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _get_sessions(client)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.listed")
        assert record["level"] == "info"

    def test_success_audit_detail_has_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _seed_session(storage_root, "sess_count", "2026-05-01T00:00:00+00:00")
        _get_sessions(client)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.listed")
        assert "count" in record["detail"]
        assert record["detail"]["count"] == 1

    def test_failure_writes_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session_dir = storage_root / "sessions" / "sess_corrupt"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text("not valid json {{{")
        _get_sessions(client)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.list.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session_dir = storage_root / "sessions" / "sess_corrupt"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text("not valid json {{{")
        _get_sessions(client)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.list.failed")
        assert record["level"] == "error"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session_dir = storage_root / "sessions" / "sess_corrupt"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text("not valid json {{{")
        _get_sessions(client)
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.list.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# Failure response
# ---------------------------------------------------------------------------


class TestListSessionsFailure:
    def test_failure_returns_500(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session_dir = storage_root / "sessions" / "sess_corrupt"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text("not valid json {{{")
        result, _ = _get_sessions(client)
        assert result["_status"] == 500

    def test_failure_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session_dir = storage_root / "sessions" / "sess_corrupt"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text("not valid json {{{")
        result, _ = _get_sessions(client)
        assert result["error"]["code"] == "session_list_failed"

    def test_failure_error_message_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        session_dir = storage_root / "sessions" / "sess_corrupt"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text("not valid json {{{")
        result, _ = _get_sessions(client)
        assert "message" in result["error"]
        assert len(result["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel instrumentation
# ---------------------------------------------------------------------------


class TestListSessionsOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _otel_exporter.clear()
        _get_sessions(client)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.list" in span_names

    def test_span_has_invocation_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _otel_exporter.clear()
        _get_sessions(client)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.list")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = _make_client(storage_root)
        session_dir = storage_root / "sessions" / "sess_corrupt"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text("not valid json {{{")
        _otel_exporter.clear()
        _get_sessions(client)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.list")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR
