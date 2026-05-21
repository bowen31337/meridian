"""
User can continue conformance suite.

Tests cover:
  - POST /v1/x/sessions/{id}/user-can-continue returns 200 with session_id, before, after, reason.
  - after is always "running".
  - reason is always "user_approved".
  - before reflects the session's phase prior to the resume (defaults to "created").
  - before reflects waiting_for_user when that was the prior phase.
  - session.phase_change event written with after="running" and reason="user_approved".
  - session.phase_change event before field matches the pre-resume phase.
  - On write failure, returns 422 with code "user_can_continue_failed".
  - On failure, audit log entry written with event "session.user_can_continue.failed".
  - On failure, audit detail includes session_id and message.
  - OTel span "session.user_can_continue" emitted on success.
  - OTel span has session.id attribute.
  - OTel span is set to ERROR status on failure.
  - create_app wires the route when storage_root and event_log are supplied.
  - create_app omits the route when storage_root is None.
  - create_app omits the route when event_log is None.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from storage_event_log import LocalEventLogWriter
from storage_reposit import LocalEventLogReader

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_writer(storage_root: Path) -> LocalEventLogWriter:
    return LocalEventLogWriter(storage_root)


def _make_client(
    storage_root: Path,
    writer: LocalEventLogWriter | None = None,
) -> TestClient:
    audit = FileAuditLog(storage_root)
    w = writer or _make_writer(storage_root)
    app = create_app(audit, storage_root=storage_root, event_log=w)
    return TestClient(app, raise_server_exceptions=False)


def _seed_phase(storage_root: Path, session_id: str, phase: str) -> None:
    async def _write() -> None:
        writer = LocalEventLogWriter(storage_root)
        await writer.append(
            session_id,
            "session.phase_change",
            {"before": "created", "after": phase, "reason": "seed", "timestamp": "t0"},
        )

    asyncio.run(_write())


def _read_events(storage_root: Path, session_id: str) -> list[Any]:
    reader = LocalEventLogReader(storage_root)
    return list(reader.read_after(session_id, -1))


def _read_audit(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


class TestUserCanContinueResponse:
    def test_returns_200_on_success(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post(
            "/v1/x/sessions/sess1/user-can-continue"
        )
        assert resp.status_code == 200

    def test_response_has_session_id(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess2/user-can-continue"
        ).json()
        assert body["session_id"] == "sess2"

    def test_response_after_is_running(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess3/user-can-continue"
        ).json()
        assert body["after"] == "running"

    def test_response_reason_is_user_approved(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess4/user-can-continue"
        ).json()
        assert body["reason"] == "user_approved"

    def test_response_before_defaults_to_created(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess5/user-can-continue"
        ).json()
        assert body["before"] == "created"

    def test_response_before_reflects_waiting_for_user(self, storage_root: Path) -> None:
        _seed_phase(storage_root, "sess6", "waiting_for_user")
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess6/user-can-continue"
        ).json()
        assert body["before"] == "waiting_for_user"

    def test_response_before_reflects_prior_phase(self, storage_root: Path) -> None:
        _seed_phase(storage_root, "sess7", "paused")
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess7/user-can-continue"
        ).json()
        assert body["before"] == "paused"


# ---------------------------------------------------------------------------
# Event log writes
# ---------------------------------------------------------------------------


class TestUserCanContinueEventLog:
    def test_phase_change_event_written(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/x/sessions/ev-sess1/user-can-continue")
        events = _read_events(storage_root, "ev-sess1")
        assert any(e.type == "session.phase_change" for e in events)

    def test_phase_change_after_is_running(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/x/sessions/ev-sess2/user-can-continue")
        events = _read_events(storage_root, "ev-sess2")
        change = next(e for e in events if e.type == "session.phase_change")
        assert change.data["after"] == "running"

    def test_phase_change_reason_is_user_approved(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/x/sessions/ev-sess3/user-can-continue")
        events = _read_events(storage_root, "ev-sess3")
        change = next(e for e in events if e.type == "session.phase_change")
        assert change.data["reason"] == "user_approved"

    def test_phase_change_before_matches_prior_phase(self, storage_root: Path) -> None:
        _seed_phase(storage_root, "ev-sess4", "waiting_for_user")
        _make_client(storage_root).post("/v1/x/sessions/ev-sess4/user-can-continue")
        events = _read_events(storage_root, "ev-sess4")
        changes = [e for e in events if e.type == "session.phase_change"]
        resume_change = changes[-1]
        assert resume_change.data["before"] == "waiting_for_user"


# ---------------------------------------------------------------------------
# Failure surfacing
# ---------------------------------------------------------------------------


class TestUserCanContinueFailure:
    def test_write_failure_returns_422(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/fail-sess/user-can-continue")
        assert resp.status_code == 422

    def test_write_failure_error_code(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        body = client.post("/v1/x/sessions/fail-sess2/user-can-continue").json()
        assert body["error"]["code"] == "user_can_continue_failed"

    def test_write_failure_error_message_non_empty(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        body = client.post("/v1/x/sessions/fail-sess3/user-can-continue").json()
        assert len(body["error"]["message"]) > 0

    def test_write_failure_writes_audit_entry(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/v1/x/sessions/fail-sess4/user-can-continue")
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.user_can_continue.failed" for r in records)

    def test_write_failure_audit_level_is_error(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/v1/x/sessions/fail-sess5/user-can-continue")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.user_can_continue.failed")
        assert rec["level"] == "error"

    def test_write_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/v1/x/sessions/fail-sess6/user-can-continue")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.user_can_continue.failed")
        assert rec["detail"]["session_id"] == "fail-sess6"

    def test_write_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/v1/x/sessions/fail-sess7/user-can-continue")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.user_can_continue.failed")
        assert len(rec["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestUserCanContinueOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_user_can_continue_span(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/x/sessions/otel-sess1/user-can-continue")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.user_can_continue" in span_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/x/sessions/otel-sess2/user-can-continue")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.user_can_continue")
        assert span is not None
        assert span.attributes["session.id"] == "otel-sess2"

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/v1/x/sessions/otel-fail/user-can-continue")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        ucc_span = spans.get("session.user_can_continue")
        assert ucc_span is not None
        assert ucc_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Router wiring via create_app
# ---------------------------------------------------------------------------


class TestUserCanContinueRouterWiring:
    def test_route_exists_with_storage_root_and_event_log(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post(
            "/v1/x/sessions/wire-sess1/user-can-continue"
        )
        assert resp.status_code != 404

    def test_no_storage_root_returns_404(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/any/user-can-continue")
        assert resp.status_code == 404

    def test_no_event_log_returns_404(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/any/user-can-continue")
        assert resp.status_code == 404
