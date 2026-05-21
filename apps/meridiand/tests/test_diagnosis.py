"""
GET /v1/sessions/{id}/diagnosis endpoint conformance suite.

Tests cover:
  - GET /v1/sessions/{id}/diagnosis returns 200 on success.
  - Response body has session_id field matching the path parameter.
  - Response body has diagnosed_at field.
  - Response body has terminal_phase field.
  - Response body has stop_reason field.
  - Response body has failure_events field (list).
  - Response body has audit_entries field (list).
  - Response body has replay_fixture_available field (bool).
  - Response body has event_count field (int).
  - Unknown session returns terminal_phase "unknown" and empty failure_events.
  - event_count is 0 for a session with no events.
  - error events are included in failure_events.
  - session.phase_change events are included in failure_events.
  - tool_call.vetoed events are included in failure_events.
  - budget.warning events are included in failure_events.
  - message.truncated events are included in failure_events.
  - session.created events are NOT included in failure_events.
  - message.added events are NOT included in failure_events.
  - terminal_phase reflects the after field of the last session.phase_change event.
  - stop_reason reflects the reason field of the last session.phase_change event.
  - terminal_phase "unknown" when no session.phase_change events exist.
  - stop_reason "" when no session.phase_change events exist.
  - audit_entries contains entries with detail.session_id matching the session.
  - audit_entries does not contain entries for a different session.
  - audit_entries is empty when no matching audit entries exist.
  - replay_fixture_available=True when model_responses.ndjson fixture exists.
  - replay_fixture_available=False when no fixture exists.
  - event_count reflects total number of events for the session.
  - failure_events include thread_id when present in the original event.
  - failure_events omit thread_id when None in the original event.
  - On read failure, returns 500 with code "session_diagnosis_failed".
  - On read failure, audit log entry written with event "session.diagnosis.failed".
  - Audit log detail includes session_id on read failure.
  - Audit log detail includes message on read failure.
  - OTel span "session.diagnosis" emitted on success.
  - OTel span has session.id attribute.
  - OTel span carries structured invocation event.
  - OTel span set to ERROR status on failure.
  - create_app wires the diagnosis router when storage_root is supplied.
  - create_app omits the diagnosis route when storage_root is None.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from storage_event_log import LocalEventLogWriter

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path, audit_log: FileAuditLog | None = None) -> TestClient:
    audit = audit_log or FileAuditLog(storage_root)
    app = create_app(audit, storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _seed_many(
    storage_root: Path,
    session_id: str,
    events: list[tuple[str, dict[str, Any]]],
    *,
    thread_ids: list[str | None] | None = None,
) -> list[int]:
    async def _go() -> list[int]:
        writer = LocalEventLogWriter(storage_root)
        seqs = []
        for i, (event_type, data) in enumerate(events):
            tid = thread_ids[i] if thread_ids else None
            seqs.append(await writer.append(session_id, event_type, data, thread_id=tid))  # type: ignore[arg-type]
        return seqs

    return asyncio.run(_go())


def _seed(
    storage_root: Path,
    session_id: str,
    event_type: str,
    data: dict[str, Any],
    *,
    thread_id: str | None = None,
) -> int:
    return _seed_many(storage_root, session_id, [(event_type, data)], thread_ids=[thread_id])[0]


def _write_audit_entry(
    storage_root: Path,
    session_id: str,
    event: str = "test.event",
    level: str = "error",
    code: str = "test_error",
) -> None:
    audit = FileAuditLog(storage_root)
    from core_errors import AuditLogEntry

    audit.write(
        AuditLogEntry(
            level=level,  # type: ignore[arg-type]
            event=event,
            code=code,
            timestamp="2024-01-01T00:00:00+00:00",
            detail={"session_id": session_id, "message": "test"},
        )
    )


def _read_audit(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Basic response shape
# ---------------------------------------------------------------------------


class TestBasicResponse:
    def test_returns_200_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/sessions/sess1/diagnosis")
        assert resp.status_code == 200

    def test_response_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/sess-shape/diagnosis").json()
        assert body["session_id"] == "sess-shape"

    def test_response_has_diagnosed_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/sess-at/diagnosis").json()
        assert "diagnosed_at" in body
        assert isinstance(body["diagnosed_at"], str)

    def test_response_has_terminal_phase(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/sess-phase/diagnosis").json()
        assert "terminal_phase" in body

    def test_response_has_stop_reason(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/sess-reason/diagnosis").json()
        assert "stop_reason" in body

    def test_response_has_failure_events(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/sess-fevt/diagnosis").json()
        assert "failure_events" in body
        assert isinstance(body["failure_events"], list)

    def test_response_has_audit_entries(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/sess-audit/diagnosis").json()
        assert "audit_entries" in body
        assert isinstance(body["audit_entries"], list)

    def test_response_has_replay_fixture_available(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/sess-replay/diagnosis").json()
        assert "replay_fixture_available" in body
        assert isinstance(body["replay_fixture_available"], bool)

    def test_response_has_event_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/sess-cnt/diagnosis").json()
        assert "event_count" in body
        assert isinstance(body["event_count"], int)


# ---------------------------------------------------------------------------
# Unknown / empty session
# ---------------------------------------------------------------------------


class TestEmptySession:
    def test_unknown_session_terminal_phase_unknown(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/no-such-session/diagnosis").json()
        assert body["terminal_phase"] == "unknown"

    def test_unknown_session_failure_events_empty(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/no-such-session/diagnosis").json()
        assert body["failure_events"] == []

    def test_unknown_session_event_count_zero(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/no-such-session/diagnosis").json()
        assert body["event_count"] == 0

    def test_unknown_session_stop_reason_empty(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/no-such-session/diagnosis").json()
        assert body["stop_reason"] == ""


# ---------------------------------------------------------------------------
# failure_events filtering
# ---------------------------------------------------------------------------


class TestFailureEventsFiltering:
    def test_error_events_included(self, storage_root: Path) -> None:
        _seed(storage_root, "err-sess", "error", {"msg": "boom"})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/err-sess/diagnosis").json()
        assert any(e["type"] == "error" for e in body["failure_events"])

    def test_phase_change_events_included(self, storage_root: Path) -> None:
        _seed(
            storage_root,
            "pc-sess",
            "session.phase_change",
            {"before": "running", "after": "terminated", "reason": "budget_exceeded"},
        )
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/pc-sess/diagnosis").json()
        assert any(e["type"] == "session.phase_change" for e in body["failure_events"])

    def test_tool_call_vetoed_events_included(self, storage_root: Path) -> None:
        _seed(storage_root, "veto-sess", "tool_call.vetoed", {"tool_name": "bash"})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/veto-sess/diagnosis").json()
        assert any(e["type"] == "tool_call.vetoed" for e in body["failure_events"])

    def test_budget_warning_events_included(self, storage_root: Path) -> None:
        _seed(storage_root, "bw-sess", "budget.warning", {"model_calls": 5})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/bw-sess/diagnosis").json()
        assert any(e["type"] == "budget.warning" for e in body["failure_events"])

    def test_message_truncated_events_included(self, storage_root: Path) -> None:
        _seed(storage_root, "trunc-sess", "message.truncated", {"model_call_number": 3})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/trunc-sess/diagnosis").json()
        assert any(e["type"] == "message.truncated" for e in body["failure_events"])

    def test_session_created_not_in_failure_events(self, storage_root: Path) -> None:
        _seed(storage_root, "created-sess", "session.created", {"agent_id": "agent-1"})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/created-sess/diagnosis").json()
        assert not any(e["type"] == "session.created" for e in body["failure_events"])

    def test_message_added_not_in_failure_events(self, storage_root: Path) -> None:
        _seed(storage_root, "madd-sess", "message.added", {"role": "user"})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/madd-sess/diagnosis").json()
        assert not any(e["type"] == "message.added" for e in body["failure_events"])

    def test_failure_event_fields(self, storage_root: Path) -> None:
        _seed(storage_root, "fields-sess", "error", {"msg": "oops"})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/fields-sess/diagnosis").json()
        evt = body["failure_events"][0]
        assert "seq" in evt
        assert "ts" in evt
        assert "type" in evt
        assert "data" in evt

    def test_failure_events_thread_id_included(self, storage_root: Path) -> None:
        _seed(storage_root, "tid-sess", "error", {}, thread_id="thread-abc")
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/tid-sess/diagnosis").json()
        assert body["failure_events"][0]["thread_id"] == "thread-abc"

    def test_failure_events_thread_id_omitted_when_none(self, storage_root: Path) -> None:
        _seed(storage_root, "notid-sess2", "error", {})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/notid-sess2/diagnosis").json()
        assert "thread_id" not in body["failure_events"][0]


# ---------------------------------------------------------------------------
# terminal_phase and stop_reason extraction
# ---------------------------------------------------------------------------


class TestPhaseExtraction:
    def test_terminal_phase_from_last_phase_change(self, storage_root: Path) -> None:
        _seed_many(
            storage_root,
            "phase-sess",
            [
                (
                    "session.phase_change",
                    {"before": "created", "after": "running", "reason": "start"},
                ),
                (
                    "session.phase_change",
                    {"before": "running", "after": "terminated", "reason": "budget_exceeded"},
                ),
            ],
        )
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/phase-sess/diagnosis").json()
        assert body["terminal_phase"] == "terminated"

    def test_stop_reason_from_last_phase_change(self, storage_root: Path) -> None:
        _seed_many(
            storage_root,
            "reason-sess",
            [
                (
                    "session.phase_change",
                    {"before": "created", "after": "running", "reason": "start"},
                ),
                (
                    "session.phase_change",
                    {"before": "running", "after": "idle", "reason": "end_turn"},
                ),
            ],
        )
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/reason-sess/diagnosis").json()
        assert body["stop_reason"] == "end_turn"

    def test_terminal_phase_unknown_with_no_phase_change(self, storage_root: Path) -> None:
        _seed(storage_root, "nopc-sess", "error", {"msg": "boom"})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/nopc-sess/diagnosis").json()
        assert body["terminal_phase"] == "unknown"

    def test_stop_reason_empty_with_no_phase_change(self, storage_root: Path) -> None:
        _seed(storage_root, "nopc-sess2", "error", {"msg": "boom"})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/nopc-sess2/diagnosis").json()
        assert body["stop_reason"] == ""


# ---------------------------------------------------------------------------
# audit_entries filtering
# ---------------------------------------------------------------------------


class TestAuditEntries:
    def test_audit_entries_for_session_included(self, storage_root: Path) -> None:
        _write_audit_entry(storage_root, "audit-sess", event="harness.run_loop.failed")
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/audit-sess/diagnosis").json()
        assert len(body["audit_entries"]) >= 1
        assert any(e.get("detail", {}).get("session_id") == "audit-sess" for e in body["audit_entries"])

    def test_audit_entries_for_other_session_excluded(self, storage_root: Path) -> None:
        _write_audit_entry(storage_root, "other-sess", event="harness.run_loop.failed")
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/my-sess/diagnosis").json()
        assert all(e.get("detail", {}).get("session_id") != "other-sess" for e in body["audit_entries"])

    def test_audit_entries_empty_when_none_for_session(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/clean-sess/diagnosis").json()
        assert body["audit_entries"] == []


# ---------------------------------------------------------------------------
# replay_fixture_available
# ---------------------------------------------------------------------------


class TestReplayFixture:
    def test_replay_fixture_true_when_exists(self, storage_root: Path) -> None:
        fixture_path = storage_root / "fixtures" / "fix-sess" / "model_responses.ndjson"
        fixture_path.parent.mkdir(parents=True)
        fixture_path.write_text("")
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/fix-sess/diagnosis").json()
        assert body["replay_fixture_available"] is True

    def test_replay_fixture_false_when_absent(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/nofix-sess/diagnosis").json()
        assert body["replay_fixture_available"] is False


# ---------------------------------------------------------------------------
# event_count
# ---------------------------------------------------------------------------


class TestEventCount:
    def test_event_count_matches_total_events(self, storage_root: Path) -> None:
        _seed_many(
            storage_root,
            "cnt-sess",
            [
                ("session.created", {}),
                ("message.added", {"role": "user"}),
                ("error", {"msg": "boom"}),
            ],
        )
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/cnt-sess/diagnosis").json()
        assert body["event_count"] == 3


# ---------------------------------------------------------------------------
# Error handling — failure surfaced to caller + audit log
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_read_failure_returns_500(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "meridiand._diagnosis.LocalEventLogReader.read_after",
            side_effect=RuntimeError("disk error"),
        ):
            resp = client.get("/v1/sessions/fail-sess/diagnosis")
        assert resp.status_code == 500

    def test_read_failure_error_code(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "meridiand._diagnosis.LocalEventLogReader.read_after",
            side_effect=RuntimeError("disk error"),
        ):
            body = client.get("/v1/sessions/fail-sess2/diagnosis").json()
        assert body.get("error", {}).get("code") == "session_diagnosis_failed"

    def test_read_failure_writes_audit_log(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "meridiand._diagnosis.LocalEventLogReader.read_after",
            side_effect=RuntimeError("disk error"),
        ):
            client.get("/v1/sessions/fail-audit-sess/diagnosis")

        entries = _read_audit(storage_root)
        assert any(e.get("event") == "session.diagnosis.failed" for e in entries)

    def test_audit_detail_includes_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "meridiand._diagnosis.LocalEventLogReader.read_after",
            side_effect=RuntimeError("disk error"),
        ):
            client.get("/v1/sessions/sid-fail-sess/diagnosis")

        entries = _read_audit(storage_root)
        diag_entry = next(
            (e for e in entries if e.get("event") == "session.diagnosis.failed"), None
        )
        assert diag_entry is not None
        assert diag_entry["detail"]["session_id"] == "sid-fail-sess"

    def test_audit_detail_includes_message(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "meridiand._diagnosis.LocalEventLogReader.read_after",
            side_effect=RuntimeError("disk error"),
        ):
            client.get("/v1/sessions/msg-fail-sess/diagnosis")

        entries = _read_audit(storage_root)
        diag_entry = next(
            (e for e in entries if e.get("event") == "session.diagnosis.failed"), None
        )
        assert diag_entry is not None
        assert "message" in diag_entry["detail"]
        assert diag_entry["detail"]["message"]


# ---------------------------------------------------------------------------
# OpenTelemetry
# ---------------------------------------------------------------------------


class TestOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_otel_span_emitted_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/sessions/otel-sess/diagnosis")
        spans = _otel_exporter.get_finished_spans()
        assert any(s.name == "session.diagnosis" for s in spans)

    def test_otel_span_has_session_id_attribute(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/sessions/otel-attr-sess/diagnosis")
        spans = _otel_exporter.get_finished_spans()
        diag_spans = [s for s in spans if s.name == "session.diagnosis"]
        assert diag_spans
        assert diag_spans[-1].attributes.get("session.id") == "otel-attr-sess"

    def test_otel_span_carries_invocation_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/sessions/otel-evt-sess/diagnosis")
        spans = _otel_exporter.get_finished_spans()
        diag_spans = [s for s in spans if s.name == "session.diagnosis"]
        assert diag_spans
        # record_invocation_event always uses "meridian.error.invocation" as event name;
        # the StructuredEvent name is carried as an attribute.
        event_names = [e.name for e in diag_spans[-1].events]
        assert "meridian.error.invocation" in event_names
        invocation_event = next(e for e in diag_spans[-1].events if e.name == "meridian.error.invocation")
        assert invocation_event.attributes.get("name") == "session.diagnosis.invocation"

    def test_otel_span_error_status_on_failure(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "meridiand._diagnosis.LocalEventLogReader.read_after",
            side_effect=RuntimeError("disk error"),
        ):
            client.get("/v1/sessions/otel-err-sess/diagnosis")

        spans = _otel_exporter.get_finished_spans()
        diag_spans = [s for s in spans if s.name == "session.diagnosis"]
        assert diag_spans
        assert diag_spans[-1].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# create_app wiring
# ---------------------------------------------------------------------------


class TestCreateAppWiring:
    def test_diagnosis_route_present_with_storage_root(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        paths = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert any("/v1/sessions/{session_id}/diagnosis" in p for p in paths)

    def test_diagnosis_route_absent_without_storage_root(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        paths = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert not any("diagnosis" in p for p in paths)
