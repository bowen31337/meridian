"""
Session cancel endpoint conformance suite.

Tests cover:
  - POST /v1/sessions/{id}/cancel returns 200 with session_id, before, after, reason.
  - after is always "terminated".
  - reason is always "cancelled".
  - before reflects the session's phase prior to cancellation.
  - Returns 404 with code "session_cancel_not_found" when session manifest is absent.
  - Session not found writes audit log entry with event "session.cancel.failed".
  - Audit log detail includes session_id on not-found failure.
  - phase_change event written to event log with after="terminated" and reason="cancelled".
  - phase_change event before field matches the pre-cancel phase.
  - Tool call cancellation: tool_call.cancelled events emitted for each pending call
    when in waiting_for_tool.
  - cancelled_tool_call_ids in response matches tool call ids from checkpoint.
  - No tool_call.cancelled events when not in waiting_for_tool phase.
  - Child sessions appear in cancelled_sessions after cancel.
  - cancelled_count equals len(cancelled_sessions).
  - Child session manifest status updated to "cancelled".
  - Audit entry written with event "child_session.completed" for each cancelled child.
  - Success writes audit log entry with event "session.cancel.accepted".
  - Audit detail on success includes session_id, before, after, reason.
  - On generic failure, 422 returned with error.code "session_cancel_failed".
  - On failure, error message is included in response body.
  - On failure, audit log entry written with event "session.cancel.failed".
  - On failure, audit detail includes session_id and message.
  - OTel span "session.cancel" emitted on success.
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

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from storage_event_log import LocalEventLogWriter
from storage_reposit import LocalEventLogReader

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path) -> TestClient:
    audit = FileAuditLog(storage_root)
    writer = LocalEventLogWriter(storage_root)
    app = create_app(audit, storage_root=storage_root, event_log=writer)
    return TestClient(app, raise_server_exceptions=False)


def _write_session(storage_root: Path, session_id: str) -> None:
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(
        json.dumps({"session_id": session_id, "status": "active"})
    )


def _write_child_manifest(
    storage_root: Path,
    parent_id: str,
    child_id: str,
    *,
    status: str = "spawned",
) -> None:
    session_dir = storage_root / "sessions" / child_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "child_session_id": child_id,
                "parent_session_id": parent_id,
                "status": status,
            }
        )
    )


def _write_checkpoint(
    storage_root: Path,
    session_id: str,
    pending_tool_calls: list[dict[str, Any]],
) -> None:
    ckpt_dir = storage_root / "checkpoints" / session_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "latest.json").write_text(
        json.dumps(
            {
                "seq": 1,
                "phase": "waiting_for_tool",
                "pending_tool_calls": pending_tool_calls,
                "message_tail": [],
                "usage": {},
                "taken_at": "2024-01-01T00:00:00+00:00",
            }
        )
    )


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


class TestSessionCancelResponse:
    def test_returns_200_on_success(self, storage_root: Path) -> None:
        _write_session(storage_root, "sc-sess1")
        resp = _make_client(storage_root).post("/v1/sessions/sc-sess1/cancel")
        assert resp.status_code == 200

    def test_response_has_session_id(self, storage_root: Path) -> None:
        _write_session(storage_root, "sc-sess2")
        body = _make_client(storage_root).post("/v1/sessions/sc-sess2/cancel").json()
        assert body["session_id"] == "sc-sess2"

    def test_response_after_is_terminated(self, storage_root: Path) -> None:
        _write_session(storage_root, "sc-sess3")
        body = _make_client(storage_root).post("/v1/sessions/sc-sess3/cancel").json()
        assert body["after"] == "terminated"

    def test_response_reason_is_cancelled(self, storage_root: Path) -> None:
        _write_session(storage_root, "sc-sess4")
        body = _make_client(storage_root).post("/v1/sessions/sc-sess4/cancel").json()
        assert body["reason"] == "cancelled"

    def test_response_before_reflects_prior_phase(self, storage_root: Path) -> None:
        _write_session(storage_root, "sc-sess5")
        _seed_phase(storage_root, "sc-sess5", "idle")
        body = _make_client(storage_root).post("/v1/sessions/sc-sess5/cancel").json()
        assert body["before"] == "idle"

    def test_response_before_defaults_to_created(self, storage_root: Path) -> None:
        _write_session(storage_root, "sc-sess6")
        body = _make_client(storage_root).post("/v1/sessions/sc-sess6/cancel").json()
        assert body["before"] == "created"


# ---------------------------------------------------------------------------
# Session not found
# ---------------------------------------------------------------------------


class TestSessionCancelNotFound:
    def test_missing_session_returns_404(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post("/v1/sessions/no-such/cancel")
        assert resp.status_code == 404

    def test_missing_session_error_code(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post("/v1/sessions/no-such2/cancel").json()
        assert body["error"]["code"] == "session_cancel_not_found"

    def test_missing_session_writes_audit_log(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/sessions/no-audit/cancel")
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.cancel.failed" for r in records)

    def test_missing_session_audit_detail_has_session_id(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/sessions/no-detail/cancel")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.cancel.failed")
        assert rec["detail"]["session_id"] == "no-detail"


# ---------------------------------------------------------------------------
# Phase transition written to event log
# ---------------------------------------------------------------------------


class TestSessionCancelPhaseTransition:
    def test_phase_change_event_written(self, storage_root: Path) -> None:
        _write_session(storage_root, "pt-sess1")
        _make_client(storage_root).post("/v1/sessions/pt-sess1/cancel")
        events = _read_events(storage_root, "pt-sess1")
        assert any(e.type == "session.phase_change" for e in events)

    def test_phase_change_after_is_terminated(self, storage_root: Path) -> None:
        _write_session(storage_root, "pt-sess2")
        _make_client(storage_root).post("/v1/sessions/pt-sess2/cancel")
        events = _read_events(storage_root, "pt-sess2")
        change = next(e for e in events if e.type == "session.phase_change")
        assert change.data["after"] == "terminated"

    def test_phase_change_reason_is_cancelled(self, storage_root: Path) -> None:
        _write_session(storage_root, "pt-sess3")
        _make_client(storage_root).post("/v1/sessions/pt-sess3/cancel")
        events = _read_events(storage_root, "pt-sess3")
        change = next(e for e in events if e.type == "session.phase_change")
        assert change.data["reason"] == "cancelled"

    def test_phase_change_before_matches_prior_phase(self, storage_root: Path) -> None:
        _write_session(storage_root, "pt-sess4")
        _seed_phase(storage_root, "pt-sess4", "waiting_for_user")
        _make_client(storage_root).post("/v1/sessions/pt-sess4/cancel")
        events = _read_events(storage_root, "pt-sess4")
        changes = [e for e in events if e.type == "session.phase_change"]
        # last change is the cancellation
        cancel_change = changes[-1]
        assert cancel_change.data["before"] == "waiting_for_user"


# ---------------------------------------------------------------------------
# Tool call cancellation
# ---------------------------------------------------------------------------


class TestSessionCancelToolCalls:
    def test_tool_call_cancelled_events_emitted_when_waiting_for_tool(
        self, storage_root: Path
    ) -> None:
        _write_session(storage_root, "tc-sess1")
        _seed_phase(storage_root, "tc-sess1", "waiting_for_tool")
        _write_checkpoint(storage_root, "tc-sess1", [{"id": "call_abc"}, {"id": "call_def"}])
        _make_client(storage_root).post("/v1/sessions/tc-sess1/cancel")
        events = _read_events(storage_root, "tc-sess1")
        cancelled = [e for e in events if e.type == "tool_call.cancelled"]
        assert len(cancelled) == 2

    def test_cancelled_tool_call_ids_in_response(self, storage_root: Path) -> None:
        _write_session(storage_root, "tc-sess2")
        _seed_phase(storage_root, "tc-sess2", "waiting_for_tool")
        _write_checkpoint(storage_root, "tc-sess2", [{"id": "call_xyz"}])
        body = _make_client(storage_root).post("/v1/sessions/tc-sess2/cancel").json()
        assert "call_xyz" in body["cancelled_tool_call_ids"]

    def test_no_tool_call_events_when_not_waiting_for_tool(self, storage_root: Path) -> None:
        _write_session(storage_root, "tc-sess3")
        _seed_phase(storage_root, "tc-sess3", "idle")
        _write_checkpoint(storage_root, "tc-sess3", [{"id": "call_idle"}])
        _make_client(storage_root).post("/v1/sessions/tc-sess3/cancel")
        events = _read_events(storage_root, "tc-sess3")
        assert not any(e.type == "tool_call.cancelled" for e in events)

    def test_empty_cancelled_tool_call_ids_when_no_checkpoint(self, storage_root: Path) -> None:
        _write_session(storage_root, "tc-sess4")
        _seed_phase(storage_root, "tc-sess4", "waiting_for_tool")
        body = _make_client(storage_root).post("/v1/sessions/tc-sess4/cancel").json()
        assert body["cancelled_tool_call_ids"] == []


# ---------------------------------------------------------------------------
# Child session cancellation
# ---------------------------------------------------------------------------


class TestSessionCancelChildren:
    def test_child_in_cancelled_sessions(self, storage_root: Path) -> None:
        _write_session(storage_root, "ch-parent1")
        _write_child_manifest(storage_root, "ch-parent1", "ch-child1")
        body = _make_client(storage_root).post("/v1/sessions/ch-parent1/cancel").json()
        assert "ch-child1" in body["cancelled_sessions"]

    def test_cancelled_count_matches_list_length(self, storage_root: Path) -> None:
        _write_session(storage_root, "ch-parent2")
        _write_child_manifest(storage_root, "ch-parent2", "ch-child2a")
        _write_child_manifest(storage_root, "ch-parent2", "ch-child2b")
        body = _make_client(storage_root).post("/v1/sessions/ch-parent2/cancel").json()
        assert body["cancelled_count"] == len(body["cancelled_sessions"])

    def test_child_manifest_status_set_to_cancelled(self, storage_root: Path) -> None:
        _write_session(storage_root, "ch-parent3")
        _write_child_manifest(storage_root, "ch-parent3", "ch-child3")
        _make_client(storage_root).post("/v1/sessions/ch-parent3/cancel")
        manifest = json.loads(
            (storage_root / "sessions" / "ch-child3" / "manifest.json").read_text()
        )
        assert manifest["status"] == "cancelled"

    def test_audit_entry_per_child(self, storage_root: Path) -> None:
        _write_session(storage_root, "ch-parent4")
        _write_child_manifest(storage_root, "ch-parent4", "ch-child4")
        _make_client(storage_root).post("/v1/sessions/ch-parent4/cancel")
        records = _read_audit(storage_root)
        assert any(
            r.get("event") == "child_session.completed"
            and r.get("detail", {}).get("child_session_id") == "ch-child4"
            for r in records
        )

    def test_no_children_returns_empty_cancelled_sessions(self, storage_root: Path) -> None:
        _write_session(storage_root, "ch-parent5")
        body = _make_client(storage_root).post("/v1/sessions/ch-parent5/cancel").json()
        assert body["cancelled_sessions"] == []
        assert body["cancelled_count"] == 0


# ---------------------------------------------------------------------------
# Audit log on success
# ---------------------------------------------------------------------------


class TestSessionCancelAuditSuccess:
    def test_success_writes_accepted_event(self, storage_root: Path) -> None:
        _write_session(storage_root, "aud-sess1")
        _make_client(storage_root).post("/v1/sessions/aud-sess1/cancel")
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.cancel.accepted" for r in records)

    def test_accepted_audit_level_is_info(self, storage_root: Path) -> None:
        _write_session(storage_root, "aud-sess2")
        _make_client(storage_root).post("/v1/sessions/aud-sess2/cancel")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.cancel.accepted")
        assert rec["level"] == "info"

    def test_accepted_audit_detail_has_session_id(self, storage_root: Path) -> None:
        _write_session(storage_root, "aud-sess3")
        _make_client(storage_root).post("/v1/sessions/aud-sess3/cancel")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.cancel.accepted")
        assert rec["detail"]["session_id"] == "aud-sess3"

    def test_accepted_audit_detail_has_after_terminated(self, storage_root: Path) -> None:
        _write_session(storage_root, "aud-sess4")
        _make_client(storage_root).post("/v1/sessions/aud-sess4/cancel")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.cancel.accepted")
        assert rec["detail"]["after"] == "terminated"

    def test_accepted_audit_detail_has_reason_cancelled(self, storage_root: Path) -> None:
        _write_session(storage_root, "aud-sess5")
        _make_client(storage_root).post("/v1/sessions/aud-sess5/cancel")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.cancel.accepted")
        assert rec["detail"]["reason"] == "cancelled"


# ---------------------------------------------------------------------------
# Failure surfacing
# ---------------------------------------------------------------------------


class TestSessionCancelFailure:
    def _make_failing_client(self, storage_root: Path) -> tuple[TestClient, Path]:
        _write_session(storage_root, "fail-sess")
        _write_child_manifest(storage_root, "fail-sess", "fail-child")
        child_path = storage_root / "sessions" / "fail-child" / "manifest.json"
        child_path.chmod(0o444)
        return _make_client(storage_root), child_path

    def test_failure_returns_422(self, storage_root: Path) -> None:
        client, child_path = self._make_failing_client(storage_root)
        try:
            resp = client.post("/v1/sessions/fail-sess/cancel")
            assert resp.status_code == 422
        finally:
            child_path.chmod(0o644)

    def test_failure_error_code_is_session_cancel_failed(self, storage_root: Path) -> None:
        client, child_path = self._make_failing_client(storage_root)
        try:
            body = client.post("/v1/sessions/fail-sess/cancel").json()
            assert body["error"]["code"] == "session_cancel_failed"
        finally:
            child_path.chmod(0o644)

    def test_failure_error_message_in_response(self, storage_root: Path) -> None:
        client, child_path = self._make_failing_client(storage_root)
        try:
            body = client.post("/v1/sessions/fail-sess/cancel").json()
            assert len(body["error"]["message"]) > 0
        finally:
            child_path.chmod(0o644)

    def test_failure_writes_audit_entry(self, storage_root: Path) -> None:
        client, child_path = self._make_failing_client(storage_root)
        try:
            client.post("/v1/sessions/fail-sess/cancel")
        finally:
            child_path.chmod(0o644)
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.cancel.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client, child_path = self._make_failing_client(storage_root)
        try:
            client.post("/v1/sessions/fail-sess/cancel")
        finally:
            child_path.chmod(0o644)
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.cancel.failed")
        assert rec["level"] == "error"

    def test_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client, child_path = self._make_failing_client(storage_root)
        try:
            client.post("/v1/sessions/fail-sess/cancel")
        finally:
            child_path.chmod(0o644)
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.cancel.failed")
        assert rec["detail"]["session_id"] == "fail-sess"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client, child_path = self._make_failing_client(storage_root)
        try:
            client.post("/v1/sessions/fail-sess/cancel")
        finally:
            child_path.chmod(0o644)
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.cancel.failed")
        assert len(rec["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestSessionCancelOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_session_cancel_span(self, storage_root: Path) -> None:
        _write_session(storage_root, "otel-sess1")
        _make_client(storage_root).post("/v1/sessions/otel-sess1/cancel")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.cancel" in span_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        _write_session(storage_root, "otel-sess2")
        _make_client(storage_root).post("/v1/sessions/otel-sess2/cancel")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        cancel_span = spans.get("session.cancel")
        assert cancel_span is not None
        assert cancel_span.attributes["session.id"] == "otel-sess2"

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        _make_client(storage_root).post("/v1/sessions/no-such-otel/cancel")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        cancel_span = spans.get("session.cancel")
        assert cancel_span is not None
        assert cancel_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Router wiring via create_app
# ---------------------------------------------------------------------------


class TestSessionCancelRouterWiring:
    def test_route_exists_with_storage_root_and_event_log(self, storage_root: Path) -> None:
        _write_session(storage_root, "wire-sess1")
        resp = _make_client(storage_root).post("/v1/sessions/wire-sess1/cancel")
        assert resp.status_code != 404

    def test_no_storage_root_returns_404(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        from fastapi.testclient import TestClient

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/sessions/any/cancel")
        assert resp.status_code == 404

    def test_no_event_log_returns_404(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        from fastapi.testclient import TestClient

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/sessions/any/cancel")
        assert resp.status_code == 404
