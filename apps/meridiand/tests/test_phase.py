"""
Phase transition endpoint conformance suite.

Tests cover:
  - POST /v1/x/sessions/{id}/phase returns 200 with session_id, before, after, reason, seq.
  - before is 'created' when no prior phase_change events exist.
  - before reflects the 'after' field of the last session.phase_change event.
  - after matches to_phase from the request body.
  - reason matches the reason from the request body.
  - seq is the event sequence number from the event log.
  - session.phase_change event is written to the NDJSON log with correct fields.
  - Event data includes before, after, timestamp, and reason fields.
  - Missing to_phase field returns 422 (FastAPI schema validation).
  - Missing reason field returns 422 (FastAPI schema validation).
  - On write failure, returns 422 with code "phase_transition_failed".
  - On write failure, an audit log entry is written with event "session.phase_transition.failed".
  - Audit log detail includes session_id and to_phase.
  - OTel span "session.phase_transition" is emitted on success.
  - OTel span is set to ERROR status on failure.
  - create_app wires the phase router when storage_root and event_log are supplied.
  - create_app omits the phase route when storage_root is None.
  - create_app omits the phase route when event_log is None.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from storage_event_log import LocalEventLogWriter

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_writer(storage_root: Path) -> LocalEventLogWriter:
    return LocalEventLogWriter(storage_root)


def _make_client(
    storage_root: Path,
    audit_log: FileAuditLog,
    writer: LocalEventLogWriter | None = None,
) -> TestClient:
    w = writer or _make_writer(storage_root)
    app = create_app(audit_log, storage_root=storage_root, event_log=w)
    return TestClient(app, raise_server_exceptions=False)


async def _async_write_phase_change(
    storage_root: Path, session_id: str, before: str, after: str
) -> None:
    writer = LocalEventLogWriter(storage_root)
    await writer.append(
        session_id,
        "session.phase_change",
        {"before": before, "after": after, "reason": "seed", "timestamp": "t0"},
    )


def _write_phase_change(storage_root: Path, session_id: str, before: str, after: str) -> None:
    """Pre-seed the event log with a session.phase_change event."""
    import asyncio

    asyncio.run(_async_write_phase_change(storage_root, session_id, before, after))


def _read_phase_events(storage_root: Path, session_id: str) -> list[dict[str, Any]]:
    """Read all session.phase_change events from the NDJSON log."""
    from storage_reposit import LocalEventLogReader

    reader = LocalEventLogReader(storage_root)
    events = reader.read_after(session_id, -1)
    return [e.data for e in events if e.type == "session.phase_change"]


def _read_audit(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Integration tests: POST /v1/x/sessions/{id}/phase
# ---------------------------------------------------------------------------


class TestPhaseTransitionEndpoint:
    def test_returns_200_on_success(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post(
            "/v1/x/sessions/sess1/phase", json={"to_phase": "running", "reason": "start"}
        )
        assert resp.status_code == 200

    def test_response_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post(
            "/v1/x/sessions/sess2/phase", json={"to_phase": "running", "reason": "go"}
        ).json()
        assert body["session_id"] == "sess2"

    def test_response_before_is_created_for_new_session(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post(
            "/v1/x/sessions/new-sess/phase", json={"to_phase": "running", "reason": "init"}
        ).json()
        assert body["before"] == "created"

    def test_response_before_reflects_last_phase_change(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        writer = _make_writer(storage_root)
        _write_phase_change(storage_root, "prev-sess", "created", "paused")
        client = _make_client(storage_root, audit, writer)
        body = client.post(
            "/v1/x/sessions/prev-sess/phase", json={"to_phase": "done", "reason": "finish"}
        ).json()
        assert body["before"] == "paused"

    def test_response_after_matches_to_phase(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post(
            "/v1/x/sessions/sess3/phase", json={"to_phase": "thinking", "reason": "x"}
        ).json()
        assert body["after"] == "thinking"

    def test_response_reason_matches_body(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post(
            "/v1/x/sessions/sess4/phase",
            json={"to_phase": "done", "reason": "task-complete"},
        ).json()
        assert body["reason"] == "task-complete"

    def test_response_has_seq(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post(
            "/v1/x/sessions/sess5/phase", json={"to_phase": "running", "reason": "x"}
        ).json()
        assert "seq" in body
        assert isinstance(body["seq"], int)

    def test_event_written_to_log(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post(
            "/v1/x/sessions/log-sess/phase", json={"to_phase": "running", "reason": "go"}
        )
        events = _read_phase_events(storage_root, "log-sess")
        assert len(events) == 1

    def test_event_data_has_before_field(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post(
            "/v1/x/sessions/bf-sess/phase", json={"to_phase": "running", "reason": "x"}
        )
        events = _read_phase_events(storage_root, "bf-sess")
        assert events[0]["before"] == "created"

    def test_event_data_has_after_field(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post(
            "/v1/x/sessions/af-sess/phase", json={"to_phase": "done", "reason": "x"}
        )
        events = _read_phase_events(storage_root, "af-sess")
        assert events[0]["after"] == "done"

    def test_event_data_has_reason_field(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post(
            "/v1/x/sessions/rs-sess/phase", json={"to_phase": "done", "reason": "my-reason"}
        )
        events = _read_phase_events(storage_root, "rs-sess")
        assert events[0]["reason"] == "my-reason"

    def test_event_data_has_timestamp_field(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post(
            "/v1/x/sessions/ts-sess/phase", json={"to_phase": "running", "reason": "x"}
        )
        events = _read_phase_events(storage_root, "ts-sess")
        assert "timestamp" in events[0]

    def test_second_transition_uses_previous_as_before(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        writer = _make_writer(storage_root)
        client = _make_client(storage_root, audit, writer)
        client.post(
            "/v1/x/sessions/chain-sess/phase", json={"to_phase": "running", "reason": "a"}
        )
        body = client.post(
            "/v1/x/sessions/chain-sess/phase", json={"to_phase": "done", "reason": "b"}
        ).json()
        assert body["before"] == "running"
        assert body["after"] == "done"

    def test_missing_to_phase_returns_422(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/sess/phase", json={"reason": "x"})
        assert resp.status_code == 422

    def test_missing_reason_returns_422(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/sess/phase", json={"to_phase": "running"})
        assert resp.status_code == 422

    def test_failure_returns_422_with_code(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/x/sessions/fail-sess/phase", json={"to_phase": "running", "reason": "x"}
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "phase_transition_failed"

    def test_failure_writes_audit_log(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post(
            "/v1/x/sessions/audit-sess/phase", json={"to_phase": "running", "reason": "x"}
        )
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.phase_transition.failed" for r in records)

    def test_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post(
            "/v1/x/sessions/detail-sess/phase", json={"to_phase": "done", "reason": "x"}
        )
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.phase_transition.failed")
        assert rec["detail"]["session_id"] == "detail-sess"

    def test_failure_audit_detail_has_to_phase(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post(
            "/v1/x/sessions/tp-fail-sess/phase", json={"to_phase": "paused", "reason": "x"}
        )
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.phase_transition.failed")
        assert rec["detail"]["to_phase"] == "paused"

    def test_no_storage_root_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/any/phase", json={"to_phase": "x", "reason": "y"})
        assert resp.status_code == 404

    def test_no_event_log_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/any/phase", json={"to_phase": "x", "reason": "y"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# OTel span tests
# ---------------------------------------------------------------------------


class TestPhaseTransitionOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _make_client(self, storage_root: Path) -> TestClient:
        audit = FileAuditLog(storage_root)
        writer = _make_writer(storage_root)
        app = create_app(audit, storage_root=storage_root, event_log=writer)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_phase_transition_span(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        client.post(
            "/v1/x/sessions/otel-sess/phase", json={"to_phase": "running", "reason": "x"}
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.phase_transition" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post(
            "/v1/x/sessions/otel-fail/phase", json={"to_phase": "done", "reason": "x"}
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        pt_span = spans.get("session.phase_transition")
        assert pt_span is not None
        assert pt_span.status.status_code == StatusCode.ERROR
