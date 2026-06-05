"""
System integration test: GET /v1/sessions/{id} returns Session with current phase projection.

Tests cover:
  - GET /v1/sessions/{id} returns 200 with session_id, id, agent_id, created_at, phase fields.
  - phase field reflects the live PhaseProjection, not a stale manifest value.
  - phase is "created" when no phase_change events have been written.
  - phase reflects a seeded phase_change event (e.g. "idle", "waiting_for_user").
  - Returns 404 with code "session_not_found" when session manifest is absent.
  - Not-found writes audit log entry with event "session.get.failed".
  - Not-found audit detail includes session_id.
  - Not-found audit detail includes message.
  - On generic failure, returns 500 with code "session_get_failed".
  - On failure, error message is included in response body.
  - On failure, audit log entry is written with event "session.get.failed".
  - On failure, audit detail includes session_id.
  - Success writes audit log entry with event "session.fetched".
  - Success audit level is "info".
  - Success audit detail includes session_id.
  - OTel span "session.get" is emitted on success.
  - OTel span carries a structured invocation event on each call.
  - OTel span has session.id attribute.
  - OTel span is set to ERROR status on failure.
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

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path) -> TestClient:
    audit = FileAuditLog(storage_root)
    writer = LocalEventLogWriter(storage_root)
    app = create_app(audit, storage_root=storage_root, event_log=writer)
    return TestClient(app, raise_server_exceptions=False)


def _write_session(
    storage_root: Path,
    session_id: str,
    *,
    agent_id: str | None = None,
    created_at: str = "2026-05-01T00:00:00+00:00",
) -> None:
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "session_id": session_id,
        "id": session_id,
        "created_at": created_at,
        "status": "idle",
    }
    if agent_id is not None:
        record["agent_id"] = agent_id
    (session_dir / "manifest.json").write_text(json.dumps(record))


def _seed_phase(storage_root: Path, session_id: str, phase: str) -> None:
    async def _write() -> None:
        writer = LocalEventLogWriter(storage_root)
        await writer.append(
            session_id,
            "session.phase_change",
            {"before": "created", "after": phase, "reason": "seed", "timestamp": "t0"},
        )

    asyncio.run(_write())


def _read_audit(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _get(client: TestClient, session_id: str) -> tuple[dict[str, Any], Any]:
    resp = client.get(f"/v1/sessions/{session_id}")
    return resp.json() | {"_status": resp.status_code}, resp


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


class TestGetSessionResponse:
    def test_returns_200(self, storage_root: Path) -> None:
        _write_session(storage_root, "gs-sess1")
        body, _ = _get(_make_client(storage_root), "gs-sess1")
        assert body["_status"] == 200

    def test_has_session_id(self, storage_root: Path) -> None:
        _write_session(storage_root, "gs-sess2")
        body, _ = _get(_make_client(storage_root), "gs-sess2")
        assert body["session_id"] == "gs-sess2"

    def test_has_id_field(self, storage_root: Path) -> None:
        _write_session(storage_root, "gs-sess3")
        body, _ = _get(_make_client(storage_root), "gs-sess3")
        assert body["id"] == "gs-sess3"

    def test_has_created_at(self, storage_root: Path) -> None:
        _write_session(storage_root, "gs-sess4", created_at="2026-05-10T00:00:00+00:00")
        body, _ = _get(_make_client(storage_root), "gs-sess4")
        assert body["created_at"] == "2026-05-10T00:00:00+00:00"

    def test_has_agent_id(self, storage_root: Path) -> None:
        _write_session(storage_root, "gs-sess5", agent_id="agent_xyz")
        body, _ = _get(_make_client(storage_root), "gs-sess5")
        assert body["agent_id"] == "agent_xyz"

    def test_has_phase_field(self, storage_root: Path) -> None:
        _write_session(storage_root, "gs-sess6")
        body, _ = _get(_make_client(storage_root), "gs-sess6")
        assert "phase" in body

    def test_phase_is_created_when_no_events(self, storage_root: Path) -> None:
        _write_session(storage_root, "gs-phase1")
        body, _ = _get(_make_client(storage_root), "gs-phase1")
        assert body["phase"] == "created"

    def test_phase_reflects_seeded_idle(self, storage_root: Path) -> None:
        _write_session(storage_root, "gs-phase2")
        _seed_phase(storage_root, "gs-phase2", "idle")
        body, _ = _get(_make_client(storage_root), "gs-phase2")
        assert body["phase"] == "idle"

    def test_phase_reflects_seeded_waiting_for_user(self, storage_root: Path) -> None:
        _write_session(storage_root, "gs-phase3")
        _seed_phase(storage_root, "gs-phase3", "waiting_for_user")
        body, _ = _get(_make_client(storage_root), "gs-phase3")
        assert body["phase"] == "waiting_for_user"

    def test_phase_reflects_seeded_terminated(self, storage_root: Path) -> None:
        _write_session(storage_root, "gs-phase4")
        _seed_phase(storage_root, "gs-phase4", "terminated")
        body, _ = _get(_make_client(storage_root), "gs-phase4")
        assert body["phase"] == "terminated"

    def test_phase_overrides_stale_manifest_value(self, storage_root: Path) -> None:
        session_dir = storage_root / "sessions" / "gs-stale"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text(
            json.dumps({"session_id": "gs-stale", "id": "gs-stale", "phase": "terminated"})
        )
        _seed_phase(storage_root, "gs-stale", "idle")
        body, _ = _get(_make_client(storage_root), "gs-stale")
        assert body["phase"] == "idle"


# ---------------------------------------------------------------------------
# Not found
# ---------------------------------------------------------------------------


class TestGetSessionNotFound:
    def test_missing_session_returns_404(self, storage_root: Path) -> None:
        body, _ = _get(_make_client(storage_root), "no-such-session")
        assert body["_status"] == 404

    def test_missing_session_error_code(self, storage_root: Path) -> None:
        body, _ = _get(_make_client(storage_root), "no-such-session2")
        assert body["error"]["code"] == "session_not_found"

    def test_missing_session_writes_audit_log(self, storage_root: Path) -> None:
        _get(_make_client(storage_root), "no-audit-sess")
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.get.failed" for r in records)

    def test_missing_session_audit_detail_has_session_id(self, storage_root: Path) -> None:
        _get(_make_client(storage_root), "no-detail-sess")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.get.failed")
        assert rec["detail"]["session_id"] == "no-detail-sess"

    def test_missing_session_audit_detail_has_message(self, storage_root: Path) -> None:
        _get(_make_client(storage_root), "no-msg-sess")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.get.failed")
        assert "message" in rec["detail"]
        assert len(rec["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# Generic failure
# ---------------------------------------------------------------------------


class TestGetSessionFailure:
    def test_corrupt_manifest_returns_500(self, storage_root: Path) -> None:
        session_dir = storage_root / "sessions" / "gs-corrupt"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text("not valid json {{{")
        body, _ = _get(_make_client(storage_root), "gs-corrupt")
        assert body["_status"] == 500

    def test_corrupt_manifest_error_code(self, storage_root: Path) -> None:
        session_dir = storage_root / "sessions" / "gs-corrupt2"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text("not valid json {{{")
        body, _ = _get(_make_client(storage_root), "gs-corrupt2")
        assert body["error"]["code"] == "session_get_failed"

    def test_corrupt_manifest_error_message_present(self, storage_root: Path) -> None:
        session_dir = storage_root / "sessions" / "gs-corrupt3"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text("not valid json {{{")
        body, _ = _get(_make_client(storage_root), "gs-corrupt3")
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0

    def test_corrupt_manifest_writes_audit_log(self, storage_root: Path) -> None:
        session_dir = storage_root / "sessions" / "gs-corrupt4"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text("not valid json {{{")
        _get(_make_client(storage_root), "gs-corrupt4")
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.get.failed" for r in records)

    def test_corrupt_manifest_audit_detail_has_session_id(self, storage_root: Path) -> None:
        session_dir = storage_root / "sessions" / "gs-corrupt5"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text("not valid json {{{")
        _get(_make_client(storage_root), "gs-corrupt5")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.get.failed")
        assert rec["detail"]["session_id"] == "gs-corrupt5"


# ---------------------------------------------------------------------------
# Audit log (success)
# ---------------------------------------------------------------------------


class TestGetSessionAudit:
    def test_success_writes_audit_entry(self, storage_root: Path) -> None:
        _write_session(storage_root, "aud-sess1")
        _get(_make_client(storage_root), "aud-sess1")
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.fetched" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        _write_session(storage_root, "aud-sess2")
        _get(_make_client(storage_root), "aud-sess2")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.fetched")
        assert rec["level"] == "info"

    def test_success_audit_detail_has_session_id(self, storage_root: Path) -> None:
        _write_session(storage_root, "aud-sess3")
        _get(_make_client(storage_root), "aud-sess3")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.fetched")
        assert rec["detail"]["session_id"] == "aud-sess3"


# ---------------------------------------------------------------------------
# OTel instrumentation
# ---------------------------------------------------------------------------


class TestGetSessionOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_span(self, storage_root: Path) -> None:
        _write_session(storage_root, "otel-sess1")
        _otel_exporter.clear()
        _get(_make_client(storage_root), "otel-sess1")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.get" in span_names

    def test_span_has_invocation_event(self, storage_root: Path) -> None:
        _write_session(storage_root, "otel-sess2")
        _otel_exporter.clear()
        _get(_make_client(storage_root), "otel-sess2")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.get")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        _write_session(storage_root, "otel-sess3")
        _otel_exporter.clear()
        _get(_make_client(storage_root), "otel-sess3")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.get")
        assert span is not None
        assert span.attributes.get("session.id") == "otel-sess3"

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        session_dir = storage_root / "sessions" / "otel-corrupt"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text("not valid json {{{")
        _otel_exporter.clear()
        _get(_make_client(storage_root), "otel-corrupt")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.get")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR
