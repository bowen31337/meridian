"""
System integration test: POST /v1/sessions creates a Session.

Tests cover:
  - POST /v1/sessions returns 201 with session_id, thread_id, agent_id,
    agent_version_id, status, created_at fields.
  - session_id starts with "sess_".
  - thread_id starts with "thread_".
  - status is "idle" after creation.
  - created_at is present and non-empty.
  - agent_id is propagated from request body.
  - agent_id is None when not supplied.
  - agent_version_id is pinned to the current agent version when agent_id resolves.
  - agent_version_id is None when agent_id is None.
  - agent_version_id is None when agent_id is provided but agent file is absent.
  - Session manifest created at sessions/{session_id}/manifest.json.
  - Manifest contains session_id, agent_id, agent_version_id, thread_id, status, created_at.
  - Initial thread file created at sessions/{session_id}/threads/{thread_id}.json.
  - Thread file contains thread_id, session_id, created_at.
  - session.created event is written to the event log.
  - Event type is "session.created".
  - Event data contains session_id, agent_id, agent_version_id, thread_id, created_at.
  - Event thread_id matches the thread_id returned in the response.
  - On failure, returns 500 with code "session_create_failed".
  - On failure, error message is surfaced in response body.
  - On failure, audit log entry is written with event "session.create.failed".
  - Audit entry level is "error" on failure.
  - Audit detail includes session_id, agent_id, and message on failure.
  - OTel span "session.create" is emitted on success.
  - OTel span has session.id attribute.
  - OTel span carries a structured invocation event on each call.
  - OTel span is set to ERROR status on failure.
  - create_app wires the sessions router when storage_root and event_log are supplied.
  - create_app omits the sessions route when storage_root is None.
  - create_app omits the sessions route when event_log is None.
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
    audit_log: FileAuditLog | None = None,
    event_log: EventLogWriter | None = None,
) -> TestClient:
    audit = audit_log or FileAuditLog(storage_root)
    writer = event_log or LocalEventLogWriter(storage_root)
    app = create_app(audit, storage_root=storage_root, event_log=writer)
    return TestClient(app, raise_server_exceptions=False)


def _post_session(
    client: TestClient,
    agent_id: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if agent_id is not None:
        body["agent_id"] = agent_id
    resp = client.post("/v1/sessions", json=body)
    return resp.json() | {"_status": resp.status_code}


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_events(storage_root: Path, session_id: str) -> list[dict[str, Any]]:
    files = list((storage_root / "events").glob(f"**/{session_id}.ndjson"))
    if not files:
        return []
    return [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]


def _seed_agent(storage_root: Path, agent_id: str, version_id: str) -> None:
    agents_dir = storage_root / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    agent_record = {
        "id": agent_id,
        "name": "Test Agent",
        "kind": "test",
        "created_at": "2026-01-01T00:00:00+00:00",
        "version": {"id": version_id},
    }
    (agents_dir / f"{agent_id}.json").write_text(json.dumps(agent_record))


# ---------------------------------------------------------------------------
# Basic response shape
# ---------------------------------------------------------------------------


class TestSessionCreateSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        assert result["_status"] == 201

    def test_session_id_in_response(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        assert "session_id" in result

    def test_session_id_starts_with_sess(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        assert result["session_id"].startswith("sess_")

    def test_thread_id_in_response(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        assert "thread_id" in result

    def test_thread_id_starts_with_thread(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        assert result["thread_id"].startswith("thread_")

    def test_status_is_idle(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        assert result["status"] == "idle"

    def test_created_at_present(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        assert "created_at" in result
        assert len(result["created_at"]) > 0

    def test_agent_id_none_when_not_provided(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        assert result["agent_id"] is None

    def test_agent_version_id_none_when_no_agent_id(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        assert result["agent_version_id"] is None


# ---------------------------------------------------------------------------
# Agent propagation and version pinning
# ---------------------------------------------------------------------------


class TestAgentVersionPinning:
    def test_agent_id_propagated_in_response(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root), agent_id="agent-42")
        assert result["agent_id"] == "agent-42"

    def test_agent_version_id_pinned_when_agent_exists(self, storage_root: Path) -> None:
        _seed_agent(storage_root, "agent-pin", "agentver_abc123")
        result = _post_session(_make_client(storage_root), agent_id="agent-pin")
        assert result["agent_version_id"] == "agentver_abc123"

    def test_agent_version_id_none_when_agent_file_absent(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root), agent_id="agent-missing")
        assert result["agent_version_id"] is None


# ---------------------------------------------------------------------------
# Session manifest creation
# ---------------------------------------------------------------------------


class TestSessionManifest:
    def test_manifest_file_created(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        session_id = result["session_id"]
        assert (storage_root / "sessions" / session_id / "manifest.json").exists()

    def test_manifest_has_session_id(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        session_id = result["session_id"]
        manifest = json.loads(
            (storage_root / "sessions" / session_id / "manifest.json").read_text()
        )
        assert manifest["session_id"] == session_id

    def test_manifest_has_thread_id(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        session_id = result["session_id"]
        manifest = json.loads(
            (storage_root / "sessions" / session_id / "manifest.json").read_text()
        )
        assert manifest["thread_id"] == result["thread_id"]

    def test_manifest_has_status_idle(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        session_id = result["session_id"]
        manifest = json.loads(
            (storage_root / "sessions" / session_id / "manifest.json").read_text()
        )
        assert manifest["status"] == "idle"

    def test_manifest_has_created_at(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        session_id = result["session_id"]
        manifest = json.loads(
            (storage_root / "sessions" / session_id / "manifest.json").read_text()
        )
        assert "created_at" in manifest
        assert len(manifest["created_at"]) > 0

    def test_manifest_agent_id_set_when_provided(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root), agent_id="agent-manifest")
        session_id = result["session_id"]
        manifest = json.loads(
            (storage_root / "sessions" / session_id / "manifest.json").read_text()
        )
        assert manifest["agent_id"] == "agent-manifest"

    def test_manifest_agent_id_none_when_not_provided(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        session_id = result["session_id"]
        manifest = json.loads(
            (storage_root / "sessions" / session_id / "manifest.json").read_text()
        )
        assert manifest["agent_id"] is None

    def test_manifest_agent_version_id_pinned(self, storage_root: Path) -> None:
        _seed_agent(storage_root, "agent-mver", "agentver_mver")
        result = _post_session(_make_client(storage_root), agent_id="agent-mver")
        session_id = result["session_id"]
        manifest = json.loads(
            (storage_root / "sessions" / session_id / "manifest.json").read_text()
        )
        assert manifest["agent_version_id"] == "agentver_mver"


# ---------------------------------------------------------------------------
# Initial thread creation
# ---------------------------------------------------------------------------


class TestInitialThread:
    def test_thread_file_created(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        session_id = result["session_id"]
        thread_id = result["thread_id"]
        assert (storage_root / "sessions" / session_id / "threads" / f"{thread_id}.json").exists()

    def test_thread_file_has_thread_id(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        session_id = result["session_id"]
        thread_id = result["thread_id"]
        thread = json.loads(
            (storage_root / "sessions" / session_id / "threads" / f"{thread_id}.json").read_text()
        )
        assert thread["thread_id"] == thread_id

    def test_thread_file_has_session_id(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        session_id = result["session_id"]
        thread_id = result["thread_id"]
        thread = json.loads(
            (storage_root / "sessions" / session_id / "threads" / f"{thread_id}.json").read_text()
        )
        assert thread["session_id"] == session_id

    def test_thread_file_has_created_at(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        session_id = result["session_id"]
        thread_id = result["thread_id"]
        thread = json.loads(
            (storage_root / "sessions" / session_id / "threads" / f"{thread_id}.json").read_text()
        )
        assert "created_at" in thread
        assert len(thread["created_at"]) > 0


# ---------------------------------------------------------------------------
# session.created event
# ---------------------------------------------------------------------------


class TestSessionCreatedEvent:
    def test_session_created_event_written(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        events = _read_events(storage_root, result["session_id"])
        assert any(e["type"] == "session.created" for e in events)

    def test_event_seq_is_zero(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        events = _read_events(storage_root, result["session_id"])
        ev = next(e for e in events if e["type"] == "session.created")
        assert ev["seq"] == 0

    def test_event_data_has_session_id(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        events = _read_events(storage_root, result["session_id"])
        ev = next(e for e in events if e["type"] == "session.created")
        assert ev["data"]["session_id"] == result["session_id"]

    def test_event_data_has_agent_id(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root), agent_id="agent-ev")
        events = _read_events(storage_root, result["session_id"])
        ev = next(e for e in events if e["type"] == "session.created")
        assert ev["data"]["agent_id"] == "agent-ev"

    def test_event_data_agent_id_none_when_not_provided(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        events = _read_events(storage_root, result["session_id"])
        ev = next(e for e in events if e["type"] == "session.created")
        assert ev["data"]["agent_id"] is None

    def test_event_data_has_thread_id(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        events = _read_events(storage_root, result["session_id"])
        ev = next(e for e in events if e["type"] == "session.created")
        assert ev["data"]["thread_id"] == result["thread_id"]

    def test_event_data_has_created_at(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        events = _read_events(storage_root, result["session_id"])
        ev = next(e for e in events if e["type"] == "session.created")
        assert "created_at" in ev["data"]

    def test_event_thread_id_matches_response(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        events = _read_events(storage_root, result["session_id"])
        ev = next(e for e in events if e["type"] == "session.created")
        assert ev["thread_id"] == result["thread_id"]

    def test_event_data_has_agent_version_id(self, storage_root: Path) -> None:
        _seed_agent(storage_root, "agent-evver", "agentver_evver")
        result = _post_session(_make_client(storage_root), agent_id="agent-evver")
        events = _read_events(storage_root, result["session_id"])
        ev = next(e for e in events if e["type"] == "session.created")
        assert ev["data"]["agent_version_id"] == "agentver_evver"


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class _FailingEventLogWriter(EventLogWriter):
    async def append(self, *args: Any, **kwargs: Any) -> int:
        raise RuntimeError("simulated write failure")


class TestSessionCreateFailure:
    def test_failure_returns_500(self, storage_root: Path) -> None:
        failing = _FailingEventLogWriter()
        result = _post_session(_make_client(storage_root, event_log=failing))
        assert result["_status"] == 500

    def test_failure_error_code(self, storage_root: Path) -> None:
        failing = _FailingEventLogWriter()
        result = _post_session(_make_client(storage_root, event_log=failing))
        assert result["error"]["code"] == "session_create_failed"

    def test_failure_error_message_present(self, storage_root: Path) -> None:
        failing = _FailingEventLogWriter()
        result = _post_session(_make_client(storage_root, event_log=failing))
        assert "message" in result["error"]
        assert len(result["error"]["message"]) > 0

    def test_failure_writes_audit(self, storage_root: Path) -> None:
        failing = _FailingEventLogWriter()
        _post_session(_make_client(storage_root, event_log=failing))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.create.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        failing = _FailingEventLogWriter()
        _post_session(_make_client(storage_root, event_log=failing))
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.create.failed")
        assert record["level"] == "error"

    def test_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        failing = _FailingEventLogWriter()
        _post_session(_make_client(storage_root, event_log=failing))
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.create.failed")
        assert "session_id" in record["detail"]
        assert record["detail"]["session_id"].startswith("sess_")

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        failing = _FailingEventLogWriter()
        _post_session(_make_client(storage_root, event_log=failing))
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.create.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# Router wiring
# ---------------------------------------------------------------------------


class TestSessionRouterWiring:
    def test_sessions_route_exists_with_storage_root_and_event_log(
        self, storage_root: Path
    ) -> None:
        result = _post_session(_make_client(storage_root))
        assert result["_status"] == 201

    def test_no_storage_root_no_route(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/sessions", json={})
        assert resp.status_code == 404

    def test_no_event_log_no_route(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/sessions", json={})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# OTel instrumentation
# ---------------------------------------------------------------------------


class TestSessionOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_session_create_span(self, storage_root: Path) -> None:
        _post_session(_make_client(storage_root))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.create" in span_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.create")
        assert span is not None
        assert span.attributes["session.id"] == result["session_id"]

    def test_span_has_invocation_event(self, storage_root: Path) -> None:
        _post_session(_make_client(storage_root))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.create")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        failing = _FailingEventLogWriter()
        _post_session(_make_client(storage_root, event_log=failing))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.create")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR
