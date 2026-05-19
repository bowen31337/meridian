"""
System integration test: full POST /v1/sessions flow.

Tests cover:
  - POST /v1/sessions → wake → model call → tool dispatch → tool result → end_turn → idle
    flow with FakeModel and FakeSandbox.
  - Returns 201 with session_id, status, phase, model_call_count, tool_call_count fields.
  - session_id starts with "sess_".
  - status is "idle" after the harness runs to end_turn.
  - phase is "idle" after the harness runs to end_turn.
  - model_call_count reflects the number of fake model calls made.
  - tool_call_count reflects the number of fake tool dispatches made.
  - Session manifest is created at sessions/{session_id}/manifest.json.
  - Manifest contains session_id, agent_id, status, created_at fields.
  - Single end_turn call: model_call_count=1, tool_call_count=0.
  - Full tool dispatch → tool result → continue → end_turn: model_call_count=2,
    tool_call_count=1.
  - agent_id propagated into session manifest when supplied in request body.
  - agent_id is None in manifest when not supplied.
  - Missing model fixture returns 422 with code "session_run_failed".
  - Missing fixture writes an audit log entry with event "session.run.failed".
  - Audit entry level is "error" on failure.
  - Audit detail includes session_id and fixture_session_id on failure.
  - Audit detail includes message on failure.
  - On failure, error message is surfaced in response body (error.message present).
  - OTel span "session.run" is emitted on success.
  - OTel span has session.id attribute.
  - OTel span has session.fixture_session_id attribute.
  - OTel span carries a structured invocation event on each call.
  - OTel span is set to ERROR status on failure.
  - create_app wires the sessions router when storage_root is supplied.
  - create_app omits the sessions route when storage_root is None.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _write_model_fixture(fixture_dir: Path, calls: list[list[dict[str, Any]]]) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(events) for events in calls]
    (fixture_dir / "model_responses.ndjson").write_text("\n".join(lines) + "\n")


def _write_tool_fixture(fixture_dir: Path, results: list[dict[str, Any]]) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in results]
    (fixture_dir / "tool_responses.ndjson").write_text("\n".join(lines) + "\n")


def _end_turn_call() -> list[dict[str, Any]]:
    return [
        {"type": "message_start", "model": "fake", "provider": "fake"},
        {"type": "text_delta", "text": "Done."},
        {"type": "message_stop", "stop_reason": "end_turn"},
    ]


def _tool_use_call(tool_id: str = "tu_1", tool_name: str = "bash") -> list[dict[str, Any]]:
    return [
        {"type": "message_start", "model": "fake", "provider": "fake"},
        {"type": "tool_use_start", "id": tool_id, "name": tool_name},
        {"type": "tool_input_delta", "id": tool_id, "partial_json": '{"cmd":"ls"}'},
        {"type": "message_stop", "stop_reason": "tool_use"},
    ]


def _post_session(
    client: TestClient,
    fixture_session_id: str,
    agent_id: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"fixture_session_id": fixture_session_id}
    if agent_id is not None:
        body["agent_id"] = agent_id
    resp = client.post("/v1/sessions", json=body)
    return resp.json() | {"_status": resp.status_code}


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Basic response shape — single end_turn call
# ---------------------------------------------------------------------------


class TestSessionCreateSuccess:
    def _setup(self, storage_root: Path, fixture_id: str = "fix-basic") -> None:
        _write_model_fixture(
            storage_root / "fixtures" / fixture_id,
            [_end_turn_call()],
        )

    def test_returns_201(self, storage_root: Path) -> None:
        self._setup(storage_root)
        result = _post_session(_make_client(storage_root), "fix-basic")
        assert result["_status"] == 201

    def test_session_id_in_response(self, storage_root: Path) -> None:
        self._setup(storage_root)
        result = _post_session(_make_client(storage_root), "fix-basic")
        assert "session_id" in result

    def test_session_id_starts_with_sess(self, storage_root: Path) -> None:
        self._setup(storage_root)
        result = _post_session(_make_client(storage_root), "fix-basic")
        assert result["session_id"].startswith("sess_")

    def test_status_is_idle(self, storage_root: Path) -> None:
        self._setup(storage_root)
        result = _post_session(_make_client(storage_root), "fix-basic")
        assert result["status"] == "idle"

    def test_phase_is_idle(self, storage_root: Path) -> None:
        self._setup(storage_root)
        result = _post_session(_make_client(storage_root), "fix-basic")
        assert result["phase"] == "idle"

    def test_model_call_count_is_one(self, storage_root: Path) -> None:
        self._setup(storage_root)
        result = _post_session(_make_client(storage_root), "fix-basic")
        assert result["model_call_count"] == 1

    def test_tool_call_count_is_zero(self, storage_root: Path) -> None:
        self._setup(storage_root)
        result = _post_session(_make_client(storage_root), "fix-basic")
        assert result["tool_call_count"] == 0


# ---------------------------------------------------------------------------
# Full flow: tool dispatch → tool result → end_turn
# ---------------------------------------------------------------------------


class TestSessionFullFlow:
    def _setup(self, storage_root: Path, fixture_id: str = "fix-full") -> None:
        fixture_dir = storage_root / "fixtures" / fixture_id
        _write_model_fixture(fixture_dir, [_tool_use_call(), _end_turn_call()])
        _write_tool_fixture(fixture_dir, [{"content": "file1.txt\nfile2.txt"}])

    def test_returns_201_on_full_flow(self, storage_root: Path) -> None:
        self._setup(storage_root)
        result = _post_session(_make_client(storage_root), "fix-full")
        assert result["_status"] == 201

    def test_model_call_count_is_two(self, storage_root: Path) -> None:
        self._setup(storage_root)
        result = _post_session(_make_client(storage_root), "fix-full")
        assert result["model_call_count"] == 2

    def test_tool_call_count_is_one(self, storage_root: Path) -> None:
        self._setup(storage_root)
        result = _post_session(_make_client(storage_root), "fix-full")
        assert result["tool_call_count"] == 1

    def test_phase_is_idle_after_full_flow(self, storage_root: Path) -> None:
        self._setup(storage_root)
        result = _post_session(_make_client(storage_root), "fix-full")
        assert result["phase"] == "idle"


# ---------------------------------------------------------------------------
# Session manifest creation
# ---------------------------------------------------------------------------


class TestSessionManifest:
    def _setup(self, storage_root: Path, fixture_id: str) -> None:
        _write_model_fixture(
            storage_root / "fixtures" / fixture_id,
            [_end_turn_call()],
        )

    def test_manifest_file_created(self, storage_root: Path) -> None:
        self._setup(storage_root, "fix-manifest")
        result = _post_session(_make_client(storage_root), "fix-manifest")
        session_id = result["session_id"]
        manifest_path = storage_root / "sessions" / session_id / "manifest.json"
        assert manifest_path.exists()

    def test_manifest_has_session_id(self, storage_root: Path) -> None:
        self._setup(storage_root, "fix-manifest-sid")
        result = _post_session(_make_client(storage_root), "fix-manifest-sid")
        session_id = result["session_id"]
        manifest = json.loads(
            (storage_root / "sessions" / session_id / "manifest.json").read_text()
        )
        assert manifest["session_id"] == session_id

    def test_manifest_has_status_active(self, storage_root: Path) -> None:
        self._setup(storage_root, "fix-manifest-status")
        result = _post_session(_make_client(storage_root), "fix-manifest-status")
        session_id = result["session_id"]
        manifest = json.loads(
            (storage_root / "sessions" / session_id / "manifest.json").read_text()
        )
        assert manifest["status"] == "active"

    def test_manifest_has_created_at(self, storage_root: Path) -> None:
        self._setup(storage_root, "fix-manifest-ts")
        result = _post_session(_make_client(storage_root), "fix-manifest-ts")
        session_id = result["session_id"]
        manifest = json.loads(
            (storage_root / "sessions" / session_id / "manifest.json").read_text()
        )
        assert "created_at" in manifest
        assert len(manifest["created_at"]) > 0

    def test_manifest_agent_id_set_when_provided(self, storage_root: Path) -> None:
        self._setup(storage_root, "fix-manifest-agent")
        result = _post_session(
            _make_client(storage_root), "fix-manifest-agent", agent_id="agent-42"
        )
        session_id = result["session_id"]
        manifest = json.loads(
            (storage_root / "sessions" / session_id / "manifest.json").read_text()
        )
        assert manifest["agent_id"] == "agent-42"

    def test_manifest_agent_id_none_when_not_provided(self, storage_root: Path) -> None:
        self._setup(storage_root, "fix-manifest-noagent")
        result = _post_session(_make_client(storage_root), "fix-manifest-noagent")
        session_id = result["session_id"]
        manifest = json.loads(
            (storage_root / "sessions" / session_id / "manifest.json").read_text()
        )
        assert manifest["agent_id"] is None


# ---------------------------------------------------------------------------
# Failure: missing fixture
# ---------------------------------------------------------------------------


class TestSessionMissingFixture:
    def test_missing_fixture_returns_422(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root), "no-such-fixture")
        assert result["_status"] == 422

    def test_missing_fixture_error_code(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root), "no-such-fixture-2")
        assert result["error"]["code"] == "session_run_failed"

    def test_missing_fixture_error_message_present(self, storage_root: Path) -> None:
        result = _post_session(_make_client(storage_root), "no-such-fixture-3")
        assert "message" in result["error"]
        assert len(result["error"]["message"]) > 0

    def test_missing_fixture_writes_audit(self, storage_root: Path) -> None:
        _post_session(_make_client(storage_root), "no-such-fixture-4")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.run.failed" for r in records)

    def test_missing_fixture_audit_level_is_error(self, storage_root: Path) -> None:
        _post_session(_make_client(storage_root), "no-such-fixture-5")
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.run.failed")
        assert record["level"] == "error"

    def test_missing_fixture_audit_detail_has_session_id(self, storage_root: Path) -> None:
        _post_session(_make_client(storage_root), "no-such-fixture-6")
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.run.failed")
        assert "session_id" in record["detail"]
        assert record["detail"]["session_id"].startswith("sess_")

    def test_missing_fixture_audit_detail_has_fixture_session_id(
        self, storage_root: Path
    ) -> None:
        _post_session(_make_client(storage_root), "no-such-fixture-7")
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.run.failed")
        assert record["detail"]["fixture_session_id"] == "no-such-fixture-7"

    def test_missing_fixture_audit_detail_has_message(self, storage_root: Path) -> None:
        _post_session(_make_client(storage_root), "no-such-fixture-8")
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.run.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# Router wiring
# ---------------------------------------------------------------------------


class TestSessionRouterWiring:
    def test_sessions_route_exists_with_storage_root(self, storage_root: Path) -> None:
        _write_model_fixture(
            storage_root / "fixtures" / "wire-fix",
            [_end_turn_call()],
        )
        result = _post_session(_make_client(storage_root), "wire-fix")
        assert result["_status"] == 201

    def test_no_storage_root_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/sessions", json={"fixture_session_id": "any"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# OTel instrumentation
# ---------------------------------------------------------------------------


class TestSessionOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _setup(self, storage_root: Path, fixture_id: str) -> None:
        _write_model_fixture(
            storage_root / "fixtures" / fixture_id,
            [_end_turn_call()],
        )

    def test_success_emits_session_run_span(self, storage_root: Path) -> None:
        self._setup(storage_root, "otel-fix-1")
        _post_session(_make_client(storage_root), "otel-fix-1")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.run" in span_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        self._setup(storage_root, "otel-fix-2")
        result = _post_session(_make_client(storage_root), "otel-fix-2")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.run")
        assert span is not None
        assert span.attributes["session.id"] == result["session_id"]

    def test_span_has_fixture_session_id_attribute(self, storage_root: Path) -> None:
        self._setup(storage_root, "otel-fix-3")
        _post_session(_make_client(storage_root), "otel-fix-3")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.run")
        assert span is not None
        assert span.attributes["session.fixture_session_id"] == "otel-fix-3"

    def test_span_has_invocation_event(self, storage_root: Path) -> None:
        self._setup(storage_root, "otel-fix-4")
        _post_session(_make_client(storage_root), "otel-fix-4")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.run")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        _post_session(_make_client(storage_root), "otel-missing-fix")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.run")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR
