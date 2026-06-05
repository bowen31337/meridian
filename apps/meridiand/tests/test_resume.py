"""
Resume endpoint conformance suite.

Tests cover:
  - POST /v1/x/sessions/{id}/resume returns 200 with session_id, status, phase,
    checkpoint_seq, tool_calls_dispatched, model_call_count, tool_call_count fields.
  - Loads latest checkpoint from $STORAGE_ROOT/checkpoints/{id}/latest.json.
  - Falls back to replay log ($STORAGE_ROOT/fixtures/{id}/model_responses.ndjson)
    when no checkpoint exists.
  - checkpoint_seq is populated from the checkpoint when one exists.
  - checkpoint_seq is None when falling back to replay log.
  - Pending tool calls from checkpoint are re-dispatched; tool_calls_dispatched reflects count.
  - Phase is "waiting_for_model" when pending tool calls exist, then transitions to "idle"
    after harness runs.
  - Phase is "idle" after harness runs with no pending tool calls.
  - Missing checkpoint and missing replay log returns 422 with code "resume_failed".
  - Missing both writes an audit log entry with event "session.resume.failed".
  - Audit log detail includes session_id.
  - OTel span "session.resume" is emitted on success.
  - OTel span is set to ERROR status on failure.
  - create_app wires the resume router when storage_root is supplied.
  - create_app omits the resume route when storage_root is None.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_checkpoint(checkpoint_dir: Path, data: dict) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "latest.json").write_text(json.dumps(data))


def _write_model_fixture(fixture_dir: Path, calls: list[list[dict]]) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(events) for events in calls]
    (fixture_dir / "model_responses.ndjson").write_text("\n".join(lines) + "\n")


def _write_tool_fixture(fixture_dir: Path, results: list[dict]) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in results]
    (fixture_dir / "tool_responses.ndjson").write_text("\n".join(lines) + "\n")


def _end_turn_call() -> list[dict]:
    return [
        {"type": "message_start", "model": "fake", "provider": "fake"},
        {"type": "text_delta", "text": "Hello!"},
        {"type": "message_stop", "stop_reason": "end_turn"},
    ]


def _tool_use_call(tool_id: str = "tu_1", tool_name: str = "bash") -> list[dict]:
    return [
        {"type": "message_start", "model": "fake", "provider": "fake"},
        {"type": "tool_use_start", "id": tool_id, "name": tool_name},
        {"type": "tool_input_delta", "id": tool_id, "partial_json": '{"cmd":"ls"}'},
        {"type": "message_stop", "stop_reason": "tool_use"},
    ]


def _make_client(storage_root: Path, audit_log: FileAuditLog) -> TestClient:
    app = create_app(audit_log, storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _make_checkpoint(
    *,
    seq: int = 1,
    phase: str = "thinking",
    pending_tool_calls: list | None = None,
) -> dict:
    return {
        "seq": seq,
        "phase": phase,
        "pending_tool_calls": pending_tool_calls or [],
        "message_tail": [{"role": "assistant", "content": "hello"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "taken_at": "2024-01-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Integration tests: POST /v1/x/sessions/{id}/resume — checkpoint path
# ---------------------------------------------------------------------------


class TestResumeEndpointFromCheckpoint:
    def _checkpoint_dir(self, storage_root: Path, session_id: str) -> Path:
        d = storage_root / "checkpoints" / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _fixture_dir(self, storage_root: Path, session_id: str) -> Path:
        d = storage_root / "fixtures" / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_returns_200_on_success(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        cd = self._checkpoint_dir(storage_root, "ck-sess1")
        _write_checkpoint(cd, _make_checkpoint(seq=3))
        fd = self._fixture_dir(storage_root, "ck-sess1")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/ck-sess1/resume")
        assert resp.status_code == 200

    def test_response_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        cd = self._checkpoint_dir(storage_root, "ck-sess2")
        _write_checkpoint(cd, _make_checkpoint(seq=1))
        fd = self._fixture_dir(storage_root, "ck-sess2")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/ck-sess2/resume").json()
        assert body["session_id"] == "ck-sess2"

    def test_response_status_is_resumed(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        cd = self._checkpoint_dir(storage_root, "ck-sess3")
        _write_checkpoint(cd, _make_checkpoint())
        fd = self._fixture_dir(storage_root, "ck-sess3")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/ck-sess3/resume").json()
        assert body["status"] == "resumed"

    def test_checkpoint_seq_populated(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        cd = self._checkpoint_dir(storage_root, "ck-sess4")
        _write_checkpoint(cd, _make_checkpoint(seq=7))
        fd = self._fixture_dir(storage_root, "ck-sess4")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/ck-sess4/resume").json()
        assert body["checkpoint_seq"] == 7

    def test_phase_idle_after_harness_runs(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        cd = self._checkpoint_dir(storage_root, "ck-sess5")
        _write_checkpoint(cd, _make_checkpoint(phase="thinking"))
        fd = self._fixture_dir(storage_root, "ck-sess5")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/ck-sess5/resume").json()
        assert body["phase"] == "idle"

    def test_model_call_count_reflects_harness(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        cd = self._checkpoint_dir(storage_root, "ck-sess6")
        _write_checkpoint(cd, _make_checkpoint())
        fd = self._fixture_dir(storage_root, "ck-sess6")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/ck-sess6/resume").json()
        assert body["model_call_count"] == 1

    def test_pending_tool_calls_dispatched(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        cd = self._checkpoint_dir(storage_root, "ck-sess7")
        calls = [{"id": "tc1", "name": "bash", "input": {"cmd": "ls"}}]
        _write_checkpoint(cd, _make_checkpoint(pending_tool_calls=calls))
        fd = self._fixture_dir(storage_root, "ck-sess7")
        _write_model_fixture(fd, [_end_turn_call()])
        _write_tool_fixture(fd, [{"content": "result"}])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/ck-sess7/resume").json()
        assert body["tool_calls_dispatched"] == 1

    def test_tool_call_count_includes_dispatched(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        cd = self._checkpoint_dir(storage_root, "ck-sess8")
        calls = [{"id": "tc1", "name": "bash", "input": {"cmd": "ls"}}]
        _write_checkpoint(cd, _make_checkpoint(pending_tool_calls=calls))
        fd = self._fixture_dir(storage_root, "ck-sess8")
        _write_model_fixture(fd, [_end_turn_call()])
        _write_tool_fixture(fd, [{"content": "result"}])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/ck-sess8/resume").json()
        assert body["tool_call_count"] >= 1

    def test_no_pending_tool_calls_dispatched_zero(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        cd = self._checkpoint_dir(storage_root, "ck-sess9")
        _write_checkpoint(cd, _make_checkpoint(pending_tool_calls=[]))
        fd = self._fixture_dir(storage_root, "ck-sess9")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/ck-sess9/resume").json()
        assert body["tool_calls_dispatched"] == 0


# ---------------------------------------------------------------------------
# Integration tests: POST /v1/x/sessions/{id}/resume — fallback replay path
# ---------------------------------------------------------------------------


class TestResumeEndpointFallback:
    def _fixture_dir(self, storage_root: Path, session_id: str) -> Path:
        d = storage_root / "fixtures" / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_fallback_returns_200(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "fb-sess1")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/fb-sess1/resume")
        assert resp.status_code == 200

    def test_fallback_checkpoint_seq_is_none(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "fb-sess2")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/fb-sess2/resume").json()
        assert body["checkpoint_seq"] is None

    def test_fallback_status_is_resumed(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "fb-sess3")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/fb-sess3/resume").json()
        assert body["status"] == "resumed"

    def test_fallback_phase_idle_after_harness(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "fb-sess4")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/fb-sess4/resume").json()
        assert body["phase"] == "idle"

    def test_fallback_no_dispatched_tool_calls(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "fb-sess5")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/fb-sess5/resume").json()
        assert body["tool_calls_dispatched"] == 0


# ---------------------------------------------------------------------------
# Integration tests: failure — no checkpoint and no replay log
# ---------------------------------------------------------------------------


class TestResumeEndpointFailure:
    def test_missing_both_returns_422(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/no-sess/resume")
        assert resp.status_code == 422

    def test_missing_both_error_code_is_resume_failed(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/no-sess2/resume").json()
        assert body["error"]["code"] == "resume_failed"

    def test_missing_both_writes_audit_log(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/no-audit-sess/resume")
        audit_path = storage_root / "audit.ndjson"
        assert audit_path.exists()
        records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        assert any(r.get("event") == "session.resume.failed" for r in records)

    def test_missing_both_audit_detail_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/no-detail-sess/resume")
        audit_path = storage_root / "audit.ndjson"
        records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        record = next(r for r in records if r.get("event") == "session.resume.failed")
        assert record["detail"]["session_id"] == "no-detail-sess"

    def test_no_storage_root_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/any/resume")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# OTel span tests
# ---------------------------------------------------------------------------


class TestResumeOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _make_client(self, storage_root: Path) -> TestClient:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_session_resume_span(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        fd = storage_root / "fixtures" / "otel-resume-sess"
        _write_model_fixture(fd, [_end_turn_call()])
        client.post("/v1/x/sessions/otel-resume-sess/resume")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.resume" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/no-fixture-otel-resume/resume")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        resume_span = spans.get("session.resume")
        assert resume_span is not None
        assert resume_span.status.status_code == StatusCode.ERROR
