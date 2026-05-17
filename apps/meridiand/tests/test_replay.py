"""
Replay endpoint conformance suite.

Tests cover:
  - POST /v1/x/sessions/{id}/replay returns 200 with run_id, session_id,
    model_call_count, tool_call_count, status fields.
  - model_call_count reflects the number of fake model calls made.
  - tool_call_count reflects the number of fake tool dispatches made.
  - Multi-turn replay (tool calls → second model call) works end-to-end.
  - Missing model_responses.ndjson returns 422 with code "replay_failed".
  - Missing model fixture writes an audit log entry.
  - FakeModelAdapter.call_count increments per call.
  - FakeSandboxAdapter.dispatch_count increments per dispatch.
  - OTel span "replay.run" is emitted on success.
  - OTel span is set to ERROR status on fixture-not-found failure.
  - create_app wires the replay router when storage_root is supplied.
  - create_app omits the replay route when storage_root is None.
  - When expected_events.ndjson is absent, replay succeeds without divergence check.
  - When expected_events.ndjson matches actual events, replay returns 200.
  - When expected_events.ndjson diverges from actual events, replay returns 422 with
    code "replay_failed" and an error message pinned to the first deviating event seq.
  - Divergence writes an audit log entry with first_deviating_seq in detail.
  - OTel span is set to ERROR status on divergence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._replay import FakeModelAdapter, FakeSandboxAdapter, _run_harness
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_model_fixture(fixture_dir: Path, calls: list[list[dict]]) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(events) for events in calls]
    (fixture_dir / "model_responses.ndjson").write_text("\n".join(lines) + "\n")


def _write_tool_fixture(fixture_dir: Path, results: list[dict]) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in results]
    (fixture_dir / "tool_responses.ndjson").write_text("\n".join(lines) + "\n")


def _write_expected_events(fixture_dir: Path, events: list[dict]) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e) for e in events]
    (fixture_dir / "expected_events.ndjson").write_text("\n".join(lines) + "\n")


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


# ---------------------------------------------------------------------------
# Unit tests: FakeModelAdapter
# ---------------------------------------------------------------------------


class TestFakeModelAdapter:
    async def test_yields_events_from_fixture(self, tmp_path: Path) -> None:
        p = tmp_path / "model_responses.ndjson"
        p.write_text(json.dumps(_end_turn_call()) + "\n")
        adapter = FakeModelAdapter(p)
        events = [ev async for ev in adapter.call()]
        assert len(events) == 3

    async def test_call_count_increments(self, tmp_path: Path) -> None:
        p = tmp_path / "model_responses.ndjson"
        p.write_text(json.dumps(_end_turn_call()) + "\n")
        adapter = FakeModelAdapter(p)
        async for _ in adapter.call():
            pass
        assert adapter.call_count == 1

    async def test_missing_fixture_yields_nothing(self, tmp_path: Path) -> None:
        adapter = FakeModelAdapter(tmp_path / "missing.ndjson")
        events = [ev async for ev in adapter.call()]
        assert events == []


# ---------------------------------------------------------------------------
# Unit tests: FakeSandboxAdapter
# ---------------------------------------------------------------------------


class TestFakeSandboxAdapter:
    def test_returns_content_from_fixture(self, tmp_path: Path) -> None:
        p = tmp_path / "tool_responses.ndjson"
        p.write_text(json.dumps({"content": "hello"}) + "\n")
        adapter = FakeSandboxAdapter(p)
        result = adapter.next_result()
        assert result["content"] == "hello"

    def test_dispatch_count_increments(self, tmp_path: Path) -> None:
        p = tmp_path / "tool_responses.ndjson"
        p.write_text(json.dumps({"content": "ok"}) + "\n")
        adapter = FakeSandboxAdapter(p)
        adapter.next_result()
        assert adapter.dispatch_count == 1

    def test_missing_fixture_returns_empty_content(self, tmp_path: Path) -> None:
        adapter = FakeSandboxAdapter(tmp_path / "missing.ndjson")
        result = adapter.next_result()
        assert result == {"content": ""}

    def test_exhausted_returns_empty_content(self, tmp_path: Path) -> None:
        p = tmp_path / "tool_responses.ndjson"
        p.write_text(json.dumps({"content": "only-one"}) + "\n")
        adapter = FakeSandboxAdapter(p)
        adapter.next_result()
        result = adapter.next_result()
        assert result == {"content": ""}


# ---------------------------------------------------------------------------
# Unit tests: _run_harness
# ---------------------------------------------------------------------------


class TestRunHarness:
    async def test_single_end_turn_call(self, tmp_path: Path) -> None:
        mf = tmp_path / "m.ndjson"
        mf.write_text(json.dumps(_end_turn_call()) + "\n")
        model_calls, tool_calls = await _run_harness(
            FakeModelAdapter(mf), FakeSandboxAdapter(tmp_path / "missing.ndjson")
        )
        assert model_calls == 1
        assert tool_calls == 0

    async def test_tool_use_then_end_turn(self, tmp_path: Path) -> None:
        mf = tmp_path / "m.ndjson"
        mf.write_text(json.dumps(_tool_use_call()) + "\n" + json.dumps(_end_turn_call()) + "\n")
        tf = tmp_path / "t.ndjson"
        tf.write_text(json.dumps({"content": "result"}) + "\n")
        model_calls, tool_calls = await _run_harness(FakeModelAdapter(mf), FakeSandboxAdapter(tf))
        assert model_calls == 2
        assert tool_calls == 1

    async def test_multiple_tool_calls_in_one_turn(self, tmp_path: Path) -> None:
        multi_tool_call = [
            {"type": "message_start", "model": "fake", "provider": "fake"},
            {"type": "tool_use_start", "id": "t1", "name": "bash"},
            {"type": "tool_input_delta", "id": "t1", "partial_json": "{}"},
            {"type": "tool_use_start", "id": "t2", "name": "read"},
            {"type": "tool_input_delta", "id": "t2", "partial_json": "{}"},
            {"type": "message_stop", "stop_reason": "tool_use"},
        ]
        mf = tmp_path / "m.ndjson"
        mf.write_text(json.dumps(multi_tool_call) + "\n" + json.dumps(_end_turn_call()) + "\n")
        tf = tmp_path / "t.ndjson"
        tf.write_text(json.dumps({"content": "r1"}) + "\n" + json.dumps({"content": "r2"}) + "\n")
        model_calls, tool_calls = await _run_harness(FakeModelAdapter(mf), FakeSandboxAdapter(tf))
        assert model_calls == 2
        assert tool_calls == 2


# ---------------------------------------------------------------------------
# Integration tests: POST /v1/x/sessions/{id}/replay
# ---------------------------------------------------------------------------


class TestReplayEndpoint:
    def _fixture_dir(self, storage_root: Path, session_id: str) -> Path:
        d = storage_root / "fixtures" / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_returns_200_on_success(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "sess1")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/sess1/replay")
        assert resp.status_code == 200

    def test_response_has_run_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "sess2")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sess2/replay").json()
        assert "run_id" in body
        assert isinstance(body["run_id"], str)
        assert len(body["run_id"]) > 0

    def test_response_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "sess3")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sess3/replay").json()
        assert body["session_id"] == "sess3"

    def test_response_status_completed(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "sess4")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sess4/replay").json()
        assert body["status"] == "completed"

    def test_model_call_count_reflects_fixture(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "sess5")
        _write_model_fixture(fd, [_tool_use_call(), _end_turn_call()])
        _write_tool_fixture(fd, [{"content": "ok"}])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sess5/replay").json()
        assert body["model_call_count"] == 2

    def test_tool_call_count_reflects_fixture(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "sess6")
        _write_model_fixture(fd, [_tool_use_call(), _end_turn_call()])
        _write_tool_fixture(fd, [{"content": "ok"}])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sess6/replay").json()
        assert body["tool_call_count"] == 1

    def test_missing_fixture_returns_422(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/unknown-session/replay")
        assert resp.status_code == 422

    def test_missing_fixture_error_code(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/no-fixture/replay").json()
        assert body["error"]["code"] == "replay_failed"

    def test_missing_fixture_writes_audit_log(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/no-audit-sess/replay")
        audit_path = storage_root / "audit.ndjson"
        assert audit_path.exists()
        records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        assert any(r.get("event") == "replay.run.failed" for r in records)

    def test_missing_fixture_audit_detail_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/audit-detail-sess/replay")
        audit_path = storage_root / "audit.ndjson"
        records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        replay_record = next(r for r in records if r.get("event") == "replay.run.failed")
        assert replay_record["detail"]["session_id"] == "audit-detail-sess"

    def test_no_storage_root_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/any/replay")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Divergence detection tests
# ---------------------------------------------------------------------------


class TestReplayDivergence:
    def _fixture_dir(self, storage_root: Path, session_id: str) -> Path:
        d = storage_root / "fixtures" / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_no_expected_events_file_returns_200(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "div-sess0")
        _write_model_fixture(fd, [_end_turn_call()])
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/div-sess0/replay")
        assert resp.status_code == 200

    def test_matching_expected_events_returns_200(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "div-sess1")
        events = _end_turn_call()
        _write_model_fixture(fd, [events])
        _write_expected_events(fd, events)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/div-sess1/replay")
        assert resp.status_code == 200

    def test_diverging_expected_events_returns_422(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "div-sess2")
        _write_model_fixture(fd, [_end_turn_call()])
        _write_expected_events(fd, [{"type": "wrong_event"}])
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/div-sess2/replay")
        assert resp.status_code == 422

    def test_divergence_error_code_is_replay_failed(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "div-sess3")
        _write_model_fixture(fd, [_end_turn_call()])
        _write_expected_events(fd, [{"type": "wrong_event"}])
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/div-sess3/replay").json()
        assert body["error"]["code"] == "replay_failed"

    def test_divergence_error_message_contains_seq(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "div-sess4")
        model_events = _end_turn_call()
        _write_model_fixture(fd, [model_events])
        # Diverge at index 1 (text_delta vs different)
        diverged = [
            model_events[0],
            {"type": "different_delta"},
            model_events[2],
        ]
        _write_expected_events(fd, diverged)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/div-sess4/replay").json()
        assert "seq 1" in body["error"]["message"]

    def test_divergence_writes_audit_log(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "div-sess5")
        _write_model_fixture(fd, [_end_turn_call()])
        _write_expected_events(fd, [{"type": "wrong_event"}])
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/div-sess5/replay")
        audit_path = storage_root / "audit.ndjson"
        assert audit_path.exists()
        records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        assert any(r.get("event") == "replay.run.failed" for r in records)

    def test_divergence_audit_detail_has_first_deviating_seq(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "div-sess6")
        model_events = _end_turn_call()
        _write_model_fixture(fd, [model_events])
        diverged = [model_events[0], {"type": "nope"}, model_events[2]]
        _write_expected_events(fd, diverged)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/div-sess6/replay")
        audit_path = storage_root / "audit.ndjson"
        records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        record = next(r for r in records if r.get("event") == "replay.run.failed")
        assert record["detail"]["first_deviating_seq"] == 1

    def test_divergence_audit_detail_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "div-sess7")
        _write_model_fixture(fd, [_end_turn_call()])
        _write_expected_events(fd, [{"type": "wrong"}])
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/div-sess7/replay")
        audit_path = storage_root / "audit.ndjson"
        records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        record = next(r for r in records if r.get("event") == "replay.run.failed")
        assert record["detail"]["session_id"] == "div-sess7"

    def test_matching_events_with_tool_use_returns_200(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        fd = self._fixture_dir(storage_root, "div-sess8")
        _write_model_fixture(fd, [_tool_use_call("tu_1"), _end_turn_call()])
        _write_tool_fixture(fd, [{"content": "result"}])
        expected = (
            list(_tool_use_call("tu_1"))
            + [{"type": "tool_result", "tool_id": "tu_1", "content": "result"}]
            + list(_end_turn_call())
        )
        _write_expected_events(fd, expected)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/div-sess8/replay")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# OTel span tests
# ---------------------------------------------------------------------------

# OTel provider is registered once in conftest.py; _otel_exporter is imported from there.


class TestReplayOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _make_client(self, storage_root: Path) -> TestClient:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_replay_run_span(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        fd = storage_root / "fixtures" / "otel-sess"
        _write_model_fixture(fd, [_end_turn_call()])
        client.post("/v1/x/sessions/otel-sess/replay")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "replay.run" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/no-fixture-otel/replay")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        replay_span = spans.get("replay.run")
        assert replay_span is not None
        assert replay_span.status.status_code == StatusCode.ERROR

    def test_divergence_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._make_client(storage_root)
        fd = storage_root / "fixtures" / "otel-div-sess"
        _write_model_fixture(fd, [_end_turn_call()])
        _write_expected_events(fd, [{"type": "wrong"}])
        client.post("/v1/x/sessions/otel-div-sess/replay")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        replay_span = spans.get("replay.run")
        assert replay_span is not None
        assert replay_span.status.status_code == StatusCode.ERROR
