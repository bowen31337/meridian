"""
CI regression suite conformance tests.

Tests cover:
  - POST /v1/x/ci/regression-run returns 200 with run_id, status, session_count, sessions
    when no fixtures directory exists.
  - POST /v1/x/ci/regression-run returns 200 with empty sessions when fixtures dir is empty.
  - Sessions without expected_events.ndjson are silently skipped.
  - A session whose events match the baseline returns status="passed" and 200.
  - A session whose events deviate from the baseline returns 422 with code="regression_failed".
  - first_deviating_seq points at the correct 0-based event index.
  - expected_event and actual_event in first_failure detail reflect the diverging events.
  - Multiple passing sessions all appear in sessions list with status="passed".
  - When one of several sessions fails, the first (alphabetically) failing session drives
    first_failure and the overall 422 response.
  - Audit log entry with event "ci.regression.run.failed" is written on failure.
  - Audit log detail contains run_id and first_failure.
  - OTel span "ci.regression.run" is emitted on success.
  - OTel span is set to ERROR status on regression failure.
  - model_calls and tool_calls are reported per session.
  - Tool-dispatch synthetic events (type="tool_result") are compared against baseline.
  - create_app wires the regression router when storage_root is supplied.
  - create_app omits the regression route when storage_root is None.
  - _find_divergence returns None for equal sequences.
  - _find_divergence returns (0, ...) when first events differ.
  - _find_divergence returns correct seq when later event differs.
  - _find_divergence detects length mismatch (expected longer).
  - _find_divergence detects length mismatch (actual longer).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._replay import (
    FakeModelAdapter,
    FakeSandboxAdapter,
    _find_divergence,
    _run_harness_capturing,
)

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
# Unit tests: _find_divergence
# ---------------------------------------------------------------------------


class TestFindDivergence:
    def test_equal_sequences_returns_none(self) -> None:
        seq = [{"type": "a"}, {"type": "b"}]
        assert _find_divergence(seq, seq.copy()) is None

    def test_empty_sequences_returns_none(self) -> None:
        assert _find_divergence([], []) is None

    def test_first_event_differs(self) -> None:
        result = _find_divergence([{"type": "a"}], [{"type": "b"}])
        assert result is not None
        seq, exp, act = result
        assert seq == 0
        assert exp == {"type": "a"}
        assert act == {"type": "b"}

    def test_later_event_differs(self) -> None:
        expected = [{"type": "a"}, {"type": "b"}, {"type": "c"}]
        actual = [{"type": "a"}, {"type": "b"}, {"type": "X"}]
        result = _find_divergence(expected, actual)
        assert result is not None
        seq, exp, act = result
        assert seq == 2
        assert exp == {"type": "c"}
        assert act == {"type": "X"}

    def test_expected_longer_returns_seq_at_end(self) -> None:
        expected = [{"type": "a"}, {"type": "b"}]
        actual = [{"type": "a"}]
        result = _find_divergence(expected, actual)
        assert result is not None
        seq, exp, act = result
        assert seq == 1
        assert exp == {"type": "b"}
        assert act is None

    def test_actual_longer_returns_seq_at_end(self) -> None:
        expected = [{"type": "a"}]
        actual = [{"type": "a"}, {"type": "extra"}]
        result = _find_divergence(expected, actual)
        assert result is not None
        seq, exp, act = result
        assert seq == 1
        assert exp is None
        assert act == {"type": "extra"}


# ---------------------------------------------------------------------------
# Unit tests: _run_harness_capturing
# ---------------------------------------------------------------------------


class TestRunHarnessCapturing:
    async def test_end_turn_captures_all_model_events(self, tmp_path: Path) -> None:
        mf = tmp_path / "m.ndjson"
        events = _end_turn_call()
        mf.write_text(json.dumps(events) + "\n")
        model_calls, tool_calls, captured = await _run_harness_capturing(
            FakeModelAdapter(mf), FakeSandboxAdapter(tmp_path / "missing.ndjson")
        )
        assert model_calls == 1
        assert tool_calls == 0
        assert captured == events

    async def test_tool_use_appends_tool_result_event(self, tmp_path: Path) -> None:
        mf = tmp_path / "m.ndjson"
        mf.write_text(
            json.dumps(_tool_use_call("tu_1")) + "\n" + json.dumps(_end_turn_call()) + "\n"
        )
        tf = tmp_path / "t.ndjson"
        tf.write_text(json.dumps({"content": "ok"}) + "\n")
        _, tool_calls, captured = await _run_harness_capturing(
            FakeModelAdapter(mf), FakeSandboxAdapter(tf)
        )
        assert tool_calls == 1
        tool_result_events = [e for e in captured if e.get("type") == "tool_result"]
        assert len(tool_result_events) == 1
        assert tool_result_events[0]["tool_id"] == "tu_1"
        assert tool_result_events[0]["content"] == "ok"

    async def test_model_calls_count_multi_turn(self, tmp_path: Path) -> None:
        mf = tmp_path / "m.ndjson"
        mf.write_text(json.dumps(_tool_use_call()) + "\n" + json.dumps(_end_turn_call()) + "\n")
        tf = tmp_path / "t.ndjson"
        tf.write_text(json.dumps({"content": "res"}) + "\n")
        model_calls, _, _ = await _run_harness_capturing(
            FakeModelAdapter(mf), FakeSandboxAdapter(tf)
        )
        assert model_calls == 2


# ---------------------------------------------------------------------------
# Integration tests: POST /v1/x/ci/regression-run
# ---------------------------------------------------------------------------


class TestCIRegressionEndpoint:
    def test_no_fixtures_dir_returns_200(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/ci/regression-run")
        assert resp.status_code == 200

    def test_no_fixtures_dir_returns_empty_sessions(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/ci/regression-run").json()
        assert body["session_count"] == 0
        assert body["sessions"] == []

    def test_response_has_run_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/ci/regression-run").json()
        assert "run_id" in body
        assert isinstance(body["run_id"], str)
        assert len(body["run_id"]) > 0

    def test_response_status_passed_when_no_sessions(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/ci/regression-run").json()
        assert body["status"] == "passed"

    def test_session_without_expected_events_is_skipped(self, storage_root: Path) -> None:
        session_dir = storage_root / "fixtures" / "sess-no-baseline"
        _write_model_fixture(session_dir, [_end_turn_call()])
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/ci/regression-run").json()
        assert body["session_count"] == 0
        assert body["sessions"] == []

    def test_passing_session_returns_200(self, storage_root: Path) -> None:
        session_dir = storage_root / "fixtures" / "sess1"
        events = _end_turn_call()
        _write_model_fixture(session_dir, [events])
        _write_expected_events(session_dir, events)
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/ci/regression-run")
        assert resp.status_code == 200

    def test_passing_session_has_status_passed(self, storage_root: Path) -> None:
        session_dir = storage_root / "fixtures" / "sess2"
        events = _end_turn_call()
        _write_model_fixture(session_dir, [events])
        _write_expected_events(session_dir, events)
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/ci/regression-run").json()
        assert body["sessions"][0]["status"] == "passed"
        assert body["sessions"][0]["session_id"] == "sess2"

    def test_passing_session_reports_model_calls(self, storage_root: Path) -> None:
        session_dir = storage_root / "fixtures" / "sess3"
        events = _end_turn_call()
        _write_model_fixture(session_dir, [events])
        _write_expected_events(session_dir, events)
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/ci/regression-run").json()
        assert body["sessions"][0]["model_calls"] == 1
        assert body["sessions"][0]["tool_calls"] == 0

    def test_diverging_session_returns_422(self, storage_root: Path) -> None:
        session_dir = storage_root / "fixtures" / "bad-sess"
        _write_model_fixture(session_dir, [_end_turn_call()])
        _write_expected_events(session_dir, [{"type": "different_event"}])
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/ci/regression-run")
        assert resp.status_code == 422

    def test_diverging_session_error_code(self, storage_root: Path) -> None:
        session_dir = storage_root / "fixtures" / "bad-sess2"
        _write_model_fixture(session_dir, [_end_turn_call()])
        _write_expected_events(session_dir, [{"type": "different_event"}])
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/ci/regression-run").json()
        assert body["error"]["code"] == "regression_failed"

    def test_first_deviating_seq_is_correct(self, storage_root: Path) -> None:
        session_dir = storage_root / "fixtures" / "seq-sess"
        model_events = _end_turn_call()
        _write_model_fixture(session_dir, [model_events])
        # Expected diverges at index 1 (text_delta → different)
        diverged = [
            model_events[0],
            {"type": "different_delta", "text": "nope"},
            model_events[2],
        ]
        _write_expected_events(session_dir, diverged)
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/ci/regression-run").json()
        assert body["error"]["message"].find("seq 1") != -1

    def test_failure_writes_audit_log(self, storage_root: Path) -> None:
        session_dir = storage_root / "fixtures" / "audit-sess"
        _write_model_fixture(session_dir, [_end_turn_call()])
        _write_expected_events(session_dir, [{"type": "wrong"}])
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/ci/regression-run")
        audit_path = storage_root / "audit.ndjson"
        assert audit_path.exists()
        records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        assert any(r.get("event") == "ci.regression.run.failed" for r in records)

    def test_audit_log_detail_has_run_id(self, storage_root: Path) -> None:
        session_dir = storage_root / "fixtures" / "audit-rid-sess"
        _write_model_fixture(session_dir, [_end_turn_call()])
        _write_expected_events(session_dir, [{"type": "wrong"}])
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/ci/regression-run")
        audit_path = storage_root / "audit.ndjson"
        records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        record = next(r for r in records if r.get("event") == "ci.regression.run.failed")
        assert "run_id" in record["detail"]
        assert isinstance(record["detail"]["run_id"], str)

    def test_audit_log_detail_has_first_failure(self, storage_root: Path) -> None:
        session_dir = storage_root / "fixtures" / "audit-ff-sess"
        _write_model_fixture(session_dir, [_end_turn_call()])
        _write_expected_events(session_dir, [{"type": "wrong"}])
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/ci/regression-run")
        audit_path = storage_root / "audit.ndjson"
        records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        record = next(r for r in records if r.get("event") == "ci.regression.run.failed")
        assert "first_failure" in record["detail"]
        assert record["detail"]["first_failure"]["session_id"] == "audit-ff-sess"

    def test_multiple_passing_sessions(self, storage_root: Path) -> None:
        for sid in ["sess-a", "sess-b", "sess-c"]:
            d = storage_root / "fixtures" / sid
            events = _end_turn_call()
            _write_model_fixture(d, [events])
            _write_expected_events(d, events)
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/ci/regression-run").json()
        assert body["status"] == "passed"
        assert body["session_count"] == 3
        statuses = {s["session_id"]: s["status"] for s in body["sessions"]}
        assert statuses == {"sess-a": "passed", "sess-b": "passed", "sess-c": "passed"}

    def test_first_failing_session_drives_error(self, storage_root: Path) -> None:
        # "aaaa-sess" sorts before "zzzz-sess"; aaaa-sess diverges, zzzz-sess passes
        bad_dir = storage_root / "fixtures" / "aaaa-sess"
        _write_model_fixture(bad_dir, [_end_turn_call()])
        _write_expected_events(bad_dir, [{"type": "wrong"}])

        good_dir = storage_root / "fixtures" / "zzzz-sess"
        events = _end_turn_call()
        _write_model_fixture(good_dir, [events])
        _write_expected_events(good_dir, events)

        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/ci/regression-run").json()
        assert body["error"]["code"] == "regression_failed"
        assert "aaaa-sess" in body["error"]["message"]

    def test_tool_result_events_compared_in_baseline(self, storage_root: Path) -> None:
        session_dir = storage_root / "fixtures" / "tool-sess"
        _write_model_fixture(session_dir, [_tool_use_call("tu_1"), _end_turn_call()])
        _write_tool_fixture(session_dir, [{"content": "actual-result"}])

        # Build expected events including the tool_result synthetic event
        expected = (
            list(_tool_use_call("tu_1"))
            + [
                {"type": "tool_result", "tool_id": "tu_1", "content": "actual-result"},
            ]
            + list(_end_turn_call())
        )
        _write_expected_events(session_dir, expected)

        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/ci/regression-run")
        assert resp.status_code == 200

    def test_tool_result_mismatch_causes_failure(self, storage_root: Path) -> None:
        session_dir = storage_root / "fixtures" / "tool-mismatch-sess"
        _write_model_fixture(session_dir, [_tool_use_call("tu_1"), _end_turn_call()])
        _write_tool_fixture(session_dir, [{"content": "actual"}])

        # Expected has different tool_result content
        expected = (
            list(_tool_use_call("tu_1"))
            + [
                {"type": "tool_result", "tool_id": "tu_1", "content": "different"},
            ]
            + list(_end_turn_call())
        )
        _write_expected_events(session_dir, expected)

        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/ci/regression-run")
        assert resp.status_code == 422

    def test_no_storage_root_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/ci/regression-run")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# OTel span tests
# ---------------------------------------------------------------------------


class TestCIRegressionOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _make_client(self, storage_root: Path) -> TestClient:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_regression_run_span(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        client.post("/v1/x/ci/regression-run")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "ci.regression.run" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        session_dir = storage_root / "fixtures" / "otel-fail-sess"
        _write_model_fixture(session_dir, [_end_turn_call()])
        _write_expected_events(session_dir, [{"type": "wrong"}])

        client = self._make_client(storage_root)
        client.post("/v1/x/ci/regression-run")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        reg_span = spans.get("ci.regression.run")
        assert reg_span is not None
        assert reg_span.status.status_code == StatusCode.ERROR

    def test_success_span_has_ok_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._make_client(storage_root)
        client.post("/v1/x/ci/regression-run")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        reg_span = spans.get("ci.regression.run")
        assert reg_span is not None
        assert reg_span.status.status_code != StatusCode.ERROR
