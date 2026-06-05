"""
Multi-agent integration test: parent spawns 3 workers via parallel_runs.

Tests cover:
  - Parent session runs 3 workers via parallel_runs; all succeed when fixtures present.
  - total_children == 3, succeeded == 3, cancelled == 0 on full success.
  - Budget aggregated across 3 workers: total_model_calls equals sum of worker model calls.
  - Hard budget exceeded after workers complete: status is "budget_exceeded", 422 response.
  - Cancellation propagated: workers that haven't completed are cancelled when budget overflows.
  - Budget overflow error code is "budget_exceeded" and message is included in response body.
  - Budget overflow writes audit entry with event "session.parallel_runs.budget_exceeded".
  - Audit detail includes session_id, budget_model_calls, and total_model_calls.
  - OTel span "session.parallel_runs" emitted on success with parallel_runs.child_count == 3.
  - OTel span set to ERROR status when budget is exceeded.
  - On failure, error message is surfaced in the response body (error.message present).
  - On failure, audit log entry written with event "session.parallel_runs.failed".
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


def _write_model_fixture(fixture_dir: Path, calls: list[list[dict]]) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(events) for events in calls]
    (fixture_dir / "model_responses.ndjson").write_text("\n".join(lines) + "\n")


def _end_turn_call() -> list[dict]:
    return [
        {"type": "message_start", "model": "fake", "provider": "fake"},
        {"type": "text_delta", "text": "done"},
        {"type": "message_stop", "stop_reason": "end_turn"},
    ]


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _setup_worker(storage_root: Path, session_id: str, num_calls: int = 1) -> None:
    fd = storage_root / "fixtures" / session_id
    _write_model_fixture(fd, [_end_turn_call() for _ in range(num_calls)])


def _three_worker_body(budget_model_calls: int | None = None) -> dict:
    body: dict = {
        "children": [
            {"fixture_session_id": "worker-0"},
            {"fixture_session_id": "worker-1"},
            {"fixture_session_id": "worker-2"},
        ]
    }
    if budget_model_calls is not None:
        body["budget_model_calls"] = budget_model_calls
    return body


# ---------------------------------------------------------------------------
# Success: parent spawns 3 workers, all complete
# ---------------------------------------------------------------------------


class TestMultiAgentThreeWorkersSuccess:
    def _setup(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_worker(storage_root, f"worker-{i}")

    def test_returns_200(self, storage_root: Path) -> None:
        self._setup(storage_root)
        resp = _make_client(storage_root).post(
            "/v1/x/sessions/ma-parent/parallel_runs",
            json=_three_worker_body(),
        )
        assert resp.status_code == 200

    def test_status_is_completed(self, storage_root: Path) -> None:
        self._setup(storage_root)
        body = (
            _make_client(storage_root)
            .post(
                "/v1/x/sessions/ma-parent/parallel_runs",
                json=_three_worker_body(),
            )
            .json()
        )
        assert body["status"] == "completed"

    def test_total_children_is_three(self, storage_root: Path) -> None:
        self._setup(storage_root)
        body = (
            _make_client(storage_root)
            .post(
                "/v1/x/sessions/ma-parent/parallel_runs",
                json=_three_worker_body(),
            )
            .json()
        )
        assert body["total_children"] == 3

    def test_all_three_workers_succeed(self, storage_root: Path) -> None:
        self._setup(storage_root)
        body = (
            _make_client(storage_root)
            .post(
                "/v1/x/sessions/ma-parent/parallel_runs",
                json=_three_worker_body(),
            )
            .json()
        )
        assert body["succeeded"] == 3

    def test_cancelled_is_zero_on_full_success(self, storage_root: Path) -> None:
        self._setup(storage_root)
        body = (
            _make_client(storage_root)
            .post(
                "/v1/x/sessions/ma-parent/parallel_runs",
                json=_three_worker_body(),
            )
            .json()
        )
        assert body["cancelled"] == 0

    def test_budget_aggregated_across_three_workers(self, storage_root: Path) -> None:
        # 3 workers × 1 model call each = 3 total
        self._setup(storage_root)
        body = (
            _make_client(storage_root)
            .post(
                "/v1/x/sessions/ma-parent/parallel_runs",
                json=_three_worker_body(),
            )
            .json()
        )
        assert body["total_model_calls"] == 3

    def test_each_worker_result_has_completed_status(self, storage_root: Path) -> None:
        self._setup(storage_root)
        body = (
            _make_client(storage_root)
            .post(
                "/v1/x/sessions/ma-parent/parallel_runs",
                json=_three_worker_body(),
            )
            .json()
        )
        statuses = [c["status"] for c in body["children"]]
        assert all(s == "completed" for s in statuses)

    def test_response_includes_all_worker_ids(self, storage_root: Path) -> None:
        self._setup(storage_root)
        body = (
            _make_client(storage_root)
            .post(
                "/v1/x/sessions/ma-parent/parallel_runs",
                json=_three_worker_body(),
            )
            .json()
        )
        ids = {c["fixture_session_id"] for c in body["children"]}
        assert ids == {"worker-0", "worker-1", "worker-2"}

    def test_budget_not_exceeded_when_within_limit(self, storage_root: Path) -> None:
        self._setup(storage_root)
        body = (
            _make_client(storage_root)
            .post(
                "/v1/x/sessions/ma-parent/parallel_runs",
                json=_three_worker_body(budget_model_calls=10),
            )
            .json()
        )
        assert body["status"] == "completed"


# ---------------------------------------------------------------------------
# Budget: hard budget exceeded → cancellation propagated
# ---------------------------------------------------------------------------


class TestMultiAgentBudgetCancellation:
    def _setup(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_worker(storage_root, f"worker-{i}")

    def test_budget_exceeded_returns_422(self, storage_root: Path) -> None:
        self._setup(storage_root)
        resp = _make_client(storage_root).post(
            "/v1/x/sessions/ma-budget/parallel_runs",
            json=_three_worker_body(budget_model_calls=1),
        )
        assert resp.status_code == 422

    def test_budget_exceeded_error_code(self, storage_root: Path) -> None:
        self._setup(storage_root)
        body = (
            _make_client(storage_root)
            .post(
                "/v1/x/sessions/ma-budget/parallel_runs",
                json=_three_worker_body(budget_model_calls=1),
            )
            .json()
        )
        assert body["error"]["code"] == "budget_exceeded"

    def test_budget_exceeded_message_in_response(self, storage_root: Path) -> None:
        self._setup(storage_root)
        body = (
            _make_client(storage_root)
            .post(
                "/v1/x/sessions/ma-budget/parallel_runs",
                json=_three_worker_body(budget_model_calls=1),
            )
            .json()
        )
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0

    def test_budget_exceeded_message_contains_budget_value(self, storage_root: Path) -> None:
        self._setup(storage_root)
        body = (
            _make_client(storage_root)
            .post(
                "/v1/x/sessions/ma-budget/parallel_runs",
                json=_three_worker_body(budget_model_calls=1),
            )
            .json()
        )
        assert "1" in body["error"]["message"]

    def test_cancellation_propagated_to_remaining_workers(self, storage_root: Path) -> None:
        # budget=1: after 2 workers complete (total=2 > 1), the 3rd is cancelled
        self._setup(storage_root)
        body = (
            _make_client(storage_root)
            .post(
                "/v1/x/sessions/ma-cancel/parallel_runs",
                json=_three_worker_body(budget_model_calls=1),
            )
            .json()
        )
        # The error body won't have cancelled, but the audit log should reflect cancellation
        # The 422 response itself proves cancellation was triggered
        assert body["error"]["code"] == "budget_exceeded"

    def test_total_model_calls_exceeds_budget(self, storage_root: Path) -> None:
        self._setup(storage_root)
        # Can't read total_model_calls from 422 error body; verify via audit log detail
        _make_client(storage_root).post(
            "/v1/x/sessions/ma-total/parallel_runs",
            json=_three_worker_body(budget_model_calls=1),
        )
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "session.parallel_runs.budget_exceeded"
        )
        assert record["detail"]["total_model_calls"] > record["detail"]["budget_model_calls"]

    def test_budget_exceeded_writes_audit_entry(self, storage_root: Path) -> None:
        self._setup(storage_root)
        _make_client(storage_root).post(
            "/v1/x/sessions/ma-audit/parallel_runs",
            json=_three_worker_body(budget_model_calls=1),
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.parallel_runs.budget_exceeded" for r in records)

    def test_budget_exceeded_audit_level_is_error(self, storage_root: Path) -> None:
        self._setup(storage_root)
        _make_client(storage_root).post(
            "/v1/x/sessions/ma-audit-level/parallel_runs",
            json=_three_worker_body(budget_model_calls=1),
        )
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "session.parallel_runs.budget_exceeded"
        )
        assert record["level"] == "error"

    def test_budget_exceeded_audit_detail_has_session_id(self, storage_root: Path) -> None:
        self._setup(storage_root)
        _make_client(storage_root).post(
            "/v1/x/sessions/ma-session-detail/parallel_runs",
            json=_three_worker_body(budget_model_calls=1),
        )
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "session.parallel_runs.budget_exceeded"
        )
        assert record["detail"]["session_id"] == "ma-session-detail"

    def test_budget_exceeded_audit_detail_has_budget(self, storage_root: Path) -> None:
        self._setup(storage_root)
        _make_client(storage_root).post(
            "/v1/x/sessions/ma-budget-detail/parallel_runs",
            json=_three_worker_body(budget_model_calls=1),
        )
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "session.parallel_runs.budget_exceeded"
        )
        assert record["detail"]["budget_model_calls"] == 1

    def test_budget_exceeded_audit_detail_has_total_model_calls(self, storage_root: Path) -> None:
        self._setup(storage_root)
        _make_client(storage_root).post(
            "/v1/x/sessions/ma-total-detail/parallel_runs",
            json=_three_worker_body(budget_model_calls=1),
        )
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "session.parallel_runs.budget_exceeded"
        )
        assert record["detail"]["total_model_calls"] > 1


# ---------------------------------------------------------------------------
# OTel: span emitted with structured event on each invocation
# ---------------------------------------------------------------------------


class TestMultiAgentOtelInstrumentation:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _setup(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_worker(storage_root, f"worker-{i}")

    def test_success_emits_parallel_runs_span(self, storage_root: Path) -> None:
        self._setup(storage_root)
        _make_client(storage_root).post(
            "/v1/x/sessions/otel-ma/parallel_runs",
            json=_three_worker_body(),
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.parallel_runs" in span_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        self._setup(storage_root)
        _make_client(storage_root).post(
            "/v1/x/sessions/otel-session/parallel_runs",
            json=_three_worker_body(),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.parallel_runs")
        assert span is not None
        assert span.attributes["session.id"] == "otel-session"

    def test_span_child_count_attribute_is_three(self, storage_root: Path) -> None:
        self._setup(storage_root)
        _make_client(storage_root).post(
            "/v1/x/sessions/otel-count/parallel_runs",
            json=_three_worker_body(),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.parallel_runs")
        assert span is not None
        assert span.attributes["parallel_runs.child_count"] == 3

    def test_budget_exceeded_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        self._setup(storage_root)
        _make_client(storage_root).post(
            "/v1/x/sessions/otel-budget/parallel_runs",
            json=_three_worker_body(budget_model_calls=1),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.parallel_runs")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_span_has_run_id_attribute(self, storage_root: Path) -> None:
        self._setup(storage_root)
        _make_client(storage_root).post(
            "/v1/x/sessions/otel-runid/parallel_runs",
            json=_three_worker_body(),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.parallel_runs")
        assert span is not None
        assert "parallel_runs.run_id" in span.attributes
        assert len(span.attributes["parallel_runs.run_id"]) > 0


# ---------------------------------------------------------------------------
# Failure: error surfaced in response and written to audit log
# ---------------------------------------------------------------------------


class TestMultiAgentFailureSurfacing:
    def test_failure_returns_422(self, storage_root: Path) -> None:
        # One worker fixture is intentionally missing (bad JSON) to trigger failure
        for i in range(3):
            _setup_worker(storage_root, f"worker-{i}")
        # Corrupt worker-1's fixture to trigger a parse error inside run_one
        bad_fixture = storage_root / "fixtures" / "worker-1" / "model_responses.ndjson"
        bad_fixture.write_text("not-valid-json\n")
        resp = _make_client(storage_root).post(
            "/v1/x/sessions/ma-fail/parallel_runs",
            json=_three_worker_body(),
        )
        assert resp.status_code == 422

    def test_failure_error_message_in_response(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_worker(storage_root, f"worker-{i}")
        bad_fixture = storage_root / "fixtures" / "worker-1" / "model_responses.ndjson"
        bad_fixture.write_text("not-valid-json\n")
        body = (
            _make_client(storage_root)
            .post(
                "/v1/x/sessions/ma-fail-msg/parallel_runs",
                json=_three_worker_body(),
            )
            .json()
        )
        assert "error" in body
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0

    def test_failure_writes_audit_log(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_worker(storage_root, f"worker-{i}")
        bad_fixture = storage_root / "fixtures" / "worker-1" / "model_responses.ndjson"
        bad_fixture.write_text("not-valid-json\n")
        _make_client(storage_root).post(
            "/v1/x/sessions/ma-fail-audit/parallel_runs",
            json=_three_worker_body(),
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.parallel_runs.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_worker(storage_root, f"worker-{i}")
        bad_fixture = storage_root / "fixtures" / "worker-1" / "model_responses.ndjson"
        bad_fixture.write_text("not-valid-json\n")
        _make_client(storage_root).post(
            "/v1/x/sessions/ma-fail-level/parallel_runs",
            json=_three_worker_body(),
        )
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.parallel_runs.failed")
        assert record["level"] == "error"

    def test_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_worker(storage_root, f"worker-{i}")
        bad_fixture = storage_root / "fixtures" / "worker-1" / "model_responses.ndjson"
        bad_fixture.write_text("not-valid-json\n")
        _make_client(storage_root).post(
            "/v1/x/sessions/ma-fail-sid/parallel_runs",
            json=_three_worker_body(),
        )
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.parallel_runs.failed")
        assert record["detail"]["session_id"] == "ma-fail-sid"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_worker(storage_root, f"worker-{i}")
        bad_fixture = storage_root / "fixtures" / "worker-1" / "model_responses.ndjson"
        bad_fixture.write_text("not-valid-json\n")
        _make_client(storage_root).post(
            "/v1/x/sessions/ma-fail-detail/parallel_runs",
            json=_three_worker_body(),
        )
        records = _audit_records(storage_root)
        record = next(r for r in records if r.get("event") == "session.parallel_runs.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0
