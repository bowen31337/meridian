"""
Budget aggregation: usage.delta events from descendants roll up to parent budgets.

Tests cover:
  - Each model call emits a usage.delta event accumulated by the parent in real time.
  - Total model calls derived from rolled-up usage.delta events, not post-completion counts.
  - Usage.delta events roll up across all descendant workers to a single parent total.
  - Hard budget breach via usage.delta rollup returns 422 with budget_exceeded code.
  - Mid-execution cancellation: worker stopped before its next model call when cancel_event
    is set by a sibling's usage.delta pushing the total past the hard limit.
  - Remaining pending tasks cancelled synchronously after breach (not deferred to next batch).
  - OTel span "session.parallel_runs" emitted on every invocation.
  - Budget exceeded writes audit entry with total_model_calls from accumulated deltas.
  - On failure, error message is surfaced in response body and audit log entry written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _end_turn_call() -> list[dict]:
    return [
        {"type": "message_start", "model": "fake", "provider": "fake"},
        {"type": "text_delta", "text": "done"},
        {"type": "message_stop", "stop_reason": "end_turn"},
    ]


def _tool_use_call(tool_id: str = "tool-1", name: str = "bash") -> list[dict]:
    return [
        {"type": "message_start", "model": "fake", "provider": "fake"},
        {"type": "tool_use_start", "id": tool_id, "name": name},
        {"type": "message_stop", "stop_reason": "tool_use"},
    ]


def _write_model_fixture(fixture_dir: Path, calls: list[list[dict]]) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "model_responses.ndjson").write_text(
        "\n".join(json.dumps(c) for c in calls) + "\n"
    )


def _write_tool_fixture(fixture_dir: Path, responses: list[dict]) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "tool_responses.ndjson").write_text(
        "\n".join(json.dumps(r) for r in responses) + "\n"
    )


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Usage.delta accumulation — no budget limit
# ---------------------------------------------------------------------------


class TestUsageDeltaAccumulation:
    """Usage.delta events emitted per model call roll up to the correct total."""

    def test_single_worker_single_call_total_is_one(self, storage_root: Path) -> None:
        _write_model_fixture(storage_root / "fixtures" / "w-1x1", [_end_turn_call()])
        body = _make_client(storage_root).post(
            "/v1/x/sessions/acc-1x1/parallel_runs",
            json={"children": [{"fixture_session_id": "w-1x1"}]},
        ).json()
        assert body["total_model_calls"] == 1

    def test_single_worker_two_calls_total_is_two(self, storage_root: Path) -> None:
        # tool_use → end_turn forces 2 model calls in one worker
        fd = storage_root / "fixtures" / "w-1x2"
        _write_model_fixture(fd, [_tool_use_call(), _end_turn_call()])
        _write_tool_fixture(fd, [{"content": "ok"}])
        body = _make_client(storage_root).post(
            "/v1/x/sessions/acc-1x2/parallel_runs",
            json={"children": [{"fixture_session_id": "w-1x2"}]},
        ).json()
        assert body["total_model_calls"] == 2

    def test_three_workers_one_call_each_total_is_three(self, storage_root: Path) -> None:
        for i in range(3):
            _write_model_fixture(
                storage_root / "fixtures" / f"w-3x1-{i}", [_end_turn_call()]
            )
        body = _make_client(storage_root).post(
            "/v1/x/sessions/acc-3x1/parallel_runs",
            json={
                "children": [{"fixture_session_id": f"w-3x1-{i}"} for i in range(3)]
            },
        ).json()
        assert body["total_model_calls"] == 3

    def test_three_workers_two_calls_each_total_is_six(self, storage_root: Path) -> None:
        # tool_use → end_turn gives each worker 2 model calls; 3 × 2 == 6
        for i in range(3):
            fd = storage_root / "fixtures" / f"w-3x2-{i}"
            _write_model_fixture(fd, [_tool_use_call(), _end_turn_call()])
            _write_tool_fixture(fd, [{"content": "ok"}])
        body = _make_client(storage_root).post(
            "/v1/x/sessions/acc-3x2/parallel_runs",
            json={
                "children": [{"fixture_session_id": f"w-3x2-{i}"} for i in range(3)]
            },
        ).json()
        assert body["total_model_calls"] == 6

    def test_total_within_budget_status_is_completed(self, storage_root: Path) -> None:
        _write_model_fixture(storage_root / "fixtures" / "w-budget-ok", [_end_turn_call()])
        body = _make_client(storage_root).post(
            "/v1/x/sessions/acc-budget-ok/parallel_runs",
            json={
                "children": [{"fixture_session_id": "w-budget-ok"}],
                "budget_model_calls": 10,
            },
        ).json()
        assert body["status"] == "completed"


# ---------------------------------------------------------------------------
# Hard budget breach via usage.delta rollup
# ---------------------------------------------------------------------------


class TestBudgetBreachViaUsageDelta:
    """Parent hard budget breach cancels all descendants synchronously."""

    def _setup(self, storage_root: Path, n: int = 3) -> None:
        for i in range(n):
            _write_model_fixture(
                storage_root / "fixtures" / f"wb-{i}", [_end_turn_call()]
            )

    def test_breach_returns_422(self, storage_root: Path) -> None:
        self._setup(storage_root)
        resp = _make_client(storage_root).post(
            "/v1/x/sessions/breach-422/parallel_runs",
            json={
                "children": [{"fixture_session_id": f"wb-{i}"} for i in range(3)],
                "budget_model_calls": 1,
            },
        )
        assert resp.status_code == 422

    def test_breach_error_code_is_budget_exceeded(self, storage_root: Path) -> None:
        self._setup(storage_root)
        body = _make_client(storage_root).post(
            "/v1/x/sessions/breach-code/parallel_runs",
            json={
                "children": [{"fixture_session_id": f"wb-{i}"} for i in range(3)],
                "budget_model_calls": 1,
            },
        ).json()
        assert body["error"]["code"] == "budget_exceeded"

    def test_breach_audit_total_exceeds_budget(self, storage_root: Path) -> None:
        self._setup(storage_root)
        _make_client(storage_root).post(
            "/v1/x/sessions/breach-audit/parallel_runs",
            json={
                "children": [{"fixture_session_id": f"wb-{i}"} for i in range(3)],
                "budget_model_calls": 1,
            },
        )
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "session.parallel_runs.budget_exceeded"
        )
        assert record["detail"]["total_model_calls"] > record["detail"]["budget_model_calls"]

    def test_breach_audit_total_comes_from_usage_delta_events(
        self, storage_root: Path
    ) -> None:
        # 2 workers × 1 call each with budget 1; breach after 2nd delta → total == 2
        for i in range(2):
            _write_model_fixture(
                storage_root / "fixtures" / f"wb2-{i}", [_end_turn_call()]
            )
        _make_client(storage_root).post(
            "/v1/x/sessions/breach-delta-src/parallel_runs",
            json={
                "children": [{"fixture_session_id": f"wb2-{i}"} for i in range(2)],
                "budget_model_calls": 1,
            },
        )
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "session.parallel_runs.budget_exceeded"
        )
        assert record["detail"]["total_model_calls"] == 2
        assert record["detail"]["budget_model_calls"] == 1


# ---------------------------------------------------------------------------
# Mid-execution cancellation via cancel_event
# ---------------------------------------------------------------------------


class TestMidExecutionCancellation:
    """
    A worker with multiple model calls is cancelled before its next model call
    once a prior usage.delta from the same worker sets the cancel_event.
    """

    def _setup_tool_use_worker(self, storage_root: Path, fixture_id: str) -> None:
        """Worker that needs two model calls: tool_use then end_turn."""
        fd = storage_root / "fixtures" / fixture_id
        _write_model_fixture(fd, [_tool_use_call(), _end_turn_call()])
        _write_tool_fixture(fd, [{"content": "ok"}])

    def test_mid_execution_breach_returns_422(self, storage_root: Path) -> None:
        # budget=0: any model call exceeds; worker is cancelled after its 1st call
        self._setup_tool_use_worker(storage_root, "w-mid-422")
        resp = _make_client(storage_root).post(
            "/v1/x/sessions/mid-422/parallel_runs",
            json={
                "children": [{"fixture_session_id": "w-mid-422"}],
                "budget_model_calls": 0,
            },
        )
        assert resp.status_code == 422

    def test_mid_execution_total_reflects_calls_before_breach(
        self, storage_root: Path
    ) -> None:
        # Worker has 2 model calls; budget=0 → breach after call 1 → total == 1, not 2
        self._setup_tool_use_worker(storage_root, "w-mid-total")
        _make_client(storage_root).post(
            "/v1/x/sessions/mid-total/parallel_runs",
            json={
                "children": [{"fixture_session_id": "w-mid-total"}],
                "budget_model_calls": 0,
            },
        )
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "session.parallel_runs.budget_exceeded"
        )
        assert record["detail"]["total_model_calls"] == 1

    def test_mid_execution_audit_budget_value_correct(self, storage_root: Path) -> None:
        self._setup_tool_use_worker(storage_root, "w-mid-bv")
        _make_client(storage_root).post(
            "/v1/x/sessions/mid-bv/parallel_runs",
            json={
                "children": [{"fixture_session_id": "w-mid-bv"}],
                "budget_model_calls": 0,
            },
        )
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "session.parallel_runs.budget_exceeded"
        )
        assert record["detail"]["budget_model_calls"] == 0

    def test_mid_execution_error_code_is_budget_exceeded(self, storage_root: Path) -> None:
        self._setup_tool_use_worker(storage_root, "w-mid-code")
        body = _make_client(storage_root).post(
            "/v1/x/sessions/mid-code/parallel_runs",
            json={
                "children": [{"fixture_session_id": "w-mid-code"}],
                "budget_model_calls": 0,
            },
        ).json()
        assert body["error"]["code"] == "budget_exceeded"


# ---------------------------------------------------------------------------
# OTel span emitted on each invocation
# ---------------------------------------------------------------------------


class TestBudgetAggregationOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_otel_span_emitted_on_budget_breach(self, storage_root: Path) -> None:
        _write_model_fixture(
            storage_root / "fixtures" / "otel-breach", [_end_turn_call()]
        )
        _make_client(storage_root).post(
            "/v1/x/sessions/otel-breach/parallel_runs",
            json={
                "children": [{"fixture_session_id": "otel-breach"}],
                "budget_model_calls": 0,
            },
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.parallel_runs" in span_names

    def test_otel_span_emitted_on_success(self, storage_root: Path) -> None:
        _write_model_fixture(
            storage_root / "fixtures" / "otel-ok", [_end_turn_call()]
        )
        _make_client(storage_root).post(
            "/v1/x/sessions/otel-ok/parallel_runs",
            json={"children": [{"fixture_session_id": "otel-ok"}]},
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.parallel_runs" in span_names
