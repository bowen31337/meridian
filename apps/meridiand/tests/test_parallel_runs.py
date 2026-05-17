"""
Parallel-runs endpoint conformance suite.

Tests cover:
  - POST /v1/x/sessions/{id}/parallel_runs returns 200 when all children complete.
  - Response fields: session_id, run_id, status, total_children, succeeded, cancelled,
    total_model_calls, total_tool_calls, children[].
  - N children run in parallel; all succeed when fixtures present.
  - Each child result includes fixture_session_id, model_call_count, tool_call_count, status.
  - Empty children list returns 200 with 0 totals.
  - budget_model_calls enforced: when total model calls exceed budget after any child
    completes, all remaining tasks are cancelled synchronously.
  - Budget exceeded: response status is 422 with code "budget_exceeded".
  - Budget exceeded writes audit log entry with event "session.parallel_runs.budget_exceeded".
  - Audit detail includes session_id, budget_model_calls, total_model_calls.
  - OTel span "session.parallel_runs" emitted on success.
  - OTel span set to ERROR status on budget exceeded.
  - create_app wires the parallel_runs router when storage_root is supplied.
  - create_app omits the route when storage_root is None.
  - Missing required fields returns 422.
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


def _make_client(storage_root: Path, audit_log: FileAuditLog) -> TestClient:
    app = create_app(audit_log, storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _setup_fixture(storage_root: Path, session_id: str, num_calls: int = 1) -> None:
    fd = storage_root / "fixtures" / session_id
    _write_model_fixture(fd, [_end_turn_call() for _ in range(num_calls)])


def _make_body(
    children: list[dict] | None = None,
    budget_model_calls: int | None = None,
) -> dict:
    body: dict = {"children": children if children is not None else []}
    if budget_model_calls is not None:
        body["budget_model_calls"] = budget_model_calls
    return body


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestParallelRunsSuccess:
    def test_returns_200_with_all_children_complete(self, storage_root: Path) -> None:
        _setup_fixture(storage_root, "s1")
        _setup_fixture(storage_root, "s2")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-1/parallel_runs",
            json=_make_body(
                children=[
                    {"fixture_session_id": "s1"},
                    {"fixture_session_id": "s2"},
                ]
            ),
        )
        assert resp.status_code == 200

    def test_response_has_session_id(self, storage_root: Path) -> None:
        _setup_fixture(storage_root, "s3")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-2/parallel_runs",
            json=_make_body(children=[{"fixture_session_id": "s3"}]),
        ).json()
        assert body["session_id"] == "parent-2"

    def test_response_has_run_id(self, storage_root: Path) -> None:
        _setup_fixture(storage_root, "s4")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-3/parallel_runs",
            json=_make_body(children=[{"fixture_session_id": "s4"}]),
        ).json()
        assert "run_id" in body
        assert isinstance(body["run_id"], str)
        assert len(body["run_id"]) > 0

    def test_response_status_is_completed(self, storage_root: Path) -> None:
        _setup_fixture(storage_root, "s5")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-4/parallel_runs",
            json=_make_body(children=[{"fixture_session_id": "s5"}]),
        ).json()
        assert body["status"] == "completed"

    def test_total_children_matches_input(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_fixture(storage_root, f"tc{i}")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-5/parallel_runs",
            json=_make_body(
                children=[{"fixture_session_id": f"tc{i}"} for i in range(3)]
            ),
        ).json()
        assert body["total_children"] == 3

    def test_succeeded_count(self, storage_root: Path) -> None:
        for i in range(2):
            _setup_fixture(storage_root, f"sc{i}")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-6/parallel_runs",
            json=_make_body(
                children=[{"fixture_session_id": f"sc{i}"} for i in range(2)]
            ),
        ).json()
        assert body["succeeded"] == 2

    def test_cancelled_zero_on_full_success(self, storage_root: Path) -> None:
        _setup_fixture(storage_root, "s6")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-7/parallel_runs",
            json=_make_body(children=[{"fixture_session_id": "s6"}]),
        ).json()
        assert body["cancelled"] == 0

    def test_children_results_contain_fixture_session_id(self, storage_root: Path) -> None:
        _setup_fixture(storage_root, "s7")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-8/parallel_runs",
            json=_make_body(children=[{"fixture_session_id": "s7"}]),
        ).json()
        assert body["children"][0]["fixture_session_id"] == "s7"

    def test_children_results_have_status_completed(self, storage_root: Path) -> None:
        _setup_fixture(storage_root, "s8")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-9/parallel_runs",
            json=_make_body(children=[{"fixture_session_id": "s8"}]),
        ).json()
        assert body["children"][0]["status"] == "completed"

    def test_model_call_count_aggregated(self, storage_root: Path) -> None:
        # 2 children × 1 model call each = 2 total
        _setup_fixture(storage_root, "agg1")
        _setup_fixture(storage_root, "agg2")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-agg/parallel_runs",
            json=_make_body(
                children=[
                    {"fixture_session_id": "agg1"},
                    {"fixture_session_id": "agg2"},
                ]
            ),
        ).json()
        assert body["total_model_calls"] == 2

    def test_empty_children_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-empty/parallel_runs",
            json=_make_body(children=[]),
        )
        assert resp.status_code == 200

    def test_empty_children_zero_totals(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-empty2/parallel_runs",
            json=_make_body(children=[]),
        ).json()
        assert body["total_model_calls"] == 0
        assert body["total_tool_calls"] == 0
        assert body["total_children"] == 0

    def test_run_ids_are_unique(self, storage_root: Path) -> None:
        _setup_fixture(storage_root, "uid1")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        id1 = client.post(
            "/v1/x/sessions/parent-uid/parallel_runs",
            json=_make_body(children=[{"fixture_session_id": "uid1"}]),
        ).json()["run_id"]
        id2 = client.post(
            "/v1/x/sessions/parent-uid/parallel_runs",
            json=_make_body(children=[{"fixture_session_id": "uid1"}]),
        ).json()["run_id"]
        assert id1 != id2

    def test_budget_not_exceeded_when_within_limit(self, storage_root: Path) -> None:
        _setup_fixture(storage_root, "bl1")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-budget-ok/parallel_runs",
            json=_make_body(
                children=[{"fixture_session_id": "bl1"}],
                budget_model_calls=10,
            ),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


class TestParallelRunsBudget:
    def test_budget_exceeded_returns_422(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_fixture(storage_root, f"be{i}")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        # budget_model_calls=1 but 3 children each use 1 model call → exceeds after 2nd
        resp = client.post(
            "/v1/x/sessions/budget-1/parallel_runs",
            json=_make_body(
                children=[{"fixture_session_id": f"be{i}"} for i in range(3)],
                budget_model_calls=1,
            ),
        )
        assert resp.status_code == 422

    def test_budget_exceeded_error_code(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_fixture(storage_root, f"bec{i}")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/budget-2/parallel_runs",
            json=_make_body(
                children=[{"fixture_session_id": f"bec{i}"} for i in range(3)],
                budget_model_calls=1,
            ),
        ).json()
        assert body["error"]["code"] == "budget_exceeded"

    def test_budget_exceeded_message_contains_budget(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_fixture(storage_root, f"bem{i}")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/budget-3/parallel_runs",
            json=_make_body(
                children=[{"fixture_session_id": f"bem{i}"} for i in range(3)],
                budget_model_calls=1,
            ),
        ).json()
        assert "1" in body["error"]["message"]

    def test_budget_exceeded_writes_audit_log(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_fixture(storage_root, f"bea{i}")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/budget-audit/parallel_runs",
            json=_make_body(
                children=[{"fixture_session_id": f"bea{i}"} for i in range(3)],
                budget_model_calls=1,
            ),
        )
        records = _audit_records(storage_root)
        assert any(
            r.get("event") == "session.parallel_runs.budget_exceeded" for r in records
        )

    def test_budget_exceeded_audit_level_error(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_fixture(storage_root, f"beal{i}")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/budget-level/parallel_runs",
            json=_make_body(
                children=[{"fixture_session_id": f"beal{i}"} for i in range(3)],
                budget_model_calls=1,
            ),
        )
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "session.parallel_runs.budget_exceeded"
        )
        assert record["level"] == "error"

    def test_budget_exceeded_audit_detail_has_session_id(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_fixture(storage_root, f"beas{i}")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/budget-session-detail/parallel_runs",
            json=_make_body(
                children=[{"fixture_session_id": f"beas{i}"} for i in range(3)],
                budget_model_calls=1,
            ),
        )
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "session.parallel_runs.budget_exceeded"
        )
        assert record["detail"]["session_id"] == "budget-session-detail"

    def test_budget_exceeded_audit_detail_has_budget(self, storage_root: Path) -> None:
        for i in range(3):
            _setup_fixture(storage_root, f"beab{i}")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/budget-detail-budget/parallel_runs",
            json=_make_body(
                children=[{"fixture_session_id": f"beab{i}"} for i in range(3)],
                budget_model_calls=1,
            ),
        )
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "session.parallel_runs.budget_exceeded"
        )
        assert record["detail"]["budget_model_calls"] == 1

    def test_budget_exceeded_audit_detail_has_total_model_calls(
        self, storage_root: Path
    ) -> None:
        for i in range(3):
            _setup_fixture(storage_root, f"beat{i}")
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/budget-detail-total/parallel_runs",
            json=_make_body(
                children=[{"fixture_session_id": f"beat{i}"} for i in range(3)],
                budget_model_calls=1,
            ),
        )
        records = _audit_records(storage_root)
        record = next(
            r for r in records if r.get("event") == "session.parallel_runs.budget_exceeded"
        )
        assert record["detail"]["total_model_calls"] > 1


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestParallelRunsSchema:
    def test_missing_children_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post("/v1/x/sessions/schema-1/parallel_runs", json={})
        assert resp.status_code == 422

    def test_missing_fixture_session_id_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/schema-2/parallel_runs",
            json={"children": [{"capabilities": []}]},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestParallelRunsRouteWiring:
    def test_no_storage_root_returns_404(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/x/sessions/any/parallel_runs",
            json=_make_body(children=[]),
        )
        assert resp.status_code == 404

    def test_with_storage_root_route_exists(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/any/parallel_runs",
            json=_make_body(children=[]),
        )
        assert resp.status_code != 404


from fastapi.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# OTel span tests
# ---------------------------------------------------------------------------


class TestParallelRunsOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _make_client(self, storage_root: Path) -> TestClient:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_span(self, storage_root: Path) -> None:
        _setup_fixture(storage_root, "otel-s1")
        client = self._make_client(storage_root)
        client.post(
            "/v1/x/sessions/otel-parent/parallel_runs",
            json=_make_body(children=[{"fixture_session_id": "otel-s1"}]),
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.parallel_runs" in span_names

    def test_budget_exceeded_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        for i in range(3):
            _setup_fixture(storage_root, f"otel-be{i}")
        client = self._make_client(storage_root)
        client.post(
            "/v1/x/sessions/otel-budget/parallel_runs",
            json=_make_body(
                children=[{"fixture_session_id": f"otel-be{i}"} for i in range(3)],
                budget_model_calls=1,
            ),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.parallel_runs")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        _setup_fixture(storage_root, "otel-s2")
        client = self._make_client(storage_root)
        client.post(
            "/v1/x/sessions/otel-attr-session/parallel_runs",
            json=_make_body(children=[{"fixture_session_id": "otel-s2"}]),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.parallel_runs")
        assert span is not None
        assert span.attributes["session.id"] == "otel-attr-session"

    def test_span_has_child_count_attribute(self, storage_root: Path) -> None:
        for i in range(2):
            _setup_fixture(storage_root, f"otel-cc{i}")
        client = self._make_client(storage_root)
        client.post(
            "/v1/x/sessions/otel-count/parallel_runs",
            json=_make_body(
                children=[{"fixture_session_id": f"otel-cc{i}"} for i in range(2)]
            ),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.parallel_runs")
        assert span is not None
        assert span.attributes["parallel_runs.child_count"] == 2
