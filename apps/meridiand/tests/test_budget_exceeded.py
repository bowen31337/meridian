"""
Budget-exceeded session termination conformance suite.

Tests cover:
  - POST /v1/x/sessions/{id}/budget-exceeded returns 200 with session_id, before, after,
    reason, dimension, limit, actual.
  - after is always "terminated".
  - reason is always "budget_exceeded".
  - before reflects the session's phase prior to termination (defaults to "created").
  - before reflects the last recorded session.phase_change event.
  - budget.exceeded event written to event log with dimension, limit, actual, session_id, timestamp.
  - session.phase_change event written with after="terminated" and reason="budget_exceeded".
  - session.phase_change event before field matches the pre-termination phase.
  - Missing dimension field returns 422 (FastAPI schema validation).
  - Missing limit field returns 422 (FastAPI schema validation).
  - Missing actual field returns 422 (FastAPI schema validation).
  - On write failure, returns 422 with code "budget_exceeded_session_failed".
  - On failure, audit log entry written with event "session.budget_exceeded.failed".
  - On failure, audit detail includes session_id and message.
  - OTel span "session.budget_exceeded" emitted on success.
  - OTel span has session.id attribute.
  - OTel span is set to ERROR status on failure.
  - create_app wires the route when storage_root and event_log are supplied.
  - create_app omits the route when storage_root is None.
  - create_app omits the route when event_log is None.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from storage_event_log import LocalEventLogWriter
from storage_reposit import LocalEventLogReader

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_VALID_BODY = {"dimension": "dollars", "limit": 10.0, "actual": 12.5}


def _make_writer(storage_root: Path) -> LocalEventLogWriter:
    return LocalEventLogWriter(storage_root)


def _make_client(
    storage_root: Path,
    writer: LocalEventLogWriter | None = None,
) -> TestClient:
    audit = FileAuditLog(storage_root)
    w = writer or _make_writer(storage_root)
    app = create_app(audit, storage_root=storage_root, event_log=w)
    return TestClient(app, raise_server_exceptions=False)


def _seed_phase(storage_root: Path, session_id: str, phase: str) -> None:
    async def _write() -> None:
        writer = LocalEventLogWriter(storage_root)
        await writer.append(
            session_id,
            "session.phase_change",
            {"before": "created", "after": phase, "reason": "seed", "timestamp": "t0"},
        )

    asyncio.run(_write())


def _read_events(storage_root: Path, session_id: str) -> list[Any]:
    reader = LocalEventLogReader(storage_root)
    return list(reader.read_after(session_id, -1))


def _read_audit(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


class TestBudgetExceededResponse:
    def test_returns_200_on_success(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post(
            "/v1/x/sessions/sess1/budget-exceeded", json=_VALID_BODY
        )
        assert resp.status_code == 200

    def test_response_has_session_id(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess2/budget-exceeded", json=_VALID_BODY
        ).json()
        assert body["session_id"] == "sess2"

    def test_response_after_is_terminated(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess3/budget-exceeded", json=_VALID_BODY
        ).json()
        assert body["after"] == "terminated"

    def test_response_reason_is_budget_exceeded(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess4/budget-exceeded", json=_VALID_BODY
        ).json()
        assert body["reason"] == "budget_exceeded"

    def test_response_before_defaults_to_created(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess5/budget-exceeded", json=_VALID_BODY
        ).json()
        assert body["before"] == "created"

    def test_response_before_reflects_prior_phase(self, storage_root: Path) -> None:
        _seed_phase(storage_root, "sess6", "running")
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess6/budget-exceeded", json=_VALID_BODY
        ).json()
        assert body["before"] == "running"

    def test_response_dimension_matches_request(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess7/budget-exceeded",
            json={"dimension": "input_tokens", "limit": 1000.0, "actual": 1500.0},
        ).json()
        assert body["dimension"] == "input_tokens"

    def test_response_limit_matches_request(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess8/budget-exceeded",
            json={"dimension": "dollars", "limit": 5.0, "actual": 7.0},
        ).json()
        assert body["limit"] == 5.0

    def test_response_actual_matches_request(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post(
            "/v1/x/sessions/sess9/budget-exceeded",
            json={"dimension": "dollars", "limit": 5.0, "actual": 7.0},
        ).json()
        assert body["actual"] == 7.0


# ---------------------------------------------------------------------------
# Event log writes
# ---------------------------------------------------------------------------


class TestBudgetExceededEventLog:
    def test_budget_exceeded_event_written(self, storage_root: Path) -> None:
        _make_client(storage_root).post(
            "/v1/x/sessions/ev-sess1/budget-exceeded", json=_VALID_BODY
        )
        events = _read_events(storage_root, "ev-sess1")
        assert any(e.type == "budget.exceeded" for e in events)

    def test_budget_exceeded_event_has_dimension(self, storage_root: Path) -> None:
        _make_client(storage_root).post(
            "/v1/x/sessions/ev-sess2/budget-exceeded",
            json={"dimension": "output_tokens", "limit": 500.0, "actual": 600.0},
        )
        events = _read_events(storage_root, "ev-sess2")
        ev = next(e for e in events if e.type == "budget.exceeded")
        assert ev.data["dimension"] == "output_tokens"

    def test_budget_exceeded_event_has_limit(self, storage_root: Path) -> None:
        _make_client(storage_root).post(
            "/v1/x/sessions/ev-sess3/budget-exceeded",
            json={"dimension": "dollars", "limit": 8.0, "actual": 9.0},
        )
        events = _read_events(storage_root, "ev-sess3")
        ev = next(e for e in events if e.type == "budget.exceeded")
        assert ev.data["limit"] == 8.0

    def test_budget_exceeded_event_has_actual(self, storage_root: Path) -> None:
        _make_client(storage_root).post(
            "/v1/x/sessions/ev-sess4/budget-exceeded",
            json={"dimension": "dollars", "limit": 8.0, "actual": 9.0},
        )
        events = _read_events(storage_root, "ev-sess4")
        ev = next(e for e in events if e.type == "budget.exceeded")
        assert ev.data["actual"] == 9.0

    def test_budget_exceeded_event_has_session_id(self, storage_root: Path) -> None:
        _make_client(storage_root).post(
            "/v1/x/sessions/ev-sess5/budget-exceeded", json=_VALID_BODY
        )
        events = _read_events(storage_root, "ev-sess5")
        ev = next(e for e in events if e.type == "budget.exceeded")
        assert ev.data["session_id"] == "ev-sess5"

    def test_budget_exceeded_event_has_timestamp(self, storage_root: Path) -> None:
        _make_client(storage_root).post(
            "/v1/x/sessions/ev-sess6/budget-exceeded", json=_VALID_BODY
        )
        events = _read_events(storage_root, "ev-sess6")
        ev = next(e for e in events if e.type == "budget.exceeded")
        assert "timestamp" in ev.data

    def test_phase_change_event_written(self, storage_root: Path) -> None:
        _make_client(storage_root).post(
            "/v1/x/sessions/ev-sess7/budget-exceeded", json=_VALID_BODY
        )
        events = _read_events(storage_root, "ev-sess7")
        assert any(e.type == "session.phase_change" for e in events)

    def test_phase_change_after_is_terminated(self, storage_root: Path) -> None:
        _make_client(storage_root).post(
            "/v1/x/sessions/ev-sess8/budget-exceeded", json=_VALID_BODY
        )
        events = _read_events(storage_root, "ev-sess8")
        change = next(e for e in events if e.type == "session.phase_change")
        assert change.data["after"] == "terminated"

    def test_phase_change_reason_is_budget_exceeded(self, storage_root: Path) -> None:
        _make_client(storage_root).post(
            "/v1/x/sessions/ev-sess9/budget-exceeded", json=_VALID_BODY
        )
        events = _read_events(storage_root, "ev-sess9")
        change = next(e for e in events if e.type == "session.phase_change")
        assert change.data["reason"] == "budget_exceeded"

    def test_phase_change_before_matches_prior_phase(self, storage_root: Path) -> None:
        _seed_phase(storage_root, "ev-sess10", "paused")
        _make_client(storage_root).post(
            "/v1/x/sessions/ev-sess10/budget-exceeded", json=_VALID_BODY
        )
        events = _read_events(storage_root, "ev-sess10")
        changes = [e for e in events if e.type == "session.phase_change"]
        termination = changes[-1]
        assert termination.data["before"] == "paused"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestBudgetExceededSchemaValidation:
    def test_missing_dimension_returns_422(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post(
            "/v1/x/sessions/val-sess1/budget-exceeded",
            json={"limit": 10.0, "actual": 12.0},
        )
        assert resp.status_code == 422

    def test_missing_limit_returns_422(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post(
            "/v1/x/sessions/val-sess2/budget-exceeded",
            json={"dimension": "dollars", "actual": 12.0},
        )
        assert resp.status_code == 422

    def test_missing_actual_returns_422(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post(
            "/v1/x/sessions/val-sess3/budget-exceeded",
            json={"dimension": "dollars", "limit": 10.0},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Failure surfacing
# ---------------------------------------------------------------------------


class TestBudgetExceededFailure:
    def test_write_failure_returns_422(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/fail-sess/budget-exceeded", json=_VALID_BODY)
        assert resp.status_code == 422

    def test_write_failure_error_code(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        body = client.post(
            "/v1/x/sessions/fail-sess2/budget-exceeded", json=_VALID_BODY
        ).json()
        assert body["error"]["code"] == "budget_exceeded_session_failed"

    def test_write_failure_error_message_non_empty(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        body = client.post(
            "/v1/x/sessions/fail-sess3/budget-exceeded", json=_VALID_BODY
        ).json()
        assert len(body["error"]["message"]) > 0

    def test_write_failure_writes_audit_entry(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/v1/x/sessions/fail-sess4/budget-exceeded", json=_VALID_BODY)
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.budget_exceeded.failed" for r in records)

    def test_write_failure_audit_level_is_error(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/v1/x/sessions/fail-sess5/budget-exceeded", json=_VALID_BODY)
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.budget_exceeded.failed")
        assert rec["level"] == "error"

    def test_write_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/v1/x/sessions/fail-sess6/budget-exceeded", json=_VALID_BODY)
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.budget_exceeded.failed")
        assert rec["detail"]["session_id"] == "fail-sess6"

    def test_write_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/v1/x/sessions/fail-sess7/budget-exceeded", json=_VALID_BODY)
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.budget_exceeded.failed")
        assert len(rec["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestBudgetExceededOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_budget_exceeded_span(self, storage_root: Path) -> None:
        _make_client(storage_root).post(
            "/v1/x/sessions/otel-sess1/budget-exceeded", json=_VALID_BODY
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.budget_exceeded" in span_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        _make_client(storage_root).post(
            "/v1/x/sessions/otel-sess2/budget-exceeded", json=_VALID_BODY
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.budget_exceeded")
        assert span is not None
        assert span.attributes["session.id"] == "otel-sess2"

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        audit = FileAuditLog(storage_root)
        failing_writer = AsyncMock()
        failing_writer.append.side_effect = OSError("disk full")
        app = create_app(audit, storage_root=storage_root, event_log=failing_writer)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/v1/x/sessions/otel-fail/budget-exceeded", json=_VALID_BODY)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        be_span = spans.get("session.budget_exceeded")
        assert be_span is not None
        assert be_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Router wiring via create_app
# ---------------------------------------------------------------------------


class TestBudgetExceededRouterWiring:
    def test_route_exists_with_storage_root_and_event_log(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post(
            "/v1/x/sessions/wire-sess1/budget-exceeded", json=_VALID_BODY
        )
        assert resp.status_code != 404

    def test_no_storage_root_returns_404(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/x/sessions/any/budget-exceeded", json=_VALID_BODY
        )
        assert resp.status_code == 404

    def test_no_event_log_returns_404(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/x/sessions/any/budget-exceeded", json=_VALID_BODY
        )
        assert resp.status_code == 404
