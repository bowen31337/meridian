"""
Submit-tool-results endpoint conformance suite.

Tests cover:
  - POST /v1/sessions/{id}/submit_tool_results returns 202 with session_id and submitted count.
  - Returns 404 with code "submit_tool_results_session_not_found" when session manifest absent.
  - Returns 422 with code "submit_tool_results_wrong_phase" when session is not waiting_for_tool.
  - Missing session writes audit log with event "session.submit_tool_results.failed".
  - Wrong phase writes audit log with event "session.submit_tool_results.failed".
  - Audit detail includes session_id on failure (not-found case).
  - Audit detail includes session_id and phase on failure (wrong-phase case).
  - Success writes audit log with event "session.submit_tool_results.accepted".
  - Audit detail includes session_id on success.
  - Audit detail includes count on success.
  - Each tool result is written as a tool_call.result event in the event log.
  - Phase change event written: before=waiting_for_tool, after=waiting_for_model.
  - harness_pool.wake is called with the correct session_id.
  - Pool wake failure surfaces as 422.
  - Pool wake failure writes audit log with event "session.submit_tool_results.failed".
  - OTel span "session.submit_tool_results" is emitted on success.
  - OTel span has session.id attribute on success.
  - OTel span is set to ERROR status on failure.
  - create_app wires the route when storage_root, event_log, and harness_pool are supplied.
  - create_app omits the route when harness_pool is None.
  - create_app omits the route when storage_root is None.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._submit_tool_results import make_submit_tool_results_router
from storage_event_log import LocalEventLogWriter
from storage_reposit import LocalEventLogReader

from tests._otel_shared import otel_exporter as _otel_exporter

from core_errors import HandlerOptions, install_error_handler
from meridiand._auth_middleware import AuthMiddleware
from meridiand._error_envelope_middleware import ErrorEnvelopeMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_session(storage_root: Path, session_id: str) -> None:
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(
        json.dumps({"session_id": session_id, "status": "active"})
    )


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


def _read_events(storage_root: Path, session_id: str) -> list[Any]:
    reader = LocalEventLogReader(storage_root)
    return list(reader.read_after(session_id, -1))


_SINGLE_RESULT = [{"tool_use_id": "tu-1", "content": "ok", "is_error": False}]
_MULTI_RESULTS = [
    {"tool_use_id": "tu-a", "content": "result-a", "is_error": False},
    {"tool_use_id": "tu-b", "content": None, "is_error": True},
]


# ---------------------------------------------------------------------------
# Fake pools
# ---------------------------------------------------------------------------


class _TrackingPool:
    def __init__(self) -> None:
        self.woken: list[str] = []

    async def wake(self, session_id: str) -> None:
        self.woken.append(session_id)


class _ErrorPool:
    async def wake(self, session_id: str) -> None:
        raise RuntimeError("injected pool failure")


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------


def _make_writer(storage_root: Path) -> LocalEventLogWriter:
    return LocalEventLogWriter(storage_root)


def _make_client_with_pool(
    storage_root: Path,
    pool: Any,
    writer: LocalEventLogWriter | None = None,
) -> TestClient:
    audit = FileAuditLog(storage_root)
    w = writer or _make_writer(storage_root)
    app = FastAPI()
    app.add_middleware(AuthMiddleware, audit_log=audit, bearer_token=None)
    app.add_middleware(ErrorEnvelopeMiddleware, audit_log=audit, hooks_dir=None)
    install_error_handler(app, HandlerOptions(audit_log=audit))
    app.include_router(
        make_submit_tool_results_router(
            audit_log=audit,
            storage_root=storage_root,
            event_log=w,
            harness_pool=pool,
        )
    )
    return TestClient(app, raise_server_exceptions=False)


def _make_client(
    storage_root: Path,
    writer: LocalEventLogWriter | None = None,
) -> TestClient:
    return _make_client_with_pool(storage_root, _TrackingPool(), writer)


# ---------------------------------------------------------------------------
# Response shape — success
# ---------------------------------------------------------------------------


class TestSubmitToolResultsSuccess:
    def _setup(self, storage_root: Path, session_id: str) -> TestClient:
        _write_session(storage_root, session_id)
        _seed_phase(storage_root, session_id, "waiting_for_tool")
        return _make_client(storage_root)

    def test_returns_202(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "str-s1")
        resp = client.post("/v1/sessions/str-s1/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        assert resp.status_code == 202

    def test_response_has_session_id(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "str-s2")
        body = client.post("/v1/sessions/str-s2/submit_tool_results", json={"tool_results": _SINGLE_RESULT}).json()
        assert body["session_id"] == "str-s2"

    def test_response_has_submitted_count(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "str-s3")
        body = client.post("/v1/sessions/str-s3/submit_tool_results", json={"tool_results": _SINGLE_RESULT}).json()
        assert body["submitted"] == 1

    def test_submitted_count_matches_multiple_results(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "str-s4")
        body = client.post("/v1/sessions/str-s4/submit_tool_results", json={"tool_results": _MULTI_RESULTS}).json()
        assert body["submitted"] == 2


# ---------------------------------------------------------------------------
# Session not found
# ---------------------------------------------------------------------------


class TestSubmitToolResultsNotFound:
    def test_missing_session_returns_404(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post(
            "/v1/sessions/no-such/submit_tool_results", json={"tool_results": _SINGLE_RESULT}
        )
        assert resp.status_code == 404

    def test_missing_session_error_code(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post(
            "/v1/sessions/no-sess2/submit_tool_results", json={"tool_results": _SINGLE_RESULT}
        ).json()
        assert body["error"]["code"] == "submit_tool_results_session_not_found"

    def test_missing_session_writes_audit_failed(self, storage_root: Path) -> None:
        _make_client(storage_root).post(
            "/v1/sessions/no-aud/submit_tool_results", json={"tool_results": _SINGLE_RESULT}
        )
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.submit_tool_results.failed" for r in records)

    def test_missing_session_audit_detail_has_session_id(self, storage_root: Path) -> None:
        _make_client(storage_root).post(
            "/v1/sessions/no-detail/submit_tool_results", json={"tool_results": _SINGLE_RESULT}
        )
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.submit_tool_results.failed")
        assert rec["detail"]["session_id"] == "no-detail"


# ---------------------------------------------------------------------------
# Wrong phase
# ---------------------------------------------------------------------------


class TestSubmitToolResultsWrongPhase:
    def _setup(self, storage_root: Path, session_id: str, phase: str) -> TestClient:
        _write_session(storage_root, session_id)
        _seed_phase(storage_root, session_id, phase)
        return _make_client(storage_root)

    def test_wrong_phase_returns_422(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "wp-s1", "idle")
        resp = client.post("/v1/sessions/wp-s1/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        assert resp.status_code == 422

    def test_wrong_phase_error_code(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "wp-s2", "idle")
        body = client.post("/v1/sessions/wp-s2/submit_tool_results", json={"tool_results": _SINGLE_RESULT}).json()
        assert body["error"]["code"] == "submit_tool_results_wrong_phase"

    def test_wrong_phase_writes_audit_failed(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "wp-aud", "waiting_for_user")
        client.post("/v1/sessions/wp-aud/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.submit_tool_results.failed" for r in records)

    def test_wrong_phase_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "wp-det", "idle")
        client.post("/v1/sessions/wp-det/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.submit_tool_results.failed")
        assert rec["detail"]["session_id"] == "wp-det"

    def test_wrong_phase_audit_detail_has_phase(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "wp-ph", "idle")
        client.post("/v1/sessions/wp-ph/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.submit_tool_results.failed")
        assert rec["detail"]["phase"] == "idle"


# ---------------------------------------------------------------------------
# Audit log on success
# ---------------------------------------------------------------------------


class TestSubmitToolResultsAuditSuccess:
    def _setup(self, storage_root: Path, session_id: str) -> TestClient:
        _write_session(storage_root, session_id)
        _seed_phase(storage_root, session_id, "waiting_for_tool")
        return _make_client(storage_root)

    def test_success_writes_accepted_audit_event(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "aud-ok1")
        client.post("/v1/sessions/aud-ok1/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.submit_tool_results.accepted" for r in records)

    def test_success_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "aud-ok2")
        client.post("/v1/sessions/aud-ok2/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.submit_tool_results.accepted")
        assert rec["detail"]["session_id"] == "aud-ok2"

    def test_success_audit_detail_has_count(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "aud-ok3")
        client.post("/v1/sessions/aud-ok3/submit_tool_results", json={"tool_results": _MULTI_RESULTS})
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.submit_tool_results.accepted")
        assert rec["detail"]["count"] == 2


# ---------------------------------------------------------------------------
# Event log writes
# ---------------------------------------------------------------------------


class TestSubmitToolResultsEventLog:
    def _setup(self, storage_root: Path, session_id: str) -> TestClient:
        _write_session(storage_root, session_id)
        _seed_phase(storage_root, session_id, "waiting_for_tool")
        return _make_client(storage_root)

    def test_tool_call_result_events_written(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "ev-s1")
        client.post("/v1/sessions/ev-s1/submit_tool_results", json={"tool_results": _MULTI_RESULTS})
        events = _read_events(storage_root, "ev-s1")
        result_events = [e for e in events if e.type == "tool_call.result"]
        assert len(result_events) == 2

    def test_tool_call_result_event_has_tool_use_id(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "ev-s2")
        client.post("/v1/sessions/ev-s2/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        events = _read_events(storage_root, "ev-s2")
        result_event = next(e for e in events if e.type == "tool_call.result")
        assert result_event.data["tool_use_id"] == "tu-1"

    def test_tool_call_result_event_has_content(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "ev-s3")
        client.post("/v1/sessions/ev-s3/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        events = _read_events(storage_root, "ev-s3")
        result_event = next(e for e in events if e.type == "tool_call.result")
        assert result_event.data["content"] == "ok"

    def test_tool_call_result_event_has_is_error(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "ev-s4")
        client.post("/v1/sessions/ev-s4/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        events = _read_events(storage_root, "ev-s4")
        result_event = next(e for e in events if e.type == "tool_call.result")
        assert result_event.data["is_error"] is False

    def test_phase_change_event_written(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "ev-pc1")
        client.post("/v1/sessions/ev-pc1/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        events = _read_events(storage_root, "ev-pc1")
        assert any(e.type == "session.phase_change" for e in events)

    def test_phase_change_before_is_waiting_for_tool(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "ev-pc2")
        client.post("/v1/sessions/ev-pc2/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        events = _read_events(storage_root, "ev-pc2")
        change = next(e for e in events if e.type == "session.phase_change" and e.data.get("reason") == "tool_result")
        assert change.data["before"] == "waiting_for_tool"

    def test_phase_change_after_is_waiting_for_model(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "ev-pc3")
        client.post("/v1/sessions/ev-pc3/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        events = _read_events(storage_root, "ev-pc3")
        change = next(e for e in events if e.type == "session.phase_change" and e.data.get("reason") == "tool_result")
        assert change.data["after"] == "waiting_for_model"


# ---------------------------------------------------------------------------
# Pool interaction
# ---------------------------------------------------------------------------


class TestSubmitToolResultsPoolInteraction:
    def _setup(self, storage_root: Path, session_id: str, pool: Any) -> TestClient:
        _write_session(storage_root, session_id)
        _seed_phase(storage_root, session_id, "waiting_for_tool")
        return _make_client_with_pool(storage_root, pool)

    def test_pool_wake_called_with_session_id(self, storage_root: Path) -> None:
        pool = _TrackingPool()
        client = self._setup(storage_root, "pi-s1", pool)
        client.post("/v1/sessions/pi-s1/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        assert "pi-s1" in pool.woken

    def test_pool_wake_failure_returns_422(self, storage_root: Path) -> None:
        pool = _ErrorPool()
        client = self._setup(storage_root, "pi-err1", pool)
        resp = client.post("/v1/sessions/pi-err1/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        assert resp.status_code == 422

    def test_pool_wake_failure_writes_audit_failed(self, storage_root: Path) -> None:
        pool = _ErrorPool()
        client = self._setup(storage_root, "pi-err2", pool)
        client.post("/v1/sessions/pi-err2/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.submit_tool_results.failed" for r in records)


# ---------------------------------------------------------------------------
# Router wiring via create_app
# ---------------------------------------------------------------------------


class TestSubmitToolResultsRouterWiring:
    def _make_full_client(self, storage_root: Path) -> TestClient:
        audit = FileAuditLog(storage_root)
        writer = _make_writer(storage_root)
        pool = _TrackingPool()
        app = create_app(audit, storage_root=storage_root, event_log=writer, harness_pool=pool)
        return TestClient(app, raise_server_exceptions=False)

    def test_route_exists_with_all_deps(self, storage_root: Path) -> None:
        _write_session(storage_root, "wire-s1")
        _seed_phase(storage_root, "wire-s1", "waiting_for_tool")
        client = self._make_full_client(storage_root)
        resp = client.post("/v1/sessions/wire-s1/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        assert resp.status_code == 202

    def test_no_harness_pool_route_absent(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        writer = _make_writer(storage_root)
        app = create_app(audit, storage_root=storage_root, event_log=writer)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/sessions/any/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        assert resp.status_code == 404

    def test_no_storage_root_route_absent(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        pool = _TrackingPool()
        app = create_app(audit, harness_pool=pool)  # type: ignore[arg-type]
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/sessions/any/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestSubmitToolResultsOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _setup(self, storage_root: Path, session_id: str) -> TestClient:
        _write_session(storage_root, session_id)
        _seed_phase(storage_root, session_id, "waiting_for_tool")
        return _make_client(storage_root)

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "otel-s1")
        client.post("/v1/sessions/otel-s1/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.submit_tool_results" in span_names

    def test_success_span_has_session_id_attribute(self, storage_root: Path) -> None:
        client = self._setup(storage_root, "otel-s2")
        client.post("/v1/sessions/otel-s2/submit_tool_results", json={"tool_results": _SINGLE_RESULT})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.submit_tool_results")
        assert span is not None
        assert span.attributes.get("session.id") == "otel-s2"

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        _make_client(storage_root).post(
            "/v1/sessions/otel-no-sess/submit_tool_results", json={"tool_results": _SINGLE_RESULT}
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.submit_tool_results")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR
