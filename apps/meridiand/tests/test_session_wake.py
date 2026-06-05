"""
Explicit session wake endpoint conformance suite.

Tests cover:
  - POST /v1/sessions/{id}/wake returns 202 with session_id and harness_instance_id.
  - Returns 404 with code "session_wake_not_found" when session manifest is absent.
  - Session not found writes audit log entry with event "session.wake.failed".
  - Audit log detail includes session_id on failure.
  - harness_instance_id starts with "harness_" on success.
  - Success writes audit log entry with event "session.wake.accepted".
  - Audit log detail includes harness_instance_id on success.
  - OTel span "session.wake" is emitted on success.
  - OTel span is set to ERROR status on failure.
  - create_app wires the route when storage_root and harness_pool are supplied.
  - create_app omits the route when storage_root is None.
  - create_app omits the route when harness_pool is None.
  - Pool wake is called with the correct session_id.
  - Pool wake failure surfaces as a 422 response.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core_errors import HandlerOptions, install_error_handler
from fastapi import FastAPI
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._auth_middleware import AuthMiddleware
from meridiand._error_envelope_middleware import ErrorEnvelopeMiddleware
from meridiand._harness_pool import HarnessPool
from meridiand._session_wake import make_session_wake_router
from storage_reposit import LocalEventLogReader, PhaseProjection

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_session(storage_root: Path, session_id: str, data: dict[str, Any]) -> None:
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(json.dumps(data))


def _read_audit(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _default_session(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "status": "active",
        "created_at": "2024-01-01T00:00:00+00:00",
    }


async def _noop_run(session_id: str) -> tuple[int, int, str]:
    return 0, 0, ""


def _make_real_pool(storage_root: Path, audit_log: FileAuditLog) -> HarnessPool:
    reader = LocalEventLogReader(storage_root)
    projection = PhaseProjection(reader)
    return HarnessPool(
        num_workers=2,
        run_session=_noop_run,
        phase_reader=projection,
        audit_log=audit_log,
        storage_root=storage_root,
    )


def _make_client(storage_root: Path, audit_log: FileAuditLog) -> TestClient:
    pool = _make_real_pool(storage_root, audit_log)
    app = create_app(audit_log, storage_root=storage_root, harness_pool=pool)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Fake pools for specialised test scenarios
# ---------------------------------------------------------------------------


class _TrackingPool:
    """Records every session_id passed to wake()."""

    def __init__(self) -> None:
        self.woken: list[str] = []

    async def wake(self, session_id: str) -> None:
        self.woken.append(session_id)


class _ErrorPool:
    """Always raises on wake()."""

    async def wake(self, session_id: str) -> None:
        raise RuntimeError("injected pool failure")


def _make_client_with_pool(storage_root: Path, audit_log: FileAuditLog, pool: Any) -> TestClient:
    app = FastAPI()
    app.add_middleware(AuthMiddleware, audit_log=audit_log, bearer_token=None)
    app.add_middleware(ErrorEnvelopeMiddleware, audit_log=audit_log, hooks_dir=None)
    install_error_handler(app, HandlerOptions(audit_log=audit_log))
    app.include_router(
        make_session_wake_router(
            audit_log=audit_log,
            storage_root=storage_root,
            harness_pool=pool,
        )
    )
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Basic response shape
# ---------------------------------------------------------------------------


class TestSessionWakeSuccess:
    def test_returns_202_on_success(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "sw-sess1", _default_session("sw-sess1"))
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/sessions/sw-sess1/wake")
        assert resp.status_code == 202

    def test_response_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "sw-sess2", _default_session("sw-sess2"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/sessions/sw-sess2/wake").json()
        assert body["session_id"] == "sw-sess2"

    def test_response_has_harness_instance_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "sw-sess3", _default_session("sw-sess3"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/sessions/sw-sess3/wake").json()
        assert "harness_instance_id" in body

    def test_harness_instance_id_starts_with_harness(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "sw-sess4", _default_session("sw-sess4"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/sessions/sw-sess4/wake").json()
        assert body["harness_instance_id"].startswith("harness_")


# ---------------------------------------------------------------------------
# Session not found
# ---------------------------------------------------------------------------


class TestSessionWakeNotFound:
    def test_missing_session_returns_404(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/sessions/no-such-sess/wake")
        assert resp.status_code == 404

    def test_missing_session_error_code(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/sessions/no-sess2/wake").json()
        assert body["error"]["code"] == "session_wake_not_found"

    def test_missing_session_writes_audit_log(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/sessions/no-audit-sess/wake")
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.wake.failed" for r in records)

    def test_missing_session_audit_detail_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/sessions/no-detail-sess/wake")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.wake.failed")
        assert rec["detail"]["session_id"] == "no-detail-sess"


# ---------------------------------------------------------------------------
# Audit log on success
# ---------------------------------------------------------------------------


class TestSessionWakeAuditSuccess:
    def test_success_writes_accepted_audit_event(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "aud-sess1", _default_session("aud-sess1"))
        client = _make_client(storage_root, audit)
        client.post("/v1/sessions/aud-sess1/wake")
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.wake.accepted" for r in records)

    def test_success_audit_detail_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "aud-sess2", _default_session("aud-sess2"))
        client = _make_client(storage_root, audit)
        client.post("/v1/sessions/aud-sess2/wake")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.wake.accepted")
        assert rec["detail"]["session_id"] == "aud-sess2"

    def test_success_audit_detail_has_harness_instance_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "aud-sess3", _default_session("aud-sess3"))
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/sessions/aud-sess3/wake")
        harness_id = resp.json()["harness_instance_id"]
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.wake.accepted")
        assert rec["detail"]["harness_instance_id"] == harness_id


# ---------------------------------------------------------------------------
# Pool interaction
# ---------------------------------------------------------------------------


class TestSessionWakePoolInteraction:
    def test_pool_wake_called_with_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "pool-sess1", _default_session("pool-sess1"))
        pool = _TrackingPool()
        client = _make_client_with_pool(storage_root, audit, pool)
        client.post("/v1/sessions/pool-sess1/wake")
        assert "pool-sess1" in pool.woken

    def test_pool_wake_failure_returns_422(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "pool-err-sess", _default_session("pool-err-sess"))
        pool = _ErrorPool()
        client = _make_client_with_pool(storage_root, audit, pool)
        resp = client.post("/v1/sessions/pool-err-sess/wake")
        assert resp.status_code == 422

    def test_pool_wake_failure_writes_audit_log(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "pool-err-aud", _default_session("pool-err-aud"))
        pool = _ErrorPool()
        client = _make_client_with_pool(storage_root, audit, pool)
        client.post("/v1/sessions/pool-err-aud/wake")
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.wake.failed" for r in records)


# ---------------------------------------------------------------------------
# Router wiring via create_app
# ---------------------------------------------------------------------------


class TestSessionWakeRouterWiring:
    def test_route_exists_with_storage_root_and_pool(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "wire-sess1", _default_session("wire-sess1"))
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/sessions/wire-sess1/wake")
        assert resp.status_code != 404

    def test_no_storage_root_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        pool = _TrackingPool()
        app = create_app(audit, harness_pool=pool)  # type: ignore[arg-type]
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/sessions/any/wake")
        assert resp.status_code == 404

    def test_no_pool_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/sessions/any/wake")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestSessionWakeOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _make_client(self, storage_root: Path) -> TestClient:
        audit = FileAuditLog(storage_root)
        return _make_client(storage_root, audit)

    def test_success_emits_session_wake_span(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        _write_session(storage_root, "otel-sw-sess1", _default_session("otel-sw-sess1"))
        client.post("/v1/sessions/otel-sw-sess1/wake")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.wake" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._make_client(storage_root)
        client.post("/v1/sessions/no-such-otel-sess/wake")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        wake_span = spans.get("session.wake")
        assert wake_span is not None
        assert wake_span.status.status_code == StatusCode.ERROR
