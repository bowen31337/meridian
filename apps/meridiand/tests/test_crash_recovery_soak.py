"""
Crash-recovery soak conformance suite (PRD §7.4).

Tests cover:
  - POST /v1/x/ci/crash-recovery-soak-run returns 200 on success.
  - Response body has run_id, status, crash_count, resume_count, failure_count,
    resume_rate, sample_failures fields.
  - status is "passed" on success.
  - run_id has "crash_soak_" prefix.
  - resume_rate is 1.0 when all synthetic sessions recover successfully.
  - resume_count equals crash_count when all sessions recover.
  - failure_count is 0 when all sessions recover.
  - sample_failures is [] when all sessions recover.
  - Returns 422 with code "crash_recovery_soak_failed" when resume_rate < threshold.
  - Error message mentions the resume rate percentage.
  - Error message mentions the threshold.
  - Error message mentions crash_count.
  - On failure: audit log entry "crash.recovery.soak.run.failed" written.
  - On failure: audit entry level is "error".
  - On failure: audit detail has run_id, crash_count, resume_count, failure_count,
    resume_rate, message.
  - On success: audit log entry "crash.recovery.soak.ran" written.
  - On success: audit entry level is "info".
  - On success: audit detail has run_id, crash_count, resume_count, failure_count,
    resume_rate.
  - OTel span "crash.recovery.soak.run" emitted on success.
  - OTel span "crash.recovery.soak.run" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries crash.recovery.soak.crash_count, resume_count, failure_count,
    resume_rate attributes.
  - create_app wires the soak router when storage_root is supplied.
  - create_app omits the soak route when storage_root is None.
  - CrashRecoverySoakError has http_status 422.
  - CRASH_COUNT constant is 10_000.
  - RESUME_RATE_THRESHOLD constant is 0.99.
  - _seed_synthetic_session writes manifest.json under sessions/{session_id}/.
  - _attempt_recovery returns True for a valid seeded session.
  - _attempt_recovery returns False when manifest is absent.
  - _attempt_recovery returns False when phase is a stop phase (idle/paused/terminated).
  - _attempt_recovery returns False on unexpected exception.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._crash_recovery_soak import (
    CRASH_COUNT,
    RESUME_RATE_THRESHOLD,
    CrashRecoverySoakError,
    _attempt_recovery,
    _seed_synthetic_session,
    make_crash_recovery_soak_router,
)
import pytest

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_soak_client(
    storage_root: Path,
    audit: FileAuditLog,
    *,
    crash_count: int,
) -> TestClient:
    """
    Build a TestClient with ONLY the crash-recovery soak router wired.

    create_app(audit) (no storage_root) installs all middleware/error-handling
    without registering the unoverridden 10k-crash soak route, so the
    _crash_count_override route we add here is the one that will match.
    """
    router = make_crash_recovery_soak_router(
        audit_log=audit,
        storage_root=storage_root,
        _crash_count_override=crash_count,
    )
    app = create_app(audit)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def _post_soak(storage_root: Path, *, crash_count: int = 5) -> Any:
    audit = FileAuditLog(storage_root)
    client = _make_soak_client(storage_root, audit, crash_count=crash_count)
    return client.post("/v1/x/ci/crash-recovery-soak-run")


def _post_failing_soak(storage_root: Path, *, crash_count: int = 10) -> None:
    """Run a soak where every recovery attempt fails."""
    audit = FileAuditLog(storage_root)
    client = _make_soak_client(storage_root, audit, crash_count=crash_count)
    with patch(
        "meridiand._crash_recovery_soak._attempt_recovery",
        return_value=False,
    ):
        client.post("/v1/x/ci/crash-recovery-soak-run")


def _read_audit(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_crash_count_is_10000(self) -> None:
        assert CRASH_COUNT == 10_000

    def test_resume_rate_threshold_is_0_99(self) -> None:
        assert RESUME_RATE_THRESHOLD == 0.99

    def test_crash_recovery_soak_error_http_status_422(self) -> None:
        err = CrashRecoverySoakError(message="fail", timestamp="t")
        assert err.http_status() == 422

    def test_crash_recovery_soak_error_code(self) -> None:
        err = CrashRecoverySoakError(message="fail", timestamp="t")
        assert err.code == "crash_recovery_soak_failed"


# ---------------------------------------------------------------------------
# Unit: _seed_synthetic_session
# ---------------------------------------------------------------------------


class TestSeedSyntheticSession:
    def test_writes_manifest_under_sessions_dir(self, storage_root: Path) -> None:
        _seed_synthetic_session(storage_root, "sess-seed-1", "2024-01-01T00:00:00+00:00")
        manifest_path = storage_root / "sessions" / "sess-seed-1" / "manifest.json"
        assert manifest_path.exists()

    def test_manifest_is_valid_json(self, storage_root: Path) -> None:
        _seed_synthetic_session(storage_root, "sess-seed-2", "2024-01-01T00:00:00+00:00")
        manifest_path = storage_root / "sessions" / "sess-seed-2" / "manifest.json"
        data = json.loads(manifest_path.read_text())
        assert isinstance(data, dict)

    def test_manifest_has_session_id(self, storage_root: Path) -> None:
        _seed_synthetic_session(storage_root, "sess-seed-3", "2024-01-01T00:00:00+00:00")
        manifest_path = storage_root / "sessions" / "sess-seed-3" / "manifest.json"
        data = json.loads(manifest_path.read_text())
        assert data["session_id"] == "sess-seed-3"

    def test_manifest_has_active_status(self, storage_root: Path) -> None:
        _seed_synthetic_session(storage_root, "sess-seed-4", "2024-01-01T00:00:00+00:00")
        manifest_path = storage_root / "sessions" / "sess-seed-4" / "manifest.json"
        data = json.loads(manifest_path.read_text())
        assert data["status"] == "active"

    def test_idempotent_for_same_session_id(self, storage_root: Path) -> None:
        _seed_synthetic_session(storage_root, "sess-seed-5", "t1")
        _seed_synthetic_session(storage_root, "sess-seed-5", "t2")
        manifest_path = storage_root / "sessions" / "sess-seed-5" / "manifest.json"
        assert manifest_path.exists()


# ---------------------------------------------------------------------------
# Unit: _attempt_recovery
# ---------------------------------------------------------------------------


class TestAttemptRecovery:
    def test_returns_true_for_valid_seeded_session(self, storage_root: Path) -> None:
        _seed_synthetic_session(storage_root, "rec-sess-1", "2024-01-01T00:00:00+00:00")
        assert _attempt_recovery(storage_root, "rec-sess-1") is True

    def test_returns_false_when_manifest_absent(self, storage_root: Path) -> None:
        assert _attempt_recovery(storage_root, "no-such-session") is False

    def test_returns_false_for_idle_phase(self, storage_root: Path) -> None:
        import asyncio

        from storage_event_log import LocalEventLogWriter

        session_id = "rec-idle-1"
        _seed_synthetic_session(storage_root, session_id, "2024-01-01T00:00:00+00:00")

        async def _write_event() -> None:
            writer = LocalEventLogWriter(storage_root)
            await writer.append(
                session_id,
                "session.phase_change",
                {"before": "created", "after": "idle", "reason": "test", "timestamp": "t"},
            )

        asyncio.run(_write_event())
        assert _attempt_recovery(storage_root, session_id) is False

    def test_returns_false_for_terminated_phase(self, storage_root: Path) -> None:
        import asyncio

        from storage_event_log import LocalEventLogWriter

        session_id = "rec-term-1"
        _seed_synthetic_session(storage_root, session_id, "2024-01-01T00:00:00+00:00")

        async def _write_event() -> None:
            writer = LocalEventLogWriter(storage_root)
            await writer.append(
                session_id,
                "session.phase_change",
                {"before": "created", "after": "terminated", "reason": "test", "timestamp": "t"},
            )

        asyncio.run(_write_event())
        assert _attempt_recovery(storage_root, session_id) is False

    def test_returns_false_for_paused_phase(self, storage_root: Path) -> None:
        import asyncio

        from storage_event_log import LocalEventLogWriter

        session_id = "rec-paused-1"
        _seed_synthetic_session(storage_root, session_id, "2024-01-01T00:00:00+00:00")

        async def _write_event() -> None:
            writer = LocalEventLogWriter(storage_root)
            await writer.append(
                session_id,
                "session.phase_change",
                {"before": "created", "after": "paused", "reason": "test", "timestamp": "t"},
            )

        asyncio.run(_write_event())
        assert _attempt_recovery(storage_root, session_id) is False

    def test_returns_false_on_unexpected_exception(
        self, storage_root: Path, monkeypatch: Any
    ) -> None:
        _seed_synthetic_session(storage_root, "rec-exc-1", "2024-01-01T00:00:00+00:00")

        def _bad_reader(*a: Any, **kw: Any) -> None:
            raise RuntimeError("simulated read error")

        monkeypatch.setattr("meridiand._crash_recovery_soak.LocalEventLogReader", _bad_reader)
        assert _attempt_recovery(storage_root, "rec-exc-1") is False


# ---------------------------------------------------------------------------
# Endpoint: success
# ---------------------------------------------------------------------------


class TestCrashRecoverySoakSuccess:
    def test_returns_200_on_success(self, storage_root: Path) -> None:
        resp = _post_soak(storage_root, crash_count=5)
        assert resp.status_code == 200

    def test_response_has_run_id(self, storage_root: Path) -> None:
        body = _post_soak(storage_root, crash_count=5).json()
        assert "run_id" in body

    def test_run_id_has_crash_soak_prefix(self, storage_root: Path) -> None:
        body = _post_soak(storage_root, crash_count=5).json()
        assert body["run_id"].startswith("crash_soak_")

    def test_response_status_is_passed(self, storage_root: Path) -> None:
        body = _post_soak(storage_root, crash_count=5).json()
        assert body["status"] == "passed"

    def test_response_has_crash_count(self, storage_root: Path) -> None:
        body = _post_soak(storage_root, crash_count=5).json()
        assert body["crash_count"] == 5

    def test_response_resume_count_equals_crash_count(self, storage_root: Path) -> None:
        body = _post_soak(storage_root, crash_count=5).json()
        assert body["resume_count"] == 5

    def test_response_failure_count_is_zero(self, storage_root: Path) -> None:
        body = _post_soak(storage_root, crash_count=5).json()
        assert body["failure_count"] == 0

    def test_response_resume_rate_is_1(self, storage_root: Path) -> None:
        body = _post_soak(storage_root, crash_count=5).json()
        assert body["resume_rate"] == pytest.approx(1.0)

    def test_response_sample_failures_is_empty(self, storage_root: Path) -> None:
        body = _post_soak(storage_root, crash_count=5).json()
        assert body["sample_failures"] == []

    def test_response_has_all_required_fields(self, storage_root: Path) -> None:
        body = _post_soak(storage_root, crash_count=3).json()
        for field in (
            "run_id",
            "status",
            "crash_count",
            "resume_count",
            "failure_count",
            "resume_rate",
            "sample_failures",
        ):
            assert field in body


# ---------------------------------------------------------------------------
# Endpoint: failure (resume_rate below threshold)
# ---------------------------------------------------------------------------


class TestCrashRecoverySoakFailure:
    def _failing_client(self, storage_root: Path, crash_count: int = 10) -> TestClient:
        audit = FileAuditLog(storage_root)
        return _make_soak_client(storage_root, audit, crash_count=crash_count)

    def test_returns_422_when_rate_below_threshold(self, storage_root: Path) -> None:
        client = self._failing_client(storage_root)
        with patch(
            "meridiand._crash_recovery_soak._attempt_recovery",
            return_value=False,
        ):
            resp = client.post("/v1/x/ci/crash-recovery-soak-run")
        assert resp.status_code == 422

    def test_error_code_is_crash_recovery_soak_failed(self, storage_root: Path) -> None:
        client = self._failing_client(storage_root)
        with patch(
            "meridiand._crash_recovery_soak._attempt_recovery",
            return_value=False,
        ):
            body = client.post("/v1/x/ci/crash-recovery-soak-run").json()
        assert body["error"]["code"] == "crash_recovery_soak_failed"

    def test_error_message_mentions_rate(self, storage_root: Path) -> None:
        client = self._failing_client(storage_root)
        with patch(
            "meridiand._crash_recovery_soak._attempt_recovery",
            return_value=False,
        ):
            body = client.post("/v1/x/ci/crash-recovery-soak-run").json()
        assert "0.00%" in body["error"]["message"]

    def test_error_message_mentions_threshold(self, storage_root: Path) -> None:
        client = self._failing_client(storage_root)
        with patch(
            "meridiand._crash_recovery_soak._attempt_recovery",
            return_value=False,
        ):
            body = client.post("/v1/x/ci/crash-recovery-soak-run").json()
        assert "99%" in body["error"]["message"]

    def test_error_message_mentions_crash_count(self, storage_root: Path) -> None:
        client = self._failing_client(storage_root, crash_count=10)
        with patch(
            "meridiand._crash_recovery_soak._attempt_recovery",
            return_value=False,
        ):
            body = client.post("/v1/x/ci/crash-recovery-soak-run").json()
        assert "10" in body["error"]["message"]


# ---------------------------------------------------------------------------
# Audit log: success
# ---------------------------------------------------------------------------


class TestCrashRecoverySoakAuditSuccess:
    def test_success_writes_audit_log_entry(self, storage_root: Path) -> None:
        _post_soak(storage_root, crash_count=3)
        records = _read_audit(storage_root)
        assert any(r.get("event") == "crash.recovery.soak.ran" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        _post_soak(storage_root, crash_count=3)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "crash.recovery.soak.ran")
        assert record["level"] == "info"

    def test_success_audit_detail_has_run_id(self, storage_root: Path) -> None:
        _post_soak(storage_root, crash_count=3)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "crash.recovery.soak.ran")
        assert "run_id" in record["detail"]

    def test_success_audit_detail_has_crash_count(self, storage_root: Path) -> None:
        _post_soak(storage_root, crash_count=3)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "crash.recovery.soak.ran")
        assert record["detail"]["crash_count"] == 3

    def test_success_audit_detail_has_resume_count(self, storage_root: Path) -> None:
        _post_soak(storage_root, crash_count=3)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "crash.recovery.soak.ran")
        assert "resume_count" in record["detail"]

    def test_success_audit_detail_has_failure_count(self, storage_root: Path) -> None:
        _post_soak(storage_root, crash_count=3)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "crash.recovery.soak.ran")
        assert "failure_count" in record["detail"]

    def test_success_audit_detail_has_resume_rate(self, storage_root: Path) -> None:
        _post_soak(storage_root, crash_count=3)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "crash.recovery.soak.ran")
        assert "resume_rate" in record["detail"]


# ---------------------------------------------------------------------------
# Audit log: failure
# ---------------------------------------------------------------------------


class TestCrashRecoverySoakAuditFailure:
    def test_failure_writes_audit_log_entry(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        assert any(r.get("event") == "crash.recovery.soak.run.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "crash.recovery.soak.run.failed")
        assert record["level"] == "error"

    def test_failure_audit_detail_has_run_id(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "crash.recovery.soak.run.failed")
        assert "run_id" in record["detail"]

    def test_failure_audit_detail_has_crash_count(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root, crash_count=10)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "crash.recovery.soak.run.failed")
        assert record["detail"]["crash_count"] == 10

    def test_failure_audit_detail_has_resume_count(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "crash.recovery.soak.run.failed")
        assert "resume_count" in record["detail"]

    def test_failure_audit_detail_has_failure_count(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "crash.recovery.soak.run.failed")
        assert "failure_count" in record["detail"]

    def test_failure_audit_detail_has_resume_rate(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "crash.recovery.soak.run.failed")
        assert "resume_rate" in record["detail"]

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "crash.recovery.soak.run.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel instrumentation
# ---------------------------------------------------------------------------


class TestCrashRecoverySoakOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_crash_recovery_soak_run_span(self, storage_root: Path) -> None:
        _post_soak(storage_root, crash_count=3)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "crash.recovery.soak.run" in span_names

    def test_failure_emits_crash_recovery_soak_run_span(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root, crash_count=5)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "crash.recovery.soak.run" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        _post_failing_soak(storage_root, crash_count=5)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("crash.recovery.soak.run")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_span_has_crash_count_attribute(self, storage_root: Path) -> None:
        _post_soak(storage_root, crash_count=7)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("crash.recovery.soak.run")
        assert span is not None
        assert span.attributes["crash.recovery.soak.crash_count"] == 7

    def test_span_has_resume_count_attribute(self, storage_root: Path) -> None:
        _post_soak(storage_root, crash_count=4)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("crash.recovery.soak.run")
        assert span is not None
        assert "crash.recovery.soak.resume_count" in span.attributes

    def test_span_has_failure_count_attribute(self, storage_root: Path) -> None:
        _post_soak(storage_root, crash_count=4)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("crash.recovery.soak.run")
        assert span is not None
        assert "crash.recovery.soak.failure_count" in span.attributes

    def test_span_has_resume_rate_attribute(self, storage_root: Path) -> None:
        _post_soak(storage_root, crash_count=4)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("crash.recovery.soak.run")
        assert span is not None
        assert "crash.recovery.soak.resume_rate" in span.attributes


# ---------------------------------------------------------------------------
# Router wiring (route-table inspection, no endpoint call)
# ---------------------------------------------------------------------------


class TestCrashRecoverySoakRouterWiring:
    def test_soak_route_registered_with_storage_root(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        routes = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert "/v1/x/ci/crash-recovery-soak-run" in routes

    def test_no_storage_root_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        routes = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert "/v1/x/ci/crash-recovery-soak-run" not in routes
