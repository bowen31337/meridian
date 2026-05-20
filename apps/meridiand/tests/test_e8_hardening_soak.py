"""
E8 hardening soak conformance suite (PRD §7.4).

Tests cover:
  - POST /v1/x/ci/e8-hardening-soak-run returns 200 on success.
  - Response body has run_id, status, total_sessions, resume_count, failure_count,
    resume_rate, layer_results, agent_count, channel_count, sessions_per_combo,
    sample_failures fields.
  - status is "passed" on success.
  - run_id has "e8_soak_" prefix.
  - resume_rate is 1.0 when all synthetic sessions recover.
  - resume_count equals total_sessions when all sessions recover.
  - failure_count is 0 when all sessions recover.
  - sample_failures is [] when all sessions recover.
  - layer_results contains all three layers: harness, tool_worker, daemon.
  - agent_count equals AGENT_COUNT constant.
  - channel_count equals CHANNEL_COUNT constant.
  - Returns 422 with code "e8_hardening_soak_failed" when resume_rate < threshold.
  - Error message mentions the resume rate percentage.
  - Error message mentions the threshold.
  - Error message mentions AGENT_COUNT and CHANNEL_COUNT.
  - On failure: audit log entry "e8.hardening.soak.run.failed" written.
  - On failure: audit entry level is "error".
  - On failure: audit detail has run_id, total_sessions, resume_count, failure_count,
    resume_rate, layer_results, message.
  - On success: audit log entry "e8.hardening.soak.ran" written.
  - On success: audit entry level is "info".
  - On success: audit detail has run_id, total_sessions, resume_count, failure_count,
    resume_rate, layer_results, agent_count, channel_count, sessions_per_combo.
  - OTel span "e8.hardening.soak.run" emitted on success.
  - OTel span "e8.hardening.soak.run" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries e8.hardening.soak.total_sessions, resume_count, failure_count,
    resume_rate, agent_count, channel_count attributes.
  - Span carries per-layer total and resumed attributes for all three layers.
  - create_app wires the e8 soak router when storage_root is supplied.
  - create_app omits the e8 soak route when storage_root is None.
  - E8HardeningSoakError has http_status 422.
  - AGENT_COUNT constant is 5.
  - CHANNEL_COUNT constant is 4.
  - SESSIONS_PER_COMBO constant is 500.
  - RESUME_RATE_THRESHOLD constant is 0.99.
  - _seed_harness_kill_session writes manifest with kill_layer "harness".
  - _seed_tool_worker_kill_session writes manifest with kill_layer "tool_worker"
    and pending_tool_call_id field.
  - _seed_daemon_kill_session writes manifest with kill_layer "daemon".
  - All three seeded session types include agent_id and channel_id in manifest.
  - _attempt_recovery returns True for a valid seeded session.
  - _attempt_recovery returns False when manifest is absent.
  - _attempt_recovery returns False when phase is a stop phase (idle/paused/terminated).
  - _attempt_recovery returns False on unexpected exception.
  - Sessions are distributed across all three kill layers via modulo-3 assignment.
  - layer_results totals sum to total_sessions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._e8_hardening_soak import (
    AGENT_COUNT,
    CHANNEL_COUNT,
    RESUME_RATE_THRESHOLD,
    SESSIONS_PER_COMBO,
    E8HardeningSoakError,
    _attempt_recovery,
    _seed_daemon_kill_session,
    _seed_harness_kill_session,
    _seed_tool_worker_kill_session,
    make_e8_hardening_soak_router,
)

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_soak_client(
    storage_root: Path,
    audit: FileAuditLog,
    *,
    sessions_per_combo: int,
) -> TestClient:
    router = make_e8_hardening_soak_router(
        audit_log=audit,
        storage_root=storage_root,
        _sessions_per_combo_override=sessions_per_combo,
    )
    app = create_app(audit)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def _post_soak(storage_root: Path, *, sessions_per_combo: int = 3) -> Any:
    audit = FileAuditLog(storage_root)
    client = _make_soak_client(storage_root, audit, sessions_per_combo=sessions_per_combo)
    return client.post("/v1/x/ci/e8-hardening-soak-run")


def _post_failing_soak(storage_root: Path, *, sessions_per_combo: int = 3) -> None:
    audit = FileAuditLog(storage_root)
    client = _make_soak_client(storage_root, audit, sessions_per_combo=sessions_per_combo)
    with patch(
        "meridiand._e8_hardening_soak._attempt_recovery",
        return_value=False,
    ):
        client.post("/v1/x/ci/e8-hardening-soak-run")


def _read_audit(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_agent_count_is_5(self) -> None:
        assert AGENT_COUNT == 5

    def test_channel_count_is_4(self) -> None:
        assert CHANNEL_COUNT == 4

    def test_sessions_per_combo_is_500(self) -> None:
        assert SESSIONS_PER_COMBO == 500

    def test_resume_rate_threshold_is_0_99(self) -> None:
        assert RESUME_RATE_THRESHOLD == 0.99

    def test_e8_hardening_soak_error_http_status_422(self) -> None:
        err = E8HardeningSoakError(message="fail", timestamp="t")
        assert err.http_status() == 422

    def test_e8_hardening_soak_error_code(self) -> None:
        err = E8HardeningSoakError(message="fail", timestamp="t")
        assert err.code == "e8_hardening_soak_failed"


# ---------------------------------------------------------------------------
# Unit: seed functions
# ---------------------------------------------------------------------------


class TestSeedHarnessKillSession:
    def test_writes_manifest_under_sessions_dir(self, storage_root: Path) -> None:
        _seed_harness_kill_session(storage_root, "sess-h-1", "agent_0", "cli", "t")
        assert (storage_root / "sessions" / "sess-h-1" / "manifest.json").exists()

    def test_manifest_kill_layer_is_harness(self, storage_root: Path) -> None:
        _seed_harness_kill_session(storage_root, "sess-h-2", "agent_0", "cli", "t")
        data = json.loads(
            (storage_root / "sessions" / "sess-h-2" / "manifest.json").read_text()
        )
        assert data["kill_layer"] == "harness"

    def test_manifest_has_agent_id(self, storage_root: Path) -> None:
        _seed_harness_kill_session(storage_root, "sess-h-3", "agent_2", "webhook", "t")
        data = json.loads(
            (storage_root / "sessions" / "sess-h-3" / "manifest.json").read_text()
        )
        assert data["agent_id"] == "agent_2"

    def test_manifest_has_channel_id(self, storage_root: Path) -> None:
        _seed_harness_kill_session(storage_root, "sess-h-4", "agent_0", "telegram", "t")
        data = json.loads(
            (storage_root / "sessions" / "sess-h-4" / "manifest.json").read_text()
        )
        assert data["channel_id"] == "telegram"

    def test_manifest_status_is_active(self, storage_root: Path) -> None:
        _seed_harness_kill_session(storage_root, "sess-h-5", "agent_0", "cli", "t")
        data = json.loads(
            (storage_root / "sessions" / "sess-h-5" / "manifest.json").read_text()
        )
        assert data["status"] == "active"


class TestSeedToolWorkerKillSession:
    def test_writes_manifest_under_sessions_dir(self, storage_root: Path) -> None:
        _seed_tool_worker_kill_session(storage_root, "sess-tw-1", "agent_0", "cli", "t")
        assert (storage_root / "sessions" / "sess-tw-1" / "manifest.json").exists()

    def test_manifest_kill_layer_is_tool_worker(self, storage_root: Path) -> None:
        _seed_tool_worker_kill_session(storage_root, "sess-tw-2", "agent_0", "cli", "t")
        data = json.loads(
            (storage_root / "sessions" / "sess-tw-2" / "manifest.json").read_text()
        )
        assert data["kill_layer"] == "tool_worker"

    def test_manifest_has_pending_tool_call_id(self, storage_root: Path) -> None:
        _seed_tool_worker_kill_session(storage_root, "sess-tw-3", "agent_0", "cli", "t")
        data = json.loads(
            (storage_root / "sessions" / "sess-tw-3" / "manifest.json").read_text()
        )
        assert "pending_tool_call_id" in data
        assert data["pending_tool_call_id"] == "tool_sess-tw-3"

    def test_manifest_has_agent_id(self, storage_root: Path) -> None:
        _seed_tool_worker_kill_session(
            storage_root, "sess-tw-4", "agent_3", "slack", "t"
        )
        data = json.loads(
            (storage_root / "sessions" / "sess-tw-4" / "manifest.json").read_text()
        )
        assert data["agent_id"] == "agent_3"

    def test_manifest_has_channel_id(self, storage_root: Path) -> None:
        _seed_tool_worker_kill_session(
            storage_root, "sess-tw-5", "agent_0", "slack", "t"
        )
        data = json.loads(
            (storage_root / "sessions" / "sess-tw-5" / "manifest.json").read_text()
        )
        assert data["channel_id"] == "slack"


class TestSeedDaemonKillSession:
    def test_writes_manifest_under_sessions_dir(self, storage_root: Path) -> None:
        _seed_daemon_kill_session(storage_root, "sess-d-1", "agent_0", "cli", "t")
        assert (storage_root / "sessions" / "sess-d-1" / "manifest.json").exists()

    def test_manifest_kill_layer_is_daemon(self, storage_root: Path) -> None:
        _seed_daemon_kill_session(storage_root, "sess-d-2", "agent_0", "cli", "t")
        data = json.loads(
            (storage_root / "sessions" / "sess-d-2" / "manifest.json").read_text()
        )
        assert data["kill_layer"] == "daemon"

    def test_manifest_has_agent_id(self, storage_root: Path) -> None:
        _seed_daemon_kill_session(storage_root, "sess-d-3", "agent_4", "webhook", "t")
        data = json.loads(
            (storage_root / "sessions" / "sess-d-3" / "manifest.json").read_text()
        )
        assert data["agent_id"] == "agent_4"

    def test_manifest_has_channel_id(self, storage_root: Path) -> None:
        _seed_daemon_kill_session(storage_root, "sess-d-4", "agent_0", "webhook", "t")
        data = json.loads(
            (storage_root / "sessions" / "sess-d-4" / "manifest.json").read_text()
        )
        assert data["channel_id"] == "webhook"

    def test_manifest_status_is_active(self, storage_root: Path) -> None:
        _seed_daemon_kill_session(storage_root, "sess-d-5", "agent_0", "cli", "t")
        data = json.loads(
            (storage_root / "sessions" / "sess-d-5" / "manifest.json").read_text()
        )
        assert data["status"] == "active"


# ---------------------------------------------------------------------------
# Unit: _attempt_recovery
# ---------------------------------------------------------------------------


class TestAttemptRecovery:
    def test_returns_true_for_valid_harness_kill_session(
        self, storage_root: Path
    ) -> None:
        _seed_harness_kill_session(storage_root, "rec-h-1", "agent_0", "cli", "t")
        assert _attempt_recovery(storage_root, "rec-h-1") is True

    def test_returns_true_for_valid_tool_worker_kill_session(
        self, storage_root: Path
    ) -> None:
        _seed_tool_worker_kill_session(storage_root, "rec-tw-1", "agent_0", "cli", "t")
        assert _attempt_recovery(storage_root, "rec-tw-1") is True

    def test_returns_true_for_valid_daemon_kill_session(
        self, storage_root: Path
    ) -> None:
        _seed_daemon_kill_session(storage_root, "rec-d-1", "agent_0", "cli", "t")
        assert _attempt_recovery(storage_root, "rec-d-1") is True

    def test_returns_false_when_manifest_absent(self, storage_root: Path) -> None:
        assert _attempt_recovery(storage_root, "no-such-session") is False

    def test_returns_false_for_idle_phase(self, storage_root: Path) -> None:
        import asyncio

        from storage_event_log import LocalEventLogWriter

        session_id = "rec-idle-1"
        _seed_harness_kill_session(storage_root, session_id, "agent_0", "cli", "t")

        async def _write() -> None:
            writer = LocalEventLogWriter(storage_root)
            await writer.append(
                session_id,
                "session.phase_change",
                {
                    "before": "created",
                    "after": "idle",
                    "reason": "test",
                    "timestamp": "t",
                },
            )

        asyncio.run(_write())
        assert _attempt_recovery(storage_root, session_id) is False

    def test_returns_false_for_terminated_phase(self, storage_root: Path) -> None:
        import asyncio

        from storage_event_log import LocalEventLogWriter

        session_id = "rec-term-1"
        _seed_daemon_kill_session(storage_root, session_id, "agent_0", "cli", "t")

        async def _write() -> None:
            writer = LocalEventLogWriter(storage_root)
            await writer.append(
                session_id,
                "session.phase_change",
                {
                    "before": "created",
                    "after": "terminated",
                    "reason": "test",
                    "timestamp": "t",
                },
            )

        asyncio.run(_write())
        assert _attempt_recovery(storage_root, session_id) is False

    def test_returns_false_for_paused_phase(self, storage_root: Path) -> None:
        import asyncio

        from storage_event_log import LocalEventLogWriter

        session_id = "rec-paused-1"
        _seed_tool_worker_kill_session(storage_root, session_id, "agent_0", "cli", "t")

        async def _write() -> None:
            writer = LocalEventLogWriter(storage_root)
            await writer.append(
                session_id,
                "session.phase_change",
                {
                    "before": "created",
                    "after": "paused",
                    "reason": "test",
                    "timestamp": "t",
                },
            )

        asyncio.run(_write())
        assert _attempt_recovery(storage_root, session_id) is False

    def test_returns_false_on_unexpected_exception(
        self, storage_root: Path, monkeypatch: Any
    ) -> None:
        _seed_harness_kill_session(storage_root, "rec-exc-1", "agent_0", "cli", "t")

        def _bad_reader(*a: Any, **kw: Any) -> None:
            raise RuntimeError("simulated read error")

        monkeypatch.setattr(
            "meridiand._e8_hardening_soak.LocalEventLogReader", _bad_reader
        )
        assert _attempt_recovery(storage_root, "rec-exc-1") is False


# ---------------------------------------------------------------------------
# Endpoint: success
# ---------------------------------------------------------------------------


class TestE8HardeningSoakSuccess:
    def test_returns_200_on_success(self, storage_root: Path) -> None:
        resp = _post_soak(storage_root)
        assert resp.status_code == 200

    def test_response_has_run_id(self, storage_root: Path) -> None:
        body = _post_soak(storage_root).json()
        assert "run_id" in body

    def test_run_id_has_e8_soak_prefix(self, storage_root: Path) -> None:
        body = _post_soak(storage_root).json()
        assert body["run_id"].startswith("e8_soak_")

    def test_response_status_is_passed(self, storage_root: Path) -> None:
        body = _post_soak(storage_root).json()
        assert body["status"] == "passed"

    def test_response_has_total_sessions(self, storage_root: Path) -> None:
        body = _post_soak(storage_root).json()
        assert "total_sessions" in body

    def test_total_sessions_equals_agents_times_channels_times_combo(
        self, storage_root: Path
    ) -> None:
        # sessions_per_combo=3 → 5 × 4 × 3 = 60
        body = _post_soak(storage_root, sessions_per_combo=3).json()
        assert body["total_sessions"] == AGENT_COUNT * CHANNEL_COUNT * 3

    def test_resume_count_equals_total_when_all_recover(
        self, storage_root: Path
    ) -> None:
        body = _post_soak(storage_root, sessions_per_combo=3).json()
        assert body["resume_count"] == body["total_sessions"]

    def test_failure_count_is_zero_when_all_recover(self, storage_root: Path) -> None:
        body = _post_soak(storage_root).json()
        assert body["failure_count"] == 0

    def test_resume_rate_is_1_when_all_recover(self, storage_root: Path) -> None:
        body = _post_soak(storage_root).json()
        assert body["resume_rate"] == pytest.approx(1.0)

    def test_sample_failures_is_empty_when_all_recover(
        self, storage_root: Path
    ) -> None:
        body = _post_soak(storage_root).json()
        assert body["sample_failures"] == []

    def test_response_has_layer_results(self, storage_root: Path) -> None:
        body = _post_soak(storage_root).json()
        assert "layer_results" in body

    def test_layer_results_has_all_three_layers(self, storage_root: Path) -> None:
        body = _post_soak(storage_root, sessions_per_combo=3).json()
        assert set(body["layer_results"].keys()) == {"harness", "tool_worker", "daemon"}

    def test_layer_results_totals_sum_to_total_sessions(
        self, storage_root: Path
    ) -> None:
        body = _post_soak(storage_root, sessions_per_combo=3).json()
        total_from_layers = sum(
            v["total"] for v in body["layer_results"].values()
        )
        assert total_from_layers == body["total_sessions"]

    def test_layer_results_each_layer_has_nonzero_total(
        self, storage_root: Path
    ) -> None:
        # sessions_per_combo=3 ensures one session per layer per (agent, channel) combo
        body = _post_soak(storage_root, sessions_per_combo=3).json()
        for layer in ("harness", "tool_worker", "daemon"):
            assert body["layer_results"][layer]["total"] > 0

    def test_response_agent_count_equals_constant(self, storage_root: Path) -> None:
        body = _post_soak(storage_root).json()
        assert body["agent_count"] == AGENT_COUNT

    def test_response_channel_count_equals_constant(self, storage_root: Path) -> None:
        body = _post_soak(storage_root).json()
        assert body["channel_count"] == CHANNEL_COUNT

    def test_response_has_sessions_per_combo(self, storage_root: Path) -> None:
        body = _post_soak(storage_root, sessions_per_combo=3).json()
        assert body["sessions_per_combo"] == 3

    def test_response_has_all_required_fields(self, storage_root: Path) -> None:
        body = _post_soak(storage_root).json()
        for field in (
            "run_id",
            "status",
            "total_sessions",
            "resume_count",
            "failure_count",
            "resume_rate",
            "layer_results",
            "agent_count",
            "channel_count",
            "sessions_per_combo",
            "sample_failures",
        ):
            assert field in body, f"missing field: {field}"


# ---------------------------------------------------------------------------
# Endpoint: failure (resume_rate below threshold)
# ---------------------------------------------------------------------------


class TestE8HardeningSoakFailure:
    def _failing_client(
        self, storage_root: Path, sessions_per_combo: int = 3
    ) -> TestClient:
        audit = FileAuditLog(storage_root)
        return _make_soak_client(
            storage_root, audit, sessions_per_combo=sessions_per_combo
        )

    def test_returns_422_when_rate_below_threshold(self, storage_root: Path) -> None:
        client = self._failing_client(storage_root)
        with patch(
            "meridiand._e8_hardening_soak._attempt_recovery",
            return_value=False,
        ):
            resp = client.post("/v1/x/ci/e8-hardening-soak-run")
        assert resp.status_code == 422

    def test_error_code_is_e8_hardening_soak_failed(self, storage_root: Path) -> None:
        client = self._failing_client(storage_root)
        with patch(
            "meridiand._e8_hardening_soak._attempt_recovery",
            return_value=False,
        ):
            body = client.post("/v1/x/ci/e8-hardening-soak-run").json()
        assert body["error"]["code"] == "e8_hardening_soak_failed"

    def test_error_message_mentions_rate(self, storage_root: Path) -> None:
        client = self._failing_client(storage_root)
        with patch(
            "meridiand._e8_hardening_soak._attempt_recovery",
            return_value=False,
        ):
            body = client.post("/v1/x/ci/e8-hardening-soak-run").json()
        assert "0.00%" in body["error"]["message"]

    def test_error_message_mentions_threshold(self, storage_root: Path) -> None:
        client = self._failing_client(storage_root)
        with patch(
            "meridiand._e8_hardening_soak._attempt_recovery",
            return_value=False,
        ):
            body = client.post("/v1/x/ci/e8-hardening-soak-run").json()
        assert "99%" in body["error"]["message"]

    def test_error_message_mentions_agent_count(self, storage_root: Path) -> None:
        client = self._failing_client(storage_root)
        with patch(
            "meridiand._e8_hardening_soak._attempt_recovery",
            return_value=False,
        ):
            body = client.post("/v1/x/ci/e8-hardening-soak-run").json()
        assert str(AGENT_COUNT) in body["error"]["message"]

    def test_error_message_mentions_sigkill_layers(self, storage_root: Path) -> None:
        client = self._failing_client(storage_root)
        with patch(
            "meridiand._e8_hardening_soak._attempt_recovery",
            return_value=False,
        ):
            body = client.post("/v1/x/ci/e8-hardening-soak-run").json()
        assert "harness" in body["error"]["message"]
        assert "tool-worker" in body["error"]["message"]
        assert "daemon" in body["error"]["message"]


# ---------------------------------------------------------------------------
# Audit log: success
# ---------------------------------------------------------------------------


class TestE8HardeningSoakAuditSuccess:
    def test_success_writes_audit_log_entry(self, storage_root: Path) -> None:
        _post_soak(storage_root)
        records = _read_audit(storage_root)
        assert any(r.get("event") == "e8.hardening.soak.ran" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        _post_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "e8.hardening.soak.ran")
        assert record["level"] == "info"

    def test_success_audit_detail_has_run_id(self, storage_root: Path) -> None:
        _post_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "e8.hardening.soak.ran")
        assert "run_id" in record["detail"]

    def test_success_audit_detail_has_total_sessions(self, storage_root: Path) -> None:
        _post_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "e8.hardening.soak.ran")
        assert "total_sessions" in record["detail"]

    def test_success_audit_detail_has_resume_count(self, storage_root: Path) -> None:
        _post_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "e8.hardening.soak.ran")
        assert "resume_count" in record["detail"]

    def test_success_audit_detail_has_failure_count(self, storage_root: Path) -> None:
        _post_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "e8.hardening.soak.ran")
        assert "failure_count" in record["detail"]

    def test_success_audit_detail_has_resume_rate(self, storage_root: Path) -> None:
        _post_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "e8.hardening.soak.ran")
        assert "resume_rate" in record["detail"]

    def test_success_audit_detail_has_layer_results(self, storage_root: Path) -> None:
        _post_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "e8.hardening.soak.ran")
        assert "layer_results" in record["detail"]

    def test_success_audit_detail_has_agent_count(self, storage_root: Path) -> None:
        _post_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "e8.hardening.soak.ran")
        assert record["detail"]["agent_count"] == AGENT_COUNT

    def test_success_audit_detail_has_channel_count(self, storage_root: Path) -> None:
        _post_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "e8.hardening.soak.ran")
        assert record["detail"]["channel_count"] == CHANNEL_COUNT

    def test_success_audit_detail_has_sessions_per_combo(
        self, storage_root: Path
    ) -> None:
        _post_soak(storage_root, sessions_per_combo=3)
        records = _read_audit(storage_root)
        record = next(r for r in records if r.get("event") == "e8.hardening.soak.ran")
        assert record["detail"]["sessions_per_combo"] == 3


# ---------------------------------------------------------------------------
# Audit log: failure
# ---------------------------------------------------------------------------


class TestE8HardeningSoakAuditFailure:
    def test_failure_writes_audit_log_entry(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        assert any(
            r.get("event") == "e8.hardening.soak.run.failed" for r in records
        )

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(
            r for r in records if r.get("event") == "e8.hardening.soak.run.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_detail_has_run_id(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(
            r for r in records if r.get("event") == "e8.hardening.soak.run.failed"
        )
        assert "run_id" in record["detail"]

    def test_failure_audit_detail_has_total_sessions(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(
            r for r in records if r.get("event") == "e8.hardening.soak.run.failed"
        )
        assert "total_sessions" in record["detail"]

    def test_failure_audit_detail_has_resume_count(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(
            r for r in records if r.get("event") == "e8.hardening.soak.run.failed"
        )
        assert "resume_count" in record["detail"]

    def test_failure_audit_detail_has_failure_count(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(
            r for r in records if r.get("event") == "e8.hardening.soak.run.failed"
        )
        assert "failure_count" in record["detail"]

    def test_failure_audit_detail_has_resume_rate(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(
            r for r in records if r.get("event") == "e8.hardening.soak.run.failed"
        )
        assert "resume_rate" in record["detail"]

    def test_failure_audit_detail_has_layer_results(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(
            r for r in records if r.get("event") == "e8.hardening.soak.run.failed"
        )
        assert "layer_results" in record["detail"]

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        _post_failing_soak(storage_root)
        records = _read_audit(storage_root)
        record = next(
            r for r in records if r.get("event") == "e8.hardening.soak.run.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel instrumentation
# ---------------------------------------------------------------------------


class TestE8HardeningSoakOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_e8_hardening_soak_run_span(
        self, storage_root: Path
    ) -> None:
        _post_soak(storage_root)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "e8.hardening.soak.run" in span_names

    def test_failure_emits_e8_hardening_soak_run_span(
        self, storage_root: Path
    ) -> None:
        _post_failing_soak(storage_root)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "e8.hardening.soak.run" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        _post_failing_soak(storage_root)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("e8.hardening.soak.run")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_span_has_total_sessions_attribute(self, storage_root: Path) -> None:
        _post_soak(storage_root, sessions_per_combo=3)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("e8.hardening.soak.run")
        assert span is not None
        assert "e8.hardening.soak.total_sessions" in span.attributes

    def test_span_has_resume_count_attribute(self, storage_root: Path) -> None:
        _post_soak(storage_root, sessions_per_combo=3)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("e8.hardening.soak.run")
        assert span is not None
        assert "e8.hardening.soak.resume_count" in span.attributes

    def test_span_has_failure_count_attribute(self, storage_root: Path) -> None:
        _post_soak(storage_root, sessions_per_combo=3)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("e8.hardening.soak.run")
        assert span is not None
        assert "e8.hardening.soak.failure_count" in span.attributes

    def test_span_has_resume_rate_attribute(self, storage_root: Path) -> None:
        _post_soak(storage_root, sessions_per_combo=3)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("e8.hardening.soak.run")
        assert span is not None
        assert "e8.hardening.soak.resume_rate" in span.attributes

    def test_span_has_agent_count_attribute(self, storage_root: Path) -> None:
        _post_soak(storage_root)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("e8.hardening.soak.run")
        assert span is not None
        assert span.attributes["e8.hardening.soak.agent_count"] == AGENT_COUNT

    def test_span_has_channel_count_attribute(self, storage_root: Path) -> None:
        _post_soak(storage_root)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("e8.hardening.soak.run")
        assert span is not None
        assert span.attributes["e8.hardening.soak.channel_count"] == CHANNEL_COUNT

    def test_span_has_harness_layer_attributes(self, storage_root: Path) -> None:
        _post_soak(storage_root, sessions_per_combo=3)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("e8.hardening.soak.run")
        assert span is not None
        assert "e8.hardening.soak.harness.total" in span.attributes
        assert "e8.hardening.soak.harness.resumed" in span.attributes

    def test_span_has_tool_worker_layer_attributes(self, storage_root: Path) -> None:
        _post_soak(storage_root, sessions_per_combo=3)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("e8.hardening.soak.run")
        assert span is not None
        assert "e8.hardening.soak.tool_worker.total" in span.attributes
        assert "e8.hardening.soak.tool_worker.resumed" in span.attributes

    def test_span_has_daemon_layer_attributes(self, storage_root: Path) -> None:
        _post_soak(storage_root, sessions_per_combo=3)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("e8.hardening.soak.run")
        assert span is not None
        assert "e8.hardening.soak.daemon.total" in span.attributes
        assert "e8.hardening.soak.daemon.resumed" in span.attributes


# ---------------------------------------------------------------------------
# Layer distribution
# ---------------------------------------------------------------------------


class TestLayerDistribution:
    def test_all_three_layers_have_nonzero_sessions(self, storage_root: Path) -> None:
        # sessions_per_combo=3 → i%3 assigns one session per layer per combo
        body = _post_soak(storage_root, sessions_per_combo=3).json()
        for layer in ("harness", "tool_worker", "daemon"):
            assert body["layer_results"][layer]["total"] > 0

    def test_layer_totals_sum_to_total_sessions(self, storage_root: Path) -> None:
        body = _post_soak(storage_root, sessions_per_combo=3).json()
        layer_sum = sum(v["total"] for v in body["layer_results"].values())
        assert layer_sum == body["total_sessions"]

    def test_layer_resumed_sum_equals_resume_count(self, storage_root: Path) -> None:
        body = _post_soak(storage_root, sessions_per_combo=3).json()
        resumed_sum = sum(v["resumed"] for v in body["layer_results"].values())
        assert resumed_sum == body["resume_count"]

    def test_harness_and_daemon_sessions_have_no_pending_tool_call(
        self, storage_root: Path
    ) -> None:
        _seed_harness_kill_session(storage_root, "ld-h", "agent_0", "cli", "t")
        _seed_daemon_kill_session(storage_root, "ld-d", "agent_0", "cli", "t")
        for sid in ("ld-h", "ld-d"):
            data = json.loads(
                (storage_root / "sessions" / sid / "manifest.json").read_text()
            )
            assert "pending_tool_call_id" not in data

    def test_tool_worker_sessions_have_pending_tool_call(
        self, storage_root: Path
    ) -> None:
        _seed_tool_worker_kill_session(storage_root, "ld-tw", "agent_0", "cli", "t")
        data = json.loads(
            (storage_root / "sessions" / "ld-tw" / "manifest.json").read_text()
        )
        assert "pending_tool_call_id" in data


# ---------------------------------------------------------------------------
# Router wiring
# ---------------------------------------------------------------------------


class TestE8HardeningSoakRouterWiring:
    def test_soak_route_registered_with_storage_root(
        self, storage_root: Path
    ) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        routes = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert "/v1/x/ci/e8-hardening-soak-run" in routes

    def test_no_storage_root_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        routes = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert "/v1/x/ci/e8-hardening-soak-run" not in routes
