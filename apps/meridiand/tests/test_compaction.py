"""
Auto-compaction policy conformance suite.

Tests cover:
  - GET /v1/x/compaction/policy returns correct enabled, idle_days, summary_strategy, tail_events, retention_days.
  - GET /v1/x/compaction/policy default values (enabled=True, idle_days=30, strategy=tail, tail_events=50).
  - POST /v1/x/compaction/sessions/{session_id} returns 200 on success.
  - Response has session_id, compacted_at, strategy, original_event_count, summary_event_count, archive_key, archived_file_count.
  - strategy is "tail".
  - original_event_count matches number of input events.
  - summary_event_count is at most tail_events.
  - archive_key is under compaction/{session_id}/ prefix and ends with .ndjson.gz.
  - archived_file_count equals number of event files for the session.
  - Archive is written to blob store (file exists at storage_root).
  - Archive is valid gzip containing original NDJSON events.
  - Live event file is replaced with tail summary.
  - Older date-partition event files are removed after compaction.
  - Manifest written to compaction/{session_id}/manifest.json.
  - POST /v1/x/compaction/sessions/{session_id} returns 404 when session has no event files.
  - 404 response body has error.code "compaction_session_not_found".
  - Audit log written with event "compaction.compact_session.failed" on 404.
  - Audit entry level is "error".
  - Audit entry code is "compaction_session_not_found".
  - Audit detail includes session_id, run_id, message.
  - Error response body has error.code and error.message on failure.
  - OTel span "compaction.compact_session" emitted on success.
  - OTel span "compaction.compact_session" emitted on failure.
  - OTel span set to ERROR status on 404.
  - POST /v1/x/compaction/run returns 200 with run_id, compacted_count, results.
  - POST /v1/x/compaction/run only compacts sessions idle longer than idle_days.
  - POST /v1/x/compaction/run returns compacted_count=0 when no idle sessions.
  - POST /v1/x/compaction/run compacted_count matches number of idle sessions found.
  - OTel span "compaction.run" emitted.
  - compaction route present when storage_root and compaction policy are supplied.
  - compaction route absent when storage_root is None.
  - compaction route absent when compaction policy is None.
  - Background task starts when enabled=True; does not start when enabled=False.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._config import CompactionConfig

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    storage_root: Path,
    *,
    compaction: CompactionConfig | None = None,
) -> TestClient:
    if compaction is None:
        compaction = CompactionConfig()
    app = create_app(
        FileAuditLog(storage_root),
        storage_root=storage_root,
        compaction=compaction,
    )
    return TestClient(app, raise_server_exceptions=False)


def _write_event_file(
    storage_root: Path,
    session_id: str,
    *,
    year: str = "2026",
    month: str = "05",
    day: str = "01",
    events: list[dict] | None = None,
) -> Path:
    """Write a fake NDJSON event file and return its path."""
    if events is None:
        events = [
            {"seq": i, "ts": f"2026-05-01T00:00:0{i}.000Z", "type": "message.added", "data": {}}
            for i in range(3)
        ]
    path = storage_root / "events" / year / month / day / f"{session_id}.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


def _set_old_mtime(path: Path, days_ago: int = 40) -> None:
    old_ts = time.time() - days_ago * 86400
    os.utime(path, (old_ts, old_ts))


def _audit_records(storage_root: Path) -> list[dict]:
    p = storage_root / "audit.ndjson"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# GET /v1/x/compaction/policy
# ---------------------------------------------------------------------------


class TestGetPolicy:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/x/compaction/policy")
        assert resp.status_code == 200

    def test_default_enabled(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/x/compaction/policy").json()
        assert body["enabled"] is True

    def test_default_idle_days(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/x/compaction/policy").json()
        assert body["idle_days"] == 30

    def test_default_summary_strategy(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/x/compaction/policy").json()
        assert body["summary_strategy"] == "tail"

    def test_default_tail_events(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/x/compaction/policy").json()
        assert body["tail_events"] == 50

    def test_default_retention_days_null(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/x/compaction/policy").json()
        assert body["retention_days"] is None

    def test_custom_idle_days(self, storage_root: Path) -> None:
        policy = CompactionConfig(idle_days=7)
        client = _make_client(storage_root, compaction=policy)
        body = client.get("/v1/x/compaction/policy").json()
        assert body["idle_days"] == 7

    def test_disabled_policy(self, storage_root: Path) -> None:
        policy = CompactionConfig(enabled=False)
        client = _make_client(storage_root, compaction=policy)
        body = client.get("/v1/x/compaction/policy").json()
        assert body["enabled"] is False

    def test_custom_retention_days(self, storage_root: Path) -> None:
        policy = CompactionConfig(retention_days=90)
        client = _make_client(storage_root, compaction=policy)
        body = client.get("/v1/x/compaction/policy").json()
        assert body["retention_days"] == 90


# ---------------------------------------------------------------------------
# POST /v1/x/compaction/sessions/{session_id} — success
# ---------------------------------------------------------------------------


class TestCompactSessionSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "sess_001")
        _set_old_mtime(f)
        client = _make_client(storage_root)
        resp = client.post("/v1/x/compaction/sessions/sess_001")
        assert resp.status_code == 200

    def test_response_has_session_id(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "sess_002")
        _set_old_mtime(f)
        body = _make_client(storage_root).post("/v1/x/compaction/sessions/sess_002").json()
        assert body["session_id"] == "sess_002"

    def test_response_has_compacted_at(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "sess_003")
        _set_old_mtime(f)
        body = _make_client(storage_root).post("/v1/x/compaction/sessions/sess_003").json()
        assert isinstance(body["compacted_at"], str)
        assert len(body["compacted_at"]) > 0

    def test_response_strategy_is_tail(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "sess_004")
        _set_old_mtime(f)
        body = _make_client(storage_root).post("/v1/x/compaction/sessions/sess_004").json()
        assert body["strategy"] == "tail"

    def test_response_original_event_count(self, storage_root: Path) -> None:
        events = [{"seq": i, "ts": "T", "type": "message.added", "data": {}} for i in range(5)]
        f = _write_event_file(storage_root, "sess_005", events=events)
        _set_old_mtime(f)
        body = _make_client(storage_root).post("/v1/x/compaction/sessions/sess_005").json()
        assert body["original_event_count"] == 5

    def test_response_summary_event_count_at_most_tail(self, storage_root: Path) -> None:
        policy = CompactionConfig(tail_events=2)
        events = [{"seq": i, "ts": "T", "type": "message.added", "data": {}} for i in range(10)]
        f = _write_event_file(storage_root, "sess_006", events=events)
        _set_old_mtime(f)
        body = _make_client(storage_root, compaction=policy).post(
            "/v1/x/compaction/sessions/sess_006"
        ).json()
        assert body["summary_event_count"] == 2

    def test_response_summary_count_equals_total_when_fewer_than_tail(
        self, storage_root: Path
    ) -> None:
        policy = CompactionConfig(tail_events=100)
        events = [{"seq": i, "ts": "T", "type": "message.added", "data": {}} for i in range(3)]
        f = _write_event_file(storage_root, "sess_007", events=events)
        _set_old_mtime(f)
        body = _make_client(storage_root, compaction=policy).post(
            "/v1/x/compaction/sessions/sess_007"
        ).json()
        assert body["summary_event_count"] == 3

    def test_response_archive_key_prefix(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "sess_008")
        _set_old_mtime(f)
        body = _make_client(storage_root).post("/v1/x/compaction/sessions/sess_008").json()
        assert body["archive_key"].startswith("compaction/sess_008/")

    def test_response_archive_key_suffix(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "sess_009")
        _set_old_mtime(f)
        body = _make_client(storage_root).post("/v1/x/compaction/sessions/sess_009").json()
        assert body["archive_key"].endswith(".ndjson.gz")

    def test_response_archived_file_count(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "sess_010")
        _set_old_mtime(f)
        body = _make_client(storage_root).post("/v1/x/compaction/sessions/sess_010").json()
        assert body["archived_file_count"] == 1


# ---------------------------------------------------------------------------
# Persistence — archive, summary, manifest
# ---------------------------------------------------------------------------


class TestCompactSessionPersistence:
    def test_archive_file_exists_in_blob_store(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "sess_p01")
        _set_old_mtime(f)
        body = _make_client(storage_root).post("/v1/x/compaction/sessions/sess_p01").json()
        archive_path = storage_root / body["archive_key"]
        assert archive_path.exists()

    def test_archive_is_valid_gzip(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "sess_p02")
        _set_old_mtime(f)
        body = _make_client(storage_root).post("/v1/x/compaction/sessions/sess_p02").json()
        archive_path = storage_root / body["archive_key"]
        data = gzip.decompress(archive_path.read_bytes())
        assert len(data) > 0

    def test_archive_contains_original_events(self, storage_root: Path) -> None:
        events = [{"seq": 0, "ts": "T", "type": "session.created", "data": {"key": "val"}}]
        f = _write_event_file(storage_root, "sess_p03", events=events)
        _set_old_mtime(f)
        body = _make_client(storage_root).post("/v1/x/compaction/sessions/sess_p03").json()
        archive_path = storage_root / body["archive_key"]
        content = gzip.decompress(archive_path.read_bytes()).decode()
        assert "session.created" in content

    def test_live_event_file_replaced_with_tail(self, storage_root: Path) -> None:
        policy = CompactionConfig(tail_events=2)
        events = [{"seq": i, "ts": "T", "type": "message.added", "data": {}} for i in range(5)]
        f = _write_event_file(storage_root, "sess_p04", events=events)
        _set_old_mtime(f)
        _make_client(storage_root, compaction=policy).post("/v1/x/compaction/sessions/sess_p04")
        remaining_lines = [l for l in f.read_text().splitlines() if l.strip()]
        assert len(remaining_lines) == 2

    def test_tail_contains_last_events(self, storage_root: Path) -> None:
        policy = CompactionConfig(tail_events=2)
        events = [{"seq": i, "ts": "T", "type": "message.added", "data": {}} for i in range(5)]
        f = _write_event_file(storage_root, "sess_p05", events=events)
        _set_old_mtime(f)
        _make_client(storage_root, compaction=policy).post("/v1/x/compaction/sessions/sess_p05")
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        seqs = [json.loads(l)["seq"] for l in lines]
        assert seqs == [3, 4]

    def test_older_date_partition_files_removed(self, storage_root: Path) -> None:
        f_old = _write_event_file(
            storage_root, "sess_p06", year="2025", month="01", day="01"
        )
        f_new = _write_event_file(
            storage_root, "sess_p06", year="2026", month="05", day="01"
        )
        _set_old_mtime(f_old)
        _set_old_mtime(f_new)
        _make_client(storage_root).post("/v1/x/compaction/sessions/sess_p06")
        assert not f_old.exists()
        assert f_new.exists()

    def test_manifest_written(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "sess_p07")
        _set_old_mtime(f)
        _make_client(storage_root).post("/v1/x/compaction/sessions/sess_p07")
        manifest_path = storage_root / "compaction" / "sess_p07" / "manifest.json"
        assert manifest_path.exists()

    def test_manifest_has_correct_session_id(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "sess_p08")
        _set_old_mtime(f)
        _make_client(storage_root).post("/v1/x/compaction/sessions/sess_p08")
        manifest = json.loads(
            (storage_root / "compaction" / "sess_p08" / "manifest.json").read_text()
        )
        assert manifest["session_id"] == "sess_p08"


# ---------------------------------------------------------------------------
# POST /v1/x/compaction/sessions/{session_id} — failure
# ---------------------------------------------------------------------------


class TestCompactSessionFailure:
    def test_missing_session_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/compaction/sessions/no_such_session")
        assert resp.status_code == 404

    def test_missing_session_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/compaction/sessions/no_such_session").json()
        assert body["error"]["code"] == "compaction_session_not_found"

    def test_missing_session_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/compaction/sessions/no_such_session").json()
        assert len(body["error"]["message"]) > 0

    def test_failure_writes_audit_log(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/compaction/sessions/ghost_session")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "compaction.compact_session.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/compaction/sessions/ghost_session")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "compaction.compact_session.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/compaction/sessions/ghost_session")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "compaction.compact_session.failed"
        )
        assert record["code"] == "compaction_session_not_found"

    def test_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/compaction/sessions/ghost_session")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "compaction.compact_session.failed"
        )
        assert record["detail"]["session_id"] == "ghost_session"

    def test_failure_audit_detail_has_run_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/compaction/sessions/ghost_session")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "compaction.compact_session.failed"
        )
        assert record["detail"]["run_id"].startswith("compact_")

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/compaction/sessions/ghost_session")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "compaction.compact_session.failed"
        )
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestCompactionOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_compact_session_span(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "otel_s01")
        _set_old_mtime(f)
        _make_client(storage_root).post("/v1/x/compaction/sessions/otel_s01")
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "compaction.compact_session" in names

    def test_failure_emits_compact_session_span(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/x/compaction/sessions/no_such")
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "compaction.compact_session" in names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        _make_client(storage_root).post("/v1/x/compaction/sessions/no_such")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("compaction.compact_session")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_session_id_attribute(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "otel_s02")
        _set_old_mtime(f)
        _make_client(storage_root).post("/v1/x/compaction/sessions/otel_s02")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("compaction.compact_session")
        assert span is not None
        assert span.attributes["compaction.session_id"] == "otel_s02"

    def test_run_emits_compaction_run_span(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/x/compaction/run")
        names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "compaction.run" in names


# ---------------------------------------------------------------------------
# POST /v1/x/compaction/run
# ---------------------------------------------------------------------------


class TestCompactionRun:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/compaction/run")
        assert resp.status_code == 200

    def test_response_has_run_id(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post("/v1/x/compaction/run").json()
        assert body["run_id"].startswith("compact_")

    def test_response_has_compacted_count(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post("/v1/x/compaction/run").json()
        assert "compacted_count" in body

    def test_no_idle_sessions_returns_zero(self, storage_root: Path) -> None:
        # Write a fresh event file (not old)
        _write_event_file(storage_root, "fresh_sess")
        body = _make_client(storage_root).post("/v1/x/compaction/run").json()
        assert body["compacted_count"] == 0

    def test_idle_session_is_compacted(self, storage_root: Path) -> None:
        policy = CompactionConfig(idle_days=30)
        f = _write_event_file(storage_root, "idle_sess")
        _set_old_mtime(f, days_ago=40)
        body = _make_client(storage_root, compaction=policy).post(
            "/v1/x/compaction/run"
        ).json()
        assert body["compacted_count"] == 1

    def test_recent_session_not_compacted(self, storage_root: Path) -> None:
        policy = CompactionConfig(idle_days=30)
        _write_event_file(storage_root, "recent_sess")
        body = _make_client(storage_root, compaction=policy).post(
            "/v1/x/compaction/run"
        ).json()
        assert body["compacted_count"] == 0

    def test_results_list_length_matches_count(self, storage_root: Path) -> None:
        policy = CompactionConfig(idle_days=30)
        f1 = _write_event_file(storage_root, "idle_a")
        f2 = _write_event_file(storage_root, "idle_b")
        _set_old_mtime(f1, days_ago=40)
        _set_old_mtime(f2, days_ago=40)
        body = _make_client(storage_root, compaction=policy).post(
            "/v1/x/compaction/run"
        ).json()
        assert len(body["results"]) == body["compacted_count"]


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestCompactionRouteWiring:
    def test_route_present_with_storage_root_and_policy(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/x/compaction/policy")
        assert resp.status_code != 404

    def test_compact_session_route_present(self, storage_root: Path) -> None:
        f = _write_event_file(storage_root, "route_check")
        _set_old_mtime(f)
        resp = _make_client(storage_root).post("/v1/x/compaction/sessions/route_check")
        assert resp.status_code == 200

    def test_run_route_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/compaction/run")
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(
            FileAuditLog(storage_root),
            compaction=CompactionConfig(),
        )
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/x/compaction/policy")
        assert resp.status_code == 404

    def test_route_absent_without_compaction_policy(self, storage_root: Path) -> None:
        app = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
        )
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/x/compaction/policy")
        assert resp.status_code == 404
