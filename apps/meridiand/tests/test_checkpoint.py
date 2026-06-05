"""
Checkpoint endpoint conformance suite.

Tests cover:
  - POST /v1/x/sessions/{id}/checkpoint returns 200 with session_id, seq, status fields.
  - <seq>.json is written to $STORAGE_ROOT/checkpoints/<session_id>/<seq>.json.
  - latest.json is written atomically alongside <seq>.json.
  - Both files contain the serialized SessionCheckpoint payload.
  - latest.json reflects the most recent checkpoint when called multiple times.
  - Missing required fields returns 422 (FastAPI schema validation).
  - On write failure, returns 422 with code "checkpoint_failed".
  - On write failure, an audit log entry is written with event "checkpoint.create.failed".
  - Audit log detail includes session_id and seq.
  - OTel span "checkpoint.create" is emitted on success.
  - OTel span is set to ERROR status on failure.
  - create_app wires the checkpoint router when storage_root is supplied.
  - create_app omits the checkpoint route when storage_root is None.
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


def _make_body(
    *,
    seq: int = 1,
    phase: str = "thinking",
    pending_tool_calls: list | None = None,
    message_tail: list | None = None,
    usage: dict | None = None,
    taken_at: str = "2024-01-01T00:00:00+00:00",
) -> dict:
    return {
        "seq": seq,
        "phase": phase,
        "pending_tool_calls": pending_tool_calls or [],
        "message_tail": message_tail or [{"role": "assistant", "content": "hello"}],
        "usage": usage or {"input_tokens": 10, "output_tokens": 5},
        "taken_at": taken_at,
    }


def _make_client(storage_root: Path, audit_log: FileAuditLog) -> TestClient:
    app = create_app(audit_log, storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Integration tests: POST /v1/x/sessions/{id}/checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointEndpoint:
    def test_returns_200_on_success(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/sess1/checkpoint", json=_make_body(seq=1))
        assert resp.status_code == 200

    def test_response_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sess2/checkpoint", json=_make_body()).json()
        assert body["session_id"] == "sess2"

    def test_response_has_seq(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sess3/checkpoint", json=_make_body(seq=7)).json()
        assert body["seq"] == 7

    def test_response_status_saved(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sess4/checkpoint", json=_make_body()).json()
        assert body["status"] == "saved"

    def test_seq_json_file_written(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/sess5/checkpoint", json=_make_body(seq=3))
        assert (storage_root / "checkpoints" / "sess5" / "3.json").exists()

    def test_latest_json_file_written(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/sess6/checkpoint", json=_make_body(seq=1))
        assert (storage_root / "checkpoints" / "sess6" / "latest.json").exists()

    def test_seq_json_content_matches_body(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        payload = _make_body(seq=5, phase="tool_use")
        client.post("/v1/x/sessions/sess7/checkpoint", json=payload)
        data = json.loads((storage_root / "checkpoints" / "sess7" / "5.json").read_text())
        assert data["seq"] == 5
        assert data["phase"] == "tool_use"

    def test_latest_json_content_matches_body(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        payload = _make_body(seq=2, phase="waiting")
        client.post("/v1/x/sessions/sess8/checkpoint", json=payload)
        data = json.loads((storage_root / "checkpoints" / "sess8" / "latest.json").read_text())
        assert data["seq"] == 2
        assert data["phase"] == "waiting"

    def test_message_tail_persisted(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        tail = [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "pong"}]
        client.post("/v1/x/sessions/sess9/checkpoint", json=_make_body(message_tail=tail))
        data = json.loads((storage_root / "checkpoints" / "sess9" / "latest.json").read_text())
        assert data["message_tail"] == tail

    def test_pending_tool_calls_persisted(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        calls = [{"id": "tc1", "name": "bash", "input": {"cmd": "ls"}}]
        client.post(
            "/v1/x/sessions/sess10/checkpoint",
            json=_make_body(pending_tool_calls=calls),
        )
        data = json.loads((storage_root / "checkpoints" / "sess10" / "latest.json").read_text())
        assert data["pending_tool_calls"] == calls

    def test_usage_persisted(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        usage = {"input_tokens": 100, "output_tokens": 42, "cache_read_input_tokens": 8}
        client.post("/v1/x/sessions/sess11/checkpoint", json=_make_body(usage=usage))
        data = json.loads((storage_root / "checkpoints" / "sess11" / "latest.json").read_text())
        assert data["usage"] == usage

    def test_latest_json_updated_on_second_checkpoint(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/update-sess/checkpoint", json=_make_body(seq=1))
        client.post(
            "/v1/x/sessions/update-sess/checkpoint",
            json=_make_body(seq=2, phase="tool_use"),
        )
        data = json.loads(
            (storage_root / "checkpoints" / "update-sess" / "latest.json").read_text()
        )
        assert data["seq"] == 2
        assert data["phase"] == "tool_use"

    def test_both_seq_files_exist_after_two_checkpoints(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/two-ckpt/checkpoint", json=_make_body(seq=1))
        client.post("/v1/x/sessions/two-ckpt/checkpoint", json=_make_body(seq=2))
        cp_dir = storage_root / "checkpoints" / "two-ckpt"
        assert (cp_dir / "1.json").exists()
        assert (cp_dir / "2.json").exists()

    def test_missing_seq_field_returns_422(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = _make_body()
        del body["seq"]
        resp = client.post("/v1/x/sessions/bad-body/checkpoint", json=body)
        assert resp.status_code == 422

    def test_missing_phase_field_returns_422(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = _make_body()
        del body["phase"]
        resp = client.post("/v1/x/sessions/bad-body2/checkpoint", json=body)
        assert resp.status_code == 422

    def test_failure_returns_422_with_code(self, storage_root: Path) -> None:
        # Place a file where the checkpoint dir would be to trigger mkdir failure.
        blocker_parent = storage_root / "checkpoints"
        blocker_parent.mkdir(parents=True, exist_ok=True)
        (blocker_parent / "fail-sess").write_text("block")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/fail-sess/checkpoint", json=_make_body())
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "checkpoint_failed"

    def test_failure_writes_audit_log(self, storage_root: Path) -> None:
        blocker_parent = storage_root / "checkpoints"
        blocker_parent.mkdir(parents=True, exist_ok=True)
        (blocker_parent / "audit-fail-sess").write_text("block")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/audit-fail-sess/checkpoint", json=_make_body())
        audit_path = storage_root / "audit.ndjson"
        assert audit_path.exists()
        records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        assert any(r.get("event") == "checkpoint.create.failed" for r in records)

    def test_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        blocker_parent = storage_root / "checkpoints"
        blocker_parent.mkdir(parents=True, exist_ok=True)
        (blocker_parent / "detail-fail-sess").write_text("block")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/detail-fail-sess/checkpoint", json=_make_body(seq=9))
        audit_path = storage_root / "audit.ndjson"
        records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        record = next(r for r in records if r.get("event") == "checkpoint.create.failed")
        assert record["detail"]["session_id"] == "detail-fail-sess"
        assert record["detail"]["seq"] == 9

    def test_no_storage_root_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/any/checkpoint", json=_make_body())
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# OTel span tests
# ---------------------------------------------------------------------------

# OTel provider is registered once in conftest.py; _otel_exporter is imported from there.


class TestCheckpointOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _make_client(self, storage_root: Path) -> TestClient:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_checkpoint_create_span(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/otel-sess/checkpoint", json=_make_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "checkpoint.create" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        blocker_parent = storage_root / "checkpoints"
        blocker_parent.mkdir(parents=True, exist_ok=True)
        (blocker_parent / "otel-fail-sess").write_text("block")
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/otel-fail-sess/checkpoint", json=_make_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        cp_span = spans.get("checkpoint.create")
        assert cp_span is not None
        assert cp_span.status.status_code == StatusCode.ERROR
