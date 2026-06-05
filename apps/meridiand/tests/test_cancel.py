"""
Cancel endpoint conformance suite.

Tests cover:
  - POST /v1/x/sessions/{id}/cancel returns 200 when session has no descendants.
  - Response includes session_id matching the route parameter.
  - Response includes cancelled_sessions list (empty when no children).
  - Response includes cancelled_count equal to 0 when no children.
  - Direct children appear in cancelled_sessions after cancel.
  - cancelled_count equals len(cancelled_sessions).
  - Grandchildren (multi-level descendants) are included via tree walk.
  - Multiple children at the same level are all cancelled.
  - Each cancelled child's manifest status updated to "cancelled".
  - Audit entry written with event "child_session.completed" for each cancelled child.
  - Audit detail includes child_session_id, reason="cancelled", and session_id.
  - Audit level is "info" for child_session.completed entries.
  - Sessions dir absent → returns 200 with empty cancelled_sessions.
  - create_app wires cancel router when storage_root is supplied.
  - create_app omits cancel route when storage_root is None.
  - OTel span "session.cancel" emitted on success.
  - OTel span has session.id attribute.
  - On failure, 422 returned with error.code "cancel_failed".
  - On failure, error message is included in response body.
  - On failure, audit log entry written with event "session.cancel.failed".
  - On failure, audit detail includes session_id and message.
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


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _write_child_manifest(
    storage_root: Path,
    parent_id: str,
    child_id: str,
    *,
    status: str = "spawned",
) -> Path:
    session_dir = storage_root / "sessions" / child_id
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "child_session_id": child_id,
        "parent_session_id": parent_id,
        "capabilities": [],
        "output_schema": None,
        "created_at": "2024-01-01T00:00:00+00:00",
        "status": status,
    }
    path = session_dir / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Success: session with no descendants
# ---------------------------------------------------------------------------


class TestCancelNoDescendants:
    def test_returns_200_with_no_children(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post("/v1/x/sessions/orphan/cancel")
        assert resp.status_code == 200

    def test_response_has_session_id(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post("/v1/x/sessions/orphan-sid/cancel").json()
        assert body["session_id"] == "orphan-sid"

    def test_cancelled_sessions_empty(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post("/v1/x/sessions/orphan-list/cancel").json()
        assert body["cancelled_sessions"] == []

    def test_cancelled_count_zero(self, storage_root: Path) -> None:
        body = _make_client(storage_root).post("/v1/x/sessions/orphan-count/cancel").json()
        assert body["cancelled_count"] == 0

    def test_no_sessions_dir_returns_200(self, storage_root: Path) -> None:
        # No sessions dir written at all
        assert not (storage_root / "sessions").exists()
        resp = _make_client(storage_root).post("/v1/x/sessions/no-dir/cancel")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Success: direct children propagation
# ---------------------------------------------------------------------------


class TestCancelDirectChildren:
    def test_single_child_in_cancelled_sessions(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-a", "child-a1")
        body = _make_client(storage_root).post("/v1/x/sessions/parent-a/cancel").json()
        assert "child-a1" in body["cancelled_sessions"]

    def test_cancelled_count_matches_list_length(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-b", "child-b1")
        _write_child_manifest(storage_root, "parent-b", "child-b2")
        body = _make_client(storage_root).post("/v1/x/sessions/parent-b/cancel").json()
        assert body["cancelled_count"] == len(body["cancelled_sessions"])

    def test_multiple_children_all_cancelled(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-c", "child-c1")
        _write_child_manifest(storage_root, "parent-c", "child-c2")
        _write_child_manifest(storage_root, "parent-c", "child-c3")
        body = _make_client(storage_root).post("/v1/x/sessions/parent-c/cancel").json()
        assert set(body["cancelled_sessions"]) == {"child-c1", "child-c2", "child-c3"}
        assert body["cancelled_count"] == 3

    def test_unrelated_sessions_not_included(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-d", "child-d1")
        _write_child_manifest(storage_root, "other-parent", "child-other")
        body = _make_client(storage_root).post("/v1/x/sessions/parent-d/cancel").json()
        assert "child-other" not in body["cancelled_sessions"]


# ---------------------------------------------------------------------------
# Tree walk: grandchildren / multi-level
# ---------------------------------------------------------------------------


class TestCancelTreeWalk:
    def test_grandchild_included_in_cancelled_sessions(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "root", "child-1")
        _write_child_manifest(storage_root, "child-1", "grandchild-1")
        body = _make_client(storage_root).post("/v1/x/sessions/root/cancel").json()
        assert "grandchild-1" in body["cancelled_sessions"]

    def test_all_levels_cancelled(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "root-2", "child-2")
        _write_child_manifest(storage_root, "child-2", "grandchild-2")
        _write_child_manifest(storage_root, "grandchild-2", "great-grand-2")
        body = _make_client(storage_root).post("/v1/x/sessions/root-2/cancel").json()
        assert set(body["cancelled_sessions"]) == {"child-2", "grandchild-2", "great-grand-2"}

    def test_wide_and_deep_tree(self, storage_root: Path) -> None:
        # parent → child-w1, child-w2; child-w1 → grandchild-w
        _write_child_manifest(storage_root, "wide-root", "child-w1")
        _write_child_manifest(storage_root, "wide-root", "child-w2")
        _write_child_manifest(storage_root, "child-w1", "grandchild-w")
        body = _make_client(storage_root).post("/v1/x/sessions/wide-root/cancel").json()
        assert set(body["cancelled_sessions"]) == {"child-w1", "child-w2", "grandchild-w"}


# ---------------------------------------------------------------------------
# Manifest status update
# ---------------------------------------------------------------------------


class TestCancelManifestUpdate:
    def _read_manifest(self, storage_root: Path, child_id: str) -> dict:
        return json.loads((storage_root / "sessions" / child_id / "manifest.json").read_text())

    def test_child_manifest_status_set_to_cancelled(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-mu", "child-mu1")
        _make_client(storage_root).post("/v1/x/sessions/parent-mu/cancel")
        assert self._read_manifest(storage_root, "child-mu1")["status"] == "cancelled"

    def test_grandchild_manifest_status_set_to_cancelled(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-gm", "child-gm")
        _write_child_manifest(storage_root, "child-gm", "grandchild-gm")
        _make_client(storage_root).post("/v1/x/sessions/parent-gm/cancel")
        assert self._read_manifest(storage_root, "grandchild-gm")["status"] == "cancelled"

    def test_multiple_children_manifests_all_cancelled(self, storage_root: Path) -> None:
        for i in range(3):
            _write_child_manifest(storage_root, "parent-mm", f"child-mm{i}")
        _make_client(storage_root).post("/v1/x/sessions/parent-mm/cancel")
        for i in range(3):
            assert self._read_manifest(storage_root, f"child-mm{i}")["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Audit log: child_session.completed events
# ---------------------------------------------------------------------------


class TestCancelAuditLog:
    def test_child_session_completed_written_per_child(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-al", "child-al1")
        _make_client(storage_root).post("/v1/x/sessions/parent-al/cancel")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "child_session.completed" for r in records)

    def test_audit_level_is_info(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-ali", "child-ali")
        _make_client(storage_root).post("/v1/x/sessions/parent-ali/cancel")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "child_session.completed"
        )
        assert record["level"] == "info"

    def test_audit_detail_has_child_session_id(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-alc", "child-alc")
        _make_client(storage_root).post("/v1/x/sessions/parent-alc/cancel")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "child_session.completed"
        )
        assert record["detail"]["child_session_id"] == "child-alc"

    def test_audit_detail_reason_is_cancelled(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-alr", "child-alr")
        _make_client(storage_root).post("/v1/x/sessions/parent-alr/cancel")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "child_session.completed"
        )
        assert record["detail"]["reason"] == "cancelled"

    def test_audit_detail_has_session_id(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-als", "child-als")
        _make_client(storage_root).post("/v1/x/sessions/parent-als/cancel")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "child_session.completed"
        )
        assert record["detail"]["session_id"] == "parent-als"

    def test_one_audit_entry_per_child(self, storage_root: Path) -> None:
        for i in range(3):
            _write_child_manifest(storage_root, "parent-alm", f"child-alm{i}")
        _make_client(storage_root).post("/v1/x/sessions/parent-alm/cancel")
        completed_records = [
            r for r in _audit_records(storage_root) if r.get("event") == "child_session.completed"
        ]
        assert len(completed_records) == 3

    def test_no_audit_entry_when_no_children(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/x/sessions/no-children/cancel")
        records = _audit_records(storage_root)
        assert not any(r.get("event") == "child_session.completed" for r in records)


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestCancelRouteWiring:
    def test_no_storage_root_returns_404(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/any/cancel")
        assert resp.status_code == 404

    def test_with_storage_root_route_exists(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).post("/v1/x/sessions/any/cancel")
        assert resp.status_code != 404


# ---------------------------------------------------------------------------
# OTel span tests
# ---------------------------------------------------------------------------


class TestCancelOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_session_cancel_span(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/x/sessions/otel-c1/cancel")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.cancel" in span_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        _make_client(storage_root).post("/v1/x/sessions/otel-c2/cancel")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        cancel_span = spans.get("session.cancel")
        assert cancel_span is not None
        assert cancel_span.attributes["session.id"] == "otel-c2"

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        _write_child_manifest(storage_root, "parent-otel-fail", "child-otel-fail")
        manifest_path = storage_root / "sessions" / "child-otel-fail" / "manifest.json"
        manifest_path.chmod(0o444)
        try:
            _make_client(storage_root).post("/v1/x/sessions/parent-otel-fail/cancel")
        finally:
            manifest_path.chmod(0o644)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        cancel_span = spans.get("session.cancel")
        assert cancel_span is not None
        assert cancel_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Failure: error surfaced in response and written to audit log
# ---------------------------------------------------------------------------


class TestCancelFailureSurfacing:
    def test_failure_returns_422(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-fail", "child-fail")
        manifest_path = storage_root / "sessions" / "child-fail" / "manifest.json"
        manifest_path.chmod(0o444)
        try:
            resp = _make_client(storage_root).post("/v1/x/sessions/parent-fail/cancel")
            assert resp.status_code == 422
        finally:
            manifest_path.chmod(0o644)

    def test_failure_error_code_is_cancel_failed(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-fail2", "child-fail2")
        manifest_path = storage_root / "sessions" / "child-fail2" / "manifest.json"
        manifest_path.chmod(0o444)
        try:
            body = _make_client(storage_root).post("/v1/x/sessions/parent-fail2/cancel").json()
            assert body["error"]["code"] == "cancel_failed"
        finally:
            manifest_path.chmod(0o644)

    def test_failure_error_message_in_response(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-fail3", "child-fail3")
        manifest_path = storage_root / "sessions" / "child-fail3" / "manifest.json"
        manifest_path.chmod(0o444)
        try:
            body = _make_client(storage_root).post("/v1/x/sessions/parent-fail3/cancel").json()
            assert "message" in body["error"]
            assert len(body["error"]["message"]) > 0
        finally:
            manifest_path.chmod(0o644)

    def test_failure_writes_audit_entry(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-fail4", "child-fail4")
        manifest_path = storage_root / "sessions" / "child-fail4" / "manifest.json"
        manifest_path.chmod(0o444)
        try:
            _make_client(storage_root).post("/v1/x/sessions/parent-fail4/cancel")
        finally:
            manifest_path.chmod(0o644)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.cancel.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-fail5", "child-fail5")
        manifest_path = storage_root / "sessions" / "child-fail5" / "manifest.json"
        manifest_path.chmod(0o444)
        try:
            _make_client(storage_root).post("/v1/x/sessions/parent-fail5/cancel")
        finally:
            manifest_path.chmod(0o644)
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "session.cancel.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-fail6", "child-fail6")
        manifest_path = storage_root / "sessions" / "child-fail6" / "manifest.json"
        manifest_path.chmod(0o444)
        try:
            _make_client(storage_root).post("/v1/x/sessions/parent-fail6/cancel")
        finally:
            manifest_path.chmod(0o644)
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "session.cancel.failed"
        )
        assert record["detail"]["session_id"] == "parent-fail6"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        _write_child_manifest(storage_root, "parent-fail7", "child-fail7")
        manifest_path = storage_root / "sessions" / "child-fail7" / "manifest.json"
        manifest_path.chmod(0o444)
        try:
            _make_client(storage_root).post("/v1/x/sessions/parent-fail7/cancel")
        finally:
            manifest_path.chmod(0o644)
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "session.cancel.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0
