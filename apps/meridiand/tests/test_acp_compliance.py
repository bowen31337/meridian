"""
ACP CI compliance suite conformance tests.

Tests cover:
  - POST /v1/x/ci/acp-compliance returns 200 with run_id, status, test_count, tests on success.
  - status is "passed" when all compliance tests pass.
  - test_count equals the number of tests in the suite (35).
  - Each test entry has name, description, and status fields.
  - Each test entry status is "passed" when the suite succeeds.
  - Returns 422 with code "acp_compliance_failed" when any compliance test fails.
  - Error message names the failing test.
  - On failure, audit log entry written with event "ci.acp.compliance.failed".
  - Audit log detail contains run_id and failed_test.
  - OTel span "ci.acp.compliance" emitted on success.
  - OTel span set to ERROR status on compliance failure.
  - create_app wires compliance route when acp_targets is supplied.
  - create_app omits compliance route when acp_targets is None.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core_errors import HandlerOptions, install_error_handler
from fastapi import FastAPI
from fastapi.testclient import TestClient
from meridiand._acp_compliance import make_acp_compliance_router
from meridiand._app import create_app
from meridiand._audit import FileAuditLog

from tests._otel_shared import otel_exporter as _otel_exporter

_DEFAULT_TARGETS = {"hermes": "http://hermes.example.com/acp"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    storage_root: Path,
    suite_fn=None,
    use_create_app: bool = True,
) -> TestClient:
    audit = FileAuditLog(storage_root)
    if suite_fn is not None:
        app = FastAPI()
        install_error_handler(app, HandlerOptions(audit_log=audit))
        app.include_router(make_acp_compliance_router(audit_log=audit, suite_fn=suite_fn))
    else:
        app = create_app(audit, acp_targets=_DEFAULT_TARGETS)
    return TestClient(app, raise_server_exceptions=False)


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def _failing_suite() -> list[dict[str, Any]]:
    return [
        {
            "name": "message_delivery",
            "description": "injected failure",
            "status": "failed",
            "reason": "injected: status 503",
        }
    ]


async def _passing_suite() -> list[dict[str, Any]]:
    return [
        {
            "name": "message_delivery",
            "description": "minimal passing test",
            "status": "passed",
        }
    ]


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestAcpComplianceSuccess:
    def test_returns_200_on_all_passing(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/ci/acp-compliance")
        assert resp.status_code == 200

    def test_response_has_run_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/ci/acp-compliance").json()
        assert "run_id" in body
        assert isinstance(body["run_id"], str)
        assert len(body["run_id"]) > 0

    def test_response_status_passed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/ci/acp-compliance").json()
        assert body["status"] == "passed"

    def test_response_test_count_is_20(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/ci/acp-compliance").json()
        assert body["test_count"] == 35

    def test_tests_list_length_matches_test_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/ci/acp-compliance").json()
        assert len(body["tests"]) == body["test_count"]

    def test_each_test_has_required_fields(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/ci/acp-compliance").json()
        for t in body["tests"]:
            assert "name" in t
            assert "description" in t
            assert "status" in t

    def test_each_test_status_is_passed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/ci/acp-compliance").json()
        failed = [t for t in body["tests"] if t["status"] != "passed"]
        assert failed == [], f"unexpected failures: {failed}"

    def test_run_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = client.post("/v1/x/ci/acp-compliance").json()["run_id"]
        id2 = client.post("/v1/x/ci/acp-compliance").json()["run_id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


class TestAcpComplianceFailure:
    def test_returns_422_on_failing_test(self, storage_root: Path) -> None:
        client = _make_client(storage_root, suite_fn=_failing_suite)
        resp = client.post("/v1/x/ci/acp-compliance")
        assert resp.status_code == 422

    def test_error_code_is_acp_compliance_failed(self, storage_root: Path) -> None:
        client = _make_client(storage_root, suite_fn=_failing_suite)
        body = client.post("/v1/x/ci/acp-compliance").json()
        assert body["error"]["code"] == "acp_compliance_failed"

    def test_error_message_names_failing_test(self, storage_root: Path) -> None:
        client = _make_client(storage_root, suite_fn=_failing_suite)
        body = client.post("/v1/x/ci/acp-compliance").json()
        assert "message_delivery" in body["error"]["message"]

    def test_error_message_includes_reason(self, storage_root: Path) -> None:
        client = _make_client(storage_root, suite_fn=_failing_suite)
        body = client.post("/v1/x/ci/acp-compliance").json()
        assert "injected" in body["error"]["message"]

    def test_failure_writes_audit_log(self, storage_root: Path) -> None:
        client = _make_client(storage_root, suite_fn=_failing_suite)
        client.post("/v1/x/ci/acp-compliance")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "ci.acp.compliance.failed" for r in records)

    def test_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root, suite_fn=_failing_suite)
        client.post("/v1/x/ci/acp-compliance")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "ci.acp.compliance.failed"
        )
        assert record["level"] == "error"

    def test_audit_code_is_acp_compliance_failed(self, storage_root: Path) -> None:
        client = _make_client(storage_root, suite_fn=_failing_suite)
        client.post("/v1/x/ci/acp-compliance")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "ci.acp.compliance.failed"
        )
        assert record["code"] == "acp_compliance_failed"

    def test_audit_detail_has_run_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root, suite_fn=_failing_suite)
        client.post("/v1/x/ci/acp-compliance")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "ci.acp.compliance.failed"
        )
        assert "run_id" in record["detail"]
        assert isinstance(record["detail"]["run_id"], str)

    def test_audit_detail_has_failed_test(self, storage_root: Path) -> None:
        client = _make_client(storage_root, suite_fn=_failing_suite)
        client.post("/v1/x/ci/acp-compliance")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "ci.acp.compliance.failed"
        )
        assert record["detail"]["failed_test"] == "message_delivery"

    def test_audit_detail_has_tests_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root, suite_fn=_failing_suite)
        client.post("/v1/x/ci/acp-compliance")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "ci.acp.compliance.failed"
        )
        assert isinstance(record["detail"]["tests"], list)
        assert len(record["detail"]["tests"]) > 0


# ---------------------------------------------------------------------------
# OTel span tests
# ---------------------------------------------------------------------------


class TestAcpComplianceOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _make_client(self, storage_root: Path, suite_fn=None) -> TestClient:
        audit = FileAuditLog(storage_root)
        if suite_fn is not None:
            app = FastAPI()
            install_error_handler(app, HandlerOptions(audit_log=audit))
            app.include_router(make_acp_compliance_router(audit_log=audit, suite_fn=suite_fn))
        else:
            app = create_app(audit, acp_targets=_DEFAULT_TARGETS)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_compliance_span(self, storage_root: Path) -> None:
        client = self._make_client(storage_root, suite_fn=_passing_suite)
        client.post("/v1/x/ci/acp-compliance")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "ci.acp.compliance" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._make_client(storage_root, suite_fn=_failing_suite)
        client.post("/v1/x/ci/acp-compliance")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("ci.acp.compliance")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_not_error(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._make_client(storage_root, suite_fn=_passing_suite)
        client.post("/v1/x/ci/acp-compliance")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("ci.acp.compliance")
        assert span is not None
        assert span.status.status_code != StatusCode.ERROR

    def test_span_has_run_id_attribute(self, storage_root: Path) -> None:
        client = self._make_client(storage_root, suite_fn=_passing_suite)
        client.post("/v1/x/ci/acp-compliance")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("ci.acp.compliance")
        assert span is not None
        assert "compliance.run_id" in span.attributes


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestAcpComplianceRouteWiring:
    def test_no_acp_targets_no_compliance_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/ci/acp-compliance")
        assert resp.status_code == 404

    def test_with_acp_targets_compliance_route_exists(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, acp_targets=_DEFAULT_TARGETS)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/ci/acp-compliance")
        assert resp.status_code != 404
