"""Unit coverage for api-capabilities telemetry and the list-failure branch."""

from __future__ import annotations

from opentelemetry import trace
from api_capabilities._registry import CapabilityRegistry
from api_capabilities._telemetry import get_tracer

from tests.conftest import CapturingAuditLog, MockTracer, make_app
from fastapi.testclient import TestClient


def test_get_tracer_returns_tracer() -> None:
    assert isinstance(get_tracer(), trace.Tracer)


class _ExplodingRegistry(CapabilityRegistry):
    def all_capabilities(self):  # type: ignore[override]
        raise RuntimeError("registry boom")


def test_list_capabilities_failure_audits_and_500(mock_tracer: MockTracer) -> None:
    audit = CapturingAuditLog()
    client = TestClient(
        make_app(audit_log=audit, registry=_ExplodingRegistry()),
        raise_server_exceptions=False,
    )
    resp = client.get("/v1/x/capabilities")
    assert resp.status_code == 500
    assert any(e.event == "capabilities.list.failed" for e in audit.entries)
