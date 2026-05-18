"""
ACP outbound adapter conformance suite.

Tests cover:
  - POST /v1/x/sessions/{id}/acp/outbound returns 200 with status "delivered" on success.
  - Response fields: call_id, session_id, target, status, response.
  - Unrestricted acp.outbound capability covers any target.
  - Returns 403 with code "acp_outbound_denied" when session lacks acp.outbound[target].
  - Returns 403 when session holds acp.outbound[other] but not acp.outbound[target].
  - Returns 403 on invalid capability string.
  - Returns 403 when target not in registry.
  - Returns 502 with code "acp_outbound_failed" when peer transport raises.
  - On denial, audit log entry written with event "acp.outbound.denied".
  - Audit detail includes session_id, target, call_id on denial.
  - On transport failure, audit log entry written with event "acp.outbound.failed".
  - Audit detail includes session_id, target, call_id on failure.
  - OTel span "acp.outbound" emitted on success.
  - OTel span set to ERROR status on denial.
  - OTel span set to ERROR status on transport failure.
  - Span has session.id and acp.target attributes.
  - create_app wires ACP router when acp_targets is supplied.
  - create_app omits ACP route when acp_targets is None.
  - Missing required fields returns 422.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

_HERMES_URL = "http://hermes.example.com/acp"
_OPENCLAW_URL = "http://openclaw.example.com/acp"
_DEFAULT_TARGETS = {"hermes": _HERMES_URL, "openclaw": _OPENCLAW_URL}
_PEER_RESPONSE = {"ack": True, "peer": "hermes"}


class FakeAcpPeerClient:
    """Test double: returns preconfigured responses or raises on error URLs."""

    def __init__(
        self,
        responses: dict[str, dict[str, Any]] | None = None,
        error_urls: set[str] | None = None,
    ) -> None:
        self._responses = responses if responses is not None else {_HERMES_URL: _PEER_RESPONSE}
        self._error_urls = error_urls or set()

    async def call(self, url: str, message: dict[str, Any]) -> dict[str, Any]:
        if url in self._error_urls:
            raise RuntimeError(f"Connection refused: {url}")
        return self._responses.get(url, {"status": "ok"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    storage_root: Path,
    targets: dict[str, str] | None = None,
    peer_client: FakeAcpPeerClient | None = None,
) -> TestClient:
    audit = FileAuditLog(storage_root)
    app = create_app(
        audit,
        acp_targets=targets if targets is not None else _DEFAULT_TARGETS,
        acp_peer_client=peer_client if peer_client is not None else FakeAcpPeerClient(),
    )
    return TestClient(app, raise_server_exceptions=False)


def _make_body(
    *,
    target: str = "hermes",
    session_capabilities: list[str] | None = None,
    message: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "target": target,
        "session_capabilities": (
            session_capabilities
            if session_capabilities is not None
            else [f"acp.outbound[{target}]"]
        ),
        "message": message if message is not None else {"action": "ping"},
    }


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestAcpOutboundSuccess:
    def test_returns_200_on_valid_call(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/sessions/s1/acp/outbound", json=_make_body())
        assert resp.status_code == 200

    def test_response_has_call_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/s2/acp/outbound", json=_make_body()).json()
        assert "call_id" in body
        assert isinstance(body["call_id"], str)
        assert len(body["call_id"]) > 0

    def test_response_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/s3/acp/outbound", json=_make_body()).json()
        assert body["session_id"] == "s3"

    def test_response_has_target(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/s4/acp/outbound", json=_make_body()).json()
        assert body["target"] == "hermes"

    def test_response_status_is_delivered(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/sessions/s5/acp/outbound", json=_make_body()).json()
        assert body["status"] == "delivered"

    def test_response_includes_peer_response(self, storage_root: Path) -> None:
        peer = FakeAcpPeerClient(responses={_HERMES_URL: {"ack": True}})
        client = _make_client(storage_root, peer_client=peer)
        body = client.post("/v1/x/sessions/s6/acp/outbound", json=_make_body()).json()
        assert body["response"] == {"ack": True}

    def test_call_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = client.post("/v1/x/sessions/s7/acp/outbound", json=_make_body()).json()["call_id"]
        id2 = client.post("/v1/x/sessions/s8/acp/outbound", json=_make_body()).json()["call_id"]
        assert id1 != id2

    def test_unrestricted_cap_covers_any_target(self, storage_root: Path) -> None:
        # acp.outbound (no param) should gate-pass for any target
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/sessions/s9/acp/outbound",
            json=_make_body(
                target="hermes",
                session_capabilities=["acp.outbound"],
            ),
        )
        assert resp.status_code == 200

    def test_exact_target_cap_accepted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/sessions/s10/acp/outbound",
            json=_make_body(
                target="openclaw",
                session_capabilities=["acp.outbound[openclaw]"],
            ),
        )
        assert resp.status_code == 200

    def test_can_call_different_registered_targets(self, storage_root: Path) -> None:
        peer = FakeAcpPeerClient(
            responses={_HERMES_URL: {"src": "hermes"}, _OPENCLAW_URL: {"src": "openclaw"}}
        )
        client = _make_client(storage_root, peer_client=peer)
        body_hermes = client.post(
            "/v1/x/sessions/multi1/acp/outbound",
            json=_make_body(target="hermes", session_capabilities=["acp.outbound"]),
        ).json()
        body_openclaw = client.post(
            "/v1/x/sessions/multi2/acp/outbound",
            json=_make_body(target="openclaw", session_capabilities=["acp.outbound"]),
        ).json()
        assert body_hermes["response"] == {"src": "hermes"}
        assert body_openclaw["response"] == {"src": "openclaw"}


# ---------------------------------------------------------------------------
# Capability denial path
# ---------------------------------------------------------------------------


class TestAcpOutboundDenial:
    def test_returns_403_when_no_acp_outbound_cap(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/sessions/d1/acp/outbound",
            json=_make_body(session_capabilities=["exec.shell", "fs.read"]),
        )
        assert resp.status_code == 403

    def test_error_code_is_acp_outbound_denied(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/sessions/d2/acp/outbound",
            json=_make_body(session_capabilities=["exec.shell"]),
        ).json()
        assert body["error"]["code"] == "acp_outbound_denied"

    def test_error_message_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/sessions/d3/acp/outbound",
            json=_make_body(target="hermes", session_capabilities=["exec.shell"]),
        ).json()
        assert "hermes" in body["error"]["message"]

    def test_wrong_target_cap_rejected(self, storage_root: Path) -> None:
        # has acp.outbound[openclaw] but calls hermes
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/sessions/d4/acp/outbound",
            json=_make_body(
                target="hermes",
                session_capabilities=["acp.outbound[openclaw]"],
            ),
        )
        assert resp.status_code == 403

    def test_empty_caps_denied(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/sessions/d5/acp/outbound",
            json=_make_body(session_capabilities=[]),
        )
        assert resp.status_code == 403

    def test_returns_403_on_invalid_cap_string(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/sessions/d6/acp/outbound",
            json=_make_body(session_capabilities=["INVALID!!"]),
        )
        assert resp.status_code == 403

    def test_invalid_cap_code_is_acp_outbound_denied(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/sessions/d7/acp/outbound",
            json=_make_body(session_capabilities=["INVALID!!"]),
        ).json()
        assert body["error"]["code"] == "acp_outbound_denied"

    def test_returns_403_for_unregistered_target(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/sessions/d8/acp/outbound",
            json=_make_body(
                target="unknown",
                session_capabilities=["acp.outbound[unknown]"],
            ),
        )
        assert resp.status_code == 403

    def test_unregistered_target_code_is_acp_outbound_denied(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/sessions/d9/acp/outbound",
            json=_make_body(
                target="unknown",
                session_capabilities=["acp.outbound[unknown]"],
            ),
        ).json()
        assert body["error"]["code"] == "acp_outbound_denied"


# ---------------------------------------------------------------------------
# Transport failure path
# ---------------------------------------------------------------------------


class TestAcpOutboundTransportFailure:
    def test_returns_502_on_peer_error(self, storage_root: Path) -> None:
        peer = FakeAcpPeerClient(error_urls={_HERMES_URL})
        client = _make_client(storage_root, peer_client=peer)
        resp = client.post(
            "/v1/x/sessions/t1/acp/outbound",
            json=_make_body(target="hermes"),
        )
        assert resp.status_code == 502

    def test_transport_error_code_is_acp_outbound_failed(self, storage_root: Path) -> None:
        peer = FakeAcpPeerClient(error_urls={_HERMES_URL})
        client = _make_client(storage_root, peer_client=peer)
        body = client.post(
            "/v1/x/sessions/t2/acp/outbound",
            json=_make_body(target="hermes"),
        ).json()
        assert body["error"]["code"] == "acp_outbound_failed"

    def test_transport_error_message_in_response(self, storage_root: Path) -> None:
        peer = FakeAcpPeerClient(error_urls={_HERMES_URL})
        client = _make_client(storage_root, peer_client=peer)
        body = client.post(
            "/v1/x/sessions/t3/acp/outbound",
            json=_make_body(target="hermes"),
        ).json()
        assert "hermes" in body["error"]["message"]


# ---------------------------------------------------------------------------
# Audit log — denial
# ---------------------------------------------------------------------------


class TestAcpOutboundAuditDenial:
    def test_denial_writes_audit_log(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/x/sessions/aud1/acp/outbound",
            json=_make_body(session_capabilities=["exec.shell"]),
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "acp.outbound.denied" for r in records)

    def test_denial_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/x/sessions/aud2/acp/outbound",
            json=_make_body(session_capabilities=["exec.shell"]),
        )
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "acp.outbound.denied")
        assert record["level"] == "error"

    def test_denial_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/x/sessions/aud3/acp/outbound",
            json=_make_body(session_capabilities=["exec.shell"]),
        )
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "acp.outbound.denied")
        assert record["code"] == "acp_outbound_denied"

    def test_denial_audit_detail_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/x/sessions/aud-session-4/acp/outbound",
            json=_make_body(session_capabilities=["exec.shell"]),
        )
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "acp.outbound.denied")
        assert record["detail"]["session_id"] == "aud-session-4"

    def test_denial_audit_detail_has_target(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/x/sessions/aud5/acp/outbound",
            json=_make_body(target="hermes", session_capabilities=["exec.shell"]),
        )
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "acp.outbound.denied")
        assert record["detail"]["target"] == "hermes"

    def test_denial_audit_detail_has_call_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/x/sessions/aud6/acp/outbound",
            json=_make_body(session_capabilities=["exec.shell"]),
        )
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "acp.outbound.denied")
        assert "call_id" in record["detail"]
        assert isinstance(record["detail"]["call_id"], str)

    def test_parse_error_writes_denied_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/x/sessions/aud7/acp/outbound",
            json=_make_body(session_capabilities=["INVALID!!"]),
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "acp.outbound.denied" for r in records)


# ---------------------------------------------------------------------------
# Audit log — transport failure
# ---------------------------------------------------------------------------


class TestAcpOutboundAuditFailure:
    def test_transport_failure_writes_audit_log(self, storage_root: Path) -> None:
        peer = FakeAcpPeerClient(error_urls={_HERMES_URL})
        client = _make_client(storage_root, peer_client=peer)
        client.post("/v1/x/sessions/faud1/acp/outbound", json=_make_body(target="hermes"))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "acp.outbound.failed" for r in records)

    def test_transport_failure_audit_level_is_error(self, storage_root: Path) -> None:
        peer = FakeAcpPeerClient(error_urls={_HERMES_URL})
        client = _make_client(storage_root, peer_client=peer)
        client.post("/v1/x/sessions/faud2/acp/outbound", json=_make_body(target="hermes"))
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "acp.outbound.failed")
        assert record["level"] == "error"

    def test_transport_failure_audit_code(self, storage_root: Path) -> None:
        peer = FakeAcpPeerClient(error_urls={_HERMES_URL})
        client = _make_client(storage_root, peer_client=peer)
        client.post("/v1/x/sessions/faud3/acp/outbound", json=_make_body(target="hermes"))
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "acp.outbound.failed")
        assert record["code"] == "acp_outbound_failed"

    def test_transport_failure_audit_detail_has_session_id(self, storage_root: Path) -> None:
        peer = FakeAcpPeerClient(error_urls={_HERMES_URL})
        client = _make_client(storage_root, peer_client=peer)
        client.post("/v1/x/sessions/faud-session-4/acp/outbound", json=_make_body(target="hermes"))
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "acp.outbound.failed")
        assert record["detail"]["session_id"] == "faud-session-4"

    def test_transport_failure_audit_detail_has_target(self, storage_root: Path) -> None:
        peer = FakeAcpPeerClient(error_urls={_HERMES_URL})
        client = _make_client(storage_root, peer_client=peer)
        client.post("/v1/x/sessions/faud5/acp/outbound", json=_make_body(target="hermes"))
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "acp.outbound.failed")
        assert record["detail"]["target"] == "hermes"

    def test_transport_failure_audit_detail_has_call_id(self, storage_root: Path) -> None:
        peer = FakeAcpPeerClient(error_urls={_HERMES_URL})
        client = _make_client(storage_root, peer_client=peer)
        client.post("/v1/x/sessions/faud6/acp/outbound", json=_make_body(target="hermes"))
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "acp.outbound.failed")
        assert "call_id" in record["detail"]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestAcpOutboundSchema:
    def test_missing_target_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/sessions/schema1/acp/outbound",
            json={"session_capabilities": ["acp.outbound[hermes]"], "message": {}},
        )
        assert resp.status_code == 422

    def test_missing_session_capabilities_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/sessions/schema2/acp/outbound",
            json={"target": "hermes", "message": {}},
        )
        assert resp.status_code == 422

    def test_missing_message_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/sessions/schema3/acp/outbound",
            json={"target": "hermes", "session_capabilities": ["acp.outbound[hermes]"]},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestAcpOutboundRouteWiring:
    def test_no_acp_targets_returns_404(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/x/sessions/any/acp/outbound",
            json=_make_body(),
        )
        assert resp.status_code == 404

    def test_with_acp_targets_route_exists(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/x/sessions/any/acp/outbound",
            json=_make_body(),
        )
        assert resp.status_code != 404


from fastapi.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# OTel span tests
# ---------------------------------------------------------------------------


class TestAcpOutboundOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _make_client(self, storage_root: Path, peer_client: FakeAcpPeerClient | None = None) -> TestClient:
        audit = FileAuditLog(storage_root)
        app = create_app(
            audit,
            acp_targets=_DEFAULT_TARGETS,
            acp_peer_client=peer_client if peer_client is not None else FakeAcpPeerClient(),
        )
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_acp_outbound_span(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/otel1/acp/outbound", json=_make_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "acp.outbound" in span_names

    def test_denial_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._make_client(storage_root)
        client.post(
            "/v1/x/sessions/otel2/acp/outbound",
            json=_make_body(session_capabilities=["exec.shell"]),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("acp.outbound")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_transport_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        peer = FakeAcpPeerClient(error_urls={_HERMES_URL})
        client = self._make_client(storage_root, peer_client=peer)
        client.post("/v1/x/sessions/otel3/acp/outbound", json=_make_body(target="hermes"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("acp.outbound")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/otel-session/acp/outbound", json=_make_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("acp.outbound")
        assert span is not None
        assert span.attributes["session.id"] == "otel-session"

    def test_span_has_acp_target_attribute(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/otel5/acp/outbound", json=_make_body(target="hermes"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("acp.outbound")
        assert span is not None
        assert span.attributes["acp.target"] == "hermes"
