"""
Spawn endpoint conformance suite.

Tests cover:
  - POST /v1/x/sessions/{id}/spawn returns 200 when child caps ⊆ parent caps.
  - Response fields: child_session_id, parent_session_id, capabilities, status.
  - Empty child_capabilities is always valid.
  - Equal capability sets are valid.
  - Returns 403 with code "spawn_denied" when child requests a cap parent lacks.
  - Returns 403 with code "spawn_denied" on invalid capability string.
  - On denial, audit log entry written with event "session.spawn.denied".
  - Audit detail includes parent_session_id and escalating_caps on escalation.
  - Error message is included in response body on denial.
  - Missing required fields returns 422.
  - create_app wires spawn router when storage_root is supplied.
  - create_app omits spawn route when storage_root is None.
  - OTel span "session.spawn" emitted on success.
  - OTel span set to ERROR status on denial.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path, audit_log: FileAuditLog) -> TestClient:
    app = create_app(audit_log, storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _make_body(
    *,
    parent_capabilities: list[str] | None = None,
    child_capabilities: list[str] | None = None,
) -> dict:
    return {
        "parent_capabilities": parent_capabilities
        if parent_capabilities is not None
        else ["exec.shell", "fs.read", "net.listen"],
        "child_capabilities": child_capabilities
        if child_capabilities is not None
        else ["exec.shell"],
    }


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Integration tests: POST /v1/x/sessions/{id}/spawn — success
# ---------------------------------------------------------------------------


class TestSpawnEndpointSuccess:
    def test_returns_200_on_valid_subset(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post("/v1/x/sessions/parent-1/spawn", json=_make_body())
        assert resp.status_code == 200

    def test_response_has_child_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post("/v1/x/sessions/parent-2/spawn", json=_make_body()).json()
        assert "child_session_id" in body
        assert isinstance(body["child_session_id"], str)
        assert len(body["child_session_id"]) > 0

    def test_response_has_parent_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post("/v1/x/sessions/parent-3/spawn", json=_make_body()).json()
        assert body["parent_session_id"] == "parent-3"

    def test_response_has_capabilities(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-4/spawn",
            json=_make_body(child_capabilities=["exec.shell", "fs.read"]),
        ).json()
        assert sorted(body["capabilities"]) == ["exec.shell", "fs.read"]

    def test_response_status_spawned(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post("/v1/x/sessions/parent-5/spawn", json=_make_body()).json()
        assert body["status"] == "spawned"

    def test_empty_child_caps_always_valid(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-6/spawn",
            json=_make_body(child_capabilities=[]),
        )
        assert resp.status_code == 200

    def test_equal_caps_is_valid(self, storage_root: Path) -> None:
        caps = ["exec.shell", "fs.read"]
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-7/spawn",
            json=_make_body(parent_capabilities=caps, child_capabilities=caps),
        )
        assert resp.status_code == 200

    def test_parameterized_child_under_unrestricted_parent(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-8/spawn",
            json=_make_body(
                parent_capabilities=["fs.read"],
                child_capabilities=["fs.read[/workspace]"],
            ),
        )
        assert resp.status_code == 200

    def test_child_session_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        id1 = client.post("/v1/x/sessions/parent-9/spawn", json=_make_body()).json()[
            "child_session_id"
        ]
        id2 = client.post("/v1/x/sessions/parent-9/spawn", json=_make_body()).json()[
            "child_session_id"
        ]
        assert id1 != id2


# ---------------------------------------------------------------------------
# Integration tests: POST /v1/x/sessions/{id}/spawn — denial
# ---------------------------------------------------------------------------


class TestSpawnEndpointDenial:
    def test_returns_403_on_escalation(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-d1/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["exec.shell", "exec.sudo"],
            ),
        )
        assert resp.status_code == 403

    def test_error_code_is_spawn_denied(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-d2/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["exec.sudo"],
            ),
        ).json()
        assert body["error"]["code"] == "spawn_denied"

    def test_error_message_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-d3/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["exec.sudo"],
            ),
        ).json()
        assert "exec.sudo" in body["error"]["message"]

    def test_fully_disjoint_child_rejected(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-d4/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["net.listen"],
            ),
        )
        assert resp.status_code == 403

    def test_scoped_parent_cannot_cover_unscoped_child(self, storage_root: Path) -> None:
        # Parent has fs.read[/workspace] but child wants unrestricted fs.read
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-d5/spawn",
            json=_make_body(
                parent_capabilities=["fs.read[/workspace]"],
                child_capabilities=["fs.read"],
            ),
        )
        assert resp.status_code == 403

    def test_returns_403_on_invalid_parent_cap_string(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-d6/spawn",
            json=_make_body(
                parent_capabilities=["INVALID!!"],
                child_capabilities=["exec.shell"],
            ),
        )
        assert resp.status_code == 403

    def test_returns_403_on_invalid_child_cap_string(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-d7/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["INVALID!!"],
            ),
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Audit log tests
# ---------------------------------------------------------------------------


class TestSpawnAuditLog:
    def test_denial_writes_audit_log(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/audit-d1/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["exec.sudo"],
            ),
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.spawn.denied" for r in records)

    def test_denial_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/audit-d2/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["exec.sudo"],
            ),
        )
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "session.spawn.denied"
        )
        assert record["level"] == "error"

    def test_denial_audit_detail_has_parent_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/audit-parent-3/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["exec.sudo"],
            ),
        )
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "session.spawn.denied"
        )
        assert record["detail"]["parent_session_id"] == "audit-parent-3"

    def test_denial_audit_detail_has_escalating_caps(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/audit-d4/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["exec.shell", "exec.sudo"],
            ),
        )
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "session.spawn.denied"
        )
        assert "exec.sudo" in record["detail"]["escalating_caps"]

    def test_denial_audit_code_is_spawn_denied(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/audit-d5/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["exec.sudo"],
            ),
        )
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "session.spawn.denied"
        )
        assert record["code"] == "spawn_denied"

    def test_parse_error_writes_audit_log(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/audit-d6/spawn",
            json=_make_body(parent_capabilities=["INVALID!!"]),
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.spawn.denied" for r in records)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSpawnSchemaValidation:
    def test_missing_parent_capabilities_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/schema-1/spawn",
            json={"child_capabilities": ["exec.shell"]},
        )
        assert resp.status_code == 422

    def test_missing_child_capabilities_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/schema-2/spawn",
            json={"parent_capabilities": ["exec.shell"]},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestSpawnRouteWiring:
    def test_no_storage_root_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/any/spawn", json=_make_body())
        assert resp.status_code == 404

    def test_with_storage_root_route_exists(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post("/v1/x/sessions/any/spawn", json=_make_body())
        assert resp.status_code != 404


# FastAPI TestClient must be importable here
from fastapi.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# OTel span tests
# ---------------------------------------------------------------------------


class TestSpawnOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _make_client(self, storage_root: Path) -> TestClient:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_session_spawn_span(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/otel-1/spawn", json=_make_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.spawn" in span_names

    def test_denial_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._make_client(storage_root)
        client.post(
            "/v1/x/sessions/otel-2/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["exec.sudo"],
            ),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        spawn_span = spans.get("session.spawn")
        assert spawn_span is not None
        assert spawn_span.status.status_code == StatusCode.ERROR

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/otel-3/spawn", json=_make_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        spawn_span = spans.get("session.spawn")
        assert spawn_span is not None
        assert spawn_span.attributes["session.id"] == "otel-3"
