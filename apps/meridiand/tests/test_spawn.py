"""
Spawn endpoint conformance suite.

Tests cover:
  - POST /v1/x/sessions/{id}/spawn returns 201 when child caps ⊆ parent caps
    and parent holds agent.spawn.
  - Response fields: child_session_id, parent_session_id, capabilities, status.
  - Empty child_capabilities is always valid (parent still needs agent.spawn).
  - Equal capability sets are valid.
  - Returns 403 with code "spawn_denied" when parent lacks agent.spawn capability.
  - Returns 403 with code "spawn_denied" when child requests a cap parent lacks.
  - Returns 403 with code "spawn_denied" on invalid capability string.
  - agent.spawn[param] (parameterized) also passes the spawn gate.
  - On denial, audit log entry written with event "session.spawn.denied".
  - Audit detail includes parent_session_id and escalating_caps on escalation.
  - Error message is included in response body on denial.
  - Missing required fields returns 422.
  - create_app wires spawn router when storage_root is supplied.
  - create_app omits spawn route when storage_root is None.
  - OTel span "session.spawn" emitted on success.
  - OTel span set to ERROR status on denial.
  - OTel span "child.session" emitted on successful spawn.
  - child.session span has a span link to parent session's root span when parent has traceparent.
  - child.session span has no span link when parent has no manifest.
  - child.session traceparent stored in child manifest.
  - Child session manifest written to storage_root/sessions/{id}/manifest.json.
  - output_schema is optional; stored in manifest when provided.
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
        else ["agent.spawn", "exec.shell", "fs.read", "net.listen"],
        "child_capabilities": child_capabilities
        if child_capabilities is not None
        else ["exec.shell"],
    }


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Integration tests: POST /v1/x/sessions/{id}/spawn — success
# ---------------------------------------------------------------------------


class TestSpawnEndpointSuccess:
    def test_returns_201_on_valid_subset(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post("/v1/x/sessions/parent-1/spawn", json=_make_body())
        assert resp.status_code == 201

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
        assert resp.status_code == 201

    def test_equal_caps_is_valid(self, storage_root: Path) -> None:
        caps = ["agent.spawn", "exec.shell", "fs.read"]
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-7/spawn",
            json=_make_body(parent_capabilities=caps, child_capabilities=caps),
        )
        assert resp.status_code == 201

    def test_parameterized_child_under_unrestricted_parent(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-8/spawn",
            json=_make_body(
                parent_capabilities=["agent.spawn", "fs.read"],
                child_capabilities=["fs.read[/workspace]"],
            ),
        )
        assert resp.status_code == 201

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
                parent_capabilities=["agent.spawn", "exec.shell"],
                child_capabilities=["exec.shell", "exec.sudo"],
            ),
        )
        assert resp.status_code == 403

    def test_error_code_is_spawn_denied(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-d2/spawn",
            json=_make_body(
                parent_capabilities=["agent.spawn", "exec.shell"],
                child_capabilities=["exec.sudo"],
            ),
        ).json()
        assert body["error"]["code"] == "spawn_denied"

    def test_error_message_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/parent-d3/spawn",
            json=_make_body(
                parent_capabilities=["agent.spawn", "exec.shell"],
                child_capabilities=["exec.sudo"],
            ),
        ).json()
        assert "exec.sudo" in body["error"]["message"]

    def test_fully_disjoint_child_rejected(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/parent-d4/spawn",
            json=_make_body(
                parent_capabilities=["agent.spawn", "exec.shell"],
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
                parent_capabilities=["agent.spawn", "fs.read[/workspace]"],
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
                parent_capabilities=["agent.spawn", "exec.shell"],
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
                parent_capabilities=["agent.spawn", "exec.shell"],
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
                parent_capabilities=["agent.spawn", "exec.shell"],
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
                parent_capabilities=["agent.spawn", "exec.shell"],
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
                parent_capabilities=["agent.spawn", "exec.shell"],
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
                parent_capabilities=["agent.spawn", "exec.shell"],
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
# agent.spawn capability gate
# ---------------------------------------------------------------------------


class TestSpawnAgentSpawnGate:
    def test_returns_403_when_parent_lacks_agent_spawn(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/gate-1/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell", "fs.read"],
                child_capabilities=["exec.shell"],
            ),
        )
        assert resp.status_code == 403

    def test_error_code_on_missing_agent_spawn(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/gate-2/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["exec.shell"],
            ),
        ).json()
        assert body["error"]["code"] == "spawn_denied"

    def test_error_message_mentions_agent_spawn(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/gate-3/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["exec.shell"],
            ),
        ).json()
        assert "agent.spawn" in body["error"]["message"]

    def test_parameterized_agent_spawn_passes_gate(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/gate-4/spawn",
            json=_make_body(
                parent_capabilities=["agent.spawn[worker]", "exec.shell"],
                child_capabilities=["exec.shell"],
            ),
        )
        assert resp.status_code == 201

    def test_gate_denial_writes_audit_log(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/gate-5/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["exec.shell"],
            ),
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.spawn.denied" for r in records)

    def test_gate_denial_audit_detail_has_parent_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/gate-session-6/spawn",
            json=_make_body(
                parent_capabilities=["exec.shell"],
                child_capabilities=["exec.shell"],
            ),
        )
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "session.spawn.denied"
        )
        assert record["detail"]["parent_session_id"] == "gate-session-6"

    def test_empty_parent_caps_denied(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post(
            "/v1/x/sessions/gate-7/spawn",
            json=_make_body(
                parent_capabilities=[],
                child_capabilities=[],
            ),
        )
        assert resp.status_code == 403


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


# ---------------------------------------------------------------------------
# Child session manifest persistence
# ---------------------------------------------------------------------------


class TestSpawnManifestPersistence:
    def _post(self, client, session_id: str, body: dict | None = None):
        return client.post(f"/v1/x/sessions/{session_id}/spawn", json=body or _make_body())

    def _manifest(self, storage_root: Path, child_session_id: str) -> dict:
        path = storage_root / "sessions" / child_session_id / "manifest.json"
        return json.loads(path.read_text())

    def test_manifest_file_created(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = self._post(client, "persist-1").json()
        child_id = body["child_session_id"]
        assert (storage_root / "sessions" / child_id / "manifest.json").exists()

    def test_manifest_has_correct_child_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = self._post(client, "persist-2").json()
        child_id = body["child_session_id"]
        manifest = self._manifest(storage_root, child_id)
        assert manifest["child_session_id"] == child_id

    def test_manifest_has_correct_parent_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = self._post(client, "persist-3").json()
        manifest = self._manifest(storage_root, body["child_session_id"])
        assert manifest["parent_session_id"] == "persist-3"

    def test_manifest_has_correct_capabilities(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = self._post(
            client,
            "persist-4",
            _make_body(child_capabilities=["exec.shell", "fs.read"]),
        ).json()
        manifest = self._manifest(storage_root, body["child_session_id"])
        assert sorted(manifest["capabilities"]) == ["exec.shell", "fs.read"]

    def test_manifest_status_is_spawned(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = self._post(client, "persist-5").json()
        manifest = self._manifest(storage_root, body["child_session_id"])
        assert manifest["status"] == "spawned"

    def test_manifest_output_schema_null_when_not_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = self._post(client, "persist-6").json()
        manifest = self._manifest(storage_root, body["child_session_id"])
        assert manifest["output_schema"] is None

    def test_manifest_not_written_on_denial(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        client.post(
            "/v1/x/sessions/persist-7/spawn",
            json=_make_body(
                parent_capabilities=["agent.spawn", "exec.shell"],
                child_capabilities=["exec.sudo"],
            ),
        )
        assert (
            not any((storage_root / "sessions").iterdir())
            if (storage_root / "sessions").exists()
            else True
        )


# ---------------------------------------------------------------------------
# output_schema field
# ---------------------------------------------------------------------------


class TestSpawnOutputSchema:
    def test_output_schema_optional_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        resp = client.post("/v1/x/sessions/schema-os-1/spawn", json=_make_body())
        assert resp.status_code == 201

    def test_output_schema_accepted_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        resp = client.post(
            "/v1/x/sessions/schema-os-2/spawn",
            json={**_make_body(), "output_schema": schema},
        )
        assert resp.status_code == 201

    def test_output_schema_persisted_in_manifest(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        body = client.post(
            "/v1/x/sessions/schema-os-3/spawn",
            json={**_make_body(), "output_schema": schema},
        ).json()
        manifest_path = storage_root / "sessions" / body["child_session_id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["output_schema"] == schema

    def test_output_schema_null_explicit(self, storage_root: Path) -> None:
        client = _make_client(storage_root, FileAuditLog(storage_root))
        body = client.post(
            "/v1/x/sessions/schema-os-4/spawn",
            json={**_make_body(), "output_schema": None},
        ).json()
        manifest_path = storage_root / "sessions" / body["child_session_id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["output_schema"] is None


# FastAPI TestClient must be importable here


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
                parent_capabilities=["agent.spawn", "exec.shell"],
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

    def test_child_session_span_emitted(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/otel-child-1/spawn", json=_make_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "child.session" in span_names

    def test_child_session_span_has_link_to_parent(self, storage_root: Path) -> None:
        # child.session should link back to parent session root span via stored traceparent.
        parent_id = "link-parent-sess"
        parent_trace_id = "cc" * 16  # 32 hex chars
        parent_span_id = "dd" * 8  # 16 hex chars
        parent_tp = f"00-{parent_trace_id}-{parent_span_id}-01"

        parent_dir = storage_root / "sessions" / parent_id
        parent_dir.mkdir(parents=True, exist_ok=True)
        (parent_dir / "manifest.json").write_text(
            json.dumps({"session_id": parent_id, "status": "active", "traceparent": parent_tp})
        )

        client = self._make_client(storage_root)
        client.post(f"/v1/x/sessions/{parent_id}/spawn", json=_make_body())

        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        child_span = spans.get("child.session")
        assert child_span is not None
        assert len(child_span.links) == 1
        assert child_span.links[0].context.trace_id == int(parent_trace_id, 16)

    def test_child_session_span_no_link_when_no_parent_manifest(self, storage_root: Path) -> None:
        # No parent manifest → no span link.
        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/no-manifest-parent/spawn", json=_make_body())

        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        child_span = spans.get("child.session")
        assert child_span is not None
        assert len(child_span.links) == 0

    def test_child_traceparent_stored_in_manifest(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        body = client.post("/v1/x/sessions/tp-parent/spawn", json=_make_body()).json()
        child_id = body["child_session_id"]

        manifest = json.loads((storage_root / "sessions" / child_id / "manifest.json").read_text())
        assert "traceparent" in manifest
        assert isinstance(manifest["traceparent"], str)
        assert manifest["traceparent"].startswith("00-")
