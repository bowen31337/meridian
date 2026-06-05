"""
Conformance suite for GET /v1/x/capabilities and POST /v1/x/capabilities.

Tests cover:
  - GET: 200 status, built-in capabilities, correct response schema, OTel span.
  - POST: 201 status, registered capabilities appear in GET, correct response schema, OTel span.
  - POST validation: 422 for invalid namespace format, built-in namespace conflict,
    duplicate namespace, empty capabilities list, invalid capability name.
  - Audit log: written on validation failure, not written on success.
  - OTel span: correct name, attributes, events, ERROR status on failure.
"""

from __future__ import annotations

from api_capabilities._registry import CapabilityRegistry
from fastapi.testclient import TestClient
from opentelemetry.trace import StatusCode

from tests.conftest import CapturingAuditLog, MockTracer, make_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(
    audit_log: CapturingAuditLog | None = None,
    registry: CapabilityRegistry | None = None,
) -> TestClient:
    return TestClient(
        make_app(audit_log=audit_log, registry=registry), raise_server_exceptions=False
    )


_VALID_BODY = {
    "namespace": "myplugin",
    "capabilities": [{"name": "do_thing", "param_expected": True}],
}


# ---------------------------------------------------------------------------
# GET /v1/x/capabilities — status and schema
# ---------------------------------------------------------------------------


class TestListStatus:
    def test_returns_200(self, mock_tracer: MockTracer) -> None:
        assert _client().get("/v1/x/capabilities").status_code == 200

    def test_response_has_capabilities_key(self, mock_tracer: MockTracer) -> None:
        assert "capabilities" in _client().get("/v1/x/capabilities").json()

    def test_capabilities_is_list(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/x/capabilities").json()
        assert isinstance(body["capabilities"], list)

    def test_capabilities_non_empty(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/x/capabilities").json()
        assert len(body["capabilities"]) > 0


class TestListBuiltins:
    def test_includes_fs_read(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/x/capabilities").json()
        ids = {c["id"] for c in body["capabilities"]}
        assert "fs.read" in ids

    def test_includes_exec_shell(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/x/capabilities").json()
        ids = {c["id"] for c in body["capabilities"]}
        assert "exec.shell" in ids

    def test_fs_read_param_expected_true(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/x/capabilities").json()
        cap = next(c for c in body["capabilities"] if c["id"] == "fs.read")
        assert cap["param_expected"] is True

    def test_exec_shell_param_expected_false(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/x/capabilities").json()
        cap = next(c for c in body["capabilities"] if c["id"] == "exec.shell")
        assert cap["param_expected"] is False

    def test_each_capability_has_required_fields(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/x/capabilities").json()
        for cap in body["capabilities"]:
            assert {"id", "namespace", "name", "param_expected"} <= cap.keys()

    def test_id_matches_namespace_dot_name(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/x/capabilities").json()
        for cap in body["capabilities"]:
            assert cap["id"] == f"{cap['namespace']}.{cap['name']}"


class TestListIncludesPlugins:
    def test_plugin_namespace_appears_after_registration(self, mock_tracer: MockTracer) -> None:
        registry = CapabilityRegistry()
        client = _client(registry=registry)
        client.post("/v1/x/capabilities", json=_VALID_BODY)
        body = client.get("/v1/x/capabilities").json()
        ids = {c["id"] for c in body["capabilities"]}
        assert "myplugin.do_thing" in ids


# ---------------------------------------------------------------------------
# GET /v1/x/capabilities — OTel
# ---------------------------------------------------------------------------


class TestListTelemetry:
    def test_span_name(self, mock_tracer: MockTracer) -> None:
        _client().get("/v1/x/capabilities")
        assert mock_tracer.spans[0].name == "capabilities.list"

    def test_span_has_invocation_event(self, mock_tracer: MockTracer) -> None:
        _client().get("/v1/x/capabilities")
        event_names = [e[0] for e in mock_tracer.spans[0].events]
        assert "capabilities.list.invocation" in event_names

    def test_invocation_event_count_is_positive(self, mock_tracer: MockTracer) -> None:
        _client().get("/v1/x/capabilities")
        ev = next(e for e in mock_tracer.spans[0].events if e[0] == "capabilities.list.invocation")
        assert ev[1]["count"] > 0

    def test_invocation_event_count_matches_response(self, mock_tracer: MockTracer) -> None:
        resp = _client().get("/v1/x/capabilities")
        ev = next(e for e in mock_tracer.spans[0].events if e[0] == "capabilities.list.invocation")
        assert ev[1]["count"] == len(resp.json()["capabilities"])

    def test_span_ended(self, mock_tracer: MockTracer) -> None:
        _client().get("/v1/x/capabilities")
        assert mock_tracer.spans[0].ended


# ---------------------------------------------------------------------------
# POST /v1/x/capabilities — status and schema
# ---------------------------------------------------------------------------


class TestRegisterStatus:
    def test_returns_201(self, mock_tracer: MockTracer) -> None:
        assert _client().post("/v1/x/capabilities", json=_VALID_BODY).status_code == 201

    def test_response_has_namespace(self, mock_tracer: MockTracer) -> None:
        body = _client().post("/v1/x/capabilities", json=_VALID_BODY).json()
        assert body["namespace"] == "myplugin"

    def test_response_has_capabilities(self, mock_tracer: MockTracer) -> None:
        body = _client().post("/v1/x/capabilities", json=_VALID_BODY).json()
        assert len(body["capabilities"]) == 1

    def test_response_capability_id(self, mock_tracer: MockTracer) -> None:
        body = _client().post("/v1/x/capabilities", json=_VALID_BODY).json()
        assert body["capabilities"][0]["id"] == "myplugin.do_thing"

    def test_response_capability_param_expected(self, mock_tracer: MockTracer) -> None:
        body = _client().post("/v1/x/capabilities", json=_VALID_BODY).json()
        assert body["capabilities"][0]["param_expected"] is True

    def test_multiple_capabilities_registered(self, mock_tracer: MockTracer) -> None:
        body_multi = {
            "namespace": "multi",
            "capabilities": [
                {"name": "alpha", "param_expected": False},
                {"name": "beta", "param_expected": True},
            ],
        }
        body = _client().post("/v1/x/capabilities", json=body_multi).json()
        assert len(body["capabilities"]) == 2


# ---------------------------------------------------------------------------
# POST /v1/x/capabilities — OTel
# ---------------------------------------------------------------------------


class TestRegisterTelemetry:
    def test_span_name(self, mock_tracer: MockTracer) -> None:
        _client().post("/v1/x/capabilities", json=_VALID_BODY)
        assert mock_tracer.spans[0].name == "capabilities.register"

    def test_span_has_namespace_attribute(self, mock_tracer: MockTracer) -> None:
        _client().post("/v1/x/capabilities", json=_VALID_BODY)
        assert mock_tracer.spans[0].attributes["namespace"] == "myplugin"

    def test_span_has_invocation_event(self, mock_tracer: MockTracer) -> None:
        _client().post("/v1/x/capabilities", json=_VALID_BODY)
        event_names = [e[0] for e in mock_tracer.spans[0].events]
        assert "capabilities.register.invocation" in event_names

    def test_invocation_event_namespace(self, mock_tracer: MockTracer) -> None:
        _client().post("/v1/x/capabilities", json=_VALID_BODY)
        ev = next(
            e for e in mock_tracer.spans[0].events if e[0] == "capabilities.register.invocation"
        )
        assert ev[1]["namespace"] == "myplugin"

    def test_invocation_event_capability_count(self, mock_tracer: MockTracer) -> None:
        _client().post("/v1/x/capabilities", json=_VALID_BODY)
        ev = next(
            e for e in mock_tracer.spans[0].events if e[0] == "capabilities.register.invocation"
        )
        assert ev[1]["capability_count"] == 1

    def test_span_ended_on_success(self, mock_tracer: MockTracer) -> None:
        _client().post("/v1/x/capabilities", json=_VALID_BODY)
        assert mock_tracer.spans[0].ended


# ---------------------------------------------------------------------------
# POST /v1/x/capabilities — validation failures
# ---------------------------------------------------------------------------


class TestRegisterValidation:
    def test_invalid_namespace_uppercase_returns_422(self, mock_tracer: MockTracer) -> None:
        body = {"namespace": "MyPlugin", "capabilities": [{"name": "act", "param_expected": False}]}
        assert _client().post("/v1/x/capabilities", json=body).status_code == 422

    def test_invalid_namespace_with_spaces_returns_422(self, mock_tracer: MockTracer) -> None:
        body = {
            "namespace": "my plugin",
            "capabilities": [{"name": "act", "param_expected": False}],
        }
        assert _client().post("/v1/x/capabilities", json=body).status_code == 422

    def test_invalid_namespace_starts_with_digit_returns_422(self, mock_tracer: MockTracer) -> None:
        body = {"namespace": "1plugin", "capabilities": [{"name": "act", "param_expected": False}]}
        assert _client().post("/v1/x/capabilities", json=body).status_code == 422

    def test_builtin_namespace_conflict_returns_422(self, mock_tracer: MockTracer) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "custom", "param_expected": False}]}
        assert _client().post("/v1/x/capabilities", json=body).status_code == 422

    def test_exec_builtin_namespace_conflict_returns_422(self, mock_tracer: MockTracer) -> None:
        body = {"namespace": "exec", "capabilities": [{"name": "custom", "param_expected": False}]}
        assert _client().post("/v1/x/capabilities", json=body).status_code == 422

    def test_duplicate_namespace_returns_422(self, mock_tracer: MockTracer) -> None:
        registry = CapabilityRegistry()
        client = _client(registry=registry)
        client.post("/v1/x/capabilities", json=_VALID_BODY)
        resp = client.post("/v1/x/capabilities", json=_VALID_BODY)
        assert resp.status_code == 422

    def test_empty_capabilities_returns_422(self, mock_tracer: MockTracer) -> None:
        body = {"namespace": "emptyplugin", "capabilities": []}
        assert _client().post("/v1/x/capabilities", json=body).status_code == 422

    def test_invalid_capability_name_uppercase_returns_422(self, mock_tracer: MockTracer) -> None:
        body = {
            "namespace": "myplugin",
            "capabilities": [{"name": "DoThing", "param_expected": False}],
        }
        assert _client().post("/v1/x/capabilities", json=body).status_code == 422

    def test_invalid_capability_name_with_hyphen_returns_422(self, mock_tracer: MockTracer) -> None:
        body = {
            "namespace": "myplugin",
            "capabilities": [{"name": "do-thing", "param_expected": False}],
        }
        assert _client().post("/v1/x/capabilities", json=body).status_code == 422

    def test_error_envelope_code_is_schema_invalid(self, mock_tracer: MockTracer) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "custom", "param_expected": False}]}
        resp = _client().post("/v1/x/capabilities", json=body)
        assert resp.json()["error"]["code"] == "schema_invalid"

    def test_error_envelope_has_message(self, mock_tracer: MockTracer) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "custom", "param_expected": False}]}
        resp = _client().post("/v1/x/capabilities", json=body)
        assert resp.json()["error"]["message"]

    def test_error_envelope_has_timestamp(self, mock_tracer: MockTracer) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "custom", "param_expected": False}]}
        resp = _client().post("/v1/x/capabilities", json=body)
        assert resp.json()["error"]["timestamp"]


# ---------------------------------------------------------------------------
# POST /v1/x/capabilities — audit log on failure
# ---------------------------------------------------------------------------


class TestRegisterAuditLog:
    def test_audit_written_on_invalid_namespace(
        self, mock_tracer: MockTracer, audit_log: CapturingAuditLog
    ) -> None:
        body = {"namespace": "Bad", "capabilities": [{"name": "act", "param_expected": False}]}
        _client(audit_log=audit_log).post("/v1/x/capabilities", json=body)
        assert len(audit_log.entries) == 1

    def test_audit_entry_level_error(
        self, mock_tracer: MockTracer, audit_log: CapturingAuditLog
    ) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "act", "param_expected": False}]}
        _client(audit_log=audit_log).post("/v1/x/capabilities", json=body)
        assert audit_log.entries[0].level == "error"

    def test_audit_entry_event_name(
        self, mock_tracer: MockTracer, audit_log: CapturingAuditLog
    ) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "act", "param_expected": False}]}
        _client(audit_log=audit_log).post("/v1/x/capabilities", json=body)
        assert audit_log.entries[0].event == "capabilities.register.failed"

    def test_audit_entry_operation(
        self, mock_tracer: MockTracer, audit_log: CapturingAuditLog
    ) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "act", "param_expected": False}]}
        _client(audit_log=audit_log).post("/v1/x/capabilities", json=body)
        assert audit_log.entries[0].operation == "register"

    def test_audit_entry_detail_contains_namespace(
        self, mock_tracer: MockTracer, audit_log: CapturingAuditLog
    ) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "act", "param_expected": False}]}
        _client(audit_log=audit_log).post("/v1/x/capabilities", json=body)
        assert audit_log.entries[0].detail is not None
        assert audit_log.entries[0].detail["namespace"] == "fs"

    def test_audit_entry_detail_has_reason(
        self, mock_tracer: MockTracer, audit_log: CapturingAuditLog
    ) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "act", "param_expected": False}]}
        _client(audit_log=audit_log).post("/v1/x/capabilities", json=body)
        assert "reason" in (audit_log.entries[0].detail or {})

    def test_no_audit_on_success(
        self, mock_tracer: MockTracer, audit_log: CapturingAuditLog
    ) -> None:
        _client(audit_log=audit_log).post("/v1/x/capabilities", json=_VALID_BODY)
        assert len(audit_log.entries) == 0


# ---------------------------------------------------------------------------
# POST /v1/x/capabilities — OTel on failure
# ---------------------------------------------------------------------------


class TestRegisterTelemetryOnFailure:
    def test_span_status_error_on_invalid_namespace(self, mock_tracer: MockTracer) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "act", "param_expected": False}]}
        _client().post("/v1/x/capabilities", json=body)
        assert mock_tracer.spans[0].status is not None
        assert mock_tracer.spans[0].status.status_code == StatusCode.ERROR

    def test_span_has_error_event_on_failure(self, mock_tracer: MockTracer) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "act", "param_expected": False}]}
        _client().post("/v1/x/capabilities", json=body)
        event_names = [e[0] for e in mock_tracer.spans[0].events]
        assert "capabilities.error" in event_names

    def test_span_error_event_operation(self, mock_tracer: MockTracer) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "act", "param_expected": False}]}
        _client().post("/v1/x/capabilities", json=body)
        ev = next(e for e in mock_tracer.spans[0].events if e[0] == "capabilities.error")
        assert ev[1]["operation"] == "register"

    def test_exception_recorded_on_span(self, mock_tracer: MockTracer) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "act", "param_expected": False}]}
        _client().post("/v1/x/capabilities", json=body)
        assert len(mock_tracer.spans[0].recorded_exceptions) == 1

    def test_span_ended_on_failure(self, mock_tracer: MockTracer) -> None:
        body = {"namespace": "fs", "capabilities": [{"name": "act", "param_expected": False}]}
        _client().post("/v1/x/capabilities", json=body)
        assert mock_tracer.spans[0].ended
