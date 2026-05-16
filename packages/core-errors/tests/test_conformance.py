"""
Core errors conformance suite.

Tests cover:
  - MeridianError base class: code, message, timestamp, cause fields.
  - Each subclass (capability_denied, schema_invalid, vault_unauthorized,
    budget_exceeded, divergence): correct code, http_status(), is-a MeridianError.
  - to_envelope(): correct JSON structure.
  - install_error_handler(): span emitted, invocation event attached, span
    status ERROR, error event added, cause recorded on span, audit entry
    written, correct HTTP status and JSON body returned.
  - Span lifecycle: span ended on every handler invocation.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core_errors import (
    AuditLogEntry,
    BudgetExceededError,
    CapabilityDeniedError,
    DivergenceError,
    HandlerOptions,
    MeridianError,
    SchemaInvalidError,
    VaultUnauthorizedError,
    install_error_handler,
)
from opentelemetry.trace import StatusCode

from .conftest import CapturingAuditLog, MockSpan

_TS = "2026-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# MeridianError base class
# ---------------------------------------------------------------------------

class TestMeridianError:
    def test_code_field(self) -> None:
        err = MeridianError(code="test_code", message="msg", timestamp=_TS)
        assert err.code == "test_code"

    def test_message_field(self) -> None:
        err = MeridianError(code="test_code", message="some message", timestamp=_TS)
        assert err.message == "some message"

    def test_timestamp_field(self) -> None:
        err = MeridianError(code="test_code", message="msg", timestamp=_TS)
        assert err.timestamp == _TS

    def test_cause_default_none(self) -> None:
        err = MeridianError(code="test_code", message="msg", timestamp=_TS)
        assert err.cause is None

    def test_cause_preserved(self) -> None:
        orig = RuntimeError("root")
        err = MeridianError(code="test_code", message="msg", timestamp=_TS, cause=orig)
        assert err.cause is orig

    def test_is_exception(self) -> None:
        err = MeridianError(code="test_code", message="msg", timestamp=_TS)
        assert isinstance(err, Exception)

    def test_str_is_message(self) -> None:
        err = MeridianError(code="test_code", message="hello world", timestamp=_TS)
        assert str(err) == "hello world"


# ---------------------------------------------------------------------------
# Subclasses
# ---------------------------------------------------------------------------

class TestCapabilityDeniedError:
    def test_code(self) -> None:
        assert CapabilityDeniedError(message="x", timestamp=_TS).code == "capability_denied"

    def test_is_meridian_error(self) -> None:
        assert isinstance(CapabilityDeniedError(message="x", timestamp=_TS), MeridianError)

    def test_http_status(self) -> None:
        assert CapabilityDeniedError(message="x", timestamp=_TS).http_status() == 403

    def test_cause_preserved(self) -> None:
        orig = RuntimeError("root")
        err = CapabilityDeniedError(message="x", timestamp=_TS, cause=orig)
        assert err.cause is orig


class TestSchemaInvalidError:
    def test_code(self) -> None:
        assert SchemaInvalidError(message="x", timestamp=_TS).code == "schema_invalid"

    def test_is_meridian_error(self) -> None:
        assert isinstance(SchemaInvalidError(message="x", timestamp=_TS), MeridianError)

    def test_http_status(self) -> None:
        assert SchemaInvalidError(message="x", timestamp=_TS).http_status() == 422


class TestVaultUnauthorizedError:
    def test_code(self) -> None:
        assert VaultUnauthorizedError(message="x", timestamp=_TS).code == "vault_unauthorized"

    def test_is_meridian_error(self) -> None:
        assert isinstance(VaultUnauthorizedError(message="x", timestamp=_TS), MeridianError)

    def test_http_status(self) -> None:
        assert VaultUnauthorizedError(message="x", timestamp=_TS).http_status() == 403


class TestBudgetExceededError:
    def test_code(self) -> None:
        assert BudgetExceededError(message="x", timestamp=_TS).code == "budget_exceeded"

    def test_is_meridian_error(self) -> None:
        assert isinstance(BudgetExceededError(message="x", timestamp=_TS), MeridianError)

    def test_http_status(self) -> None:
        assert BudgetExceededError(message="x", timestamp=_TS).http_status() == 429


class TestDivergenceError:
    def test_code(self) -> None:
        assert DivergenceError(message="x", timestamp=_TS).code == "divergence"

    def test_is_meridian_error(self) -> None:
        assert isinstance(DivergenceError(message="x", timestamp=_TS), MeridianError)

    def test_http_status(self) -> None:
        assert DivergenceError(message="x", timestamp=_TS).http_status() == 409


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------

class TestErrorEnvelope:
    def test_envelope_has_error_key(self) -> None:
        env = MeridianError(code="test_code", message="msg", timestamp=_TS).to_envelope()
        assert "error" in env

    def test_envelope_code(self) -> None:
        env = MeridianError(code="test_code", message="msg", timestamp=_TS).to_envelope()
        assert env["error"]["code"] == "test_code"

    def test_envelope_message(self) -> None:
        env = MeridianError(code="test_code", message="the message", timestamp=_TS).to_envelope()
        assert env["error"]["message"] == "the message"

    def test_envelope_timestamp(self) -> None:
        env = MeridianError(code="test_code", message="msg", timestamp=_TS).to_envelope()
        assert env["error"]["timestamp"] == _TS

    def test_envelope_code_for_subclass(self) -> None:
        env = CapabilityDeniedError(message="denied", timestamp=_TS).to_envelope()
        assert env["error"]["code"] == "capability_denied"


# ---------------------------------------------------------------------------
# Helpers for handler tests
# ---------------------------------------------------------------------------

def _make_app(audit: CapturingAuditLog) -> FastAPI:
    app = FastAPI()
    install_error_handler(app, HandlerOptions(audit_log=audit))

    @app.get("/capability-denied")
    def raise_capability_denied() -> None:
        raise CapabilityDeniedError(message="denied", timestamp=_TS)

    @app.get("/schema-invalid")
    def raise_schema_invalid() -> None:
        raise SchemaInvalidError(message="bad schema", timestamp=_TS)

    @app.get("/vault-unauthorized")
    def raise_vault_unauthorized() -> None:
        raise VaultUnauthorizedError(message="no access", timestamp=_TS)

    @app.get("/budget-exceeded")
    def raise_budget_exceeded() -> None:
        raise BudgetExceededError(message="over budget", timestamp=_TS)

    @app.get("/divergence")
    def raise_divergence() -> None:
        raise DivergenceError(message="state diverged", timestamp=_TS)

    return app


# ---------------------------------------------------------------------------
# Handler — HTTP status codes
# ---------------------------------------------------------------------------

class TestHandlerHttpStatus:
    def test_capability_denied_returns_403(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        assert client.get("/capability-denied").status_code == 403

    def test_schema_invalid_returns_422(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        assert client.get("/schema-invalid").status_code == 422

    def test_vault_unauthorized_returns_403(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        assert client.get("/vault-unauthorized").status_code == 403

    def test_budget_exceeded_returns_429(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        assert client.get("/budget-exceeded").status_code == 429

    def test_divergence_returns_409(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        assert client.get("/divergence").status_code == 409


# ---------------------------------------------------------------------------
# Handler — JSON error envelope
# ---------------------------------------------------------------------------

class TestHandlerEnvelope:
    def test_json_has_error_key(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        assert "error" in client.get("/capability-denied").json()

    def test_json_error_code(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        assert client.get("/capability-denied").json()["error"]["code"] == "capability_denied"

    def test_json_error_message(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        assert client.get("/capability-denied").json()["error"]["message"] == "denied"

    def test_json_error_has_timestamp(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        assert "timestamp" in client.get("/capability-denied").json()["error"]

    def test_budget_exceeded_envelope_code(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        assert client.get("/budget-exceeded").json()["error"]["code"] == "budget_exceeded"

    def test_divergence_envelope_code(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        assert client.get("/divergence").json()["error"]["code"] == "divergence"


# ---------------------------------------------------------------------------
# Handler — OTel telemetry
# ---------------------------------------------------------------------------

class TestHandlerTelemetry:
    def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        assert mock_span.name == "meridian.error_handler"

    def test_span_attributes_error_code(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        assert mock_span.attributes["error.code"] == "capability_denied"

    def test_invocation_event_attached(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        event_names = [e[0] for e in mock_span.events]
        assert "meridian.error.invocation" in event_names

    def test_invocation_event_code(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        inv = next(e for e in mock_span.events if e[0] == "meridian.error.invocation")
        assert inv[1]["code"] == "capability_denied"

    def test_error_event_attached(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        event_names = [e[0] for e in mock_span.events]
        assert "meridian.error" in event_names

    def test_error_event_code(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        err_evt = next(e for e in mock_span.events if e[0] == "meridian.error")
        assert err_evt[1]["error.code"] == "capability_denied"

    def test_span_marked_error(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        assert mock_span.ended

    def test_cause_recorded_on_span(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = RuntimeError("root cause")
        app = FastAPI()
        install_error_handler(app, HandlerOptions(audit_log=audit_log))

        @app.get("/with-cause")
        def raise_with_cause() -> None:
            raise CapabilityDeniedError(message="denied", timestamp=_TS, cause=orig)

        client = TestClient(app, raise_server_exceptions=False)
        client.get("/with-cause")
        assert orig in mock_span.recorded_exceptions

    def test_no_cause_no_recorded_exception(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        assert mock_span.recorded_exceptions == []


# ---------------------------------------------------------------------------
# Handler — audit log
# ---------------------------------------------------------------------------

class TestHandlerAuditLog:
    def test_audit_entry_written(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        assert len(audit_log.entries) == 1

    def test_audit_entry_level_error(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        assert audit_log.entries[0].level == "error"

    def test_audit_entry_event_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        assert audit_log.entries[0].event == "meridian.error.handled"

    def test_audit_entry_code(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        assert audit_log.entries[0].code == "capability_denied"

    def test_audit_entry_detail_message(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/capability-denied")
        assert audit_log.entries[0].detail == {"message": "denied"}

    def test_audit_entry_per_error_type(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        client = TestClient(_make_app(audit_log), raise_server_exceptions=False)
        client.get("/budget-exceeded")
        assert audit_log.entries[0].code == "budget_exceeded"

    def test_noop_audit_log_does_not_raise(self, mock_span: MockSpan) -> None:
        app = FastAPI()
        install_error_handler(app)

        @app.get("/test")
        def raise_err() -> None:
            raise CapabilityDeniedError(message="denied", timestamp=_TS)

        client = TestClient(app, raise_server_exceptions=False)
        assert client.get("/test").status_code == 403
