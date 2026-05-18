"""
Error-envelope middleware conformance suite.

Tests cover:
  - Non-HTTP scope passes through unchanged.
  - Successful response is passed through unmodified.
  - MeridianError escaped from the app returns the error's HTTP status code.
  - MeridianError envelope has the correct error code.
  - MeridianError envelope has the correct error message.
  - MeridianError envelope has a details field.
  - MeridianError catch writes an audit log entry with the error code.
  - MeridianError catch audit entry level is "error".
  - Generic Exception returns HTTP 500.
  - Generic Exception envelope has code "internal_server_error".
  - Generic Exception envelope has a details field.
  - Generic Exception catch writes an audit log entry.
  - Generic Exception catch audit entry level is "error".
  - OTel span "error_envelope.catch" emitted on MeridianError.
  - OTel span "error_envelope.catch" emitted on generic Exception.
  - Span carries a "meridian.error.invocation" event.
  - Span status is ERROR on exception catch.
  - No OTel span emitted when no exception is raised.
  - No OTel span emitted for non-HTTP scope.
  - ErrorEnvelopeMiddleware is registered in create_app.
  - ErrorEnvelopeMiddleware is the outermost middleware in create_app.
  - Send failure writes audit log entry with code "error_envelope_send_failed".
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from core_errors import AuditLog, AuditLogEntry, MeridianError, NoopAuditLog
from meridiand._app import create_app
from meridiand._error_envelope_middleware import ErrorEnvelopeMiddleware

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CapturingAuditLog(AuditLog):
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


class _SampleError(MeridianError):
    def __init__(self, *, message: str = "something went wrong") -> None:
        super().__init__(code="sample_error", message=message, timestamp="2026-01-01T00:00:00+00:00")

    def http_status(self) -> int:
        return 418


def _make_raising_app(exc: Exception):  # type: ignore[no-untyped-def]
    async def _app(scope: Any, receive: Any, send: Any) -> None:
        raise exc

    return _app


def _make_ok_app():  # type: ignore[no-untyped-def]
    async def _app(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"ok":true}', "more_body": False})

    return _app


def _make_middleware(
    *,
    inner_app: Any = None,
    audit_log: AuditLog | None = None,
) -> ErrorEnvelopeMiddleware:
    app = inner_app if inner_app is not None else _make_ok_app()
    return ErrorEnvelopeMiddleware(app, audit_log=audit_log or NoopAuditLog())


async def _invoke(
    middleware: ErrorEnvelopeMiddleware,
    *,
    scope_type: str = "http",
) -> tuple[int | None, bytes]:
    scope: dict[str, Any] = {
        "type": scope_type,
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8888),
    }
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg: dict[str, Any]) -> None:
        messages.append(msg)

    await middleware(scope, receive, send)

    status = next(
        (m["status"] for m in messages if m.get("type") == "http.response.start"), None
    )
    body = next(
        (m.get("body", b"") for m in messages if m.get("type") == "http.response.body"), b""
    )
    return status, body


# ---------------------------------------------------------------------------
# TestNonHttpPassthrough
# ---------------------------------------------------------------------------


class TestNonHttpPassthrough:
    async def test_websocket_scope_is_passed_through(self) -> None:
        forwarded: list[dict[str, Any]] = []

        async def _capture(scope: Any, receive: Any, send: Any) -> None:
            forwarded.append(scope)

        mw = ErrorEnvelopeMiddleware(_capture, audit_log=NoopAuditLog())
        scope = {"type": "websocket", "headers": [], "client": None}

        async def _receive() -> dict[str, Any]:
            return {}

        async def _send(msg: Any) -> None:
            pass

        await mw(scope, _receive, _send)
        assert forwarded == [scope]


# ---------------------------------------------------------------------------
# TestSuccessPassthrough
# ---------------------------------------------------------------------------


class TestSuccessPassthrough:
    async def test_successful_response_returned_unchanged(self) -> None:
        mw = _make_middleware()
        status, body = await _invoke(mw)
        assert status == 200
        assert json.loads(body) == {"ok": True}

    async def test_no_audit_log_on_success(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit_log=audit)
        await _invoke(mw)
        assert not audit.entries


# ---------------------------------------------------------------------------
# TestMeridianError
# ---------------------------------------------------------------------------


class TestMeridianError:
    async def test_meridian_error_returns_correct_status(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(_SampleError()))
        status, _ = await _invoke(mw)
        assert status == 418

    async def test_meridian_error_envelope_has_code(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(_SampleError()))
        _, body = await _invoke(mw)
        data = json.loads(body)
        assert data["error"]["code"] == "sample_error"

    async def test_meridian_error_envelope_has_message(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(_SampleError(message="oh no")))
        _, body = await _invoke(mw)
        data = json.loads(body)
        assert data["error"]["message"] == "oh no"

    async def test_meridian_error_envelope_has_details_field(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(_SampleError()))
        _, body = await _invoke(mw)
        data = json.loads(body)
        assert "details" in data["error"]
        assert isinstance(data["error"]["details"], dict)

    async def test_meridian_error_writes_audit_log(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(inner_app=_make_raising_app(_SampleError()), audit_log=audit)
        await _invoke(mw)
        assert any(e.code == "sample_error" for e in audit.entries)

    async def test_meridian_error_audit_level_is_error(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(inner_app=_make_raising_app(_SampleError()), audit_log=audit)
        await _invoke(mw)
        entry = next(e for e in audit.entries if e.code == "sample_error")
        assert entry.level == "error"

    async def test_meridian_error_audit_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(inner_app=_make_raising_app(_SampleError()), audit_log=audit)
        await _invoke(mw)
        entry = next(e for e in audit.entries if e.code == "sample_error")
        assert entry.event == "error_envelope.catch.meridian"

    async def test_meridian_error_audit_detail_has_message(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(
            inner_app=_make_raising_app(_SampleError(message="test msg")), audit_log=audit
        )
        await _invoke(mw)
        entry = next(e for e in audit.entries if e.code == "sample_error")
        assert entry.detail is not None
        assert entry.detail["message"] == "test msg"


# ---------------------------------------------------------------------------
# TestGenericException
# ---------------------------------------------------------------------------


class TestGenericException:
    async def test_generic_exception_returns_500(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(RuntimeError("boom")))
        status, _ = await _invoke(mw)
        assert status == 500

    async def test_generic_exception_code_is_internal_server_error(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(ValueError("oops")))
        _, body = await _invoke(mw)
        data = json.loads(body)
        assert data["error"]["code"] == "internal_server_error"

    async def test_generic_exception_envelope_has_details_field(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(RuntimeError("x")))
        _, body = await _invoke(mw)
        data = json.loads(body)
        assert "details" in data["error"]
        assert isinstance(data["error"]["details"], dict)

    async def test_generic_exception_writes_audit_log(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(inner_app=_make_raising_app(TypeError("bad")), audit_log=audit)
        await _invoke(mw)
        assert any(e.code == "internal_server_error" for e in audit.entries)

    async def test_generic_exception_audit_level_is_error(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(inner_app=_make_raising_app(RuntimeError("x")), audit_log=audit)
        await _invoke(mw)
        entry = next(e for e in audit.entries if e.code == "internal_server_error")
        assert entry.level == "error"

    async def test_generic_exception_audit_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(inner_app=_make_raising_app(RuntimeError("x")), audit_log=audit)
        await _invoke(mw)
        entry = next(e for e in audit.entries if e.code == "internal_server_error")
        assert entry.event == "error_envelope.catch.unexpected"

    async def test_generic_exception_audit_detail_has_error_type(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(
            inner_app=_make_raising_app(ValueError("bad value")), audit_log=audit
        )
        await _invoke(mw)
        entry = next(e for e in audit.entries if e.code == "internal_server_error")
        assert entry.detail is not None
        assert entry.detail["error_type"] == "ValueError"


# ---------------------------------------------------------------------------
# TestOtelSpan
# ---------------------------------------------------------------------------


class TestOtelSpan:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    async def test_span_emitted_on_meridian_error(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(_SampleError()))
        await _invoke(mw)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "error_envelope.catch" in span_names

    async def test_span_emitted_on_generic_exception(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(RuntimeError("boom")))
        await _invoke(mw)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "error_envelope.catch" in span_names

    async def test_span_has_invocation_event(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(_SampleError()))
        await _invoke(mw)
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "error_envelope.catch"
        )
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    async def test_span_has_meridian_error_event(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(_SampleError()))
        await _invoke(mw)
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "error_envelope.catch"
        )
        event_names = [e.name for e in span.events]
        assert "meridian.error" in event_names

    async def test_span_status_is_error_on_meridian_error(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(_SampleError()))
        await _invoke(mw)
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "error_envelope.catch"
        )
        assert span.status.status_code == StatusCode.ERROR

    async def test_span_status_is_error_on_generic_exception(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(RuntimeError("x")))
        await _invoke(mw)
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "error_envelope.catch"
        )
        assert span.status.status_code == StatusCode.ERROR

    async def test_span_carries_error_code_attribute(self) -> None:
        mw = _make_middleware(inner_app=_make_raising_app(_SampleError()))
        await _invoke(mw)
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "error_envelope.catch"
        )
        assert span.attributes["error.code"] == "sample_error"

    async def test_no_span_on_successful_request(self) -> None:
        mw = _make_middleware()
        await _invoke(mw)
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "error_envelope.catch"]
        assert not spans

    async def test_no_span_on_non_http_scope(self) -> None:
        forwarded: list[Any] = []

        async def _capture(scope: Any, receive: Any, send: Any) -> None:
            forwarded.append(scope)

        mw = ErrorEnvelopeMiddleware(_capture, audit_log=NoopAuditLog())
        scope = {"type": "lifespan", "headers": [], "client": None}

        async def _receive() -> dict[str, Any]:
            return {}

        async def _send(msg: Any) -> None:
            pass

        await mw(scope, _receive, _send)
        spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "error_envelope.catch"]
        assert not spans


# ---------------------------------------------------------------------------
# TestSendFailure
# ---------------------------------------------------------------------------


class TestSendFailure:
    async def test_send_failure_writes_audit_log(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(inner_app=_make_raising_app(RuntimeError("boom")), audit_log=audit)

        scope: dict[str, Any] = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8888),
        }

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def failing_send(msg: dict[str, Any]) -> None:
            raise IOError("connection lost")

        await mw(scope, receive, failing_send)
        assert any(e.code == "error_envelope_send_failed" for e in audit.entries)

    async def test_send_failure_audit_level_is_error(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(inner_app=_make_raising_app(_SampleError()), audit_log=audit)

        scope: dict[str, Any] = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8888),
        }

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def failing_send(msg: dict[str, Any]) -> None:
            raise IOError("connection lost")

        await mw(scope, receive, failing_send)
        entry = next(e for e in audit.entries if e.code == "error_envelope_send_failed")
        assert entry.level == "error"


# ---------------------------------------------------------------------------
# TestMiddlewareRegistration
# ---------------------------------------------------------------------------


class TestMiddlewareRegistration:
    def test_error_envelope_middleware_registered_in_create_app(self) -> None:
        app = create_app(NoopAuditLog())
        assert any(m.cls is ErrorEnvelopeMiddleware for m in app.user_middleware)

    def test_error_envelope_middleware_is_outermost(self) -> None:
        app = create_app(NoopAuditLog())
        # user_middleware[0] is the outermost (last added via add_middleware).
        assert app.user_middleware[0].cls is ErrorEnvelopeMiddleware
