"""
Auth middleware conformance suite.

Tests cover:
  - Non-HTTP scope (websocket) is passed through without checks.
  - TCP connection from loopback IPv4 (127.0.0.1) is allowed.
  - TCP connection from loopback IPv6 (::1) is allowed.
  - TCP connection from non-loopback address returns 403.
  - Non-loopback rejection has error code auth_non_loopback.
  - Non-loopback rejection writes audit log entry with code auth_non_loopback.
  - UDS connection (client=None) is allowed without any bearer token.
  - UDS connection bypasses bearer check even when bearer_token is configured.
  - Loopback TCP without bearer configured is allowed (no Authorization header needed).
  - Loopback TCP with bearer configured and correct token is allowed.
  - Loopback TCP with bearer configured and wrong token returns 401.
  - Loopback TCP with bearer configured and missing header returns 401.
  - Bearer rejection has error code auth_bearer_invalid.
  - Bearer rejection writes audit log entry with code auth_bearer_invalid.
  - Empty bearer_token config value is treated as unconfigured (no check).
  - OTel span "auth.check" is emitted on each HTTP request.
  - Span carries meridian.error.invocation event.
  - Span carries auth.check.allowed event on success.
  - Span carries auth.check.rejected event on rejection.
  - AuthMiddleware is registered in create_app.
  - AuthMiddleware is the second outermost middleware in create_app
    (ErrorEnvelopeMiddleware is outermost).
"""

from __future__ import annotations

import json
from typing import Any

from core_errors import AuditLog, AuditLogEntry, NoopAuditLog
from meridiand._app import create_app
from meridiand._auth_middleware import AuthMiddleware
from meridiand._config import AuthConfig

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CapturingAuditLog(AuditLog):
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


async def _invoke(
    middleware: AuthMiddleware,
    *,
    client: tuple[str, int] | None = ("127.0.0.1", 50000),
    headers: list[tuple[bytes, bytes]] | None = None,
    scope_type: str = "http",
) -> tuple[int | None, bytes]:
    scope: dict[str, Any] = {
        "type": scope_type,
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": headers or [],
        "client": client,
        "server": ("127.0.0.1", 8888),
    }
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg: dict[str, Any]) -> None:
        messages.append(msg)

    await middleware(scope, receive, send)

    status = next((m["status"] for m in messages if m.get("type") == "http.response.start"), None)
    body = next(
        (m.get("body", b"") for m in messages if m.get("type") == "http.response.body"), b""
    )
    return status, body


def _make_middleware(
    *,
    bearer_token: str | None = None,
    audit_log: AuditLog | None = None,
) -> AuthMiddleware:
    async def _handler(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"ok":true}', "more_body": False})

    return AuthMiddleware(
        _handler,
        audit_log=audit_log or NoopAuditLog(),
        bearer_token=bearer_token,
    )


def _auth_header(token: str) -> list[tuple[bytes, bytes]]:
    return [(b"authorization", f"Bearer {token}".encode())]


# ---------------------------------------------------------------------------
# TestNonHttpPassthrough
# ---------------------------------------------------------------------------


class TestNonHttpPassthrough:
    async def test_websocket_scope_is_passed_through(self) -> None:
        forwarded: list[dict[str, Any]] = []

        async def _capture(scope: Any, receive: Any, send: Any) -> None:
            forwarded.append(scope)

        mw = AuthMiddleware(_capture, audit_log=NoopAuditLog())
        scope = {"type": "websocket", "headers": [], "client": ("10.0.0.1", 50000)}

        async def _receive() -> dict[str, Any]:
            return {}

        async def _send(msg: Any) -> None:
            pass

        await mw(scope, _receive, _send)
        assert forwarded == [scope]


# ---------------------------------------------------------------------------
# TestLoopbackEnforcement
# ---------------------------------------------------------------------------


class TestLoopbackEnforcement:
    async def test_loopback_ipv4_is_allowed(self) -> None:
        mw = _make_middleware()
        status, _ = await _invoke(mw, client=("127.0.0.1", 50000))
        assert status == 200

    async def test_loopback_127_x_x_x_is_allowed(self) -> None:
        mw = _make_middleware()
        status, _ = await _invoke(mw, client=("127.0.0.2", 50000))
        assert status == 200

    async def test_loopback_ipv6_is_allowed(self) -> None:
        mw = _make_middleware()
        status, _ = await _invoke(mw, client=("::1", 50000))
        assert status == 200

    async def test_non_loopback_returns_403(self) -> None:
        mw = _make_middleware()
        status, _ = await _invoke(mw, client=("10.0.0.1", 50000))
        assert status == 403

    async def test_invalid_host_returns_403(self) -> None:
        """Non-IP host string (e.g. 'notanip') hits the ValueError fallback."""
        mw = _make_middleware()
        status, _ = await _invoke(mw, client=("notanip", 50000))
        assert status == 403

    async def test_bearer_extraction_skips_non_auth_headers(self) -> None:
        """A non-Authorization header before Authorization is skipped (covers 26->25)."""
        mw = _make_middleware(bearer_token="secret")
        headers = [
            (b"x-trace-id", b"abc"),
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer secret"),
        ]
        status, _ = await _invoke(mw, headers=headers)
        assert status == 200

    async def test_non_loopback_error_code(self) -> None:
        mw = _make_middleware()
        _, body = await _invoke(mw, client=("192.168.1.50", 50000))
        data = json.loads(body)
        assert data["error"]["code"] == "auth_non_loopback"

    async def test_non_loopback_writes_audit_log(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit_log=audit)
        await _invoke(mw, client=("172.16.0.1", 50000))
        assert any(e.code == "auth_non_loopback" for e in audit.entries)

    async def test_non_loopback_audit_entry_level_is_error(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit_log=audit)
        await _invoke(mw, client=("8.8.8.8", 50000))
        entry = next(e for e in audit.entries if e.code == "auth_non_loopback")
        assert entry.level == "error"

    async def test_non_loopback_audit_detail_has_client_host(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit_log=audit)
        await _invoke(mw, client=("203.0.113.1", 50000))
        entry = next(e for e in audit.entries if e.code == "auth_non_loopback")
        assert entry.detail is not None
        assert entry.detail["client_host"] == "203.0.113.1"


# ---------------------------------------------------------------------------
# TestUdsBypass
# ---------------------------------------------------------------------------


class TestUdsBypass:
    async def test_uds_connection_is_allowed(self) -> None:
        mw = _make_middleware()
        status, _ = await _invoke(mw, client=None)
        assert status == 200

    async def test_uds_bypasses_bearer_check_when_configured(self) -> None:
        mw = _make_middleware(bearer_token="secret-token")
        # No Authorization header — should still pass because UDS bypasses bearer check.
        status, _ = await _invoke(mw, client=None, headers=[])
        assert status == 200

    async def test_uds_does_not_write_audit_log_on_allow(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit_log=audit)
        await _invoke(mw, client=None)
        assert not audit.entries


# ---------------------------------------------------------------------------
# TestBearerTokenCheck
# ---------------------------------------------------------------------------


class TestBearerTokenCheck:
    async def test_no_bearer_configured_loopback_passes_without_header(self) -> None:
        mw = _make_middleware(bearer_token=None)
        status, _ = await _invoke(mw, client=("127.0.0.1", 50000))
        assert status == 200

    async def test_correct_bearer_token_is_allowed(self) -> None:
        mw = _make_middleware(bearer_token="my-secret")
        status, _ = await _invoke(
            mw,
            client=("127.0.0.1", 50000),
            headers=_auth_header("my-secret"),
        )
        assert status == 200

    async def test_wrong_bearer_token_returns_401(self) -> None:
        mw = _make_middleware(bearer_token="correct-token")
        status, _ = await _invoke(
            mw,
            client=("127.0.0.1", 50000),
            headers=_auth_header("wrong-token"),
        )
        assert status == 401

    async def test_missing_bearer_header_returns_401(self) -> None:
        mw = _make_middleware(bearer_token="required-token")
        status, _ = await _invoke(mw, client=("127.0.0.1", 50000), headers=[])
        assert status == 401

    async def test_bearer_invalid_error_code(self) -> None:
        mw = _make_middleware(bearer_token="correct-token")
        _, body = await _invoke(
            mw,
            client=("127.0.0.1", 50000),
            headers=_auth_header("wrong-token"),
        )
        data = json.loads(body)
        assert data["error"]["code"] == "auth_bearer_invalid"

    async def test_bearer_invalid_writes_audit_log(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(bearer_token="correct-token", audit_log=audit)
        await _invoke(mw, client=("127.0.0.1", 50000), headers=_auth_header("wrong"))
        assert any(e.code == "auth_bearer_invalid" for e in audit.entries)

    async def test_bearer_invalid_audit_level_is_error(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(bearer_token="correct-token", audit_log=audit)
        await _invoke(mw, client=("127.0.0.1", 50000), headers=[])
        entry = next(e for e in audit.entries if e.code == "auth_bearer_invalid")
        assert entry.level == "error"

    async def test_empty_bearer_token_config_is_unconfigured(self) -> None:
        mw = _make_middleware(bearer_token="")
        # Empty token → treated as unconfigured → no bearer check → pass without header.
        status, _ = await _invoke(mw, client=("127.0.0.1", 50000), headers=[])
        assert status == 200

    async def test_non_bearer_auth_scheme_returns_401(self) -> None:
        mw = _make_middleware(bearer_token="token")
        # "Basic ..." header — not a Bearer token.
        headers = [(b"authorization", b"Basic dXNlcjpwYXNz")]
        status, _ = await _invoke(mw, client=("127.0.0.1", 50000), headers=headers)
        assert status == 401


# ---------------------------------------------------------------------------
# TestOtelSpan
# ---------------------------------------------------------------------------


class TestOtelSpan:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    async def test_emits_auth_check_span_on_request(self) -> None:
        mw = _make_middleware()
        await _invoke(mw, client=("127.0.0.1", 50000))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "auth.check" in span_names

    async def test_span_has_invocation_event(self) -> None:
        mw = _make_middleware()
        await _invoke(mw, client=("127.0.0.1", 50000))
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "auth.check")
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    async def test_allowed_request_has_allowed_event(self) -> None:
        mw = _make_middleware()
        await _invoke(mw, client=("127.0.0.1", 50000))
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "auth.check")
        event_names = [e.name for e in span.events]
        assert "auth.check.allowed" in event_names

    async def test_rejected_request_has_rejected_event(self) -> None:
        mw = _make_middleware()
        await _invoke(mw, client=("10.0.0.1", 50000))
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "auth.check")
        event_names = [e.name for e in span.events]
        assert "auth.check.rejected" in event_names

    async def test_span_carries_is_uds_false_for_tcp(self) -> None:
        mw = _make_middleware()
        await _invoke(mw, client=("127.0.0.1", 50000))
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "auth.check")
        assert span.attributes["auth.is_uds"] is False

    async def test_span_carries_is_uds_true_for_uds(self) -> None:
        mw = _make_middleware()
        await _invoke(mw, client=None)
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "auth.check")
        assert span.attributes["auth.is_uds"] is True

    async def test_span_emitted_per_request(self) -> None:
        mw = _make_middleware()
        await _invoke(mw, client=("127.0.0.1", 50000))
        await _invoke(mw, client=("127.0.0.1", 50000))
        auth_spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "auth.check"]
        assert len(auth_spans) == 2

    async def test_no_span_emitted_for_non_http_scope(self) -> None:
        async def _handler(scope: Any, receive: Any, send: Any) -> None:
            pass

        mw = AuthMiddleware(_handler, audit_log=NoopAuditLog())
        scope = {"type": "lifespan", "headers": [], "client": None}

        async def _receive() -> dict[str, Any]:
            return {}

        async def _send(msg: Any) -> None:
            pass

        await mw(scope, _receive, _send)
        auth_spans = [s for s in _otel_exporter.get_finished_spans() if s.name == "auth.check"]
        assert not auth_spans


# ---------------------------------------------------------------------------
# TestMiddlewareRegistration
# ---------------------------------------------------------------------------


class TestMiddlewareRegistration:
    def test_auth_middleware_registered_in_create_app(self) -> None:
        app = create_app(NoopAuditLog())
        assert any(m.cls is AuthMiddleware for m in app.user_middleware)

    def test_auth_middleware_is_second_outermost(self) -> None:
        app = create_app(NoopAuditLog())
        # user_middleware[0] is ErrorEnvelopeMiddleware (outermost).
        # user_middleware[1] is AuthMiddleware (second outermost).
        assert app.user_middleware[1].cls is AuthMiddleware

    def test_auth_config_bearer_token_passed_to_middleware(self) -> None:
        auth = AuthConfig(bearer_token="my-token")
        app = create_app(NoopAuditLog(), auth_config=auth)
        mw_entry = next(m for m in app.user_middleware if m.cls is AuthMiddleware)
        assert mw_entry.kwargs["bearer_token"] == "my-token"

    def test_no_auth_config_means_no_bearer_check(self) -> None:
        app = create_app(NoopAuditLog(), auth_config=None)
        mw_entry = next(m for m in app.user_middleware if m.cls is AuthMiddleware)
        assert mw_entry.kwargs["bearer_token"] is None
