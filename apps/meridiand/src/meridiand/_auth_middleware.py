from __future__ import annotations

import ipaddress
import json
from datetime import UTC, datetime

from core_errors import AuditLog, AuditLogEntry, StructuredEvent, record_invocation_event
from starlette.types import ASGIApp, Receive, Scope, Send

from ._telemetry import get_tracer


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _extract_bearer(scope: Scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name == b"authorization":
            val = value.decode("latin-1", errors="replace")
            if val.startswith("Bearer "):
                return val[7:]
    return None


class AuthMiddleware:
    """ASGI middleware for connection authentication.

    Responsibilities:
    - Rejects TCP connections from non-loopback addresses with 403; writes
      to the audit log on rejection.
    - UDS connections (scope client is None) bypass all checks.
    - When bearer_token is configured, validates the Authorization: Bearer
      <token> header on TCP connections; rejects with 401 on mismatch or
      absence and writes to the audit log.
    - Emits an OpenTelemetry span and structured invocation event on every
      HTTP request.
    - On rejection: surfaces a JSON error response to the caller.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        audit_log: AuditLog,
        bearer_token: str | None = None,
    ) -> None:
        self._app = app
        self._audit_log = audit_log
        self._bearer_token = bearer_token or None  # treat empty string as unconfigured

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        now = _now()
        tracer = get_tracer()
        client = scope.get("client")
        is_uds = client is None
        client_host = client[0] if client is not None else ""

        with tracer.start_as_current_span(
            "auth.check",
            attributes={
                "auth.is_uds": is_uds,
                "auth.client_host": client_host,
                "auth.bearer_configured": self._bearer_token is not None,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="auth.check.invocation",
                    code="auth_check",
                    timestamp=now,
                ),
            )

            if not is_uds:
                if not _is_loopback(client_host):
                    await self._send_error(
                        send,
                        403,
                        "auth_non_loopback",
                        "Connection rejected: non-loopback source address",
                    )
                    span.add_event(
                        "auth.check.rejected",
                        {
                            "auth.reject_reason": "non_loopback",
                            "auth.client_host": client_host,
                        },
                    )
                    self._audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="auth.rejected.non_loopback",
                            code="auth_non_loopback",
                            timestamp=now,
                            detail={"client_host": client_host},
                        )
                    )
                    return

                if self._bearer_token is not None:
                    presented = _extract_bearer(scope)
                    if presented != self._bearer_token:
                        await self._send_error(
                            send,
                            401,
                            "auth_bearer_invalid",
                            "Invalid or missing bearer token",
                        )
                        span.add_event(
                            "auth.check.rejected",
                            {
                                "auth.reject_reason": "bearer_invalid",
                                "auth.client_host": client_host,
                            },
                        )
                        self._audit_log.write(
                            AuditLogEntry(
                                level="error",
                                event="auth.rejected.bearer_invalid",
                                code="auth_bearer_invalid",
                                timestamp=now,
                                detail={"client_host": client_host},
                            )
                        )
                        return

            span.add_event(
                "auth.check.allowed",
                {"auth.is_uds": is_uds, "auth.client_host": client_host},
            )
            await self._app(scope, receive, send)

    async def _send_error(
        self,
        send: Send,
        status_code: int,
        code: str,
        message: str,
    ) -> None:
        body = json.dumps(
            {"error": {"code": code, "message": message}},
            separators=(",", ":"),
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )
