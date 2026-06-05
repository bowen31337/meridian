from __future__ import annotations

import contextlib
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    record_invocation_event,
)
from opentelemetry.trace import Status, StatusCode
from sdk_sandbox import ExecutionContext
from starlette.types import ASGIApp, Receive, Scope, Send

from ._hook_dispatch import dispatch_hooks
from ._telemetry import get_tracer


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ErrorEnvelopeMiddleware:
    """ASGI middleware that catches uncaught exceptions and returns a JSON error envelope.

    Responsibilities:
    - Catches any MeridianError that escapes the inner ASGI stack and returns
      {"error": {"code", "message", "details": {}}} with the error's HTTP status.
    - Catches any other Exception and returns HTTP 500 with code
      "internal_server_error".
    - On each catch: opens an OTel span "error_envelope.catch", emits a structured
      invocation event, records the error on the span, and writes to the audit log.
    - On failure to send the error envelope: writes to the audit log.
    """

    def __init__(self, app: ASGIApp, *, audit_log: AuditLog, hooks_dir: Path | None = None) -> None:
        self._app = app
        self._audit_log = audit_log
        self._hooks_dir = hooks_dir

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        try:
            await self._app(scope, receive, send)
        except MeridianError as exc:
            await self._handle_meridian(send, exc)
        except Exception as exc:
            await self._handle_unexpected(send, exc)

    async def _handle_meridian(self, send: Send, exc: MeridianError) -> None:
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "error_envelope.catch",
            attributes={"error.code": exc.code},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="error_envelope.catch.invocation",
                    code=exc.code,
                    timestamp=now,
                ),
            )
            span.set_status(Status(StatusCode.ERROR, exc.message))
            span.add_event(
                "meridian.error",
                {"error.code": exc.code, "error.message": exc.message},
            )
            if exc.cause is not None:
                span.record_exception(exc.cause)
            self._audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="error_envelope.catch.meridian",
                    code=exc.code,
                    timestamp=now,
                    detail={"message": exc.message},
                )
            )

        if self._hooks_dir is not None:
            # on_error hooks must never block the error response.
            with contextlib.suppress(Exception):
                await dispatch_hooks(
                    "on_error",
                    {"error_code": exc.code, "error_message": exc.message},
                    ExecutionContext(session_id=""),
                    hooks_dir=self._hooks_dir,
                    audit_log=self._audit_log,
                )

        try:
            await self._send_envelope(send, exc.http_status(), exc.code, exc.message, {})
        except Exception as send_exc:
            self._audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="error_envelope.send.failed",
                    code="error_envelope_send_failed",
                    timestamp=_now(),
                    detail={"error": str(send_exc), "original_code": exc.code},
                )
            )

    async def _handle_unexpected(self, send: Send, exc: Exception) -> None:
        now = _now()
        code = "internal_server_error"
        message = "An unexpected error occurred"
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "error_envelope.catch",
            attributes={"error.code": code},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="error_envelope.catch.invocation",
                    code=code,
                    timestamp=now,
                ),
            )
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.add_event(
                "meridian.error",
                {"error.code": code, "error.message": str(exc)},
            )
            span.record_exception(exc)
            self._audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="error_envelope.catch.unexpected",
                    code=code,
                    timestamp=now,
                    detail={"error_type": type(exc).__name__, "error": str(exc)},
                )
            )

        if self._hooks_dir is not None:
            # on_error hooks must never block the error response.
            with contextlib.suppress(Exception):
                await dispatch_hooks(
                    "on_error",
                    {"error_code": code, "error_message": str(exc)},
                    ExecutionContext(session_id=""),
                    hooks_dir=self._hooks_dir,
                    audit_log=self._audit_log,
                )

        try:
            await self._send_envelope(send, 500, code, message, {})
        except Exception as send_exc:
            self._audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="error_envelope.send.failed",
                    code="error_envelope_send_failed",
                    timestamp=_now(),
                    detail={"error": str(send_exc), "original_code": code},
                )
            )

    async def _send_envelope(
        self,
        send: Send,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any],
    ) -> None:
        body = json.dumps(
            {"error": {"code": code, "message": message, "details": details}},
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
