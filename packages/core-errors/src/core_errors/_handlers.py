from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ._audit import AuditLog, NoopAuditLog
from ._telemetry import get_tracer, record_error, record_invocation_event
from ._types import AuditLogEntry, MeridianError, StructuredEvent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class HandlerOptions:
    """Options supplied by the host application for the global exception handler."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)


def install_error_handler(app: FastAPI, options: HandlerOptions | None = None) -> None:
    """
    Registers a global FastAPI exception handler for MeridianError.

    Per-invocation:
      1. Opens OTel span "meridian.error_handler" with error.code attribute.
      2. Attaches a "meridian.error.invocation" structured event to the span.
      3. Sets span status to ERROR and adds a "meridian.error" event; records
         the root cause exception on the span when present.
      4. Writes an audit log entry (level "error", event "meridian.error.handled").
      5. Returns a JSON error envelope with the appropriate HTTP status code.
    """
    opts = options or HandlerOptions()

    @app.exception_handler(MeridianError)
    async def _handler(request: Request, exc: MeridianError) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "meridian.error_handler",
            attributes={"error.code": exc.code},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="meridian.error.invocation",
                    code=exc.code,
                    timestamp=now,
                ),
            )
            record_error(span, exc)

            opts.audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="meridian.error.handled",
                    code=exc.code,
                    timestamp=now,
                    detail={"message": exc.message},
                )
            )

        return JSONResponse(status_code=exc.http_status(), content=exc.to_envelope())
