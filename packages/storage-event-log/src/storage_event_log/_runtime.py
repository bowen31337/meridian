from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ._audit import AuditLog, NoopAuditLog
from ._contract import EventLogWriter
from ._telemetry import get_tracer, record_event_log_failure, record_invocation_event
from ._types import AuditLogEntry, EventLogFailure, StructuredEvent


@dataclass
class EventLogOptions:
    """Options supplied by the host application for each runtime call."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)
    on_error: Callable[[EventLogFailure], None] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventLogRuntime:
    """
    Thin wrapper around an EventLogWriter that adds OTel spans, structured
    invocation events, and audit-log writes on failure.

    Instantiate once with a concrete EventLogWriter backend (e.g.
    LocalEventLogWriter), then call append through the runtime so every
    operation is traced and any failure is recorded in the audit log before
    being raised.
    """

    def __init__(self, writer: EventLogWriter) -> None:
        self._writer = writer

    def _fail(
        self,
        span: object,
        failure: EventLogFailure,
        options: EventLogOptions,
        audit_event: str,
    ) -> None:
        record_event_log_failure(span, failure)  # type: ignore[arg-type]
        options.audit_log.write(
            AuditLogEntry(
                level="error",
                event=audit_event,
                session_id=failure.session_id,
                timestamp=failure.timestamp,
                detail={"code": failure.code, "message": failure.message},
            )
        )
        if options.on_error is not None:
            options.on_error(failure)

    async def append(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
        *,
        thread_id: str | None = None,
        options: EventLogOptions | None = None,
    ) -> int:
        """
        Append one event to the session's NDJSON log and return its seq number.

        Per-invocation:
          1. Opens OTel span "event_log.append" with event_log.session_id attribute.
          2. Attaches an "event_log.invocation" structured event.
          3. Dispatches to the writer; wraps unexpected exceptions as
             EVENT_LOG_APPEND_FAILED.
        """
        opts = options or EventLogOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "event_log.append",
            attributes={"event_log.session_id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="event_log.invocation",
                    session_id=session_id,
                    timestamp=now,
                    operation="append",
                ),
            )

            try:
                return await self._writer.append(
                    session_id, event_type, data, thread_id=thread_id
                )
            except EventLogFailure as failure:
                self._fail(span, failure, opts, "event_log.append.failed")
                raise
            except Exception as exc:
                failure = EventLogFailure(
                    code="EVENT_LOG_APPEND_FAILED",
                    message=str(exc),
                    session_id=session_id,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "event_log.append.failed")
                raise failure from exc
