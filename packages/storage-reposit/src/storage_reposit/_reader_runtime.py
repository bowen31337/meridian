from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from storage_event_log import SessionEvent

from ._audit import AuditLog, NoopAuditLog
from ._reader import LocalEventLogReader
from ._telemetry import get_tracer, record_invocation_event, record_reader_failure
from ._types import AuditLogEntry, EventSeq, IndexerFailure, StructuredEvent


@dataclass
class ReaderOptions:
    """Options supplied by the host application for each reader runtime call."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)
    on_error: Callable[[IndexerFailure], None] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReaderRuntime:
    """
    Thin wrapper around LocalEventLogReader that adds OTel spans, structured
    invocation events, and audit-log writes on failure.

    Instantiate once with a LocalEventLogReader, then call read_events through
    the runtime so every operation is traced and any failure is recorded in the
    audit log before being raised.
    """

    def __init__(self, reader: LocalEventLogReader) -> None:
        self._reader = reader

    def _fail(
        self,
        span: object,
        failure: IndexerFailure,
        options: ReaderOptions,
        audit_event: str,
    ) -> None:
        record_reader_failure(span, failure)  # type: ignore[arg-type]
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

    async def read_events(
        self,
        session_id: str,
        since: EventSeq = -1,
        *,
        follow: bool = False,
        options: ReaderOptions | None = None,
    ) -> AsyncIterator[SessionEvent]:
        """
        Stream events for session_id from the event log.

        Per-invocation:
          1. Opens OTel span "reader.read_events" with reader.session_id attribute.
          2. Attaches a "reader.invocation" structured event.
          3. Yields events from the reader; re-audits IndexerFailure (e.g. bad NDJSON),
             wraps unexpected exceptions as READER_READ_EVENTS_FAILED.
        """
        opts = options or ReaderOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "reader.read_events",
            attributes={"reader.session_id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="reader.invocation",
                    session_id=session_id,
                    timestamp=now,
                    operation="read_events",
                ),
            )

            try:
                async for event in self._reader.read_events(session_id, since, follow=follow):
                    yield event
            except IndexerFailure as failure:
                self._fail(span, failure, opts, "reader.read_events.failed")
                raise
            except Exception as exc:
                failure = IndexerFailure(
                    code="READER_READ_EVENTS_FAILED",
                    message=str(exc),
                    session_id=session_id,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "reader.read_events.failed")
                raise failure from exc
