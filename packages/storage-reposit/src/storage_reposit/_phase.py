from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ._audit import AuditLog, NoopAuditLog
from ._reader import LocalEventLogReader
from ._telemetry import get_tracer, record_invocation_event, record_phase_failure
from ._types import AuditLogEntry, IndexerFailure, StructuredEvent

_DEFAULT_PHASE = "created"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PhaseProjection:
    """
    Derives the current phase of a session by scanning its event log tail.

    Reads all events for a session in ascending seq order and returns the
    'after' field from the last session.phase_change event found. Returns
    'created' if no session.phase_change events exist yet.
    """

    def __init__(self, reader: LocalEventLogReader) -> None:
        self._reader = reader

    def current_phase(self, session_id: str) -> str:
        """Return the current phase derived from the session's event log."""
        events = self._reader.read_after(session_id, -1)
        phase = _DEFAULT_PHASE
        for event in events:
            if event.type == "session.phase_change":
                after = event.data.get("after")
                if isinstance(after, str) and after:
                    phase = after
        return phase


@dataclass
class PhaseProjectionOptions:
    """Options supplied by the host application for each phase projection call."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)
    on_error: Callable[[IndexerFailure], None] | None = None


class PhaseProjectionRuntime:
    """
    Thin wrapper around PhaseProjection that adds OTel spans, structured
    invocation events, and audit-log writes on failure.

    Instantiate once with a PhaseProjection, then call current_phase through
    the runtime so every operation is traced and any failure is recorded in the
    audit log before being raised.
    """

    def __init__(self, projection: PhaseProjection) -> None:
        self._projection = projection

    def _fail(
        self,
        span: object,
        failure: IndexerFailure,
        options: PhaseProjectionOptions,
        audit_event: str,
    ) -> None:
        record_phase_failure(span, failure)  # type: ignore[arg-type]
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

    def current_phase(
        self,
        session_id: str,
        *,
        options: PhaseProjectionOptions | None = None,
    ) -> str:
        """
        Return the current phase for session_id derived from the event log.

        Per-invocation:
          1. Opens OTel span "phase.current_phase" with phase.session_id attribute.
          2. Attaches a "phase.invocation" structured event.
          3. Dispatches to the projection; re-audits IndexerFailure (e.g. bad NDJSON),
             wraps unexpected exceptions as PHASE_PROJECT_FAILED.
        """
        opts = options or PhaseProjectionOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "phase.current_phase",
            attributes={"phase.session_id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="phase.invocation",
                    session_id=session_id,
                    timestamp=now,
                    operation="current_phase",
                ),
            )

            try:
                return self._projection.current_phase(session_id)
            except IndexerFailure as failure:
                self._fail(span, failure, opts, "phase.current_phase.failed")
                raise
            except Exception as exc:
                failure = IndexerFailure(
                    code="PHASE_PROJECT_FAILED",
                    message=str(exc),
                    session_id=session_id,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "phase.current_phase.failed")
                raise failure from exc
