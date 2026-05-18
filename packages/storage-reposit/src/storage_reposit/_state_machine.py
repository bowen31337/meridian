from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ._audit import AuditLog, NoopAuditLog
from ._telemetry import get_tracer, record_invocation_event, record_state_machine_failure
from ._types import AuditLogEntry, IndexerFailure, StructuredEvent

PHASES: frozenset[str] = frozenset({"created", "running", "paused", "terminated"})
EVENTS: frozenset[str] = frozenset({"start", "pause", "resume", "terminate"})

_TRANSITIONS: dict[tuple[str, str], str] = {
    ("created", "start"): "running",
    ("created", "pause"): "created",
    ("created", "resume"): "created",
    ("created", "terminate"): "terminated",
    ("running", "start"): "running",
    ("running", "pause"): "paused",
    ("running", "resume"): "running",
    ("running", "terminate"): "terminated",
    ("paused", "start"): "paused",
    ("paused", "pause"): "paused",
    ("paused", "resume"): "running",
    ("paused", "terminate"): "terminated",
    ("terminated", "start"): "terminated",
    ("terminated", "pause"): "terminated",
    ("terminated", "resume"): "terminated",
    ("terminated", "terminate"): "terminated",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PhaseStateMachine:
    """
    Deterministic phase state machine.

    Given a current phase and an event type, returns the next phase.
    'terminated' is a final absorbing state: any event leaves it unchanged.

    Raises ValueError for unrecognised phases or event types.
    """

    def next_phase(self, current: str, event_type: str) -> str:
        """Return the next phase given the current phase and the triggering event."""
        if current not in PHASES:
            raise ValueError(
                f"Unknown phase: {current!r}. Valid phases: {sorted(PHASES)}"
            )
        if event_type not in EVENTS:
            raise ValueError(
                f"Unknown event type: {event_type!r}. Valid events: {sorted(EVENTS)}"
            )
        return _TRANSITIONS[(current, event_type)]


@dataclass
class PhaseStateMachineOptions:
    """Options supplied by the host application for each state machine call."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)
    on_error: Callable[[IndexerFailure], None] | None = None


class PhaseStateMachineRuntime:
    """
    Thin wrapper around PhaseStateMachine that adds OTel spans, structured
    invocation events, and audit-log writes on failure.

    Instantiate once with a PhaseStateMachine, then call next_phase through
    the runtime so every operation is traced and any failure is recorded in
    the audit log before being raised.
    """

    def __init__(self, machine: PhaseStateMachine) -> None:
        self._machine = machine

    def _fail(
        self,
        span: object,
        failure: IndexerFailure,
        options: PhaseStateMachineOptions,
        audit_event: str,
    ) -> None:
        record_state_machine_failure(span, failure)  # type: ignore[arg-type]
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

    def next_phase(
        self,
        session_id: str,
        current: str,
        event_type: str,
        *,
        options: PhaseStateMachineOptions | None = None,
    ) -> str:
        """
        Return the next phase for the given current phase and event type.

        Per-invocation:
          1. Opens OTel span "phase.next_phase" with session_id, current, and
             event_type attributes.
          2. Attaches a "phase.state_machine.invocation" structured event.
          3. Dispatches to the machine; re-audits IndexerFailure, wraps
             unexpected exceptions as PHASE_NEXT_PHASE_FAILED.
        """
        opts = options or PhaseStateMachineOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "phase.next_phase",
            attributes={
                "phase.session_id": session_id,
                "phase.current": current,
                "phase.event_type": event_type,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="phase.state_machine.invocation",
                    session_id=session_id,
                    timestamp=now,
                    operation="next_phase",
                ),
            )

            try:
                return self._machine.next_phase(current, event_type)
            except IndexerFailure as failure:
                self._fail(span, failure, opts, "phase.next_phase.failed")
                raise
            except Exception as exc:
                failure = IndexerFailure(
                    code="PHASE_NEXT_PHASE_FAILED",
                    message=str(exc),
                    session_id=session_id,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "phase.next_phase.failed")
                raise failure from exc
