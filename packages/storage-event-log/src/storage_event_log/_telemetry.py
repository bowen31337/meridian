from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ._types import EventLogFailure, StructuredEvent
from ._version import EVENT_LOG_SDK_VERSION

_TRACER_NAME = "meridian.storage-event-log"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, EVENT_LOG_SDK_VERSION)


def record_invocation_event(span: Span, event: StructuredEvent) -> None:
    """Attaches a structured "event_log.invocation" event to the active span."""
    attrs: dict[str, str | int | float | bool] = {}
    for k, v in vars(event).items():
        if isinstance(v, (str, int, float, bool)):
            attrs[k] = v
    span.add_event("event_log.invocation", attrs)


def record_event_log_failure(span: Span, failure: EventLogFailure) -> None:
    """Records a failure on the span: sets status to ERROR, adds an "event_log.error" event,
    and records the underlying exception if present."""
    span.set_status(Status(StatusCode.ERROR, failure.message))
    span.add_event(
        "event_log.error",
        {
            "event_log.session_id": failure.session_id,
            "error.code": failure.code,
            "error.message": failure.message,
        },
    )
    if failure.cause is not None:
        span.record_exception(failure.cause)
