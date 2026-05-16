from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ._types import MeridianError, StructuredEvent
from ._version import CORE_ERRORS_VERSION

_TRACER_NAME = "meridian.core-errors"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, CORE_ERRORS_VERSION)


def record_invocation_event(span: Span, event: StructuredEvent) -> None:
    """Attaches a structured "meridian.error.invocation" event to the active span."""
    attrs: dict[str, str | int | float | bool] = {}
    for k, v in vars(event).items():
        if isinstance(v, (str, int, float, bool)):
            attrs[k] = v
    span.add_event("meridian.error.invocation", attrs)


def record_error(span: Span, error: MeridianError) -> None:
    """Sets span status to ERROR, adds a structured error event, and records the cause."""
    span.set_status(Status(StatusCode.ERROR, error.message))
    span.add_event(
        "meridian.error",
        {
            "error.code": error.code,
            "error.message": error.message,
        },
    )
    if error.cause is not None:
        span.record_exception(error.cause)
