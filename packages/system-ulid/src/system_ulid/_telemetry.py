from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ._types import StructuredEvent, UlidFailure
from ._version import ULID_SDK_VERSION

_TRACER_NAME = "meridian.system-ulid"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, ULID_SDK_VERSION)


def record_invocation_event(span: Span, event: StructuredEvent) -> None:
    """Attaches a structured "ulid.invocation" event to the active span."""
    attrs: dict[str, str | int | float | bool] = {}
    for k, v in vars(event).items():
        if isinstance(v, (str, int, float, bool)):
            attrs[k] = v
    span.add_event("ulid.invocation", attrs)


def record_ulid_failure(span: Span, failure: UlidFailure) -> None:
    """Records a failure on the span: sets status to ERROR, adds a "ulid.error" event,
    and records the underlying exception if present."""
    span.set_status(Status(StatusCode.ERROR, failure.message))
    span.add_event(
        "ulid.error",
        {
            "ulid.prefix": failure.prefix,
            "error.code": failure.code,
            "error.message": failure.message,
        },
    )
    if failure.cause is not None:
        span.record_exception(failure.cause)
