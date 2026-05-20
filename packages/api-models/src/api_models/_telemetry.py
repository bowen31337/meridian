from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ._version import API_MODELS_VERSION

_TRACER_NAME = "meridian.api-models"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, API_MODELS_VERSION)


def record_list_event(span: Span, *, count: int) -> None:
    """Attach a structured models.list.invocation event to the active span."""
    span.add_event("models.list.invocation", {"count": count})


def record_failure(span: Span, error: Exception, *, operation: str) -> None:
    """Set span status to ERROR, add a models.error event, and record the exception."""
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.add_event(
        "models.error",
        {
            "operation": operation,
            "error.type": type(error).__name__,
            "error.message": str(error),
        },
    )
    span.record_exception(error)
