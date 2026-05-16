from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ._types import EnvironmentFailure, StructuredEvent
from ._version import ENVIRONMENT_SDK_VERSION

_TRACER_NAME = "meridian.sdk-environment"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, ENVIRONMENT_SDK_VERSION)


def record_invocation_event(span: Span, event: StructuredEvent) -> None:
    """
    Attaches a structured "environment.invocation" event to the active span.
    Called once per runtime operation regardless of success or failure.
    """
    attrs: dict[str, str | int | float | bool] = {}
    for k, v in vars(event).items():
        if isinstance(v, (str, int, float, bool)):
            attrs[k] = v
    span.add_event("environment.invocation", attrs)


def record_environment_failure(span: Span, failure: EnvironmentFailure) -> None:
    """
    Records a failure on the span: sets status to ERROR, adds an
    "environment.error" event, and records the underlying exception if present.
    """
    span.set_status(Status(StatusCode.ERROR, failure.message))
    span.add_event(
        "environment.error",
        {
            "environment.id": failure.environment_id,
            "environment.kind": failure.environment_kind,
            "session.id": failure.session_id,
            "error.code": failure.code,
            "error.message": failure.message,
        },
    )
    if failure.cause is not None:
        span.record_exception(failure.cause)
