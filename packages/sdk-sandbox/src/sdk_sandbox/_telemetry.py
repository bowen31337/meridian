from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ._types import SandboxFailure, StructuredEvent
from ._version import SANDBOX_SDK_VERSION

_TRACER_NAME = "meridian.sdk-sandbox"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, SANDBOX_SDK_VERSION)


def record_invocation_event(span: Span, event: StructuredEvent) -> None:
    """
    Attaches a structured "sandbox.invocation" event to the active span.
    Called once per execute() call regardless of success or failure.
    """
    attrs: dict[str, str | int | float | bool] = {}
    for k, v in vars(event).items():
        if isinstance(v, (str, int, float, bool)):
            attrs[k] = v
    span.add_event("sandbox.invocation", attrs)


def record_sandbox_failure(span: Span, failure: SandboxFailure) -> None:
    """
    Records a failure on the span: sets status to ERROR, adds a
    "sandbox.error" event, and records the underlying exception if present.
    """
    span.set_status(Status(StatusCode.ERROR, failure.message))
    span.add_event(
        "sandbox.error",
        {
            "tool.name": failure.tool_name,
            "session.id": failure.session_id,
            "error.code": failure.code,
            "error.message": failure.message,
        },
    )
    if failure.cause is not None:
        span.record_exception(failure.cause)
