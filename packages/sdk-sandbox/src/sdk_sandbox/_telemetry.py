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


def record_capability_denial(
    span: Span,
    tool_name: str,
    session_id: str,
    required: frozenset[str],
    missing: frozenset[str],
    granted: frozenset[str],
    message: str,
) -> None:
    """
    Records a capability denial on the span: sets status to ERROR and adds
    a "capability.denied" event with required/missing/granted detail.
    Does not record an exception (denial is policy, not an unexpected error).
    """
    span.set_status(Status(StatusCode.ERROR, message))
    span.add_event(
        "capability.denied",
        {
            "tool.name": tool_name,
            "session.id": session_id,
            "error.code": "capability_denied",
            "capability.required": ", ".join(sorted(required)),
            "capability.missing": ", ".join(sorted(missing)),
            "capability.granted": ", ".join(sorted(granted)),
        },
    )


def record_env_mismatch(
    span: Span,
    tool_name: str,
    session_id: str,
    requires_env: str,
    actual_env: str | None,
    message: str,
) -> None:
    """
    Records an environment mismatch on the span: sets status to ERROR and adds
    an "env.mismatch" event with requires/actual detail.
    """
    span.set_status(Status(StatusCode.ERROR, message))
    span.add_event(
        "env.mismatch",
        {
            "tool.name": tool_name,
            "session.id": session_id,
            "error.code": "env_mismatch",
            "env.required": requires_env,
            "env.actual": actual_env or "",
        },
    )


def record_tool_timeout(
    span: Span,
    tool_name: str,
    session_id: str,
    timeout_ms: int,
    message: str,
) -> None:
    """
    Records a tool execution timeout on the span: sets status to ERROR and
    adds a "tool.timeout" event.
    """
    span.set_status(Status(StatusCode.ERROR, message))
    span.add_event(
        "tool.timeout",
        {
            "tool.name": tool_name,
            "session.id": session_id,
            "error.code": "timeout",
            "timeout.ms": timeout_ms,
        },
    )


def record_input_schema_failure(
    span: Span,
    tool_name: str,
    session_id: str,
    offending_path: str,
    message: str,
) -> None:
    """
    Records a pre-dispatch input schema failure on the span: sets status to
    ERROR and adds an "input.schema.failed" event with the offending field path.
    """
    span.set_status(Status(StatusCode.ERROR, message))
    span.add_event(
        "input.schema.failed",
        {
            "tool.name": tool_name,
            "session.id": session_id,
            "error.code": "input_validation_failed",
            "error.message": message,
            "schema.offending_path": offending_path,
        },
    )


def record_output_schema_failure(
    span: Span,
    tool_name: str,
    session_id: str,
    offending_path: str,
    message: str,
) -> None:
    """
    Records a post-dispatch output schema failure on the span: sets status to
    ERROR and adds an "output.schema.failed" event with the offending field path.
    """
    span.set_status(Status(StatusCode.ERROR, message))
    span.add_event(
        "output.schema.failed",
        {
            "tool.name": tool_name,
            "session.id": session_id,
            "error.code": "output_validation_failed",
            "error.message": message,
            "schema.offending_path": offending_path,
        },
    )


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
