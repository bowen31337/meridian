from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ._version import MERIDIAN_CLI_VERSION

_TRACER_NAME = "meridian.cli"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, MERIDIAN_CLI_VERSION)


def record_invocation_event(span: Span, event: dict[str, object]) -> None:
    attrs: dict[str, str | int | float | bool] = {}
    for k, v in event.items():
        if isinstance(v, (str, int, float, bool)):
            attrs[k] = v
    span.add_event("meridian.cli.invocation", attrs)


def record_failure(span: Span, code: str, message: str) -> None:
    span.set_status(Status(StatusCode.ERROR, message))
    span.add_event("meridian.cli.failure", {"error.code": code, "error.message": message})
