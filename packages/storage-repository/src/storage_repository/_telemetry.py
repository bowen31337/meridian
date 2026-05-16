from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ._types import RepositoryFailure, StructuredEvent
from ._version import REPOSITORY_SDK_VERSION

_TRACER_NAME = "meridian.storage-repository"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, REPOSITORY_SDK_VERSION)


def record_invocation_event(span: Span, event: StructuredEvent) -> None:
    """Attaches a structured "repo.invocation" event to the active span."""
    attrs: dict[str, str | int | float | bool] = {}
    for k, v in vars(event).items():
        if isinstance(v, (str, int, float, bool)):
            attrs[k] = v
    span.add_event("repo.invocation", attrs)


def record_repo_failure(span: Span, failure: RepositoryFailure) -> None:
    """Records a failure on the span: sets status to ERROR, adds a "repo.error" event,
    and records the underlying exception if present."""
    span.set_status(Status(StatusCode.ERROR, failure.message))
    span.add_event(
        "repo.error",
        {
            "entity.type": failure.entity_type,
            "entity.id": failure.entity_id,
            "repo.operation": failure.operation,
            "error.code": failure.code,
            "error.message": failure.message,
        },
    )
    if failure.cause is not None:
        span.record_exception(failure.cause)
