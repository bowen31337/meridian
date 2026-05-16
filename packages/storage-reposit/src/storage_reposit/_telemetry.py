from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ._types import IndexerFailure, StructuredEvent
from ._version import INDEXER_SDK_VERSION

_TRACER_NAME = "meridian.storage-reposit"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, INDEXER_SDK_VERSION)


def record_invocation_event(span: Span, event: StructuredEvent) -> None:
    """Attaches a structured "indexer.invocation" event to the active span."""
    attrs: dict[str, str | int | float | bool] = {}
    for k, v in vars(event).items():
        if isinstance(v, (str, int, float, bool)):
            attrs[k] = v
    span.add_event("indexer.invocation", attrs)


def record_indexer_failure(span: Span, failure: IndexerFailure) -> None:
    """Records a failure on the span: sets status to ERROR, adds an "indexer.error" event,
    and records the underlying exception if present."""
    span.set_status(Status(StatusCode.ERROR, failure.message))
    span.add_event(
        "indexer.error",
        {
            "indexer.session_id": failure.session_id,
            "error.code": failure.code,
            "error.message": failure.message,
        },
    )
    if failure.cause is not None:
        span.record_exception(failure.cause)
