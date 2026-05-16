from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ._version import KB_INDEXER_VERSION

TRACER_NAME = "meridian.kb-indexer"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME, KB_INDEXER_VERSION)


def record_invocation_event(
    span: Span,
    *,
    operation: str,
    file_path: str | None = None,
    chunk_count: int | None = None,
) -> None:
    """Attach a structured ``indexer.invocation`` event to the active span."""
    attrs: dict[str, str | int] = {"indexer.operation": operation}
    if file_path:
        attrs["indexer.file_path"] = file_path
    if chunk_count is not None:
        attrs["indexer.chunk_count"] = chunk_count
    span.add_event("indexer.invocation", attrs)


def record_indexer_failure(
    span: Span,
    error: Exception,
    *,
    operation: str,
    file_path: str | None = None,
) -> None:
    """Set span status to ERROR and record an ``indexer.error`` event."""
    span.set_status(Status(StatusCode.ERROR, str(error)))
    event_attrs: dict[str, str] = {
        "indexer.operation": operation,
        "error.type": type(error).__name__,
        "error.message": str(error),
    }
    if file_path:
        event_attrs["indexer.file_path"] = file_path
    span.add_event("indexer.error", event_attrs)
    span.record_exception(error)
