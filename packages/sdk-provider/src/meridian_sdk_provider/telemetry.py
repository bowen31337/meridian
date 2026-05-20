from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ._version import SDK_PROVIDER_VERSION

TRACER_NAME = "meridian.sdk-provider"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME, SDK_PROVIDER_VERSION)


def record_invocation_event(
    span: Span,
    *,
    provider_name: str,
    provider_kind: str,
    model: str,
    session_id: str | None,
    routing_rule: str | None,
) -> None:
    """Attach a structured ``provider.invocation`` event to the active span.

    Called once per routing attempt (primary and each fallback) so the span
    carries a complete audit trail of which providers were tried.
    """
    attrs: dict[str, str] = {
        "provider.name": provider_name,
        "provider.kind": provider_kind,
        "model": model,
    }
    if session_id:
        attrs["session.id"] = session_id
    if routing_rule:
        attrs["routing.rule"] = routing_rule
    span.add_event("provider.invocation", attrs)


def record_cache_metrics(
    span: Span,
    *,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
) -> None:
    """Attach a provider.cache_metrics event carrying cache hit/miss counters."""
    span.add_event(
        "provider.cache_metrics",
        {
            "cache.creation_tokens": cache_creation_input_tokens,
            "cache.read_tokens": cache_read_input_tokens,
            "cache.hit": cache_read_input_tokens > 0,
        },
    )


def record_provider_failure(
    span: Span,
    error: Exception,
    *,
    provider_name: str,
    model: str,
) -> None:
    """Set span status to ERROR and record a ``provider.error`` event.

    The underlying exception is also attached via ``record_exception`` so
    exporters that support stack traces can surface it.
    """
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.add_event(
        "provider.error",
        {
            "provider.name": provider_name,
            "model": model,
            "error.type": type(error).__name__,
            "error.message": str(error),
        },
    )
    span.record_exception(error)
