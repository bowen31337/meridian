from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# opentelemetry-api is an optional dependency.  We degrade gracefully when it
# is absent so the SDK works in minimal environments (e.g. a bare subprocess
# tool server that only needs the subprocess_server helper).
try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False


@asynccontextmanager
async def tool_span(
    tool_name: str,
    session_id: str | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> AsyncGenerator[Any, None]:
    """Async context manager that wraps a tool invocation in an OTel span.

    Yields the active span (or None when opentelemetry-api is not installed).
    On an unhandled exception the span status is set to ERROR before re-raise.
    """
    if not _OTEL_AVAILABLE:
        yield None
        return

    tracer = trace.get_tracer(
        "meridian.sdk_tool", schema_url="https://opentelemetry.io/schemas/1.24.0"
    )
    with tracer.start_as_current_span("tool.call") as span:
        span.set_attribute("tool.name", tool_name)
        if session_id:
            span.set_attribute("meridian.session_id", session_id)
        if extra_attrs:
            for key, value in extra_attrs.items():
                span.set_attribute(key, str(value))
        try:
            yield span
        except Exception:
            span.set_status(StatusCode.ERROR)
            raise
