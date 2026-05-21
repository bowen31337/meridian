from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ._version import MERIDIAND_VERSION

_TRACER_NAME = "meridian.meridiand"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, MERIDIAND_VERSION)


def record_create_event(
    span: Span,
    *,
    cors_enabled: bool,
    gzip_enabled: bool,
    router_count: int,
    serve_ui_enabled: bool = False,
) -> None:
    span.add_event(
        "app.factory.create",
        {
            "cors_enabled": cors_enabled,
            "gzip_enabled": gzip_enabled,
            "router_count": router_count,
            "serve_ui_enabled": serve_ui_enabled,
        },
    )


def record_factory_failure(span: Span, error: Exception) -> None:
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.add_event(
        "app.factory.error",
        {
            "error.type": type(error).__name__,
            "error.message": str(error),
        },
    )
    span.record_exception(error)


def record_daemon_start_event(
    span: Span,
    *,
    bind_mode: str,
    bind_socket: str,
    bind_host: str,
    bind_port: int,
) -> None:
    span.add_event(
        "daemon.start",
        {
            "daemon.bind_mode": bind_mode,
            "daemon.socket": bind_socket,
            "daemon.host": bind_host,
            "daemon.port": bind_port,
        },
    )


def record_daemon_failure(span: Span, error: Exception) -> None:
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.add_event(
        "daemon.start.error",
        {
            "error.type": type(error).__name__,
            "error.message": str(error),
        },
    )
    span.record_exception(error)
