from __future__ import annotations

from datetime import UTC, datetime

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest


def _now() -> str:
    return datetime.now(UTC).isoformat()


class MetricsError(MeridianError):
    def __init__(
        self, *, message: str, timestamp: str, cause: BaseException | None = None
    ) -> None:
        super().__init__(
            code="metrics_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


def make_metrics_router(*, audit_log: AuditLog) -> APIRouter:
    router = APIRouter()

    @router.get("/metrics")
    async def get_metrics() -> Response:
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span("metrics.scrape") as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="metrics.scrape.invocation",
                    code="metrics_scrape",
                    timestamp=now,
                ),
            )
            try:
                data = generate_latest()
                return Response(content=data, media_type=CONTENT_TYPE_LATEST)
            except Exception as exc:
                err = MetricsError(
                    message=f"Metrics scrape failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="metrics.scrape.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"message": err.message},
                    )
                )
                raise err

    return router
