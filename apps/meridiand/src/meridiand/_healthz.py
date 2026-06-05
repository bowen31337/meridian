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
from fastapi.responses import JSONResponse


def _now() -> str:
    return datetime.now(UTC).isoformat()


class HealthzError(MeridianError):
    def __init__(self, *, message: str, timestamp: str, cause: BaseException | None = None) -> None:
        super().__init__(
            code="healthz_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


def make_healthz_router(*, audit_log: AuditLog) -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    async def get_healthz() -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span("health.liveness") as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="health.liveness.invocation",
                    code="health_liveness",
                    timestamp=now,
                ),
            )
            try:
                return JSONResponse(status_code=200, content={"status": "ok"})
            except Exception as exc:
                err = HealthzError(
                    message=f"Liveness probe failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="health.liveness.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"message": err.message},
                    )
                )
                raise err from exc

    return router
