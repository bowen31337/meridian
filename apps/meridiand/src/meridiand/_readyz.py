from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class ReadyzState:
    """Mutable readiness state shared between the lifespan and the /readyz handler."""

    storage: bool = False
    providers: bool = False
    plugins: bool = False


class ReadyzError(MeridianError):
    def __init__(self, *, message: str, timestamp: str, cause: BaseException | None = None) -> None:
        super().__init__(
            code="readyz_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


def make_readyz_router(*, audit_log: AuditLog, state: ReadyzState) -> APIRouter:
    router = APIRouter()

    @router.get("/readyz")
    async def get_readyz() -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span("health.readiness") as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="health.readiness.invocation",
                    code="health_readiness",
                    timestamp=now,
                ),
            )
            try:
                components = [
                    {"name": "storage", "ready": state.storage},
                    {"name": "providers", "ready": state.providers},
                    {"name": "plugins", "ready": state.plugins},
                ]
                if state.storage and state.providers and state.plugins:
                    return JSONResponse(status_code=200, content={"status": "ok"})
                return JSONResponse(
                    status_code=503,
                    content={"status": "not_ready", "components": components},
                )
            except Exception as exc:
                err = ReadyzError(
                    message=f"Readiness probe failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="health.readiness.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"message": err.message},
                    )
                )
                raise err

    return router
