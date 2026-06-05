from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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
from pydantic import BaseModel
from storage_event_log import EventLogWriter
from storage_reposit import LocalEventLogReader, PhaseProjection


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class BudgetExceededSessionError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="budget_exceeded_session_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class BudgetExceededRequest(BaseModel):
    dimension: str
    limit: float
    actual: float


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_budget_exceeded_router(
    *, audit_log: AuditLog, storage_root: Path, event_log: EventLogWriter
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/sessions/{session_id}/budget-exceeded")
    async def budget_exceeded(session_id: str, body: BudgetExceededRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "session.budget_exceeded",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.budget_exceeded.invocation",
                    code="session_budget_exceeded",
                    timestamp=now,
                ),
            )

            try:
                reader = LocalEventLogReader(storage_root)
                projection = PhaseProjection(reader)
                before = projection.current_phase(session_id)

                await event_log.append(
                    session_id,
                    "budget.exceeded",
                    {
                        "session_id": session_id,
                        "dimension": body.dimension,
                        "limit": body.limit,
                        "actual": body.actual,
                        "timestamp": now,
                    },
                )

                await event_log.append(
                    session_id,
                    "session.phase_change",
                    {
                        "before": before,
                        "after": "terminated",
                        "timestamp": now,
                        "reason": "budget_exceeded",
                    },
                )

            except BudgetExceededSessionError:
                raise
            except Exception as exc:
                err = BudgetExceededSessionError(
                    message=f"Budget exceeded termination failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.budget_exceeded.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

        return JSONResponse(
            content={
                "session_id": session_id,
                "before": before,
                "after": "terminated",
                "reason": "budget_exceeded",
                "dimension": body.dimension,
                "limit": body.limit,
                "actual": body.actual,
            }
        )

    return router
