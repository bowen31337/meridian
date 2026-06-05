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
from sdk_sandbox import ExecutionContext
from storage_event_log import EventLogWriter
from storage_reposit import LocalEventLogReader, PhaseProjection

from ._hook_dispatch import dispatch_hooks


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class SoftBudgetExceededSessionError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="soft_budget_exceeded_session_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class SoftBudgetExceededRequest(BaseModel):
    dimension: str
    limit: float
    actual: float


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_soft_budget_exceeded_router(
    *, audit_log: AuditLog, storage_root: Path, event_log: EventLogWriter
) -> APIRouter:
    router = APIRouter()
    hooks_dir = storage_root / "hooks"

    @router.post("/v1/x/sessions/{session_id}/soft-budget-exceeded")
    async def soft_budget_exceeded(
        session_id: str, body: SoftBudgetExceededRequest
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "session.soft_budget_exceeded",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.soft_budget_exceeded.invocation",
                    code="session_soft_budget_exceeded",
                    timestamp=now,
                ),
            )

            try:
                reader = LocalEventLogReader(storage_root)
                projection = PhaseProjection(reader)
                before = projection.current_phase(session_id)

                await dispatch_hooks(
                    "pre_message",
                    {
                        "session_id": session_id,
                        "budget_warning": {
                            "dimension": body.dimension,
                            "limit": body.limit,
                            "actual": body.actual,
                        },
                    },
                    ExecutionContext(session_id=session_id),
                    hooks_dir=hooks_dir,
                    audit_log=audit_log,
                )

                await event_log.append(
                    session_id,
                    "budget.warning",
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
                        "after": "waiting_for_user",
                        "timestamp": now,
                        "reason": "budget_warning",
                    },
                )

            except SoftBudgetExceededSessionError:
                raise
            except Exception as exc:
                err = SoftBudgetExceededSessionError(
                    message=(
                        f"Soft budget exceeded handling failed for session {session_id!r}: {exc}"
                    ),
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.soft_budget_exceeded.failed",
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
                "after": "waiting_for_user",
                "reason": "budget_warning",
                "dimension": body.dimension,
                "limit": body.limit,
                "actual": body.actual,
            }
        )

    return router
