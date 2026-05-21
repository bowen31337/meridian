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
from storage_event_log import EventLogWriter
from storage_reposit import LocalEventLogReader, PhaseProjection


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class UserCanContinueError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="user_can_continue_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_user_can_continue_router(
    *, audit_log: AuditLog, storage_root: Path, event_log: EventLogWriter
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/sessions/{session_id}/user-can-continue")
    async def user_can_continue(session_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "session.user_can_continue",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.user_can_continue.invocation",
                    code="session_user_can_continue",
                    timestamp=now,
                ),
            )

            try:
                reader = LocalEventLogReader(storage_root)
                projection = PhaseProjection(reader)
                before = projection.current_phase(session_id)

                await event_log.append(
                    session_id,
                    "session.phase_change",
                    {
                        "before": before,
                        "after": "running",
                        "timestamp": now,
                        "reason": "user_approved",
                    },
                )

            except UserCanContinueError:
                raise
            except Exception as exc:
                err = UserCanContinueError(
                    message=f"User can continue failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.user_can_continue.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

        return JSONResponse(
            content={
                "session_id": session_id,
                "before": before,
                "after": "running",
                "reason": "user_approved",
            }
        )

    return router
