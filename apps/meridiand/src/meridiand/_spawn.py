from __future__ import annotations

import uuid
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
from sdk_capabilities import CapabilityParseError, is_subset, missing, parse_set


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class SpawnError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="spawn_denied", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 403


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class SpawnRequest(BaseModel):
    parent_capabilities: list[str]
    child_capabilities: list[str]


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_spawn_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/sessions/{session_id}/spawn")
    async def spawn_session(session_id: str, body: SpawnRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        child_session_id = str(uuid.uuid4())

        with tracer.start_as_current_span(
            "session.spawn",
            attributes={
                "session.id": session_id,
                "session.child_id": child_session_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.spawn.invocation",
                    code="session_spawn",
                    timestamp=now,
                ),
            )

            try:
                parent_caps = parse_set(body.parent_capabilities)
                child_caps = parse_set(body.child_capabilities)
            except CapabilityParseError as exc:
                err = SpawnError(
                    message=f"Invalid capability string for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.spawn.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "parent_session_id": session_id,
                            "child_session_id": child_session_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

            escalating = missing(child_caps, parent_caps)
            if escalating:
                escalating_strs = sorted(str(c) for c in escalating)
                err = SpawnError(
                    message=(
                        f"Capability escalation denied for child of session {session_id!r}: "
                        f"caps not held by parent: {', '.join(escalating_strs)}"
                    ),
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.spawn.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "parent_session_id": session_id,
                            "child_session_id": child_session_id,
                            "escalating_caps": escalating_strs,
                            "message": err.message,
                        },
                    )
                )
                raise err

        return JSONResponse(
            content={
                "child_session_id": child_session_id,
                "parent_session_id": session_id,
                "capabilities": sorted(str(c) for c in child_caps),
                "status": "spawned",
            }
        )

    return router
