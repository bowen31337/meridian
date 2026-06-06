from __future__ import annotations

from datetime import UTC, datetime
import json
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

from ._metrics_registry import active_sessions, session_duration_seconds, sessions_total

_TERMINAL_PHASES = frozenset({"terminated", "completed"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class PhaseTransitionError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="phase_transition_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class PhaseTransitionRequest(BaseModel):
    to_phase: str
    reason: str


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_phase_router(
    *, audit_log: AuditLog, storage_root: Path, event_log: EventLogWriter
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/sessions/{session_id}/phase")
    async def transition_phase(session_id: str, body: PhaseTransitionRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "session.phase_transition",
            attributes={
                "session.id": session_id,
                "phase.to": body.to_phase,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.phase_transition.invocation",
                    code="phase_transition",
                    timestamp=now,
                ),
            )

            try:
                reader = LocalEventLogReader(storage_root)
                projection = PhaseProjection(reader)
                before = projection.current_phase(session_id)

                seq = await event_log.append(
                    session_id,
                    "session.phase_change",
                    {
                        "before": before,
                        "after": body.to_phase,
                        "timestamp": now,
                        "reason": body.reason,
                    },
                )
                sessions_total.labels(phase=body.to_phase).inc()
                active_sessions.labels(phase=body.to_phase).inc()
                active_sessions.labels(phase=before).dec()
                if body.to_phase in _TERMINAL_PHASES:
                    manifest_path = storage_root / "sessions" / session_id / "manifest.json"
                    try:
                        manifest = json.loads(manifest_path.read_text())
                        created_at = manifest.get("created_at", "")
                        if created_at:
                            started = datetime.fromisoformat(created_at)
                            duration = (datetime.now(UTC) - started).total_seconds()
                            session_duration_seconds.labels(result=body.to_phase).observe(duration)
                    except Exception:
                        pass
            except PhaseTransitionError:
                raise
            except Exception as exc:
                err = PhaseTransitionError(
                    message=f"Phase transition failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.phase_transition.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "to_phase": body.to_phase,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

        return JSONResponse(
            content={
                "session_id": session_id,
                "before": before,
                "after": body.to_phase,
                "reason": body.reason,
                "seq": seq,
            }
        )

    return router
