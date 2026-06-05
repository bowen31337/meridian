from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
import uuid

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
from opentelemetry import metrics

from ._version import MERIDIAND_VERSION

_meter = metrics.get_meter("meridian.meridiand", MERIDIAND_VERSION)
_wakes_counter = _meter.create_counter(
    "meridian_session_explicit_wakes_total",
    description="Total number of explicit session wake invocations",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Protocol — allows test stubs without requiring a full HarnessPool
# ---------------------------------------------------------------------------


class _WakeDispatcher(Protocol):
    async def wake(self, session_id: str) -> None: ...


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class SessionWakeNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="session_wake_not_found", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 404


class SessionWakeError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="session_wake_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_session_wake_router(
    *, audit_log: AuditLog, storage_root: Path, harness_pool: _WakeDispatcher
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/sessions/{session_id}/wake", status_code=202)
    async def wake_session(session_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        harness_instance_id = f"harness_{uuid.uuid4().hex}"
        _wakes_counter.add(1, {"session.id": session_id})

        with tracer.start_as_current_span(
            "session.wake",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.wake.invocation",
                    code="session_wake",
                    timestamp=now,
                ),
            )

            try:
                manifest_path = storage_root / "sessions" / session_id / "manifest.json"
                if not manifest_path.exists():
                    err = SessionWakeNotFoundError(
                        message=f"Session {session_id!r} not found",
                        timestamp=_now(),
                    )
                    record_error(span, err)
                    audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="session.wake.failed",
                            code=err.code,
                            timestamp=err.timestamp,
                            detail={
                                "session_id": session_id,
                                "message": err.message,
                            },
                        )
                    )
                    raise err

                await harness_pool.wake(session_id)

                span.set_attribute("session.wake.harness_instance_id", harness_instance_id)
                span.set_attribute("session.wake.success", True)

                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="session.wake.accepted",
                        code="session_wake_accepted",
                        timestamp=_now(),
                        detail={
                            "session_id": session_id,
                            "harness_instance_id": harness_instance_id,
                        },
                    )
                )

            except (SessionWakeNotFoundError, SessionWakeError):
                raise
            except Exception as exc:
                err2 = SessionWakeError(
                    message=f"Wake failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.wake.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "session_id": session_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2 from exc

        return JSONResponse(
            content={
                "session_id": session_id,
                "harness_instance_id": harness_instance_id,
            },
            status_code=202,
        )

    return router
