from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

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
from pydantic import BaseModel
from storage_event_log import EventLogWriter
from storage_reposit import LocalEventLogReader, PhaseProjection

from ._version import MERIDIAND_VERSION

_meter = metrics.get_meter("meridian.meridiand", MERIDIAND_VERSION)
_submit_counter = _meter.create_counter(
    "meridian_tool_results_submitted_total",
    description="Total number of submit_tool_results invocations",
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


class SubmitToolResultsSessionNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="submit_tool_results_session_not_found",
            message=message,
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class SubmitToolResultsWrongPhaseError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="submit_tool_results_wrong_phase",
            message=message,
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 422


class SubmitToolResultsError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="submit_tool_results_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ToolResultItem(BaseModel):
    tool_use_id: str
    content: Any = None
    is_error: bool = False


class SubmitToolResultsRequest(BaseModel):
    tool_results: list[ToolResultItem]


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_submit_tool_results_router(
    *,
    audit_log: AuditLog,
    storage_root: Path,
    event_log: EventLogWriter,
    harness_pool: _WakeDispatcher,
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/sessions/{session_id}/submit_tool_results", status_code=202)
    async def submit_tool_results(
        session_id: str, body: SubmitToolResultsRequest
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        _submit_counter.add(1, {"session.id": session_id})

        with tracer.start_as_current_span(
            "session.submit_tool_results",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.submit_tool_results.invocation",
                    code="submit_tool_results",
                    timestamp=now,
                ),
            )

            try:
                manifest_path = storage_root / "sessions" / session_id / "manifest.json"
                if not manifest_path.exists():
                    err = SubmitToolResultsSessionNotFoundError(
                        message=f"Session {session_id!r} not found",
                        timestamp=_now(),
                    )
                    record_error(span, err)
                    audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="session.submit_tool_results.failed",
                            code=err.code,
                            timestamp=err.timestamp,
                            detail={
                                "session_id": session_id,
                                "message": err.message,
                            },
                        )
                    )
                    raise err

                reader = LocalEventLogReader(storage_root)
                projection = PhaseProjection(reader)
                phase = projection.current_phase(session_id)

                if phase != "waiting_for_tool":
                    err2 = SubmitToolResultsWrongPhaseError(
                        message=(
                            f"Session {session_id!r} is in phase {phase!r},"
                            " expected 'waiting_for_tool'"
                        ),
                        timestamp=_now(),
                    )
                    record_error(span, err2)
                    audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="session.submit_tool_results.failed",
                            code=err2.code,
                            timestamp=err2.timestamp,
                            detail={
                                "session_id": session_id,
                                "phase": phase,
                                "message": err2.message,
                            },
                        )
                    )
                    raise err2

                for item in body.tool_results:
                    await event_log.append(
                        session_id,
                        "tool_call.result",
                        {
                            "tool_use_id": item.tool_use_id,
                            "content": item.content,
                            "is_error": item.is_error,
                            "timestamp": _now(),
                        },
                    )

                await event_log.append(
                    session_id,
                    "session.phase_change",
                    {
                        "before": "waiting_for_tool",
                        "after": "waiting_for_model",
                        "timestamp": _now(),
                        "reason": "tool_result",
                    },
                )

                await harness_pool.wake(session_id)

                span.set_attribute("session.submit_tool_results.count", len(body.tool_results))
                span.set_attribute("session.submit_tool_results.success", True)

                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="session.submit_tool_results.accepted",
                        code="submit_tool_results_accepted",
                        timestamp=_now(),
                        detail={
                            "session_id": session_id,
                            "count": len(body.tool_results),
                        },
                    )
                )

            except (
                SubmitToolResultsSessionNotFoundError,
                SubmitToolResultsWrongPhaseError,
                SubmitToolResultsError,
            ):
                raise
            except Exception as exc:
                err3 = SubmitToolResultsError(
                    message=f"Submit tool results failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err3)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.submit_tool_results.failed",
                        code=err3.code,
                        timestamp=err3.timestamp,
                        detail={
                            "session_id": session_id,
                            "message": err3.message,
                        },
                    )
                )
                raise err3

        return JSONResponse(
            content={
                "session_id": session_id,
                "submitted": len(body.tool_results),
            },
            status_code=202,
        )

    return router
