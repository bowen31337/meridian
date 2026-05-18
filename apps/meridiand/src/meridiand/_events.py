from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from storage_event_log import SessionEvent
from storage_reposit import LocalEventLogReader


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class SessionEventsError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="session_events_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _event_to_dict(event: SessionEvent) -> dict[str, Any]:
    d: dict[str, Any] = {
        "seq": event.seq,
        "ts": event.ts,
        "type": event.type,
        "data": event.data,
    }
    if event.thread_id is not None:
        d["thread_id"] = event.thread_id
    return d


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_events_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/sessions/{session_id}/events")
    async def get_session_events(
        session_id: str,
        request: Request,
        since: int = Query(default=-1),
        type: str | None = Query(default=None),
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "session.events.read",
            attributes={
                "session.id": session_id,
                "events.since": since,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.events.read.invocation",
                    code="session_events_read",
                    timestamp=now,
                ),
            )

            try:
                reader = LocalEventLogReader(storage_root)
                raw_events = reader.read_after(session_id, since)

                type_set: set[str] | None = None
                if type is not None:
                    type_set = {t.strip() for t in type.split(",") if t.strip()}

                events = [e for e in raw_events if type_set is None or e.type in type_set]

            except SessionEventsError:
                raise
            except Exception as exc:
                err = SessionEventsError(
                    message=f"Failed to read events for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.events.read.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "since": since,
                            "message": err.message,
                        },
                    )
                )
                raise err

        accept = request.headers.get("accept", "")
        if "application/x-ndjson" in accept:

            def _ndjson() -> Iterator[str]:
                for event in events:
                    yield json.dumps(_event_to_dict(event), separators=(",", ":")) + "\n"

            return StreamingResponse(_ndjson(), media_type="application/x-ndjson")  # type: ignore[return-value]

        return JSONResponse(content=[_event_to_dict(e) for e in events])

    return router
