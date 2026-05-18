from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
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


def _format_sse(event: SessionEvent) -> str:
    data = json.dumps(_event_to_dict(event), separators=(",", ":"))
    return f"event: {event.type}\nid: {event.seq}\ndata: {data}\n\n"


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
        stream: bool = Query(default=False),
    ) -> JSONResponse:
        type_set: set[str] | None = None
        if type is not None:
            type_set = {t.strip() for t in type.split(",") if t.strip()}

        # SSE streaming branch
        if stream:
            effective_since = since
            last_event_id = request.headers.get("last-event-id")
            if last_event_id is not None:
                try:
                    effective_since = int(last_event_id)
                except ValueError:
                    pass

            now = _now()
            tracer = get_tracer()
            with tracer.start_as_current_span(
                "session.events.stream",
                attributes={"session.id": session_id, "events.since": effective_since},
            ) as span:
                record_invocation_event(
                    span,
                    StructuredEvent(
                        name="session.events.stream.invocation",
                        code="session_events_stream",
                        timestamp=now,
                    ),
                )

            async def _sse() -> AsyncIterator[str]:
                reader = LocalEventLogReader(storage_root)
                try:
                    async for event in reader.read_events(
                        session_id, effective_since, follow=False
                    ):
                        if type_set is None or event.type in type_set:
                            yield _format_sse(event)
                except Exception as exc:
                    err = SessionEventsError(
                        message=f"Failed to stream events for session {session_id!r}: {exc}",
                        timestamp=_now(),
                        cause=exc,
                    )
                    audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="session.events.stream.failed",
                            code=err.code,
                            timestamp=err.timestamp,
                            detail={
                                "session_id": session_id,
                                "since": effective_since,
                                "message": err.message,
                            },
                        )
                    )
                    error_data = json.dumps({"code": err.code, "message": err.message})
                    yield f"event: error\ndata: {error_data}\n\n"

            return StreamingResponse(_sse(), media_type="text/event-stream")  # type: ignore[return-value]

        # Non-streaming branch
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
