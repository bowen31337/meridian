from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
import contextlib
from datetime import UTC, datetime
import json
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
from storage_event_log import SUBSCRIBER_CHANNEL_SIZE, SessionEvent, SubscriberBus
from storage_reposit import LocalEventLogReader

# ---------------------------------------------------------------------------
# SDK event kind mapping
# ---------------------------------------------------------------------------

# Maps internal event log types to the SDK-facing SessionEventKind discriminant.
# Only events listed here are surfaced on the SDK endpoint; all others are omitted.
_SDK_KIND_MAP: dict[str, str] = {
    "message.added": "message",
    "tool_call.requested": "tool_call",
    "tool_call.result": "tool_result",
    "canvas_op": "canvas_op",
    "error": "error",
}


def _to_sdk_event(event: SessionEvent, session_id: str) -> dict[str, Any] | None:
    """Convert an internal SessionEvent to SDK-facing format, or None to skip it."""
    kind = _SDK_KIND_MAP.get(event.type)
    if kind is None:
        return None
    return {
        "id": f"{session_id}:{event.seq}",
        "session_id": session_id,
        "kind": kind,
        "payload": event.data,
        "timestamp": event.ts,
    }


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


def make_events_router(
    *,
    audit_log: AuditLog,
    storage_root: Path,
    subscriber_bus: SubscriberBus | None = None,
) -> APIRouter:
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
                with contextlib.suppress(ValueError):
                    effective_since = int(last_event_id)

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

            if subscriber_bus is not None:
                # Live streaming: subscribe first, replay history from disk, then
                # follow the in-process queue.  The harness never blocks: if the
                # queue fills up the subscriber is dropped with a subscriber_lagged
                # sentinel and this generator terminates.
                queue = subscriber_bus.subscribe(session_id)

                async def _sse_live() -> AsyncIterator[str]:
                    try:
                        reader = LocalEventLogReader(storage_root)
                        watermark = effective_since
                        try:
                            async for event in reader.read_events(
                                session_id, effective_since, follow=False
                            ):
                                if type_set is None or event.type in type_set:
                                    yield _format_sse(event)
                                watermark = event.seq
                        except Exception as exc:
                            err = SessionEventsError(
                                message=(
                                    f"Failed to stream events for session {session_id!r}: {exc}"
                                ),
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
                            return

                        # Follow live events from the in-process queue.
                        while True:
                            item = await queue.get()
                            if item is None:
                                # Subscriber was dropped because the queue overflowed.
                                # Durable log is the truth; client must re-subscribe.
                                err_ts = _now()
                                audit_log.write(
                                    AuditLogEntry(
                                        level="warn",
                                        event="session.events.stream.subscriber_lagged",
                                        code="subscriber_lagged",
                                        timestamp=err_ts,
                                        detail={
                                            "session_id": session_id,
                                            "since": effective_since,
                                            "capacity": SUBSCRIBER_CHANNEL_SIZE,
                                        },
                                    )
                                )
                                lagged_data = json.dumps(
                                    {
                                        "code": "subscriber_lagged",
                                        "message": (
                                            f"Subscriber dropped: in-process queue overflowed "
                                            f"(capacity={SUBSCRIBER_CHANNEL_SIZE}). "
                                            "Re-subscribe using the last received seq as since."
                                        ),
                                    }
                                )
                                yield f"event: subscriber_lagged\ndata: {lagged_data}\n\n"
                                return

                            event: SessionEvent = item
                            if event.seq <= watermark:
                                # Already delivered during history replay; skip.
                                continue
                            watermark = event.seq
                            if type_set is None or event.type in type_set:
                                yield _format_sse(event)
                    finally:
                        subscriber_bus.unsubscribe(session_id, queue)

                return StreamingResponse(_sse_live(), media_type="text/event-stream")  # type: ignore[return-value]

            # No subscriber bus: historical-only streaming (original behaviour).
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
                raise err from exc

        accept = request.headers.get("accept", "")
        if "application/x-ndjson" in accept:

            def _ndjson() -> Iterator[str]:
                for event in events:
                    yield json.dumps(_event_to_dict(event), separators=(",", ":")) + "\n"

            return StreamingResponse(_ndjson(), media_type="application/x-ndjson")  # type: ignore[return-value]

        return JSONResponse(content=[_event_to_dict(e) for e in events])

    # ------------------------------------------------------------------
    # SDK-facing events endpoint: GET /sessions/{session_id}/events
    #
    # Returns events in the SDK SessionEventList schema format with kind
    # and payload fields.  Only event types that map to a SessionEventKind
    # are included; internal-only events are omitted.  Supports limit and
    # offset pagination.  On failure surfaces an error response to the
    # caller and writes to the audit log.
    # ------------------------------------------------------------------

    @router.get("/sessions/{session_id}/events")
    async def list_session_events_sdk(
        session_id: str,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "session.events.sdk.read",
            attributes={
                "session.id": session_id,
                "events.limit": limit,
                "events.offset": offset,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.events.sdk.read.invocation",
                    code="session_events_sdk_read",
                    timestamp=now,
                ),
            )

            try:
                reader = LocalEventLogReader(storage_root)
                raw_events = reader.read_after(session_id, -1)
                sdk_events = [
                    ev for e in raw_events if (ev := _to_sdk_event(e, session_id)) is not None
                ]
                total = len(sdk_events)
                page = sdk_events[offset : offset + limit]

            except SessionEventsError:
                raise
            except Exception as exc:
                err = SessionEventsError(
                    message=(f"Failed to read SDK events for session {session_id!r}: {exc}"),
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.events.sdk.read.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

        return JSONResponse(content={"events": page, "total": total})

    return router
