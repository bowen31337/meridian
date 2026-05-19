from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._contract import EventLogWriter
from ._subscriber_bus import SubscriberBus
from ._telemetry import get_tracer, record_event_log_failure, record_fsync_event
from ._types import EventLogFailure, EventType, FsyncPolicy, SessionEvent, StructuredEvent


def _now_dt() -> datetime:
    return datetime.now(UTC)


class LocalEventLogWriter(EventLogWriter):
    """
    Local filesystem event log writer.

    Each session maps to $storage_root/events/<YYYY>/<MM>/<DD>/<session_id>.ndjson.
    Events are appended with O_APPEND | O_CREAT so concurrent writers on the same
    session file are safe at the OS level. The seq counter is tracked in memory;
    callers that share one LocalEventLogWriter instance across concurrent coroutines
    must serialize appends themselves.

    Session IDs must not contain path separators or ".." components.

    After each write, if either of the following conditions is met the writer issues
    an fsync, emits an "event_log.fsync" OTel span, and resets both counters:
      - ``fsync_policy.every_n_events`` events have been written since the last fsync.
      - ``fsync_policy.every_ms`` milliseconds have elapsed since the last fsync.
    On fsync failure an EventLogFailure(EVENT_LOG_FSYNC_FAILED) is raised after the
    span is marked ERROR.
    """

    def __init__(
        self,
        storage_root: str | Path,
        *,
        fsync_policy: FsyncPolicy | None = None,
        subscriber_bus: SubscriberBus | None = None,
    ) -> None:
        self._root = Path(storage_root)
        self._seq: dict[str, int] = {}
        self._fsync_policy: FsyncPolicy = fsync_policy or FsyncPolicy()
        self._events_since_fsync: int = 0
        self._last_fsync_mono: float = time.monotonic()
        self._bus = subscriber_bus

    def _validate_session_id(self, session_id: str, timestamp: str) -> None:
        if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
            raise EventLogFailure(
                code="EVENT_LOG_SESSION_ID_INVALID",
                message=f"Session ID contains illegal characters: {session_id!r}",
                session_id=session_id,
                timestamp=timestamp,
            )

    def _path(self, session_id: str, dt: datetime) -> Path:
        return (
            self._root
            / "events"
            / dt.strftime("%Y")
            / dt.strftime("%m")
            / dt.strftime("%d")
            / f"{session_id}.ndjson"
        )

    def _should_fsync(self) -> bool:
        policy = self._fsync_policy
        if self._events_since_fsync >= policy.every_n_events:
            return True
        elapsed_ms = (time.monotonic() - self._last_fsync_mono) * 1000
        return elapsed_ms >= policy.every_ms

    def _do_fsync(self, fd: int, session_id: str, timestamp: str) -> None:
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "event_log.fsync",
            attributes={"event_log.session_id": session_id},
        ) as span:
            record_fsync_event(
                span,
                StructuredEvent(
                    name="event_log.fsync.invocation",
                    session_id=session_id,
                    timestamp=timestamp,
                    operation="fsync",
                ),
            )
            try:
                os.fsync(fd)
                self._events_since_fsync = 0
                self._last_fsync_mono = time.monotonic()
            except OSError as exc:
                failure = EventLogFailure(
                    code="EVENT_LOG_FSYNC_FAILED",
                    message=str(exc),
                    session_id=session_id,
                    timestamp=timestamp,
                    cause=exc,
                )
                record_event_log_failure(span, failure)
                raise failure from exc

    async def append(
        self,
        session_id: str,
        event_type: EventType,
        data: dict[str, Any],
        *,
        thread_id: str | None = None,
    ) -> int:
        dt = _now_dt()
        timestamp = dt.isoformat(timespec="milliseconds")
        self._validate_session_id(session_id, timestamp)

        seq = self._seq.get(session_id, 0)
        event = SessionEvent(
            seq=seq,
            ts=timestamp,
            type=event_type,
            data=data,
            thread_id=thread_id,
        )

        path = self._path(session_id, dt)
        path.parent.mkdir(parents=True, exist_ok=True)

        record: dict[str, Any] = {
            "seq": event.seq,
            "ts": event.ts,
            "type": event.type,
            "data": event.data,
        }
        if event.thread_id is not None:
            record["thread_id"] = event.thread_id

        line = json.dumps(record, separators=(",", ":")) + "\n"
        encoded = line.encode()

        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, encoded)
            self._seq[session_id] = seq + 1
            self._events_since_fsync += 1
            if self._should_fsync():
                self._do_fsync(fd, session_id, timestamp)
        finally:
            os.close(fd)

        # Fan out to live subscribers after the durable write succeeds.
        # Non-blocking: overflow drops the subscriber rather than blocking the harness.
        if self._bus is not None:
            self._bus.publish(session_id, event)

        return seq
