from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._contract import EventLogWriter
from ._types import EventLogFailure, SessionEvent


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _now() -> str:
    return _now_dt().isoformat()


class LocalEventLogWriter(EventLogWriter):
    """
    Local filesystem event log writer.

    Each session maps to $storage_root/events/<YYYY>/<MM>/<DD>/<session_id>.ndjson.
    Events are appended with O_APPEND | O_CREAT so concurrent writers on the same
    session file are safe at the OS level. The seq counter is tracked in memory;
    callers that share one LocalEventLogWriter instance across concurrent coroutines
    must serialize appends themselves.

    Session IDs must not contain path separators or ".." components.
    """

    def __init__(self, storage_root: str | Path) -> None:
        self._root = Path(storage_root)
        self._seq: dict[str, int] = {}

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

    async def append(
        self,
        session_id: str,
        event_type: str,
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
        finally:
            os.close(fd)

        self._seq[session_id] = seq + 1
        return seq
