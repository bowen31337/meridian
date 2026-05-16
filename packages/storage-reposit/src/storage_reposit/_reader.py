from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from storage_event_log import SessionEvent

from ._types import IndexerFailure


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalEventLogReader:
    """
    Reads events from date-partitioned NDJSON event log files after a watermark.

    Scans all matching files under $storage_root/events/<YYYY>/<MM>/<DD>/<session_id>.ndjson
    and returns events with seq > after_seq, sorted by seq.  Raises IndexerFailure
    if any line cannot be parsed as JSON.
    """

    def __init__(self, storage_root: str | Path) -> None:
        self._root = Path(storage_root)

    def read_after(self, session_id: str, after_seq: int) -> list[SessionEvent]:
        """Return events for session_id with seq > after_seq, in ascending seq order."""
        events: list[SessionEvent] = []
        for path in sorted(self._root.rglob(f"{session_id}.ndjson")):
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise IndexerFailure(
                        code="INDEXER_READ_FAILED",
                        message=f"Invalid JSON in {path}: {exc}",
                        session_id=session_id,
                        timestamp=_now(),
                        cause=exc,
                    ) from exc
                if record["seq"] > after_seq:
                    events.append(
                        SessionEvent(
                            seq=record["seq"],
                            ts=record["ts"],
                            type=record["type"],
                            data=record["data"],
                            thread_id=record.get("thread_id"),
                        )
                    )
        return sorted(events, key=lambda e: e.seq)
