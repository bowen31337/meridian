"""UsageRollupProjector — aggregates usage.delta events into the usage_rollups table."""

from __future__ import annotations

import sqlite3

from storage_event_log import SessionEvent

from ._store import SQLiteProjectionStore


def _hour(ts: str) -> str:
    """Truncate an ISO-8601 timestamp to the hour bucket (YYYY-MM-DDTHH)."""
    return ts[:13] + ":00:00"


class UsageRollupProjector:
    """EventHandler that folds usage.delta events into the usage_rollups projection.

    Non-usage.delta events are silently skipped.  Call apply_usage_delta on the
    store inside the transaction provided by BackgroundIndexer so the watermark
    and projection row advance atomically.
    """

    def __init__(self, store: SQLiteProjectionStore) -> None:
        self._store = store

    async def handle(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        event: SessionEvent,
    ) -> None:
        if event.type != "usage.delta":
            return
        data = event.data
        self._store.apply_usage_delta(
            conn,
            session_id=session_id,
            hour=_hour(event.ts),
            input_tokens=int(data.get("prompt_tokens", 0)),
            output_tokens=int(data.get("completion_tokens", 0)),
            cache_creation_tokens=int(data.get("cache_creation_tokens", 0)),
            cache_read_tokens=int(data.get("cache_read_tokens", 0)),
        )
