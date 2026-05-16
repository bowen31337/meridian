from __future__ import annotations

from ._contract import EventHandler
from ._reader import LocalEventLogReader
from ._store import SQLiteProjectionStore


class BackgroundIndexer:
    """
    Reads new events from the event log and updates SQLite projection rows.

    For each session index_session:
      1. Reads the current watermark (last processed seq; -1 means never).
      2. Reads all events from the NDJSON log with seq > watermark.
      3. For each event, opens a transaction, calls the handler, then advances
         the watermark — all atomically so projections never contradict the log.

    Projections therefore trail the log by at most one poll cycle.  Handler
    failures propagate without advancing the watermark, so the same event is
    retried on the next call.
    """

    def __init__(
        self,
        reader: LocalEventLogReader,
        store: SQLiteProjectionStore,
        handler: EventHandler,
    ) -> None:
        self._reader = reader
        self._store = store
        self._handler = handler

    async def index_session(self, session_id: str) -> int:
        """
        Apply any new events for session_id to the projection store.

        Returns the number of events applied.  Raises IndexerFailure on read
        errors; propagates handler exceptions unchanged (no watermark advance).
        """
        watermark = self._store.get_watermark(session_id)
        new_events = self._reader.read_after(session_id, watermark)

        for event in new_events:
            with self._store.transaction() as conn:
                await self._handler.handle(conn, session_id, event)
                self._store.set_watermark(conn, session_id, event.seq, event.ts)

        return len(new_events)
