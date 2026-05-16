from __future__ import annotations

import sqlite3
from typing import Protocol

from storage_event_log import SessionEvent


class EventHandler(Protocol):
    """
    Applied to every new event during indexing.

    The handler receives an active SQLite connection inside a transaction.
    It must not commit or rollback; the indexer manages the transaction so
    the watermark and projection updates are atomic.
    """

    async def handle(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        event: SessionEvent,
    ) -> None: ...
