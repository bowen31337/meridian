from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


class SQLiteProjectionStore:
    """
    Manages the _watermarks table in a SQLite database.

    The watermarks table tracks how far the indexer has read into each
    session's event log (last_seq processed, -1 means nothing yet).

    Callers extend the database with their own projection tables; this class
    only owns _watermarks.  set_watermark must be called inside a transaction
    obtained from transaction() so the watermark and projection update are
    committed atomically.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _watermarks (
                    session_id TEXT PRIMARY KEY,
                    last_seq   INTEGER NOT NULL DEFAULT -1,
                    updated_at TEXT    NOT NULL
                )
                """
            )
            conn.commit()

    def get_watermark(self, session_id: str) -> int:
        """Return last_seq for session_id, or -1 if the session has not been indexed."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT last_seq FROM _watermarks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row[0] if row else -1

    def set_watermark(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        last_seq: int,
        updated_at: str,
    ) -> None:
        """Upsert the watermark for session_id within an active transaction."""
        conn.execute(
            """
            INSERT INTO _watermarks(session_id, last_seq, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id)
            DO UPDATE SET last_seq = excluded.last_seq,
                          updated_at = excluded.updated_at
            """,
            (session_id, last_seq, updated_at),
        )

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield an open connection inside a transaction; commit on exit, rollback on error."""
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
