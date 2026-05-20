from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from . import _migrations as _mig


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SQLiteProjectionStore:
    """
    Manages the projection-store SQLite database.

    Call migrate() once on startup before any reads or writes.  set_watermark
    must be called inside a transaction obtained from transaction() so the
    watermark and projection update are committed atomically.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    def migrate(self) -> int:
        """
        Apply any pending schema migrations and return the count applied.

        Idempotent: safe to call on every startup.  Raises sqlite3.Error on
        statement failure.
        """
        with sqlite3.connect(self._db_path) as conn:
            count = _mig.migrate(conn, applied_at=_now())
            conn.commit()
        return count

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

    def apply_usage_delta(
        self,
        conn: sqlite3.Connection,
        *,
        session_id: str,
        hour: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> None:
        """Upsert a usage.delta increment into usage_rollups within an active transaction."""
        conn.execute(
            """
            INSERT INTO usage_rollups(
                session_id, hour,
                input_tokens, output_tokens,
                cache_tokens, cache_creation_tokens, cache_read_tokens
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, hour)
            DO UPDATE SET
                input_tokens          = input_tokens + excluded.input_tokens,
                output_tokens         = output_tokens + excluded.output_tokens,
                cache_tokens          = cache_tokens
                                        + excluded.cache_creation_tokens
                                        + excluded.cache_read_tokens,
                cache_creation_tokens = cache_creation_tokens + excluded.cache_creation_tokens,
                cache_read_tokens     = cache_read_tokens + excluded.cache_read_tokens
            """,
            (
                session_id,
                hour,
                input_tokens,
                output_tokens,
                cache_creation_tokens + cache_read_tokens,
                cache_creation_tokens,
                cache_read_tokens,
            ),
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
