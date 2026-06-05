from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 3

_TRACKING_DDL = """
CREATE TABLE IF NOT EXISTS _schema_migrations (
    version    INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    applied_at TEXT    NOT NULL
)
"""

_MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "create_watermarks",
        """
        CREATE TABLE IF NOT EXISTS _watermarks (
            session_id TEXT    PRIMARY KEY,
            last_seq   INTEGER NOT NULL DEFAULT -1,
            updated_at TEXT    NOT NULL
        )
        """,
    ),
    (
        2,
        "create_projection_tables",
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id     TEXT    PRIMARY KEY,
            phase          TEXT    NOT NULL DEFAULT 'created',
            last_event_seq INTEGER NOT NULL DEFAULT -1,
            updated_at     TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS sessions_phase
            ON sessions (phase);

        CREATE TABLE IF NOT EXISTS tool_calls (
            id         TEXT    PRIMARY KEY,
            session_id TEXT    NOT NULL,
            tool_name  TEXT    NOT NULL,
            seq        INTEGER NOT NULL,
            ts         TEXT    NOT NULL,
            status     TEXT    NOT NULL DEFAULT 'pending',
            result     TEXT
        );

        CREATE INDEX IF NOT EXISTS tool_calls_session_id
            ON tool_calls (session_id);

        CREATE TABLE IF NOT EXISTS usage_rollups (
            session_id    TEXT    NOT NULL,
            hour          TEXT    NOT NULL,
            input_tokens  INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_tokens  INTEGER NOT NULL DEFAULT 0,
            dollars       REAL    NOT NULL DEFAULT 0.0,
            PRIMARY KEY (session_id, hour)
        );

        CREATE INDEX IF NOT EXISTS usage_rollups_session_id
            ON usage_rollups (session_id);

        CREATE TABLE IF NOT EXISTS message_index (
            id         TEXT    PRIMARY KEY,
            session_id TEXT    NOT NULL,
            thread_id  TEXT,
            seq        INTEGER NOT NULL,
            ts         TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS message_index_session_id
            ON message_index (session_id)
        """,
    ),
    (
        3,
        "add_cache_columns_to_usage_rollups",
        """
        ALTER TABLE usage_rollups ADD COLUMN cache_creation_tokens INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE usage_rollups ADD COLUMN cache_read_tokens INTEGER NOT NULL DEFAULT 0
        """,
    ),
]


def migrate(conn: sqlite3.Connection, *, applied_at: str) -> int:
    """
    Apply any pending migrations and return the count applied.

    Creates _schema_migrations on first call.  Already-applied migrations are
    skipped (idempotent).  Each migration's statements are run sequentially;
    caller manages the surrounding transaction.  Raises sqlite3.Error on
    statement failure.
    """
    conn.execute(_TRACKING_DDL)
    applied: set[int] = {
        row[0] for row in conn.execute("SELECT version FROM _schema_migrations").fetchall()
    }
    count = 0
    for version, name, sql in _MIGRATIONS:
        if version in applied:
            continue
        for stmt in (s.strip() for s in sql.split(";") if s.strip()):
            conn.execute(stmt)
        conn.execute(
            "INSERT INTO _schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
            (version, name, applied_at),
        )
        count += 1
    return count
