"""
Shared DDL migrations in lowest-common SQL dialect (SQLite >= 3.24 / Postgres 9.5+).

All tables use TEXT primary keys (UUID strings), TEXT for timestamps (ISO 8601),
and TEXT for JSON columns.  Every statement is idempotent (IF NOT EXISTS).

Run these in order before any repository operations.
"""
from __future__ import annotations

MIGRATIONS: list[str] = [
    # ------------------------------------------------------------------
    # agents
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS agents (
        id           TEXT PRIMARY KEY,
        kind         TEXT NOT NULL,
        name         TEXT NOT NULL,
        config       TEXT NOT NULL DEFAULT '{}',
        capabilities TEXT NOT NULL DEFAULT '[]',
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    )
    """,

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id         TEXT PRIMARY KEY,
        agent_id   TEXT NOT NULL,
        status     TEXT NOT NULL DEFAULT 'active',
        metadata   TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS sessions_agent_id ON sessions (agent_id)",

    # ------------------------------------------------------------------
    # threads
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS threads (
        id         TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        title      TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS threads_session_id ON threads (session_id)",

    # ------------------------------------------------------------------
    # messages
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS messages (
        id         TEXT PRIMARY KEY,
        thread_id  TEXT NOT NULL,
        session_id TEXT NOT NULL,
        role       TEXT NOT NULL,
        content    TEXT NOT NULL DEFAULT '[]',
        sequence   INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS messages_thread_id  ON messages (thread_id)",
    "CREATE INDEX IF NOT EXISTS messages_session_id ON messages (session_id)",

    # ------------------------------------------------------------------
    # tool_calls
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS tool_calls (
        id         TEXT PRIMARY KEY,
        message_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        tool_name  TEXT NOT NULL,
        input      TEXT NOT NULL DEFAULT '{}',
        output     TEXT,
        status     TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS tool_calls_message_id  ON tool_calls (message_id)",
    "CREATE INDEX IF NOT EXISTS tool_calls_session_id  ON tool_calls (session_id)",

    # ------------------------------------------------------------------
    # skills
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS skills (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL UNIQUE,
        description  TEXT NOT NULL DEFAULT '',
        capabilities TEXT NOT NULL DEFAULT '[]',
        config       TEXT NOT NULL DEFAULT '{}',
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    )
    """,

    # ------------------------------------------------------------------
    # environments
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS environments (
        id         TEXT PRIMARY KEY,
        kind       TEXT NOT NULL,
        status     TEXT NOT NULL DEFAULT 'provisioned',
        config     TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,

    # ------------------------------------------------------------------
    # memory_entries
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS memory_entries (
        id         TEXT PRIMARY KEY,
        scope      TEXT NOT NULL,
        key        TEXT NOT NULL,
        value      TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (scope, key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS memory_entries_scope ON memory_entries (scope)",

    # ------------------------------------------------------------------
    # vault_entries  (metadata only — no secret values)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS vault_entries (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL UNIQUE,
        description TEXT,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )
    """,

    # ------------------------------------------------------------------
    # user_profiles
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS user_profiles (
        id           TEXT PRIMARY KEY,
        username     TEXT NOT NULL UNIQUE,
        display_name TEXT,
        email        TEXT,
        metadata     TEXT,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    )
    """,

    # ------------------------------------------------------------------
    # channels
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS channels (
        id         TEXT PRIMARY KEY,
        kind       TEXT NOT NULL,
        name       TEXT NOT NULL,
        config     TEXT NOT NULL DEFAULT '{}',
        status     TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,

    # ------------------------------------------------------------------
    # webhooks
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS webhooks (
        id         TEXT PRIMARY KEY,
        url        TEXT NOT NULL,
        events     TEXT NOT NULL DEFAULT '[]',
        secret_ref TEXT,
        status     TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
]
