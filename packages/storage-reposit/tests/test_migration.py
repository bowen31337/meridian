"""
Migration system tests.

Covers:
  _migrations module:
    - SCHEMA_VERSION is 2.
    - migrate() creates _schema_migrations table.
    - migrate() applies migration 1: _watermarks table exists.
    - migrate() applies migration 2: sessions table exists.
    - migrate() applies migration 2: tool_calls table exists.
    - migrate() applies migration 2: usage_rollups table exists.
    - migrate() applies migration 2: message_index table exists.
    - migrate() creates sessions_phase index.
    - migrate() creates tool_calls_session_id index.
    - migrate() creates usage_rollups_session_id index.
    - migrate() creates message_index_session_id index.
    - migrate() returns 2 on a fresh database.
    - migrate() returns 0 when called again (idempotent).
    - _schema_migrations records both applied versions.

  SQLiteProjectionStore.migrate():
    - Returns 2 on a fresh database.
    - Returns 0 on second call (idempotent).
    - All projection tables exist after migrate().
    - sessions table has expected columns.
    - tool_calls table has expected columns.
    - usage_rollups table has expected columns.
    - message_index table has expected columns.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from storage_reposit import SCHEMA_VERSION, SQLiteProjectionStore
from storage_reposit import _migrations as _mig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def open_fresh(tmp_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(tmp_path / "m.db")


def table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def index_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }


def column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


# ===========================================================================
# _migrations module
# ===========================================================================


class TestSchemaVersion:
    def test_schema_version_is_2(self) -> None:
        assert SCHEMA_VERSION == 2


class TestMigrateFunction:
    def test_creates_schema_migrations_table(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
            _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        with sqlite3.connect(tmp_path / "m.db") as conn:
            assert "_schema_migrations" in table_names(conn)

    def test_migration_1_creates_watermarks(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
            _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        with sqlite3.connect(tmp_path / "m.db") as conn:
            assert "_watermarks" in table_names(conn)

    def test_migration_2_creates_sessions(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
            _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        with sqlite3.connect(tmp_path / "m.db") as conn:
            assert "sessions" in table_names(conn)

    def test_migration_2_creates_tool_calls(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
            _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        with sqlite3.connect(tmp_path / "m.db") as conn:
            assert "tool_calls" in table_names(conn)

    def test_migration_2_creates_usage_rollups(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
            _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        with sqlite3.connect(tmp_path / "m.db") as conn:
            assert "usage_rollups" in table_names(conn)

    def test_migration_2_creates_message_index(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
            _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        with sqlite3.connect(tmp_path / "m.db") as conn:
            assert "message_index" in table_names(conn)

    def test_sessions_phase_index_created(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
            _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        with sqlite3.connect(tmp_path / "m.db") as conn:
            assert "sessions_phase" in index_names(conn)

    def test_tool_calls_session_id_index_created(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
            _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        with sqlite3.connect(tmp_path / "m.db") as conn:
            assert "tool_calls_session_id" in index_names(conn)

    def test_usage_rollups_session_id_index_created(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
            _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        with sqlite3.connect(tmp_path / "m.db") as conn:
            assert "usage_rollups_session_id" in index_names(conn)

    def test_message_index_session_id_index_created(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
            _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        with sqlite3.connect(tmp_path / "m.db") as conn:
            assert "message_index_session_id" in index_names(conn)

    def test_returns_2_on_fresh_database(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
            count = _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        assert count == 2

    def test_returns_0_on_second_call(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
            _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        with sqlite3.connect(tmp_path / "m.db") as conn:
            count = _mig.migrate(conn, applied_at="2024-01-01T00:00:01+00:00")
        assert count == 0

    def test_schema_migrations_records_both_versions(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
            _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        with sqlite3.connect(tmp_path / "m.db") as conn:
            versions = {
                row[0]
                for row in conn.execute("SELECT version FROM _schema_migrations").fetchall()
            }
        assert versions == {1, 2}

    def test_idempotent_on_existing_watermarks(self, tmp_path: Path) -> None:
        with open_fresh(tmp_path) as conn:
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
        with sqlite3.connect(tmp_path / "m.db") as conn:
            count = _mig.migrate(conn, applied_at="2024-01-01T00:00:00+00:00")
        assert count == 2
        with sqlite3.connect(tmp_path / "m.db") as conn:
            assert "_watermarks" in table_names(conn)


# ===========================================================================
# SQLiteProjectionStore.migrate()
# ===========================================================================


class TestSQLiteProjectionStoreMigrate:
    def test_returns_2_on_fresh_database(self, tmp_path: Path) -> None:
        store = SQLiteProjectionStore(tmp_path / "p.db")
        assert store.migrate() == 2

    def test_returns_0_on_second_call(self, tmp_path: Path) -> None:
        store = SQLiteProjectionStore(tmp_path / "p.db")
        store.migrate()
        assert store.migrate() == 0

    def test_all_projection_tables_exist(self, tmp_path: Path) -> None:
        store = SQLiteProjectionStore(tmp_path / "p.db")
        store.migrate()
        with sqlite3.connect(tmp_path / "p.db") as conn:
            names = table_names(conn)
        assert {"_watermarks", "sessions", "tool_calls", "usage_rollups", "message_index"} <= names

    def test_sessions_columns(self, tmp_path: Path) -> None:
        store = SQLiteProjectionStore(tmp_path / "p.db")
        store.migrate()
        with sqlite3.connect(tmp_path / "p.db") as conn:
            cols = column_names(conn, "sessions")
        assert {"session_id", "phase", "last_event_seq", "updated_at"} <= cols

    def test_tool_calls_columns(self, tmp_path: Path) -> None:
        store = SQLiteProjectionStore(tmp_path / "p.db")
        store.migrate()
        with sqlite3.connect(tmp_path / "p.db") as conn:
            cols = column_names(conn, "tool_calls")
        assert {"id", "session_id", "tool_name", "seq", "ts", "status", "result"} <= cols

    def test_usage_rollups_columns(self, tmp_path: Path) -> None:
        store = SQLiteProjectionStore(tmp_path / "p.db")
        store.migrate()
        with sqlite3.connect(tmp_path / "p.db") as conn:
            cols = column_names(conn, "usage_rollups")
        assert {
            "session_id",
            "hour",
            "input_tokens",
            "output_tokens",
            "cache_tokens",
            "dollars",
        } <= cols

    def test_message_index_columns(self, tmp_path: Path) -> None:
        store = SQLiteProjectionStore(tmp_path / "p.db")
        store.migrate()
        with sqlite3.connect(tmp_path / "p.db") as conn:
            cols = column_names(conn, "message_index")
        assert {"id", "session_id", "thread_id", "seq", "ts"} <= cols

    def test_sessions_defaults(self, tmp_path: Path) -> None:
        store = SQLiteProjectionStore(tmp_path / "p.db")
        store.migrate()
        ts = "2024-01-01T00:00:00+00:00"
        with sqlite3.connect(tmp_path / "p.db") as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, updated_at) VALUES (?, ?)",
                ("s1", ts),
            )
            row = conn.execute(
                "SELECT phase, last_event_seq FROM sessions WHERE session_id = 's1'"
            ).fetchone()
        assert row == ("created", -1)

    def test_usage_rollups_composite_primary_key(self, tmp_path: Path) -> None:
        store = SQLiteProjectionStore(tmp_path / "p.db")
        store.migrate()
        ts = "2024-01-01T00:00:00+00:00"
        with sqlite3.connect(tmp_path / "p.db") as conn:
            conn.execute(
                "INSERT INTO usage_rollups (session_id, hour) VALUES (?, ?)",
                ("s1", ts),
            )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO usage_rollups (session_id, hour) VALUES (?, ?)",
                    ("s1", ts),
                )
