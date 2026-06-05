"""
UsageRollupProjector conformance suite.

Covers:

  _hour() helper:
    - Truncates full ISO-8601 timestamp to the hour bucket.
    - Preserves the YYYY-MM-DDTHH prefix exactly.
    - Works across midnight/hour boundaries.

  UsageRollupProjector.handle():
    - Non-usage.delta events are silently ignored.
    - usage.delta events write input_tokens to usage_rollups.
    - usage.delta events write output_tokens to usage_rollups.
    - usage.delta events write cache_creation_tokens to usage_rollups.
    - usage.delta events write cache_read_tokens to usage_rollups.
    - cache_tokens (legacy sum column) equals cache_creation + cache_read.
    - Second call in the same hour accumulates input_tokens.
    - Second call in the same hour accumulates output_tokens.
    - Second call in the same hour accumulates cache_creation_tokens.
    - Second call in the same hour accumulates cache_read_tokens.
    - cache_tokens accumulates across multiple calls.
    - Calls in different hours produce separate rows.
    - Different sessions produce separate rows.
    - Missing token fields default to zero.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

import pytest
from storage_event_log import SessionEvent
from storage_reposit import SQLiteProjectionStore, UsageRollupProjector
from storage_reposit._usage_rollup import _hour

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def open_migrated(tmp_path: Path) -> tuple[SQLiteProjectionStore, sqlite3.Connection]:
    db_path = tmp_path / "test.db"
    store = SQLiteProjectionStore(db_path)
    store.migrate()
    conn = sqlite3.connect(db_path)
    return store, conn


def usage_event(
    seq: int,
    ts: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> SessionEvent:
    return SessionEvent(
        seq=seq,
        ts=ts,
        type="usage.delta",
        data={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_read_tokens": cache_read_tokens,
        },
    )


def other_event(seq: int, event_type: str = "message.added") -> SessionEvent:
    return SessionEvent(seq=seq, ts="2024-01-01T10:00:00.000+00:00", type=event_type, data={})


def row(conn: sqlite3.Connection, session_id: str, hour: str) -> dict[str, Any] | None:
    cur = conn.execute(
        """
        SELECT input_tokens, output_tokens, cache_tokens,
               cache_creation_tokens, cache_read_tokens
        FROM usage_rollups
        WHERE session_id = ? AND hour = ?
        """,
        (session_id, hour),
    )
    r = cur.fetchone()
    if r is None:
        return None
    keys = (
        "input_tokens",
        "output_tokens",
        "cache_tokens",
        "cache_creation_tokens",
        "cache_read_tokens",
    )
    return dict(zip(keys, r, strict=False))


def all_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = conn.execute(
        "SELECT session_id, hour, input_tokens, output_tokens, cache_tokens, "
        "cache_creation_tokens, cache_read_tokens FROM usage_rollups ORDER BY session_id, hour"
    )
    keys = (
        "session_id",
        "hour",
        "input_tokens",
        "output_tokens",
        "cache_tokens",
        "cache_creation_tokens",
        "cache_read_tokens",
    )
    return [dict(zip(keys, r, strict=False)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# _hour() helper
# ---------------------------------------------------------------------------


class TestHourHelper:
    def test_truncates_to_hour_bucket(self) -> None:
        assert _hour("2024-03-15T14:37:52.123+00:00") == "2024-03-15T14:00:00"

    def test_preserves_year_month_day_hour(self) -> None:
        result = _hour("2024-03-15T14:37:52.123+00:00")
        assert result.startswith("2024-03-15T14")

    def test_midnight_hour(self) -> None:
        assert _hour("2024-01-01T00:59:59.999+00:00") == "2024-01-01T00:00:00"

    def test_last_hour_of_day(self) -> None:
        assert _hour("2024-01-01T23:00:00.000+00:00") == "2024-01-01T23:00:00"


# ---------------------------------------------------------------------------
# UsageRollupProjector.handle()
# ---------------------------------------------------------------------------


class TestUsageRollupProjectorIgnoresNonUsageEvents:
    @pytest.mark.asyncio
    async def test_message_added_not_written(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        await projector.handle(conn, "s1", other_event(0, "message.added"))
        assert all_rows(conn) == []

    @pytest.mark.asyncio
    async def test_session_phase_change_not_written(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        await projector.handle(conn, "s1", other_event(0, "session.phase_change"))
        assert all_rows(conn) == []

    @pytest.mark.asyncio
    async def test_model_call_completed_not_written(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        await projector.handle(conn, "s1", other_event(0, "model_call.completed"))
        assert all_rows(conn) == []


class TestUsageRollupProjectorSingleEvent:
    TS = "2024-01-15T10:30:00.000+00:00"
    HOUR = "2024-01-15T10:00:00"

    @pytest.mark.asyncio
    async def test_writes_input_tokens(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        await projector.handle(conn, "s1", usage_event(0, self.TS, prompt_tokens=42))
        assert row(conn, "s1", self.HOUR)["input_tokens"] == 42

    @pytest.mark.asyncio
    async def test_writes_output_tokens(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        await projector.handle(conn, "s1", usage_event(0, self.TS, completion_tokens=7))
        assert row(conn, "s1", self.HOUR)["output_tokens"] == 7

    @pytest.mark.asyncio
    async def test_writes_cache_creation_tokens(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        await projector.handle(conn, "s1", usage_event(0, self.TS, cache_creation_tokens=150))
        assert row(conn, "s1", self.HOUR)["cache_creation_tokens"] == 150

    @pytest.mark.asyncio
    async def test_writes_cache_read_tokens(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        await projector.handle(conn, "s1", usage_event(0, self.TS, cache_read_tokens=200))
        assert row(conn, "s1", self.HOUR)["cache_read_tokens"] == 200

    @pytest.mark.asyncio
    async def test_cache_tokens_sum_of_creation_and_read(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        await projector.handle(
            conn, "s1", usage_event(0, self.TS, cache_creation_tokens=100, cache_read_tokens=50)
        )
        assert row(conn, "s1", self.HOUR)["cache_tokens"] == 150

    @pytest.mark.asyncio
    async def test_missing_fields_default_to_zero(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        event = SessionEvent(
            seq=0,
            ts=self.TS,
            type="usage.delta",
            data={},
        )
        await projector.handle(conn, "s1", event)
        r = row(conn, "s1", self.HOUR)
        assert r is not None
        assert r["input_tokens"] == 0
        assert r["cache_creation_tokens"] == 0
        assert r["cache_read_tokens"] == 0


class TestUsageRollupProjectorAccumulation:
    TS1 = "2024-01-15T10:10:00.000+00:00"
    TS2 = "2024-01-15T10:55:00.000+00:00"
    HOUR = "2024-01-15T10:00:00"

    @pytest.mark.asyncio
    async def test_accumulates_input_tokens(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        await projector.handle(conn, "s1", usage_event(0, self.TS1, prompt_tokens=10))
        await projector.handle(conn, "s1", usage_event(1, self.TS2, prompt_tokens=20))
        assert row(conn, "s1", self.HOUR)["input_tokens"] == 30

    @pytest.mark.asyncio
    async def test_accumulates_output_tokens(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        await projector.handle(conn, "s1", usage_event(0, self.TS1, completion_tokens=5))
        await projector.handle(conn, "s1", usage_event(1, self.TS2, completion_tokens=3))
        assert row(conn, "s1", self.HOUR)["output_tokens"] == 8

    @pytest.mark.asyncio
    async def test_accumulates_cache_creation_tokens(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        await projector.handle(conn, "s1", usage_event(0, self.TS1, cache_creation_tokens=100))
        await projector.handle(conn, "s1", usage_event(1, self.TS2, cache_creation_tokens=50))
        assert row(conn, "s1", self.HOUR)["cache_creation_tokens"] == 150

    @pytest.mark.asyncio
    async def test_accumulates_cache_read_tokens(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        await projector.handle(conn, "s1", usage_event(0, self.TS1, cache_read_tokens=200))
        await projector.handle(conn, "s1", usage_event(1, self.TS2, cache_read_tokens=300))
        assert row(conn, "s1", self.HOUR)["cache_read_tokens"] == 500

    @pytest.mark.asyncio
    async def test_cache_tokens_accumulates_across_calls(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        await projector.handle(
            conn, "s1", usage_event(0, self.TS1, cache_creation_tokens=100, cache_read_tokens=200)
        )
        await projector.handle(
            conn, "s1", usage_event(1, self.TS2, cache_creation_tokens=50, cache_read_tokens=25)
        )
        assert row(conn, "s1", self.HOUR)["cache_tokens"] == 375


class TestUsageRollupProjectorIsolation:
    @pytest.mark.asyncio
    async def test_different_hours_produce_separate_rows(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        ts_h10 = "2024-01-15T10:30:00.000+00:00"
        ts_h11 = "2024-01-15T11:05:00.000+00:00"
        await projector.handle(conn, "s1", usage_event(0, ts_h10, prompt_tokens=10))
        await projector.handle(conn, "s1", usage_event(1, ts_h11, prompt_tokens=20))
        rows = all_rows(conn)
        assert len(rows) == 2
        hours = {r["hour"] for r in rows}
        assert hours == {"2024-01-15T10:00:00", "2024-01-15T11:00:00"}

    @pytest.mark.asyncio
    async def test_different_sessions_produce_separate_rows(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        ts = "2024-01-15T10:30:00.000+00:00"
        await projector.handle(conn, "s1", usage_event(0, ts, prompt_tokens=10))
        await projector.handle(conn, "s2", usage_event(0, ts, prompt_tokens=20))
        rows = all_rows(conn)
        assert len(rows) == 2
        by_session = {r["session_id"]: r for r in rows}
        assert by_session["s1"]["input_tokens"] == 10
        assert by_session["s2"]["input_tokens"] == 20

    @pytest.mark.asyncio
    async def test_hour_events_not_bleed_across_sessions(self, tmp_path: Path) -> None:
        store, conn = open_migrated(tmp_path)
        projector = UsageRollupProjector(store)
        ts = "2024-01-15T10:00:00.000+00:00"
        hour = "2024-01-15T10:00:00"
        await projector.handle(conn, "s1", usage_event(0, ts, cache_creation_tokens=100))
        await projector.handle(conn, "s2", usage_event(0, ts, cache_read_tokens=200))
        assert row(conn, "s1", hour)["cache_creation_tokens"] == 100
        assert row(conn, "s1", hour)["cache_read_tokens"] == 0
        assert row(conn, "s2", hour)["cache_creation_tokens"] == 0
        assert row(conn, "s2", hour)["cache_read_tokens"] == 200
