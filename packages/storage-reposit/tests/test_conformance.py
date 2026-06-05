"""
storage-reposit conformance suite.

Covers:

  LocalEventLogReader:
    - Returns [] when no NDJSON files exist.
    - Returns all events when watermark is -1.
    - Filters out events with seq <= watermark.
    - Returns events sorted by seq across multiple files.
    - Handles multiple date-partitioned files for the same session.
    - Raises IndexerFailure(INDEXER_READ_FAILED) on malformed JSON.
    - Omits thread_id from SessionEvent when absent from record.
    - Includes thread_id when present.

  LocalEventLogReader.read_events (cat mode, follow=False):
    - Yields [] when no NDJSON files exist.
    - Yields all events with since=-1.
    - Filters events at or below since.
    - Yields events in ascending seq order.
    - Raises IndexerFailure(INDEXER_READ_FAILED) on malformed JSON.
    - Stops after all existing events (does not poll).

  LocalEventLogReader.read_events (tail mode, follow=True):
    - Yields existing events then polls for new ones.
    - New events appended to the file are yielded after the next poll.

  SQLiteProjectionStore:
    - get_watermark returns -1 for an unknown session.
    - set_watermark upserts correctly (insert then update).
    - transaction() commits on success.
    - transaction() rolls back and re-raises on exception.

  BackgroundIndexer:
    - index_session returns 0 when no NDJSON files exist.
    - index_session returns 0 when all events are at or below watermark.
    - index_session returns N when N new events are available.
    - Handler is called once per new event.
    - Events are delivered to the handler in ascending seq order.
    - Watermark advances to last processed seq after success.
    - Second index_session call returns 0 (idempotent after full index).
    - Two independent sessions do not share watermarks.
    - Handler exception propagates; watermark is not advanced.
    - IndexerFailure from reader propagates unchanged.

  IndexerRuntime:
    - index_session returns count on success.
    - Span name is "indexer.index_session".
    - Span carries indexer.session_id attribute.
    - "indexer.invocation" event is attached to the span.
    - Invocation event has operation="index_session".
    - No audit entries written on success.
    - Span is ended on success.
    - IndexerFailure from indexer is re-raised and audited.
    - Audit entry level is "error" and event is "indexer.index_session.failed".
    - Span is marked ERROR on IndexerFailure.
    - Span is ended on IndexerFailure.
    - Unexpected exception is wrapped as INDEXER_INDEX_SESSION_FAILED.
    - Cause is preserved in wrapped failure.
    - Audit entry written for unexpected exception.
    - "indexer.error" event added to span on unexpected exception.
    - Exception recorded on span via record_exception.
    - on_error callback invoked for every failure.
    - Span is ended on unexpected exception.

  ReaderRuntime:
    - read_events yields events on success.
    - Span name is "reader.read_events".
    - Span carries reader.session_id attribute.
    - "reader.invocation" event is attached to the span.
    - Invocation event has operation="read_events".
    - No audit entries written on success.
    - Span is ended on success.
    - IndexerFailure from reader is re-raised and audited.
    - Audit entry level is "error" and event is "reader.read_events.failed".
    - Span is marked ERROR on IndexerFailure.
    - Span is ended on IndexerFailure.
    - Unexpected exception is wrapped as READER_READ_EVENTS_FAILED.
    - Cause is preserved in wrapped failure.
    - Audit entry written for unexpected exception.
    - "reader.error" event added to span on unexpected exception.
    - Exception recorded on span via record_exception.
    - on_error callback invoked for every failure.
    - Span is ended on unexpected exception.

  MigrationRuntime:
    - migrate returns count on success.
    - Span name is "migration.migrate".
    - "migration.invocation" event is attached to the span.
    - Invocation event has operation="migrate".
    - No audit entries written on success.
    - Span is ended on success.
    - Unexpected exception is wrapped as MIGRATION_FAILED.
    - Cause is preserved in wrapped failure.
    - Audit entry written for unexpected exception.
    - Audit entry level is "error" and event is "migration.migrate.failed".
    - "migration.error" event added to span on failure.
    - Exception recorded on span via record_exception.
    - on_error callback invoked for every failure.
    - Span is marked ERROR on failure.
    - Span is ended on failure.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
from typing import Any

from opentelemetry.trace import StatusCode
import pytest
from storage_event_log import SessionEvent
from storage_reposit import (
    AuditLogEntry,
    BackgroundIndexer,
    IndexerFailure,
    IndexerOptions,
    IndexerRuntime,
    LocalEventLogReader,
    MigrationOptions,
    MigrationRuntime,
    ReaderOptions,
    ReaderRuntime,
    SQLiteProjectionStore,
)

from .conftest import (
    CapturingAuditLog,
    CapturingEventHandler,
    FailingEventHandler,
    MockSpan,
    StubReader,
    StubStore,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_ndjson(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def make_record(seq: int, event_type: str = "message.added", **extra: Any) -> dict[str, Any]:
    r: dict[str, Any] = {
        "seq": seq,
        "ts": f"2024-01-01T00:00:0{seq}.000+00:00",
        "type": event_type,
        "data": {},
    }
    r.update(extra)
    return r


def make_store(tmp_path: Path, db_name: str = "proj.db") -> SQLiteProjectionStore:
    store = SQLiteProjectionStore(tmp_path / db_name)
    store.migrate()
    return store


def make_indexer(
    tmp_path: Path,
    handler: Any = None,
    db_name: str = "proj.db",
) -> tuple[BackgroundIndexer, SQLiteProjectionStore]:
    reader = LocalEventLogReader(tmp_path)
    store = make_store(tmp_path, db_name)
    h = handler or CapturingEventHandler()
    return BackgroundIndexer(reader, store, h), store


def ndjson_path(root: Path, session_id: str, date: str = "2024/01/01") -> Path:
    return root / "events" / date / f"{session_id}.ndjson"


def make_options(
    audit: CapturingAuditLog,
    errors: list[IndexerFailure] | None = None,
) -> IndexerOptions:
    return IndexerOptions(
        audit_log=audit,
        on_error=(lambda e: errors.append(e)) if errors is not None else None,
    )


def make_reader_options(
    audit: CapturingAuditLog,
    errors: list[IndexerFailure] | None = None,
) -> ReaderOptions:
    return ReaderOptions(
        audit_log=audit,
        on_error=(lambda e: errors.append(e)) if errors is not None else None,
    )


def make_migration_options(
    audit: CapturingAuditLog,
    errors: list[IndexerFailure] | None = None,
) -> MigrationOptions:
    return MigrationOptions(
        audit_log=audit,
        on_error=(lambda e: errors.append(e)) if errors is not None else None,
    )


def make_session_event(seq: int) -> SessionEvent:
    return SessionEvent(
        seq=seq, ts=f"2024-01-01T00:00:0{seq}.000+00:00", type="message.added", data={}
    )


class StubIndexer:
    """BackgroundIndexer substitute with configurable failure injection."""

    def __init__(self, *, raises: Exception | None = None, returns: int = 0) -> None:
        self._raises = raises
        self._returns = returns

    async def index_session(self, session_id: str) -> int:
        if self._raises:
            raise self._raises
        return self._returns


def make_runtime(stub: StubIndexer | None = None) -> IndexerRuntime:
    return IndexerRuntime(stub or StubIndexer())  # type: ignore[arg-type]


def make_migration_runtime(stub: StubStore | None = None) -> MigrationRuntime:
    return MigrationRuntime(stub or StubStore())  # type: ignore[arg-type]


# ===========================================================================
# LocalEventLogReader
# ===========================================================================


class TestLocalEventLogReader:
    def test_empty_when_no_files(self, tmp_path: Path) -> None:
        reader = LocalEventLogReader(tmp_path)
        assert reader.read_after("sess1", -1) == []

    def test_returns_all_events_watermark_minus_one(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1)])
        reader = LocalEventLogReader(tmp_path)
        events = reader.read_after("s1", -1)
        assert [e.seq for e in events] == [0, 1]

    def test_filters_events_at_or_below_watermark(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1), make_record(2)])
        reader = LocalEventLogReader(tmp_path)
        events = reader.read_after("s1", 1)
        assert [e.seq for e in events] == [2]

    def test_returns_empty_when_all_at_or_below_watermark(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1)])
        reader = LocalEventLogReader(tmp_path)
        assert reader.read_after("s1", 5) == []

    def test_sorted_by_seq(self, tmp_path: Path) -> None:
        p = ndjson_path(tmp_path, "s1")
        write_ndjson(p, [make_record(2), make_record(0), make_record(1)])
        reader = LocalEventLogReader(tmp_path)
        events = reader.read_after("s1", -1)
        assert [e.seq for e in events] == [0, 1, 2]

    def test_multiple_date_files_merged(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1", "2024/01/01"), [make_record(0), make_record(1)])
        write_ndjson(ndjson_path(tmp_path, "s1", "2024/01/02"), [make_record(2), make_record(3)])
        reader = LocalEventLogReader(tmp_path)
        events = reader.read_after("s1", -1)
        assert [e.seq for e in events] == [0, 1, 2, 3]

    def test_multiple_date_files_filtered_by_watermark(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1", "2024/01/01"), [make_record(0), make_record(1)])
        write_ndjson(ndjson_path(tmp_path, "s1", "2024/01/02"), [make_record(2), make_record(3)])
        reader = LocalEventLogReader(tmp_path)
        events = reader.read_after("s1", 1)
        assert [e.seq for e in events] == [2, 3]

    def test_different_sessions_isolated(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0)])
        write_ndjson(ndjson_path(tmp_path, "s2"), [make_record(0), make_record(1)])
        reader = LocalEventLogReader(tmp_path)
        assert len(reader.read_after("s1", -1)) == 1
        assert len(reader.read_after("s2", -1)) == 2

    def test_raises_on_bad_json(self, tmp_path: Path) -> None:
        p = ndjson_path(tmp_path, "s1")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("not-json\n", encoding="utf-8")
        reader = LocalEventLogReader(tmp_path)
        with pytest.raises(IndexerFailure) as exc_info:
            reader.read_after("s1", -1)
        assert exc_info.value.code == "INDEXER_READ_FAILED"
        assert exc_info.value.session_id == "s1"

    def test_raises_on_bad_json_cause_preserved(self, tmp_path: Path) -> None:
        p = ndjson_path(tmp_path, "s1")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{bad\n", encoding="utf-8")
        reader = LocalEventLogReader(tmp_path)
        with pytest.raises(IndexerFailure) as exc_info:
            reader.read_after("s1", -1)
        assert isinstance(exc_info.value.cause, json.JSONDecodeError)

    def test_thread_id_omitted_when_absent(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0)])
        reader = LocalEventLogReader(tmp_path)
        events = reader.read_after("s1", -1)
        assert events[0].thread_id is None

    def test_thread_id_included_when_present(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0, thread_id="t-42")])
        reader = LocalEventLogReader(tmp_path)
        events = reader.read_after("s1", -1)
        assert events[0].thread_id == "t-42"

    def test_event_fields_preserved(self, tmp_path: Path) -> None:
        rec = make_record(0, event_type="hook.invoked", data={"key": "val"})
        rec["data"] = {"key": "val"}
        write_ndjson(ndjson_path(tmp_path, "s1"), [rec])
        reader = LocalEventLogReader(tmp_path)
        events = reader.read_after("s1", -1)
        assert events[0].type == "hook.invoked"
        assert events[0].data == {"key": "val"}

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        p = ndjson_path(tmp_path, "s1")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(make_record(0)) + "\n\n" + json.dumps(make_record(1)) + "\n",
            encoding="utf-8",
        )
        reader = LocalEventLogReader(tmp_path)
        assert len(reader.read_after("s1", -1)) == 2


# ===========================================================================
# SQLiteProjectionStore
# ===========================================================================


class TestSQLiteProjectionStore:
    def test_watermark_minus_one_for_unknown_session(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        assert store.get_watermark("unknown") == -1

    def test_set_watermark_inserts(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with store.transaction() as conn:
            store.set_watermark(conn, "s1", 3, "2024-01-01T00:00:00+00:00")
        assert store.get_watermark("s1") == 3

    def test_set_watermark_updates_existing(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with store.transaction() as conn:
            store.set_watermark(conn, "s1", 3, "2024-01-01T00:00:00+00:00")
        with store.transaction() as conn:
            store.set_watermark(conn, "s1", 7, "2024-01-01T00:00:01+00:00")
        assert store.get_watermark("s1") == 7

    def test_transaction_commits_on_success(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with store.transaction() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS foo (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO foo VALUES (1)")
        with sqlite3.connect(tmp_path / "proj.db") as c:
            assert c.execute("SELECT COUNT(*) FROM foo").fetchone()[0] == 1

    def test_transaction_rolls_back_on_exception(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with store.transaction() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS bar (id INTEGER PRIMARY KEY)")
        try:
            with store.transaction() as conn:
                conn.execute("INSERT INTO bar VALUES (1)")
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        with sqlite3.connect(tmp_path / "proj.db") as c:
            assert c.execute("SELECT COUNT(*) FROM bar").fetchone()[0] == 0

    def test_transaction_reraises_exception(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(ValueError, match="oops"), store.transaction() as _conn:
            raise ValueError("oops")

    def test_tables_created_on_migrate(self, tmp_path: Path) -> None:
        store = SQLiteProjectionStore(tmp_path / "p.db")
        store.migrate()
        with sqlite3.connect(tmp_path / "p.db") as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "_watermarks" in tables
        assert "sessions" in tables
        assert "tool_calls" in tables
        assert "usage_rollups" in tables
        assert "message_index" in tables


# ===========================================================================
# BackgroundIndexer
# ===========================================================================


class TestBackgroundIndexerNoEvents:
    async def test_returns_zero_no_files(self, tmp_path: Path) -> None:
        indexer, _ = make_indexer(tmp_path)
        assert await indexer.index_session("s1") == 0

    async def test_returns_zero_all_already_indexed(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1)])
        indexer, store = make_indexer(tmp_path)
        await indexer.index_session("s1")
        assert await indexer.index_session("s1") == 0

    async def test_returns_zero_when_watermark_at_latest(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1)])
        handler = CapturingEventHandler()
        indexer, store = make_indexer(tmp_path, handler)
        with store.transaction() as conn:
            store.set_watermark(conn, "s1", 1, "2024-01-01T00:00:00+00:00")
        assert await indexer.index_session("s1") == 0
        assert handler.calls == []


class TestBackgroundIndexerNewEvents:
    async def test_returns_event_count(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1), make_record(2)])
        indexer, _ = make_indexer(tmp_path)
        assert await indexer.index_session("s1") == 3

    async def test_handler_called_once_per_event(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1)])
        handler = CapturingEventHandler()
        indexer, _ = make_indexer(tmp_path, handler)
        await indexer.index_session("s1")
        assert len(handler.calls) == 2

    async def test_events_delivered_in_seq_order(self, tmp_path: Path) -> None:
        p = ndjson_path(tmp_path, "s1")
        write_ndjson(p, [make_record(2), make_record(0), make_record(1)])
        handler = CapturingEventHandler()
        indexer, _ = make_indexer(tmp_path, handler)
        await indexer.index_session("s1")
        seqs = [c["event"].seq for c in handler.calls]
        assert seqs == [0, 1, 2]

    async def test_handler_receives_correct_session_id(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "sess-abc"), [make_record(0)])
        handler = CapturingEventHandler()
        indexer, _ = make_indexer(tmp_path, handler)
        await indexer.index_session("sess-abc")
        assert handler.calls[0]["session_id"] == "sess-abc"

    async def test_watermark_advances_to_last_seq(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1), make_record(2)])
        indexer, store = make_indexer(tmp_path)
        await indexer.index_session("s1")
        assert store.get_watermark("s1") == 2

    async def test_second_call_returns_zero(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1)])
        indexer, _ = make_indexer(tmp_path)
        await indexer.index_session("s1")
        assert await indexer.index_session("s1") == 0

    async def test_incremental_indexing(self, tmp_path: Path) -> None:
        p = ndjson_path(tmp_path, "s1")
        write_ndjson(p, [make_record(0), make_record(1)])
        indexer, store = make_indexer(tmp_path)
        assert await indexer.index_session("s1") == 2

        # Append two more events (O_APPEND style: rewrite file with all lines)
        write_ndjson(p, [make_record(0), make_record(1), make_record(2), make_record(3)])
        assert await indexer.index_session("s1") == 2
        assert store.get_watermark("s1") == 3

    async def test_independent_sessions(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "sA"), [make_record(0)])
        write_ndjson(ndjson_path(tmp_path, "sB"), [make_record(0), make_record(1)])
        indexer, store = make_indexer(tmp_path)
        await indexer.index_session("sA")
        await indexer.index_session("sB")
        assert store.get_watermark("sA") == 0
        assert store.get_watermark("sB") == 1


class TestBackgroundIndexerFailure:
    async def test_handler_exception_propagates(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0)])
        exc = RuntimeError("boom")
        indexer, _ = make_indexer(tmp_path, FailingEventHandler(exc=exc))
        with pytest.raises(RuntimeError, match="boom"):
            await indexer.index_session("s1")

    async def test_watermark_not_advanced_on_handler_failure(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1)])
        indexer, store = make_indexer(tmp_path, FailingEventHandler(fail_on=1))
        with pytest.raises(RuntimeError):
            await indexer.index_session("s1")
        assert store.get_watermark("s1") == -1

    async def test_partial_failure_watermark_at_last_success(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1), make_record(2)])
        indexer, store = make_indexer(tmp_path, FailingEventHandler(fail_on=2))
        with pytest.raises(RuntimeError):
            await indexer.index_session("s1")
        assert store.get_watermark("s1") == 0

    async def test_reader_indexer_failure_propagates(self, tmp_path: Path) -> None:
        p = ndjson_path(tmp_path, "s1")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("not-json\n", encoding="utf-8")
        indexer, _ = make_indexer(tmp_path)
        with pytest.raises(IndexerFailure) as exc_info:
            await indexer.index_session("s1")
        assert exc_info.value.code == "INDEXER_READ_FAILED"


# ===========================================================================
# IndexerRuntime
# ===========================================================================


class TestIndexerRuntimeSuccess:
    async def test_returns_count(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = make_runtime(StubIndexer(returns=3))
        result = await rt.index_session("s1", options=make_options(audit_log))
        assert result == 3

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await make_runtime().index_session("s1", options=make_options(audit_log))
        assert mock_span.name == "indexer.index_session"

    async def test_span_session_id_attribute(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await make_runtime().index_session("s1", options=make_options(audit_log))
        assert mock_span.attributes["indexer.session_id"] == "s1"

    async def test_invocation_event_attached(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await make_runtime().index_session("s1", options=make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "indexer.invocation" in event_names

    async def test_invocation_event_operation(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await make_runtime().index_session("s1", options=make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "indexer.invocation")
        assert inv[1]["operation"] == "index_session"

    async def test_no_audit_entries_on_success(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await make_runtime().index_session("s1", options=make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await make_runtime().index_session("s1", options=make_options(audit_log))
        assert mock_span.ended


class TestIndexerRuntimeIndexerFailure:
    def _make_failure(self) -> IndexerFailure:
        return IndexerFailure(
            code="INDEXER_READ_FAILED",
            message="bad json",
            session_id="s1",
            timestamp="2024-01-01T00:00:00+00:00",
        )

    async def test_re_raises_indexer_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubIndexer(raises=self._make_failure()))
        with pytest.raises(IndexerFailure) as exc_info:
            await rt.index_session("s1", options=make_options(audit_log))
        assert exc_info.value.code == "INDEXER_READ_FAILED"

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubIndexer(raises=self._make_failure()))
        with pytest.raises(IndexerFailure):
            await rt.index_session("s1", options=make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "indexer.index_session.failed"

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubIndexer(raises=self._make_failure()))
        with pytest.raises(IndexerFailure):
            await rt.index_session("s1", options=make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_span_ended_on_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubIndexer(raises=self._make_failure()))
        with pytest.raises(IndexerFailure):
            await rt.index_session("s1", options=make_options(audit_log))
        assert mock_span.ended


class TestIndexerRuntimeUnexpectedException:
    async def test_wraps_as_index_session_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubIndexer(raises=OSError("disk full")))
        with pytest.raises(IndexerFailure) as exc_info:
            await rt.index_session("s1", options=make_options(audit_log))
        assert exc_info.value.code == "INDEXER_INDEX_SESSION_FAILED"

    async def test_cause_preserved(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = OSError("disk full")
        rt = make_runtime(StubIndexer(raises=orig))
        with pytest.raises(IndexerFailure) as exc_info:
            await rt.index_session("s1", options=make_options(audit_log))
        assert exc_info.value.cause is orig

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubIndexer(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            await rt.index_session("s1", options=make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "indexer.index_session.failed"
        assert entry.session_id == "s1"

    async def test_indexer_error_event_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubIndexer(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            await rt.index_session("s1", options=make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "indexer.error" in event_names

    async def test_exception_recorded_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = OSError("boom")
        rt = make_runtime(StubIndexer(raises=orig))
        with pytest.raises(IndexerFailure):
            await rt.index_session("s1", options=make_options(audit_log))
        assert orig in mock_span.recorded_exceptions

    async def test_on_error_callback(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[IndexerFailure] = []
        rt = make_runtime(StubIndexer(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            await rt.index_session("s1", options=make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "INDEXER_INDEX_SESSION_FAILED"

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubIndexer(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            await rt.index_session("s1", options=make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_span_ended_on_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubIndexer(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            await rt.index_session("s1", options=make_options(audit_log))
        assert mock_span.ended


# ===========================================================================
# LocalEventLogReader.read_events
# ===========================================================================


class TestLocalEventLogReaderReadEvents:
    async def test_empty_when_no_files(self, tmp_path: Path) -> None:
        reader = LocalEventLogReader(tmp_path)
        events = [e async for e in reader.read_events("s1", since=-1)]
        assert events == []

    async def test_yields_all_events_since_minus_one(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1)])
        reader = LocalEventLogReader(tmp_path)
        events = [e async for e in reader.read_events("s1", since=-1)]
        assert [e.seq for e in events] == [0, 1]

    async def test_filters_events_at_or_below_since(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1), make_record(2)])
        reader = LocalEventLogReader(tmp_path)
        events = [e async for e in reader.read_events("s1", since=1)]
        assert [e.seq for e in events] == [2]

    async def test_yields_events_in_seq_order(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(2), make_record(0), make_record(1)])
        reader = LocalEventLogReader(tmp_path)
        events = [e async for e in reader.read_events("s1", since=-1)]
        assert [e.seq for e in events] == [0, 1, 2]

    async def test_empty_when_all_below_since(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1)])
        reader = LocalEventLogReader(tmp_path)
        events = [e async for e in reader.read_events("s1", since=5)]
        assert events == []

    async def test_multiple_date_files_merged(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1", "2024/01/01"), [make_record(0), make_record(1)])
        write_ndjson(ndjson_path(tmp_path, "s1", "2024/01/02"), [make_record(2), make_record(3)])
        reader = LocalEventLogReader(tmp_path)
        events = [e async for e in reader.read_events("s1", since=-1)]
        assert [e.seq for e in events] == [0, 1, 2, 3]

    async def test_raises_on_bad_json(self, tmp_path: Path) -> None:
        p = ndjson_path(tmp_path, "s1")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("not-json\n", encoding="utf-8")
        reader = LocalEventLogReader(tmp_path)
        with pytest.raises(IndexerFailure) as exc_info:
            async for _ in reader.read_events("s1", since=-1):
                pass
        assert exc_info.value.code == "INDEXER_READ_FAILED"

    async def test_stops_after_existing_events_cat_mode(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [make_record(0), make_record(1)])
        reader = LocalEventLogReader(tmp_path)
        count = 0
        async for _ in reader.read_events("s1", since=-1, follow=False):
            count += 1
        assert count == 2

    async def test_tail_yields_new_events(self, tmp_path: Path) -> None:
        p = ndjson_path(tmp_path, "s1")
        write_ndjson(p, [make_record(0)])
        reader = LocalEventLogReader(tmp_path)

        gen = reader.read_events("s1", since=-1, follow=True, poll_interval=0.01)
        first = await gen.__anext__()
        assert first.seq == 0

        write_ndjson(p, [make_record(0), make_record(1)])
        second = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
        assert second.seq == 1

        await gen.aclose()

    async def test_tail_yields_events_from_multiple_polls(self, tmp_path: Path) -> None:
        p = ndjson_path(tmp_path, "s1")
        write_ndjson(p, [make_record(0)])
        reader = LocalEventLogReader(tmp_path)

        gen = reader.read_events("s1", since=-1, follow=True, poll_interval=0.01)
        await gen.__anext__()

        write_ndjson(p, [make_record(0), make_record(1)])
        e1 = await asyncio.wait_for(gen.__anext__(), timeout=2.0)

        write_ndjson(p, [make_record(0), make_record(1), make_record(2)])
        e2 = await asyncio.wait_for(gen.__anext__(), timeout=2.0)

        await gen.aclose()
        assert e1.seq == 1
        assert e2.seq == 2


# ===========================================================================
# ReaderRuntime
# ===========================================================================


def make_reader_runtime(stub: StubReader | None = None) -> ReaderRuntime:
    return ReaderRuntime(stub or StubReader())  # type: ignore[arg-type]


class TestReaderRuntimeSuccess:
    async def test_yields_events(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        events_in = [make_session_event(0), make_session_event(1)]
        rt = make_reader_runtime(StubReader(events=events_in))
        out = [e async for e in rt.read_events("s1", options=make_reader_options(audit_log))]
        assert [e.seq for e in out] == [0, 1]

    async def test_span_name(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        async for _ in make_reader_runtime().read_events(
            "s1", options=make_reader_options(audit_log)
        ):
            pass
        assert mock_reader_span.name == "reader.read_events"

    async def test_span_session_id_attribute(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        async for _ in make_reader_runtime().read_events(
            "s1", options=make_reader_options(audit_log)
        ):
            pass
        assert mock_reader_span.attributes["reader.session_id"] == "s1"

    async def test_invocation_event_attached(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        async for _ in make_reader_runtime().read_events(
            "s1", options=make_reader_options(audit_log)
        ):
            pass
        event_names = [e[0] for e in mock_reader_span.events]
        assert "reader.invocation" in event_names

    async def test_invocation_event_operation(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        async for _ in make_reader_runtime().read_events(
            "s1", options=make_reader_options(audit_log)
        ):
            pass
        inv = next(e for e in mock_reader_span.events if e[0] == "reader.invocation")
        assert inv[1]["operation"] == "read_events"

    async def test_no_audit_entries_on_success(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        async for _ in make_reader_runtime().read_events(
            "s1", options=make_reader_options(audit_log)
        ):
            pass
        assert audit_log.entries == []

    async def test_span_ended(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        async for _ in make_reader_runtime().read_events(
            "s1", options=make_reader_options(audit_log)
        ):
            pass
        assert mock_reader_span.ended


class TestReaderRuntimeIndexerFailure:
    def _make_failure(self) -> IndexerFailure:
        return IndexerFailure(
            code="INDEXER_READ_FAILED",
            message="bad json",
            session_id="s1",
            timestamp="2024-01-01T00:00:00+00:00",
        )

    async def test_re_raises_indexer_failure(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_reader_runtime(StubReader(raises=self._make_failure()))
        with pytest.raises(IndexerFailure) as exc_info:
            async for _ in rt.read_events("s1", options=make_reader_options(audit_log)):
                pass
        assert exc_info.value.code == "INDEXER_READ_FAILED"

    async def test_audit_entry_written(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_reader_runtime(StubReader(raises=self._make_failure()))
        with pytest.raises(IndexerFailure):
            async for _ in rt.read_events("s1", options=make_reader_options(audit_log)):
                pass
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "reader.read_events.failed"

    async def test_span_marked_error(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_reader_runtime(StubReader(raises=self._make_failure()))
        with pytest.raises(IndexerFailure):
            async for _ in rt.read_events("s1", options=make_reader_options(audit_log)):
                pass
        assert mock_reader_span.status.status_code == StatusCode.ERROR

    async def test_span_ended_on_failure(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_reader_runtime(StubReader(raises=self._make_failure()))
        with pytest.raises(IndexerFailure):
            async for _ in rt.read_events("s1", options=make_reader_options(audit_log)):
                pass
        assert mock_reader_span.ended


class TestReaderRuntimeUnexpectedException:
    async def test_wraps_as_reader_read_events_failed(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_reader_runtime(StubReader(raises=OSError("disk full")))
        with pytest.raises(IndexerFailure) as exc_info:
            async for _ in rt.read_events("s1", options=make_reader_options(audit_log)):
                pass
        assert exc_info.value.code == "READER_READ_EVENTS_FAILED"

    async def test_cause_preserved(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = OSError("disk full")
        rt = make_reader_runtime(StubReader(raises=orig))
        with pytest.raises(IndexerFailure) as exc_info:
            async for _ in rt.read_events("s1", options=make_reader_options(audit_log)):
                pass
        assert exc_info.value.cause is orig

    async def test_audit_entry_written(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_reader_runtime(StubReader(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            async for _ in rt.read_events("s1", options=make_reader_options(audit_log)):
                pass
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "reader.read_events.failed"
        assert entry.session_id == "s1"

    async def test_reader_error_event_on_span(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_reader_runtime(StubReader(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            async for _ in rt.read_events("s1", options=make_reader_options(audit_log)):
                pass
        event_names = [e[0] for e in mock_reader_span.events]
        assert "reader.error" in event_names

    async def test_exception_recorded_on_span(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = OSError("boom")
        rt = make_reader_runtime(StubReader(raises=orig))
        with pytest.raises(IndexerFailure):
            async for _ in rt.read_events("s1", options=make_reader_options(audit_log)):
                pass
        assert orig in mock_reader_span.recorded_exceptions

    async def test_on_error_callback(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[IndexerFailure] = []
        rt = make_reader_runtime(StubReader(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            async for _ in rt.read_events("s1", options=make_reader_options(audit_log, errors)):
                pass
        assert len(errors) == 1
        assert errors[0].code == "READER_READ_EVENTS_FAILED"

    async def test_span_marked_error(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_reader_runtime(StubReader(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            async for _ in rt.read_events("s1", options=make_reader_options(audit_log)):
                pass
        assert mock_reader_span.status.status_code == StatusCode.ERROR

    async def test_span_ended_on_failure(
        self, mock_reader_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_reader_runtime(StubReader(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            async for _ in rt.read_events("s1", options=make_reader_options(audit_log)):
                pass
        assert mock_reader_span.ended


# ===========================================================================
# MigrationRuntime
# ===========================================================================


class TestMigrationRuntimeSuccess:
    def test_returns_count(
        self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_migration_runtime(StubStore(returns=2))
        result = rt.migrate(options=make_migration_options(audit_log))
        assert result == 2

    def test_span_name(self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        make_migration_runtime().migrate(options=make_migration_options(audit_log))
        assert mock_migration_span.name == "migration.migrate"

    def test_invocation_event_attached(
        self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_migration_runtime().migrate(options=make_migration_options(audit_log))
        event_names = [e[0] for e in mock_migration_span.events]
        assert "migration.invocation" in event_names

    def test_invocation_event_operation(
        self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_migration_runtime().migrate(options=make_migration_options(audit_log))
        inv = next(e for e in mock_migration_span.events if e[0] == "migration.invocation")
        assert inv[1]["operation"] == "migrate"

    def test_no_audit_entries_on_success(
        self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_migration_runtime().migrate(options=make_migration_options(audit_log))
        assert audit_log.entries == []

    def test_span_ended(self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        make_migration_runtime().migrate(options=make_migration_options(audit_log))
        assert mock_migration_span.ended


class TestMigrationRuntimeUnexpectedException:
    def test_wraps_as_migration_failed(
        self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_migration_runtime(StubStore(raises=OSError("disk full")))
        with pytest.raises(IndexerFailure) as exc_info:
            rt.migrate(options=make_migration_options(audit_log))
        assert exc_info.value.code == "MIGRATION_FAILED"

    def test_cause_preserved(
        self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = OSError("disk full")
        rt = make_migration_runtime(StubStore(raises=orig))
        with pytest.raises(IndexerFailure) as exc_info:
            rt.migrate(options=make_migration_options(audit_log))
        assert exc_info.value.cause is orig

    def test_audit_entry_written(
        self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_migration_runtime(StubStore(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            rt.migrate(options=make_migration_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "migration.migrate.failed"

    def test_migration_error_event_on_span(
        self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_migration_runtime(StubStore(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            rt.migrate(options=make_migration_options(audit_log))
        event_names = [e[0] for e in mock_migration_span.events]
        assert "migration.error" in event_names

    def test_exception_recorded_on_span(
        self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = OSError("boom")
        rt = make_migration_runtime(StubStore(raises=orig))
        with pytest.raises(IndexerFailure):
            rt.migrate(options=make_migration_options(audit_log))
        assert orig in mock_migration_span.recorded_exceptions

    def test_on_error_callback(
        self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[IndexerFailure] = []
        rt = make_migration_runtime(StubStore(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            rt.migrate(options=make_migration_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "MIGRATION_FAILED"

    def test_span_marked_error(
        self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_migration_runtime(StubStore(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            rt.migrate(options=make_migration_options(audit_log))
        assert mock_migration_span.status.status_code == StatusCode.ERROR

    def test_span_ended_on_failure(
        self, mock_migration_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_migration_runtime(StubStore(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            rt.migrate(options=make_migration_options(audit_log))
        assert mock_migration_span.ended
