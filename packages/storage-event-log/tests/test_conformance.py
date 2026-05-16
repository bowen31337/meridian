"""
Event-log conformance suite.

Covers EventLogRuntime (via a StubEventLogWriter) and LocalEventLogWriter
(via tmp_path):

  EventLogRuntime:
    - append success: span emitted, invocation event attached, no audit entries,
      correct seq returned.
    - Writer raises EventLogFailure (e.g. invalid session ID): audited and re-raised.
    - Writer raises unexpected exception: wrapped as EVENT_LOG_APPEND_FAILED,
      cause preserved, audit entry written, span marked ERROR.
    - on_error callback invoked on every failure.
    - Span lifecycle: span ended on both success and failure paths.

  LocalEventLogWriter:
    - append creates NDJSON file at expected path and returns seq 0.
    - Seq increments monotonically per session.
    - Different sessions are independent.
    - Written line is valid JSON with required fields.
    - thread_id omitted from JSON when None, included when set.
    - O_APPEND: multiple appends produce multiple lines.
    - Invalid session IDs raise EventLogFailure(EVENT_LOG_SESSION_ID_INVALID).
    - Nested date directories are created automatically.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from storage_event_log import (
    AuditLogEntry,
    EventLogFailure,
    EventLogOptions,
    EventLogRuntime,
    EventLogWriter,
    LocalEventLogWriter,
)
from opentelemetry.trace import StatusCode

from .conftest import CapturingAuditLog, MockSpan


# ---------------------------------------------------------------------------
# Stub writer
# ---------------------------------------------------------------------------

class StubEventLogWriter(EventLogWriter):
    """In-memory writer with configurable failure injection."""

    def __init__(self, *, append_raises: Exception | None = None) -> None:
        self._append_raises = append_raises
        self._calls: list[dict[str, Any]] = []
        self._seq: dict[str, int] = {}

    async def append(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
        *,
        thread_id: str | None = None,
    ) -> int:
        if self._append_raises:
            raise self._append_raises
        seq = self._seq.get(session_id, 0)
        self._calls.append(
            {"session_id": session_id, "type": event_type, "data": data, "thread_id": thread_id, "seq": seq}
        )
        self._seq[session_id] = seq + 1
        return seq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_options(audit: CapturingAuditLog, errors: list[EventLogFailure] | None = None) -> EventLogOptions:
    return EventLogOptions(
        audit_log=audit,
        on_error=(lambda e: errors.append(e)) if errors is not None else None,
    )


def make_runtime(writer: EventLogWriter | None = None) -> EventLogRuntime:
    return EventLogRuntime(writer or StubEventLogWriter())


# ---------------------------------------------------------------------------
# append — success
# ---------------------------------------------------------------------------

class TestAppendSuccess:
    async def test_seq_returned(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        writer = StubEventLogWriter()
        rt = EventLogRuntime(writer)
        seq = await rt.append("s1", "session.created", {}, options=make_options(audit_log))
        assert seq == 0

    async def test_seq_increments(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        writer = StubEventLogWriter()
        rt = EventLogRuntime(writer)
        s0 = await rt.append("s1", "session.created", {}, options=make_options(audit_log))
        s1 = await rt.append("s1", "message.added", {"text": "hi"}, options=make_options(audit_log))
        assert s0 == 0
        assert s1 == 1

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await make_runtime().append("s1", "session.created", {}, options=make_options(audit_log))
        assert mock_span.name == "event_log.append"

    async def test_span_session_id_attribute(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await make_runtime().append("s1", "session.created", {}, options=make_options(audit_log))
        assert mock_span.attributes["event_log.session_id"] == "s1"

    async def test_invocation_event_attached(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await make_runtime().append("s1", "session.created", {}, options=make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "event_log.invocation" in event_names

    async def test_invocation_event_operation(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await make_runtime().append("s1", "session.created", {}, options=make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "event_log.invocation")
        assert inv[1]["operation"] == "append"

    async def test_no_audit_entries_on_success(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await make_runtime().append("s1", "session.created", {}, options=make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await make_runtime().append("s1", "session.created", {}, options=make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# append — writer raises EventLogFailure (e.g. invalid session ID)
# ---------------------------------------------------------------------------

class TestAppendEventLogFailure:
    def _make_failure(self) -> EventLogFailure:
        return EventLogFailure(
            code="EVENT_LOG_SESSION_ID_INVALID",
            message="bad id",
            session_id="bad/id",
            timestamp="2024-01-01T00:00:00+00:00",
        )

    async def test_re_raises_event_log_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EventLogRuntime(StubEventLogWriter(append_raises=self._make_failure()))
        with pytest.raises(EventLogFailure) as exc_info:
            await rt.append("bad/id", "session.created", {}, options=make_options(audit_log))
        assert exc_info.value.code == "EVENT_LOG_SESSION_ID_INVALID"

    async def test_audit_entry_written(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EventLogRuntime(StubEventLogWriter(append_raises=self._make_failure()))
        with pytest.raises(EventLogFailure):
            await rt.append("bad/id", "session.created", {}, options=make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "event_log.append.failed"

    async def test_span_marked_error(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EventLogRuntime(StubEventLogWriter(append_raises=self._make_failure()))
        with pytest.raises(EventLogFailure):
            await rt.append("bad/id", "session.created", {}, options=make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_span_ended_on_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EventLogRuntime(StubEventLogWriter(append_raises=self._make_failure()))
        with pytest.raises(EventLogFailure):
            await rt.append("bad/id", "session.created", {}, options=make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# append — writer raises unexpected exception
# ---------------------------------------------------------------------------

class TestAppendStoreRaises:
    async def test_wraps_as_append_failed(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EventLogRuntime(StubEventLogWriter(append_raises=OSError("disk full")))
        with pytest.raises(EventLogFailure) as exc_info:
            await rt.append("s1", "session.created", {}, options=make_options(audit_log))
        assert exc_info.value.code == "EVENT_LOG_APPEND_FAILED"

    async def test_cause_preserved(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = OSError("disk full")
        rt = EventLogRuntime(StubEventLogWriter(append_raises=orig))
        with pytest.raises(EventLogFailure) as exc_info:
            await rt.append("s1", "session.created", {}, options=make_options(audit_log))
        assert exc_info.value.cause is orig

    async def test_audit_entry_written(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EventLogRuntime(StubEventLogWriter(append_raises=OSError("boom")))
        with pytest.raises(EventLogFailure):
            await rt.append("s1", "session.created", {}, options=make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "event_log.append.failed"
        assert entry.session_id == "s1"

    async def test_span_marked_error(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EventLogRuntime(StubEventLogWriter(append_raises=OSError("boom")))
        with pytest.raises(EventLogFailure):
            await rt.append("s1", "session.created", {}, options=make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_error_event_on_span(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EventLogRuntime(StubEventLogWriter(append_raises=OSError("boom")))
        with pytest.raises(EventLogFailure):
            await rt.append("s1", "session.created", {}, options=make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "event_log.error" in event_names

    async def test_exception_recorded_on_span(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = OSError("boom")
        rt = EventLogRuntime(StubEventLogWriter(append_raises=orig))
        with pytest.raises(EventLogFailure):
            await rt.append("s1", "session.created", {}, options=make_options(audit_log))
        assert orig in mock_span.recorded_exceptions

    async def test_on_error_callback(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        errors: list[EventLogFailure] = []
        rt = EventLogRuntime(StubEventLogWriter(append_raises=OSError("boom")))
        with pytest.raises(EventLogFailure):
            await rt.append("s1", "session.created", {}, options=make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "EVENT_LOG_APPEND_FAILED"

    async def test_span_ended_on_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EventLogRuntime(StubEventLogWriter(append_raises=OSError("boom")))
        with pytest.raises(EventLogFailure):
            await rt.append("s1", "session.created", {}, options=make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# LocalEventLogWriter — filesystem integration
# ---------------------------------------------------------------------------

class TestLocalEventLogWriter:
    async def test_append_returns_seq_zero(self, tmp_path: Path) -> None:
        writer = LocalEventLogWriter(tmp_path)
        seq = await writer.append("sess1", "session.created", {})
        assert seq == 0

    async def test_seq_increments_per_session(self, tmp_path: Path) -> None:
        writer = LocalEventLogWriter(tmp_path)
        s0 = await writer.append("sess1", "session.created", {})
        s1 = await writer.append("sess1", "message.added", {"text": "hi"})
        s2 = await writer.append("sess1", "message.added", {"text": "bye"})
        assert (s0, s1, s2) == (0, 1, 2)

    async def test_different_sessions_are_independent(self, tmp_path: Path) -> None:
        writer = LocalEventLogWriter(tmp_path)
        assert await writer.append("sessA", "session.created", {}) == 0
        assert await writer.append("sessB", "session.created", {}) == 0
        assert await writer.append("sessA", "message.added", {}) == 1

    async def test_file_created_at_expected_path(self, tmp_path: Path) -> None:
        from datetime import datetime, timezone
        from unittest.mock import patch

        fixed_dt = datetime(2024, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        with patch("storage_event_log._local._now_dt", return_value=fixed_dt):
            writer = LocalEventLogWriter(tmp_path)
            await writer.append("my-session", "session.created", {})

        expected = tmp_path / "events" / "2024" / "03" / "15" / "my-session.ndjson"
        assert expected.exists()

    async def test_written_line_is_valid_json(self, tmp_path: Path) -> None:
        writer = LocalEventLogWriter(tmp_path)
        await writer.append("sess1", "session.created", {"key": "val"})

        ndjson_files = list(tmp_path.rglob("*.ndjson"))
        assert len(ndjson_files) == 1
        lines = ndjson_files[0].read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["seq"] == 0
        assert record["type"] == "session.created"
        assert record["data"] == {"key": "val"}
        assert "ts" in record

    async def test_thread_id_omitted_when_none(self, tmp_path: Path) -> None:
        writer = LocalEventLogWriter(tmp_path)
        await writer.append("sess1", "session.created", {}, thread_id=None)
        ndjson_files = list(tmp_path.rglob("*.ndjson"))
        record = json.loads(ndjson_files[0].read_text().strip())
        assert "thread_id" not in record

    async def test_thread_id_included_when_set(self, tmp_path: Path) -> None:
        writer = LocalEventLogWriter(tmp_path)
        await writer.append("sess1", "session.created", {}, thread_id="t-42")
        ndjson_files = list(tmp_path.rglob("*.ndjson"))
        record = json.loads(ndjson_files[0].read_text().strip())
        assert record["thread_id"] == "t-42"

    async def test_multiple_appends_produce_multiple_lines(self, tmp_path: Path) -> None:
        writer = LocalEventLogWriter(tmp_path)
        await writer.append("sess1", "session.created", {})
        await writer.append("sess1", "message.added", {"n": 1})
        await writer.append("sess1", "message.added", {"n": 2})
        ndjson_files = list(tmp_path.rglob("*.ndjson"))
        lines = ndjson_files[0].read_text().strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["seq"] == 0
        assert json.loads(lines[1])["seq"] == 1
        assert json.loads(lines[2])["seq"] == 2

    async def test_lines_end_with_newline(self, tmp_path: Path) -> None:
        writer = LocalEventLogWriter(tmp_path)
        await writer.append("sess1", "session.created", {})
        ndjson_files = list(tmp_path.rglob("*.ndjson"))
        raw = ndjson_files[0].read_bytes()
        assert raw.endswith(b"\n")

    async def test_date_dirs_created_automatically(self, tmp_path: Path) -> None:
        writer = LocalEventLogWriter(tmp_path)
        await writer.append("sess1", "session.created", {})
        events_dir = tmp_path / "events"
        assert events_dir.is_dir()

    async def test_invalid_session_id_slash(self, tmp_path: Path) -> None:
        writer = LocalEventLogWriter(tmp_path)
        with pytest.raises(EventLogFailure) as exc_info:
            await writer.append("bad/id", "session.created", {})
        assert exc_info.value.code == "EVENT_LOG_SESSION_ID_INVALID"

    async def test_invalid_session_id_dotdot(self, tmp_path: Path) -> None:
        writer = LocalEventLogWriter(tmp_path)
        with pytest.raises(EventLogFailure) as exc_info:
            await writer.append("../../etc", "session.created", {})
        assert exc_info.value.code == "EVENT_LOG_SESSION_ID_INVALID"

    async def test_invalid_session_id_empty(self, tmp_path: Path) -> None:
        writer = LocalEventLogWriter(tmp_path)
        with pytest.raises(EventLogFailure) as exc_info:
            await writer.append("", "session.created", {})
        assert exc_info.value.code == "EVENT_LOG_SESSION_ID_INVALID"

    async def test_data_roundtrip(self, tmp_path: Path) -> None:
        payload = {"nested": {"a": 1, "b": [1, 2, 3]}, "flag": True}
        writer = LocalEventLogWriter(tmp_path)
        await writer.append("sess1", "hook.invoked", payload)
        ndjson_files = list(tmp_path.rglob("*.ndjson"))
        record = json.loads(ndjson_files[0].read_text().strip())
        assert record["data"] == payload
