"""
PhaseProjection and PhaseProjectionRuntime conformance suite.

Covers:

  PhaseProjection:
    - Returns 'created' when no event log files exist.
    - Returns 'created' when events exist but none are session.phase_change.
    - Returns 'after' from the single session.phase_change event.
    - Returns 'after' from the last of multiple session.phase_change events.
    - Ignores phase_change events with a non-string 'after' field.
    - Ignores phase_change events with an empty string 'after' field.
    - Handles events across multiple date-partitioned files.
    - Propagates IndexerFailure from the reader on malformed JSON.

  PhaseProjectionRuntime:
    - Returns the current phase on success.
    - Span name is "phase.current_phase".
    - Span carries phase.session_id attribute.
    - "phase.invocation" structured event is attached to the span.
    - Invocation event has operation="current_phase".
    - No audit entries written on success.
    - Span is ended on success.
    - IndexerFailure from projection is re-raised and audited.
    - Audit entry level is "error" and event is "phase.current_phase.failed".
    - Span is marked ERROR on IndexerFailure.
    - Span is ended on IndexerFailure.
    - Unexpected exception is wrapped as PHASE_PROJECT_FAILED.
    - Cause is preserved in wrapped failure.
    - Audit entry written for unexpected exception.
    - "phase.error" event added to span on unexpected exception.
    - Exception recorded on span via record_exception.
    - on_error callback invoked for every failure.
    - Span is ended on unexpected exception.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opentelemetry.trace import StatusCode
import pytest
from storage_reposit import (
    AuditLogEntry,
    IndexerFailure,
    LocalEventLogReader,
    PhaseProjection,
    PhaseProjectionOptions,
    PhaseProjectionRuntime,
)

from .conftest import CapturingAuditLog, MockSpan, StubPhaseProjection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_ndjson(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def ndjson_path(root: Path, session_id: str, date: str = "2024/01/01") -> Path:
    return root / "events" / date / f"{session_id}.ndjson"


def phase_change_record(seq: int, before: str, after: str) -> dict[str, Any]:
    return {
        "seq": seq,
        "ts": f"2024-01-01T00:00:0{seq}.000+00:00",
        "type": "session.phase_change",
        "data": {"before": before, "after": after, "reason": "test"},
    }


def other_record(seq: int) -> dict[str, Any]:
    return {
        "seq": seq,
        "ts": f"2024-01-01T00:00:0{seq}.000+00:00",
        "type": "message.added",
        "data": {},
    }


def make_projection(root: Path) -> PhaseProjection:
    return PhaseProjection(LocalEventLogReader(root))


def make_runtime(stub: StubPhaseProjection | None = None) -> PhaseProjectionRuntime:
    return PhaseProjectionRuntime(stub or StubPhaseProjection())  # type: ignore[arg-type]


def make_options(
    audit: CapturingAuditLog,
    errors: list[IndexerFailure] | None = None,
) -> PhaseProjectionOptions:
    return PhaseProjectionOptions(
        audit_log=audit,
        on_error=(lambda e: errors.append(e)) if errors is not None else None,
    )


# ===========================================================================
# PhaseProjection
# ===========================================================================


class TestPhaseProjectionDefault:
    def test_returns_created_when_no_files(self, tmp_path: Path) -> None:
        assert make_projection(tmp_path).current_phase("s1") == "created"

    def test_returns_created_when_no_phase_change_events(self, tmp_path: Path) -> None:
        write_ndjson(ndjson_path(tmp_path, "s1"), [other_record(0), other_record(1)])
        assert make_projection(tmp_path).current_phase("s1") == "created"


class TestPhaseProjectionFromEvents:
    def test_single_phase_change(self, tmp_path: Path) -> None:
        write_ndjson(
            ndjson_path(tmp_path, "s1"),
            [phase_change_record(0, "created", "running")],
        )
        assert make_projection(tmp_path).current_phase("s1") == "running"

    def test_last_phase_change_wins(self, tmp_path: Path) -> None:
        write_ndjson(
            ndjson_path(tmp_path, "s1"),
            [
                phase_change_record(0, "created", "running"),
                phase_change_record(1, "running", "paused"),
                phase_change_record(2, "paused", "done"),
            ],
        )
        assert make_projection(tmp_path).current_phase("s1") == "done"

    def test_non_phase_events_ignored(self, tmp_path: Path) -> None:
        write_ndjson(
            ndjson_path(tmp_path, "s1"),
            [
                other_record(0),
                phase_change_record(1, "created", "running"),
                other_record(2),
            ],
        )
        assert make_projection(tmp_path).current_phase("s1") == "running"

    def test_non_string_after_ignored(self, tmp_path: Path) -> None:
        record: dict[str, Any] = {
            "seq": 0,
            "ts": "2024-01-01T00:00:00.000+00:00",
            "type": "session.phase_change",
            "data": {"before": "created", "after": 42, "reason": "bad"},
        }
        write_ndjson(ndjson_path(tmp_path, "s1"), [record])
        assert make_projection(tmp_path).current_phase("s1") == "created"

    def test_empty_string_after_ignored(self, tmp_path: Path) -> None:
        record: dict[str, Any] = {
            "seq": 0,
            "ts": "2024-01-01T00:00:00.000+00:00",
            "type": "session.phase_change",
            "data": {"before": "created", "after": "", "reason": "bad"},
        }
        write_ndjson(ndjson_path(tmp_path, "s1"), [record])
        assert make_projection(tmp_path).current_phase("s1") == "created"

    def test_multiple_date_partitions(self, tmp_path: Path) -> None:
        write_ndjson(
            ndjson_path(tmp_path, "s1", "2024/01/01"),
            [phase_change_record(0, "created", "running")],
        )
        write_ndjson(
            ndjson_path(tmp_path, "s1", "2024/01/02"),
            [phase_change_record(1, "running", "done")],
        )
        assert make_projection(tmp_path).current_phase("s1") == "done"

    def test_sessions_isolated(self, tmp_path: Path) -> None:
        write_ndjson(
            ndjson_path(tmp_path, "s1"),
            [phase_change_record(0, "created", "running")],
        )
        write_ndjson(
            ndjson_path(tmp_path, "s2"),
            [phase_change_record(0, "created", "paused")],
        )
        assert make_projection(tmp_path).current_phase("s1") == "running"
        assert make_projection(tmp_path).current_phase("s2") == "paused"


class TestPhaseProjectionFailure:
    def test_propagates_indexer_failure_on_bad_json(self, tmp_path: Path) -> None:
        p = ndjson_path(tmp_path, "s1")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("not-json\n", encoding="utf-8")
        with pytest.raises(IndexerFailure) as exc_info:
            make_projection(tmp_path).current_phase("s1")
        assert exc_info.value.code == "INDEXER_READ_FAILED"


# ===========================================================================
# PhaseProjectionRuntime
# ===========================================================================


class TestPhaseProjectionRuntimeSuccess:
    def test_returns_phase(self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = make_runtime(StubPhaseProjection(returns="running"))
        assert rt.current_phase("s1", options=make_options(audit_log)) == "running"

    def test_span_name(self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        make_runtime().current_phase("s1", options=make_options(audit_log))
        assert mock_phase_span.name == "phase.current_phase"

    def test_span_session_id_attribute(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().current_phase("s1", options=make_options(audit_log))
        assert mock_phase_span.attributes["phase.session_id"] == "s1"

    def test_invocation_event_attached(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().current_phase("s1", options=make_options(audit_log))
        event_names = [e[0] for e in mock_phase_span.events]
        assert "phase.invocation" in event_names

    def test_invocation_event_operation(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().current_phase("s1", options=make_options(audit_log))
        inv = next(e for e in mock_phase_span.events if e[0] == "phase.invocation")
        assert inv[1]["operation"] == "current_phase"

    def test_no_audit_entries_on_success(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().current_phase("s1", options=make_options(audit_log))
        assert audit_log.entries == []

    def test_span_ended(self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        make_runtime().current_phase("s1", options=make_options(audit_log))
        assert mock_phase_span.ended


class TestPhaseProjectionRuntimeIndexerFailure:
    def _make_failure(self) -> IndexerFailure:
        return IndexerFailure(
            code="INDEXER_READ_FAILED",
            message="bad json",
            session_id="s1",
            timestamp="2024-01-01T00:00:00+00:00",
        )

    def test_re_raises_indexer_failure(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseProjection(raises=self._make_failure()))
        with pytest.raises(IndexerFailure) as exc_info:
            rt.current_phase("s1", options=make_options(audit_log))
        assert exc_info.value.code == "INDEXER_READ_FAILED"

    def test_audit_entry_written(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseProjection(raises=self._make_failure()))
        with pytest.raises(IndexerFailure):
            rt.current_phase("s1", options=make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "phase.current_phase.failed"

    def test_span_marked_error(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseProjection(raises=self._make_failure()))
        with pytest.raises(IndexerFailure):
            rt.current_phase("s1", options=make_options(audit_log))
        assert mock_phase_span.status.status_code == StatusCode.ERROR

    def test_span_ended_on_failure(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseProjection(raises=self._make_failure()))
        with pytest.raises(IndexerFailure):
            rt.current_phase("s1", options=make_options(audit_log))
        assert mock_phase_span.ended


class TestPhaseProjectionRuntimeUnexpectedException:
    def test_wraps_as_phase_project_failed(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseProjection(raises=OSError("disk full")))
        with pytest.raises(IndexerFailure) as exc_info:
            rt.current_phase("s1", options=make_options(audit_log))
        assert exc_info.value.code == "PHASE_PROJECT_FAILED"

    def test_cause_preserved(self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = OSError("disk full")
        rt = make_runtime(StubPhaseProjection(raises=orig))
        with pytest.raises(IndexerFailure) as exc_info:
            rt.current_phase("s1", options=make_options(audit_log))
        assert exc_info.value.cause is orig

    def test_audit_entry_written(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseProjection(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            rt.current_phase("s1", options=make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "phase.current_phase.failed"
        assert entry.session_id == "s1"

    def test_phase_error_event_on_span(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseProjection(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            rt.current_phase("s1", options=make_options(audit_log))
        event_names = [e[0] for e in mock_phase_span.events]
        assert "phase.error" in event_names

    def test_exception_recorded_on_span(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = OSError("boom")
        rt = make_runtime(StubPhaseProjection(raises=orig))
        with pytest.raises(IndexerFailure):
            rt.current_phase("s1", options=make_options(audit_log))
        assert orig in mock_phase_span.recorded_exceptions

    def test_on_error_callback(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[IndexerFailure] = []
        rt = make_runtime(StubPhaseProjection(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            rt.current_phase("s1", options=make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "PHASE_PROJECT_FAILED"

    def test_span_marked_error(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseProjection(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            rt.current_phase("s1", options=make_options(audit_log))
        assert mock_phase_span.status.status_code == StatusCode.ERROR

    def test_span_ended_on_failure(
        self, mock_phase_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseProjection(raises=OSError("boom")))
        with pytest.raises(IndexerFailure):
            rt.current_phase("s1", options=make_options(audit_log))
        assert mock_phase_span.ended
