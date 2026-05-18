"""
PhaseStateMachine and PhaseStateMachineRuntime property conformance suite.

Covers:

  PhaseStateMachine — transition table properties:
    - Every (phase, event_type) pair yields a deterministic next phase.
    - created + start -> running.
    - created + pause -> created (no-op).
    - created + resume -> created (no-op).
    - created + terminate -> terminated.
    - running + start -> running (no-op).
    - running + pause -> paused.
    - running + resume -> running (no-op).
    - running + terminate -> terminated.
    - paused + start -> paused (no-op).
    - paused + pause -> paused (no-op).
    - paused + resume -> running.
    - paused + terminate -> terminated.
    - terminated is absorbing: terminate + start -> terminated.
    - terminated is absorbing: terminate + pause -> terminated.
    - terminated is absorbing: terminate + resume -> terminated.
    - terminated is absorbing: terminate + terminate -> terminated.
    - Unknown phase raises ValueError.
    - Unknown event type raises ValueError.

  PhaseStateMachineRuntime:
    - Returns the next phase on success.
    - Span name is "phase.next_phase".
    - Span carries phase.session_id attribute.
    - Span carries phase.current attribute.
    - Span carries phase.event_type attribute.
    - "phase.state_machine.invocation" structured event is attached to the span.
    - Invocation event has operation="next_phase".
    - No audit entries written on success.
    - Span is ended on success.
    - IndexerFailure from machine is re-raised and audited.
    - Audit entry level is "error" and event is "phase.next_phase.failed".
    - Span is marked ERROR on IndexerFailure.
    - Span is ended on IndexerFailure.
    - Unexpected exception is wrapped as PHASE_NEXT_PHASE_FAILED.
    - Cause is preserved in wrapped failure.
    - Audit entry written for unexpected exception.
    - "phase.state_machine.error" event added to span on unexpected exception.
    - Exception recorded on span via record_exception.
    - on_error callback invoked for every failure.
    - Span is ended on unexpected exception.
"""

from __future__ import annotations

import pytest
from opentelemetry.trace import StatusCode
from storage_reposit import (
    EVENTS,
    PHASES,
    AuditLogEntry,
    IndexerFailure,
    PhaseStateMachine,
    PhaseStateMachineOptions,
    PhaseStateMachineRuntime,
)

from .conftest import CapturingAuditLog, MockSpan, StubPhaseStateMachine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_runtime(stub: StubPhaseStateMachine | None = None) -> PhaseStateMachineRuntime:
    return PhaseStateMachineRuntime(stub or StubPhaseStateMachine())  # type: ignore[arg-type]


def make_options(
    audit: CapturingAuditLog,
    errors: list[IndexerFailure] | None = None,
) -> PhaseStateMachineOptions:
    return PhaseStateMachineOptions(
        audit_log=audit,
        on_error=(lambda e: errors.append(e)) if errors is not None else None,
    )


# ===========================================================================
# PhaseStateMachine — transition table property tests
# ===========================================================================


@pytest.mark.parametrize(
    "phase,event_type,expected",
    [
        ("created", "start", "running"),
        ("created", "pause", "created"),
        ("created", "resume", "created"),
        ("created", "terminate", "terminated"),
        ("running", "start", "running"),
        ("running", "pause", "paused"),
        ("running", "resume", "running"),
        ("running", "terminate", "terminated"),
        ("paused", "start", "paused"),
        ("paused", "pause", "paused"),
        ("paused", "resume", "running"),
        ("paused", "terminate", "terminated"),
        ("terminated", "start", "terminated"),
        ("terminated", "pause", "terminated"),
        ("terminated", "resume", "terminated"),
        ("terminated", "terminate", "terminated"),
    ],
)
class TestPhaseStateMachineTransitions:
    def test_deterministic_derivation(self, phase: str, event_type: str, expected: str) -> None:
        machine = PhaseStateMachine()
        assert machine.next_phase(phase, event_type) == expected
        assert machine.next_phase(phase, event_type) == expected


@pytest.mark.parametrize("event_type", sorted(EVENTS))
class TestPhaseStateMachineTerminatedAbsorbing:
    def test_terminated_is_final(self, event_type: str) -> None:
        assert PhaseStateMachine().next_phase("terminated", event_type) == "terminated"


class TestPhaseStateMachineInvalidInputs:
    def test_unknown_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown phase"):
            PhaseStateMachine().next_phase("unknown_phase", "start")

    def test_unknown_event_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown event type"):
            PhaseStateMachine().next_phase("created", "unknown_event")

    def test_phases_constant_matches_transition_table(self) -> None:
        for phase in PHASES:
            for event_type in EVENTS:
                result = PhaseStateMachine().next_phase(phase, event_type)
                assert result in PHASES, f"next_phase({phase!r}, {event_type!r}) = {result!r} not in PHASES"

    def test_terminated_not_reachable_from_itself_except_via_terminate_events(self) -> None:
        machine = PhaseStateMachine()
        for event_type in EVENTS:
            assert machine.next_phase("terminated", event_type) == "terminated"


# ===========================================================================
# PhaseStateMachineRuntime
# ===========================================================================


class TestPhaseStateMachineRuntimeSuccess:
    def test_returns_next_phase(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseStateMachine(returns="paused"))
        result = rt.next_phase("s1", "running", "pause", options=make_options(audit_log))
        assert result == "paused"

    def test_span_name(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().next_phase("s1", "created", "start", options=make_options(audit_log))
        assert mock_state_machine_span.name == "phase.next_phase"

    def test_span_session_id_attribute(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().next_phase("s1", "created", "start", options=make_options(audit_log))
        assert mock_state_machine_span.attributes["phase.session_id"] == "s1"

    def test_span_current_attribute(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().next_phase("s1", "running", "pause", options=make_options(audit_log))
        assert mock_state_machine_span.attributes["phase.current"] == "running"

    def test_span_event_type_attribute(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().next_phase("s1", "running", "pause", options=make_options(audit_log))
        assert mock_state_machine_span.attributes["phase.event_type"] == "pause"

    def test_invocation_event_attached(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().next_phase("s1", "created", "start", options=make_options(audit_log))
        event_names = [e[0] for e in mock_state_machine_span.events]
        assert "phase.state_machine.invocation" in event_names

    def test_invocation_event_operation(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().next_phase("s1", "created", "start", options=make_options(audit_log))
        inv = next(
            e for e in mock_state_machine_span.events if e[0] == "phase.state_machine.invocation"
        )
        assert inv[1]["operation"] == "next_phase"

    def test_no_audit_entries_on_success(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().next_phase("s1", "created", "start", options=make_options(audit_log))
        assert audit_log.entries == []

    def test_span_ended(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().next_phase("s1", "created", "start", options=make_options(audit_log))
        assert mock_state_machine_span.ended


class TestPhaseStateMachineRuntimeIndexerFailure:
    def _make_failure(self) -> IndexerFailure:
        return IndexerFailure(
            code="PHASE_INVALID_STATE",
            message="bad phase",
            session_id="s1",
            timestamp="2024-01-01T00:00:00+00:00",
        )

    def test_re_raises_indexer_failure(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseStateMachine(raises=self._make_failure()))
        with pytest.raises(IndexerFailure) as exc_info:
            rt.next_phase("s1", "created", "start", options=make_options(audit_log))
        assert exc_info.value.code == "PHASE_INVALID_STATE"

    def test_audit_entry_written(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseStateMachine(raises=self._make_failure()))
        with pytest.raises(IndexerFailure):
            rt.next_phase("s1", "created", "start", options=make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "phase.next_phase.failed"

    def test_span_marked_error(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseStateMachine(raises=self._make_failure()))
        with pytest.raises(IndexerFailure):
            rt.next_phase("s1", "created", "start", options=make_options(audit_log))
        assert mock_state_machine_span.status.status_code == StatusCode.ERROR

    def test_span_ended_on_failure(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseStateMachine(raises=self._make_failure()))
        with pytest.raises(IndexerFailure):
            rt.next_phase("s1", "created", "start", options=make_options(audit_log))
        assert mock_state_machine_span.ended


class TestPhaseStateMachineRuntimeUnexpectedException:
    def test_wraps_as_phase_next_phase_failed(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseStateMachine(raises=ValueError("unknown phase")))
        with pytest.raises(IndexerFailure) as exc_info:
            rt.next_phase("s1", "bad", "start", options=make_options(audit_log))
        assert exc_info.value.code == "PHASE_NEXT_PHASE_FAILED"

    def test_cause_preserved(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = ValueError("unknown phase")
        rt = make_runtime(StubPhaseStateMachine(raises=orig))
        with pytest.raises(IndexerFailure) as exc_info:
            rt.next_phase("s1", "bad", "start", options=make_options(audit_log))
        assert exc_info.value.cause is orig

    def test_audit_entry_written(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseStateMachine(raises=ValueError("boom")))
        with pytest.raises(IndexerFailure):
            rt.next_phase("s1", "bad", "start", options=make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "phase.next_phase.failed"
        assert entry.session_id == "s1"

    def test_state_machine_error_event_on_span(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseStateMachine(raises=ValueError("boom")))
        with pytest.raises(IndexerFailure):
            rt.next_phase("s1", "bad", "start", options=make_options(audit_log))
        event_names = [e[0] for e in mock_state_machine_span.events]
        assert "phase.state_machine.error" in event_names

    def test_exception_recorded_on_span(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = ValueError("boom")
        rt = make_runtime(StubPhaseStateMachine(raises=orig))
        with pytest.raises(IndexerFailure):
            rt.next_phase("s1", "bad", "start", options=make_options(audit_log))
        assert orig in mock_state_machine_span.recorded_exceptions

    def test_on_error_callback(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[IndexerFailure] = []
        rt = make_runtime(StubPhaseStateMachine(raises=ValueError("boom")))
        with pytest.raises(IndexerFailure):
            rt.next_phase("s1", "bad", "start", options=make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "PHASE_NEXT_PHASE_FAILED"

    def test_span_marked_error(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseStateMachine(raises=ValueError("boom")))
        with pytest.raises(IndexerFailure):
            rt.next_phase("s1", "bad", "start", options=make_options(audit_log))
        assert mock_state_machine_span.status.status_code == StatusCode.ERROR

    def test_span_ended_on_failure(
        self, mock_state_machine_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime(StubPhaseStateMachine(raises=ValueError("boom")))
        with pytest.raises(IndexerFailure):
            rt.next_phase("s1", "bad", "start", options=make_options(audit_log))
        assert mock_state_machine_span.ended
