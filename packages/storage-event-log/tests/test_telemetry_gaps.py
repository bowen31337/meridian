"""Covers the non-scalar attribute filter in record_invocation_event and
record_fsync_event — the isinstance-False branch the all-string runtime events
never exercise."""

from __future__ import annotations

from storage_event_log._telemetry import record_fsync_event, record_invocation_event
from storage_event_log._types import StructuredEvent

from .conftest import MockSpan


def _event() -> StructuredEvent:
    return StructuredEvent(
        name="event_log.invocation",
        session_id="s1",
        timestamp=None,  # type: ignore[arg-type]
        operation="append",
    )


def test_record_invocation_event_skips_non_scalar() -> None:
    span = MockSpan()
    record_invocation_event(span, _event())  # type: ignore[arg-type]
    _name, attrs = span.events[0]
    assert "timestamp" not in attrs
    assert attrs["session_id"] == "s1"


def test_record_fsync_event_skips_non_scalar() -> None:
    span = MockSpan()
    record_fsync_event(span, _event())  # type: ignore[arg-type]
    _name, attrs = span.events[0]
    assert "timestamp" not in attrs
    assert attrs["session_id"] == "s1"
