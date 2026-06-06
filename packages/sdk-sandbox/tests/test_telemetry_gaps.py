"""Covers record_invocation_event's non-scalar attribute filter — the
isinstance-False branch the all-string runtime events never exercise."""

from __future__ import annotations

from sdk_sandbox._telemetry import record_invocation_event
from sdk_sandbox._types import StructuredEvent

from .conftest import MockSpan


def test_record_invocation_event_skips_non_scalar() -> None:
    span = MockSpan()
    event = StructuredEvent(
        name="sandbox.invocation",
        tool_name="kb_search",
        session_id="s1",
        timestamp=None,  # type: ignore[arg-type]
        operation="execute",
    )
    record_invocation_event(span, event)  # type: ignore[arg-type]
    _name, attrs = span.events[0]
    assert "timestamp" not in attrs
    assert attrs["tool_name"] == "kb_search"
