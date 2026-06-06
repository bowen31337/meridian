"""Covers record_invocation_event's non-scalar attribute filter — the
isinstance-False branch the all-string runtime events never exercise."""

from __future__ import annotations

from sdk_environment._telemetry import record_invocation_event
from sdk_environment._types import StructuredEvent

from .conftest import MockSpan


def test_record_invocation_event_skips_non_scalar() -> None:
    span = MockSpan()
    event = StructuredEvent(
        name="environment.invocation",
        environment_id="e1",
        environment_kind="local",
        session_id="s1",
        timestamp=None,  # type: ignore[arg-type]
        operation="execute",
    )
    record_invocation_event(span, event)  # type: ignore[arg-type]
    _name, attrs = span.events[0]
    assert "timestamp" not in attrs
    assert attrs["environment_id"] == "e1"
