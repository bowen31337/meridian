"""Covers the record_invocation_event attribute-filter branch where a non
scalar value (None) is skipped, which the runtime's all-string events never
exercise."""

from __future__ import annotations

from storage_repository._telemetry import record_invocation_event
from storage_repository._types import StructuredEvent

from .conftest import MockSpan


def test_record_invocation_event_skips_non_scalar() -> None:
    span = MockSpan()
    # timestamp=None is a non-scalar value, exercising the isinstance-False
    # branch that filters it out of the emitted span attributes.
    event = StructuredEvent(
        name="repo.invocation",
        entity_type="agent",
        entity_id="a1",
        operation="get",
        timestamp=None,  # type: ignore[arg-type]
    )
    record_invocation_event(span, event)

    assert span.events
    _name, attrs = span.events[0]
    assert "timestamp" not in attrs
    assert attrs["entity_id"] == "a1"
