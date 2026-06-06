"""Unit coverage for storage-reposit telemetry, noop audit, and failure path."""

from __future__ import annotations

import pytest
from opentelemetry import trace

from storage_reposit._audit import NoopAuditLog
from storage_reposit._migration_runtime import MigrationOptions, MigrationRuntime
from storage_reposit._telemetry import get_tracer, record_invocation_event
from storage_reposit._types import AuditLogEntry, IndexerFailure, StructuredEvent

from .conftest import MockSpan


def test_get_tracer_returns_tracer() -> None:
    assert isinstance(get_tracer(), trace.Tracer)


def test_record_invocation_event_skips_non_scalar() -> None:
    span = MockSpan()
    event = StructuredEvent(
        name="reposit.invocation",
        session_id="s1",
        timestamp=None,  # type: ignore[arg-type]
        operation="apply",
    )
    record_invocation_event(span, event)  # type: ignore[arg-type]
    _name, attrs = span.events[0]
    assert "timestamp" not in attrs
    assert attrs["session_id"] == "s1"


def test_noop_audit_log_write_is_silent() -> None:
    entry = AuditLogEntry(
        level="error",
        event="test.event",
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
    )
    assert NoopAuditLog().write(entry) is None


class _RaisingStore:
    def migrate(self) -> int:
        raise IndexerFailure(
            code="MIGRATION_FAILED",
            message="boom",
            session_id="",
            timestamp="2026-01-01T00:00:00Z",
        )


def test_migrate_reraises_indexer_failure_and_audits() -> None:
    captured: list[AuditLogEntry] = []

    class _Capturing(NoopAuditLog):
        def write(self, entry: AuditLogEntry) -> None:
            captured.append(entry)

    runtime = MigrationRuntime(_RaisingStore())  # type: ignore[arg-type]
    with pytest.raises(IndexerFailure):
        runtime.migrate(options=MigrationOptions(audit_log=_Capturing()))

    assert any(e.event == "migration.migrate.failed" for e in captured)
