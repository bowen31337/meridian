"""Unit coverage for the NoopAuditLog fallback."""

from __future__ import annotations

from storage_event_log._audit import NoopAuditLog
from storage_event_log._types import AuditLogEntry


def test_noop_audit_log_write_is_silent() -> None:
    entry = AuditLogEntry(
        level="error",
        event="test.event",
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
    )
    # Should accept the entry and do nothing without raising.
    assert NoopAuditLog().write(entry) is None
