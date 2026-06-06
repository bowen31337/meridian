"""Unit coverage for storage-blob telemetry, noop audit, and BlobFailure re-raise paths."""

from __future__ import annotations

import pytest
from opentelemetry import trace
from storage_blob import BlobFailure, BlobRuntime
from storage_blob._audit import NoopAuditLog
from storage_blob._telemetry import get_tracer, record_invocation_event
from storage_blob._types import AuditLogEntry, StructuredEvent

from .conftest import CapturingAuditLog, MockSpan
from .test_conformance import StubBlobStore, make_options


def test_get_tracer_returns_tracer() -> None:
    assert isinstance(get_tracer(), trace.Tracer)


def test_record_invocation_event_skips_non_scalar() -> None:
    span = MockSpan()
    event = StructuredEvent(
        name="blob.invocation",
        key="k1",
        timestamp=None,  # type: ignore[arg-type]
        operation="put",
    )
    record_invocation_event(span, event)  # type: ignore[arg-type]
    _name, attrs = span.events[0]
    assert "timestamp" not in attrs
    assert attrs["key"] == "k1"


def test_noop_audit_log_write_is_silent() -> None:
    entry = AuditLogEntry(
        level="error",
        event="test.event",
        key="k1",
        timestamp="2026-01-01T00:00:00Z",
    )
    assert NoopAuditLog().write(entry) is None


def _failure() -> BlobFailure:
    return BlobFailure(
        code="BLOB_CUSTOM",
        message="store said no",
        key="k1",
        timestamp="2026-01-01T00:00:00Z",
    )


async def test_put_reraises_blob_failure(
    mock_span: MockSpan, audit_log: CapturingAuditLog
) -> None:
    orig = _failure()
    rt = BlobRuntime(StubBlobStore(put_raises=orig))
    with pytest.raises(BlobFailure) as exc_info:
        await rt.put("k1", b"x", make_options(audit_log))
    assert exc_info.value is orig
    assert any(e.event == "blob.put.failed" for e in audit_log.entries)


async def test_delete_reraises_blob_failure(
    mock_span: MockSpan, audit_log: CapturingAuditLog
) -> None:
    orig = _failure()
    rt = BlobRuntime(StubBlobStore(delete_raises=orig))
    with pytest.raises(BlobFailure) as exc_info:
        await rt.delete("k1", make_options(audit_log))
    assert exc_info.value is orig
    assert any(e.event == "blob.delete.failed" for e in audit_log.entries)
