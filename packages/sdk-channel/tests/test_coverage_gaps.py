"""Unit coverage for sdk-channel telemetry, noop audit, manifest, and re-raise paths."""

from __future__ import annotations

import pytest
from opentelemetry import trace
from sdk_channel import (
    ChannelCapabilities,
    ChannelFailure,
    ChannelManifest,
    ChannelRuntime,
    validate_manifest,
)
from sdk_channel._audit import NoopAuditLog
from sdk_channel._telemetry import get_tracer
from sdk_channel._types import AuditLogEntry

from .conftest import CapturingAuditLog, MockSpan
from .test_conformance import StubDriver, make_options, make_send, make_start, make_stop


def test_get_tracer_returns_tracer() -> None:
    assert isinstance(get_tracer(), trace.Tracer)


def test_noop_audit_log_write_is_silent() -> None:
    entry = AuditLogEntry(
        level="error",
        event="test.event",
        channel_id="c1",
        channel_kind="test.stub",
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
    )
    assert NoopAuditLog().write(entry) is None


def test_validate_manifest_empty_version_raises() -> None:
    m = ChannelManifest(
        kind="meridian.slack",
        version="",
        display_name="Slack",
        platforms=("linux",),
        auth_schemes=(),
        capabilities=ChannelCapabilities(),
    )
    with pytest.raises(ChannelFailure) as exc_info:
        validate_manifest(m)
    assert "version" in exc_info.value.message


def test_validate_manifest_empty_display_name_raises() -> None:
    m = ChannelManifest(
        kind="meridian.slack",
        version="1.0.0",
        display_name="",
        platforms=("linux",),
        auth_schemes=(),
        capabilities=ChannelCapabilities(),
    )
    with pytest.raises(ChannelFailure) as exc_info:
        validate_manifest(m)
    assert "display_name" in exc_info.value.message


def _failure(code: str) -> ChannelFailure:
    return ChannelFailure(
        code=code,
        message="driver said no",
        channel_id="chan-1",
        channel_kind="test.stub",
        session_id="sess-1",
        timestamp="2026-01-01T00:00:00Z",
    )


class TestDriverRaisesChannelFailureReraised:
    async def test_start_reraises_channel_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = _failure("CHAN_CUSTOM")
        rt = ChannelRuntime()
        rt.register(StubDriver(start_raises=orig))
        with pytest.raises(ChannelFailure) as exc_info:
            await rt.start(make_start(), make_options(audit_log))
        assert exc_info.value is orig

    async def test_send_reraises_channel_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = _failure("CHAN_CUSTOM")
        rt = ChannelRuntime()
        rt.register(StubDriver(send_raises=orig))
        with pytest.raises(ChannelFailure) as exc_info:
            await rt.send(make_send(), make_options(audit_log))
        assert exc_info.value is orig

    async def test_stop_reraises_channel_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = _failure("CHAN_CUSTOM")
        rt = ChannelRuntime()
        rt.register(StubDriver(stop_raises=orig))
        with pytest.raises(ChannelFailure) as exc_info:
            await rt.stop(make_stop(), make_options(audit_log))
        assert exc_info.value is orig
