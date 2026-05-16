"""
Channel Driver conformance suite.

Every implementation of ChannelDriver must satisfy these tests when
exercised through ChannelRuntime. The suite covers:

  - Successful start / send / stop: span emitted, invocation event attached,
    no audit entries, correct results returned.
  - Unknown kind (CHAN_KIND_NOT_REGISTERED): ChannelFailure raised, audit
    entry written at level "error", span status set to ERROR.
  - Driver exceptions (CHAN_START_FAILED / CHAN_SEND_FAILED / CHAN_STOP_FAILED):
    wrapped in ChannelFailure with cause, audit entry written, span marked
    ERROR, on_error callback called.
  - capabilities() retrieval.
  - Duplicate registration guard.
  - on_error callback invocation.
  - Span lifecycle: span ended on both success and failure paths.
  - Manifest loading and validation.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sdk_channel import (
    AuditLogEntry,
    ChannelCapabilities,
    ChannelDriver,
    ChannelFailure,
    ChannelManifest,
    ChannelRuntime,
    RuntimeOptions,
    SendRequest,
    SendResult,
    StartRequest,
    StopRequest,
    load_manifest,
    validate_manifest,
)
from opentelemetry.trace import StatusCode

from .conftest import CapturingAuditLog, MockSpan


# ---------------------------------------------------------------------------
# Stub driver
# ---------------------------------------------------------------------------

class StubDriver(ChannelDriver):
    kind = "test.stub"

    def __init__(
        self,
        *,
        start_raises: Exception | None = None,
        send_raises: Exception | None = None,
        stop_raises: Exception | None = None,
    ) -> None:
        self._start_raises = start_raises
        self._send_raises = send_raises
        self._stop_raises = stop_raises
        self.starts: list[StartRequest] = []
        self.sends: list[SendRequest] = []
        self.stops: list[StopRequest] = []

    async def start(self, request: StartRequest) -> None:
        if self._start_raises:
            raise self._start_raises
        self.starts.append(request)

    async def send(self, request: SendRequest) -> SendResult:
        if self._send_raises:
            raise self._send_raises
        self.sends.append(request)
        return SendResult(message_id="msg-1", timestamp="2026-01-01T00:00:00Z", delivered=True)

    async def stop(self, request: StopRequest) -> None:
        if self._stop_raises:
            raise self._stop_raises
        self.stops.append(request)

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            can_send_text=True,
            can_send_files=True,
            can_thread=True,
            rate_limit_per_minute=60,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_start(kind: str = "test.stub") -> StartRequest:
    return StartRequest(channel_id="chan-1", channel_kind=kind, session_id="sess-1")


def make_send(kind: str = "test.stub") -> SendRequest:
    return SendRequest(
        channel_id="chan-1",
        channel_kind=kind,
        session_id="sess-1",
        recipient="user-42",
        content="hello",
    )


def make_stop(kind: str = "test.stub") -> StopRequest:
    return StopRequest(channel_id="chan-1", channel_kind=kind, session_id="sess-1")


def make_options(audit: CapturingAuditLog, errors: list[ChannelFailure] | None = None) -> RuntimeOptions:
    return RuntimeOptions(
        audit_log=audit,
        on_error=(lambda e: errors.append(e)) if errors is not None else None,
    )


def registered_runtime() -> ChannelRuntime:
    rt = ChannelRuntime()
    rt.register(StubDriver())
    return rt


# ---------------------------------------------------------------------------
# start — success
# ---------------------------------------------------------------------------

class TestStartSuccess:
    async def test_dispatches_to_driver(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver()
        rt = ChannelRuntime()
        rt.register(driver)
        await rt.start(make_start(), make_options(audit_log))
        assert len(driver.starts) == 1

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().start(make_start(), make_options(audit_log))
        assert mock_span.name == "channel.start"

    async def test_span_attributes(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().start(make_start(), make_options(audit_log))
        assert mock_span.attributes["channel.id"] == "chan-1"
        assert mock_span.attributes["channel.kind"] == "test.stub"
        assert mock_span.attributes["session.id"] == "sess-1"

    async def test_invocation_event_attached(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().start(make_start(), make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "channel.invocation" in event_names

    async def test_invocation_event_operation(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().start(make_start(), make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "channel.invocation")
        assert inv[1]["operation"] == "start"

    async def test_no_audit_entries_on_success(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().start(make_start(), make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().start(make_start(), make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# start — CHAN_KIND_NOT_REGISTERED
# ---------------------------------------------------------------------------

class TestStartUnknownKind:
    async def test_raises_channel_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = ChannelRuntime()
        with pytest.raises(ChannelFailure) as exc_info:
            await rt.start(make_start("acme.unknown"), make_options(audit_log))
        assert exc_info.value.code == "CHAN_KIND_NOT_REGISTERED"

    async def test_audit_entry_written(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = ChannelRuntime()
        with pytest.raises(ChannelFailure):
            await rt.start(make_start("acme.unknown"), make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "channel.start.failed"

    async def test_span_marked_error(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = ChannelRuntime()
        with pytest.raises(ChannelFailure):
            await rt.start(make_start("acme.unknown"), make_options(audit_log))
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_error_event_on_span(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = ChannelRuntime()
        with pytest.raises(ChannelFailure):
            await rt.start(make_start("acme.unknown"), make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "channel.error" in event_names

    async def test_on_error_callback(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        errors: list[ChannelFailure] = []
        rt = ChannelRuntime()
        with pytest.raises(ChannelFailure):
            await rt.start(make_start("acme.unknown"), make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "CHAN_KIND_NOT_REGISTERED"

    async def test_span_ended_on_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = ChannelRuntime()
        with pytest.raises(ChannelFailure):
            await rt.start(make_start("acme.unknown"), make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# start — driver raises
# ---------------------------------------------------------------------------

class TestStartDriverRaises:
    async def test_wraps_as_start_failed(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(start_raises=RuntimeError("connection refused"))
        rt = ChannelRuntime()
        rt.register(driver)
        with pytest.raises(ChannelFailure) as exc_info:
            await rt.start(make_start(), make_options(audit_log))
        assert exc_info.value.code == "CHAN_START_FAILED"

    async def test_cause_preserved(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = RuntimeError("connection refused")
        driver = StubDriver(start_raises=orig)
        rt = ChannelRuntime()
        rt.register(driver)
        with pytest.raises(ChannelFailure) as exc_info:
            await rt.start(make_start(), make_options(audit_log))
        assert exc_info.value.cause is orig

    async def test_audit_entry_written(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(start_raises=RuntimeError("boom"))
        rt = ChannelRuntime()
        rt.register(driver)
        with pytest.raises(ChannelFailure):
            await rt.start(make_start(), make_options(audit_log))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].level == "error"

    async def test_span_marked_error(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(start_raises=RuntimeError("boom"))
        rt = ChannelRuntime()
        rt.register(driver)
        with pytest.raises(ChannelFailure):
            await rt.start(make_start(), make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_exception_recorded_on_span(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = RuntimeError("boom")
        driver = StubDriver(start_raises=orig)
        rt = ChannelRuntime()
        rt.register(driver)
        with pytest.raises(ChannelFailure):
            await rt.start(make_start(), make_options(audit_log))
        assert orig in mock_span.recorded_exceptions


# ---------------------------------------------------------------------------
# send — success
# ---------------------------------------------------------------------------

class TestSendSuccess:
    async def test_returns_result(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        result = await registered_runtime().send(make_send(), make_options(audit_log))
        assert result.message_id == "msg-1"
        assert result.delivered is True

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().send(make_send(), make_options(audit_log))
        assert mock_span.name == "channel.send"

    async def test_invocation_event_operation(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().send(make_send(), make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "channel.invocation")
        assert inv[1]["operation"] == "send"

    async def test_no_audit_entries_on_success(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().send(make_send(), make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().send(make_send(), make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# send — CHAN_KIND_NOT_REGISTERED
# ---------------------------------------------------------------------------

class TestSendUnknownKind:
    async def test_raises_channel_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = ChannelRuntime()
        with pytest.raises(ChannelFailure) as exc_info:
            await rt.send(make_send("acme.unknown"), make_options(audit_log))
        assert exc_info.value.code == "CHAN_KIND_NOT_REGISTERED"

    async def test_audit_event_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = ChannelRuntime()
        with pytest.raises(ChannelFailure):
            await rt.send(make_send("acme.unknown"), make_options(audit_log))
        assert audit_log.entries[0].event == "channel.send.failed"

    async def test_span_marked_error(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = ChannelRuntime()
        with pytest.raises(ChannelFailure):
            await rt.send(make_send("acme.unknown"), make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# send — driver raises
# ---------------------------------------------------------------------------

class TestSendDriverRaises:
    async def test_wraps_as_send_failed(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(send_raises=RuntimeError("rate limited"))
        rt = ChannelRuntime()
        rt.register(driver)
        with pytest.raises(ChannelFailure) as exc_info:
            await rt.send(make_send(), make_options(audit_log))
        assert exc_info.value.code == "CHAN_SEND_FAILED"

    async def test_cause_preserved(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = RuntimeError("rate limited")
        driver = StubDriver(send_raises=orig)
        rt = ChannelRuntime()
        rt.register(driver)
        with pytest.raises(ChannelFailure) as exc_info:
            await rt.send(make_send(), make_options(audit_log))
        assert exc_info.value.cause is orig

    async def test_audit_entry_written(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(send_raises=RuntimeError("boom"))
        rt = ChannelRuntime()
        rt.register(driver)
        with pytest.raises(ChannelFailure):
            await rt.send(make_send(), make_options(audit_log))
        assert len(audit_log.entries) == 1


# ---------------------------------------------------------------------------
# stop — success
# ---------------------------------------------------------------------------

class TestStopSuccess:
    async def test_dispatches_to_driver(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver()
        rt = ChannelRuntime()
        rt.register(driver)
        await rt.stop(make_stop(), make_options(audit_log))
        assert len(driver.stops) == 1

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().stop(make_stop(), make_options(audit_log))
        assert mock_span.name == "channel.stop"

    async def test_invocation_event_operation(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().stop(make_stop(), make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "channel.invocation")
        assert inv[1]["operation"] == "stop"

    async def test_no_audit_entries_on_success(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().stop(make_stop(), make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().stop(make_stop(), make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# stop — CHAN_KIND_NOT_REGISTERED
# ---------------------------------------------------------------------------

class TestStopUnknownKind:
    async def test_raises_channel_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = ChannelRuntime()
        with pytest.raises(ChannelFailure) as exc_info:
            await rt.stop(make_stop("acme.unknown"), make_options(audit_log))
        assert exc_info.value.code == "CHAN_KIND_NOT_REGISTERED"

    async def test_audit_event_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = ChannelRuntime()
        with pytest.raises(ChannelFailure):
            await rt.stop(make_stop("acme.unknown"), make_options(audit_log))
        assert audit_log.entries[0].event == "channel.stop.failed"

    async def test_span_marked_error(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = ChannelRuntime()
        with pytest.raises(ChannelFailure):
            await rt.stop(make_stop("acme.unknown"), make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# stop — driver raises
# ---------------------------------------------------------------------------

class TestStopDriverRaises:
    async def test_wraps_as_stop_failed(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(stop_raises=RuntimeError("still connected"))
        rt = ChannelRuntime()
        rt.register(driver)
        with pytest.raises(ChannelFailure) as exc_info:
            await rt.stop(make_stop(), make_options(audit_log))
        assert exc_info.value.code == "CHAN_STOP_FAILED"

    async def test_audit_entry_written(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(stop_raises=RuntimeError("boom"))
        rt = ChannelRuntime()
        rt.register(driver)
        with pytest.raises(ChannelFailure):
            await rt.stop(make_stop(), make_options(audit_log))
        assert len(audit_log.entries) == 1


# ---------------------------------------------------------------------------
# capabilities()
# ---------------------------------------------------------------------------

class TestDriverCapabilities:
    def test_capabilities_registered(self) -> None:
        rt = registered_runtime()
        caps = rt.capabilities("test.stub")
        assert isinstance(caps, ChannelCapabilities)
        assert caps.can_send_text is True
        assert caps.can_send_files is True
        assert caps.can_thread is True
        assert caps.rate_limit_per_minute == 60

    def test_capabilities_unknown_kind(self) -> None:
        rt = ChannelRuntime()
        with pytest.raises(ChannelFailure) as exc_info:
            rt.capabilities("acme.unknown")
        assert exc_info.value.code == "CHAN_KIND_NOT_REGISTERED"


# ---------------------------------------------------------------------------
# Registry guard
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_duplicate_registration_raises(self) -> None:
        rt = ChannelRuntime()
        rt.register(StubDriver())
        with pytest.raises(ValueError, match="already registered"):
            rt.register(StubDriver())

    def test_get_returns_driver(self) -> None:
        rt = ChannelRuntime()
        driver = StubDriver()
        rt.register(driver)
        assert rt.get("test.stub") is driver

    def test_get_returns_none_for_unknown(self) -> None:
        rt = ChannelRuntime()
        assert rt.get("acme.unknown") is None


# ---------------------------------------------------------------------------
# Manifest — load_manifest
# ---------------------------------------------------------------------------

class TestLoadManifest:
    def _write_manifest(self, tmp_path: Path, data: dict) -> Path:
        (tmp_path / "channel.json").write_text(json.dumps(data))
        return tmp_path

    def test_loads_valid_manifest(self, tmp_path: Path) -> None:
        self._write_manifest(tmp_path, {
            "kind": "meridian.slack",
            "version": "1.0.0",
            "display_name": "Slack",
            "platforms": ["linux", "darwin"],
            "auth_schemes": ["oauth2"],
            "capabilities": {"can_send_files": True, "can_thread": True},
        })
        manifest = load_manifest(tmp_path)
        assert manifest.kind == "meridian.slack"
        assert manifest.version == "1.0.0"
        assert manifest.display_name == "Slack"
        assert "linux" in manifest.platforms
        assert manifest.capabilities.can_send_files is True
        assert manifest.capabilities.can_thread is True

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ChannelFailure) as exc_info:
            load_manifest(tmp_path)
        assert exc_info.value.code == "CHAN_MANIFEST_INVALID"
        assert "not found" in exc_info.value.message

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / "channel.json").write_text("{not valid json")
        with pytest.raises(ChannelFailure) as exc_info:
            load_manifest(tmp_path)
        assert exc_info.value.code == "CHAN_MANIFEST_INVALID"

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        self._write_manifest(tmp_path, {"kind": "meridian.slack", "version": "1.0.0"})
        with pytest.raises(ChannelFailure) as exc_info:
            load_manifest(tmp_path)
        assert exc_info.value.code == "CHAN_MANIFEST_INVALID"

    def test_capability_defaults(self, tmp_path: Path) -> None:
        self._write_manifest(tmp_path, {
            "kind": "meridian.sms",
            "version": "0.1.0",
            "display_name": "SMS",
            "platforms": ["linux"],
        })
        manifest = load_manifest(tmp_path)
        assert manifest.capabilities.can_send_text is True
        assert manifest.capabilities.can_send_files is False

    def test_rate_limit_per_minute(self, tmp_path: Path) -> None:
        self._write_manifest(tmp_path, {
            "kind": "meridian.slack",
            "version": "1.0.0",
            "display_name": "Slack",
            "platforms": ["linux"],
            "rate_limit_per_minute": 30,
        })
        manifest = load_manifest(tmp_path)
        assert manifest.rate_limit_per_minute == 30


# ---------------------------------------------------------------------------
# Manifest — validate_manifest
# ---------------------------------------------------------------------------

class TestValidateManifest:
    def _base(self) -> ChannelManifest:
        return ChannelManifest(
            kind="meridian.slack",
            version="1.0.0",
            display_name="Slack",
            platforms=("linux",),
            auth_schemes=("oauth2",),
            capabilities=ChannelCapabilities(),
        )

    def test_valid_manifest_passes(self) -> None:
        validate_manifest(self._base())

    def test_empty_kind_raises(self) -> None:
        m = ChannelManifest(
            kind="",
            version="1.0.0",
            display_name="Slack",
            platforms=("linux",),
            auth_schemes=(),
            capabilities=ChannelCapabilities(),
        )
        with pytest.raises(ChannelFailure) as exc_info:
            validate_manifest(m)
        assert exc_info.value.code == "CHAN_MANIFEST_INVALID"
        assert "kind" in exc_info.value.message

    def test_empty_platforms_raises(self) -> None:
        m = ChannelManifest(
            kind="meridian.slack",
            version="1.0.0",
            display_name="Slack",
            platforms=(),
            auth_schemes=(),
            capabilities=ChannelCapabilities(),
        )
        with pytest.raises(ChannelFailure) as exc_info:
            validate_manifest(m)
        assert exc_info.value.code == "CHAN_MANIFEST_INVALID"
        assert "platforms" in exc_info.value.message
