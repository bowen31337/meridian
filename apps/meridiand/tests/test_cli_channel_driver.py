"""
Tests for the CLI channel driver (meridian.cli).

Covers:
  - Driver capabilities: kind, can_send_text, can_thread, max_message_length.
  - Outbound send (happy path): text/plain with and without trailing newline,
    token-stream JSON, tool-call JSON with and without id, assistant_prefix.
  - Outbound send (failures): missing config, bad token-stream JSON, bad
    tool-call JSON, stdout write exception.
  - Error surface: error line written to stdout on failure.
  - Audit log: written on failure with correct event, level, and code.
  - OTel spans: emitted on success and error, attributes, invocation event.
  - start(): config validated, stdin reader task started, idempotent re-start.
  - stop(): stdin reader task cancelled, stop() is a no-op when no task.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from meridiand._cli_channel_driver import (
    CLI_TOKEN_STREAM_CONTENT_TYPE,
    CLI_TOOL_CALL_CONTENT_TYPE,
    CliChannelDriver,
    NoopStdinReaderClient,
)
from opentelemetry.trace import StatusCode
from sdk_channel import (
    ChannelCapabilities,
    ChannelFailure,
    SendRequest,
    StartRequest,
    StopRequest,
)

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CLI_KIND = "meridian.cli"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CapturingStdoutWriter:
    """Records all write() calls; never raises."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.flush_count: int = 0

    def write(self, text: str) -> None:
        self.chunks.append(text)

    def flush(self) -> None:
        self.flush_count += 1

    @property
    def output(self) -> str:
        return "".join(self.chunks)


class FailingStdoutWriter:
    """Raises on write() to simulate a broken stdout."""

    def write(self, text: str) -> None:
        raise OSError("broken pipe")

    def flush(self) -> None:
        pass


class CapturingAuditLog:
    def __init__(self) -> None:
        self.entries: list[Any] = []

    def write(self, entry: Any) -> None:
        self.entries.append(entry)


def _make_channel_file(
    storage_root: Path,
    *,
    channel_id: str = "ch_cli_1",
    assistant_prefix: str | None = None,
    egress_policy: str = "enabled",
) -> str:
    channels_dir = storage_root / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"token_vault_ref": "vault/tok"}
    if assistant_prefix is not None:
        config["assistant_prefix"] = assistant_prefix
    record: dict[str, Any] = {
        "id": channel_id,
        "kind": _CLI_KIND,
        "config": config,
        "egress_policy": egress_policy,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    (channels_dir / f"{channel_id}.json").write_text(json.dumps(record))
    return channel_id


def _make_send_request(
    channel_id: str = "ch_cli_1",
    *,
    content: str = "hello cli",
    content_type: str = "text/plain",
) -> SendRequest:
    return SendRequest(
        channel_id=channel_id,
        channel_kind=_CLI_KIND,
        session_id="sess_test",
        recipient="tty-user",
        content=content,
        content_type=content_type,
    )


def _make_driver(
    storage_root: Path,
    *,
    stdout_writer: Any = None,
    stdin_reader_client: Any = None,
    audit_log: Any = None,
) -> CliChannelDriver:
    return CliChannelDriver(
        storage_root=storage_root,
        audit_log=audit_log,
        stdout_writer=stdout_writer,
        stdin_reader_client=stdin_reader_client,
    )


# ---------------------------------------------------------------------------
# Driver: capabilities
# ---------------------------------------------------------------------------


class TestDriverCapabilities:
    def test_kind_is_meridian_cli(self, storage_root: Path) -> None:
        assert _make_driver(storage_root).kind == "meridian.cli"

    def test_capabilities_returns_channel_capabilities(self, storage_root: Path) -> None:
        caps = _make_driver(storage_root).capabilities()
        assert isinstance(caps, ChannelCapabilities)

    def test_capabilities_can_send_text(self, storage_root: Path) -> None:
        assert _make_driver(storage_root).capabilities().can_send_text is True

    def test_capabilities_cannot_thread(self, storage_root: Path) -> None:
        assert _make_driver(storage_root).capabilities().can_thread is False

    def test_capabilities_no_max_message_length(self, storage_root: Path) -> None:
        assert _make_driver(storage_root).capabilities().max_message_length is None


# ---------------------------------------------------------------------------
# Driver: outbound send — happy path
# ---------------------------------------------------------------------------


class TestDriverSendSuccess:
    @pytest.fixture(autouse=True)
    def _otel_clear(self) -> None:
        _otel_exporter.clear()

    async def test_plain_text_written_to_stdout(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request(content="hello world")
        )
        assert "hello world" in writer.output

    async def test_plain_text_gets_trailing_newline(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request(content="no newline")
        )
        assert writer.output.endswith("\n")

    async def test_plain_text_already_ending_with_newline_not_doubled(
        self, storage_root: Path
    ) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request(content="has newline\n")
        )
        assert writer.output == "has newline\n"

    async def test_stdout_flushed_after_plain_text(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request(content="flush me")
        )
        assert writer.flush_count >= 1

    async def test_result_delivered_true(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        result = await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request()
        )
        assert result.delivered is True

    async def test_result_message_id_starts_with_cli(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        result = await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request()
        )
        assert result.message_id.startswith("cli_")

    async def test_result_has_timestamp(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        result = await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request()
        )
        assert isinstance(result.timestamp, str) and len(result.timestamp) > 0

    async def test_token_stream_tokens_written_in_order(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        tokens = ["Hello", ", ", "world", "!"]
        await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request(
                content=json.dumps(tokens),
                content_type=CLI_TOKEN_STREAM_CONTENT_TYPE,
            )
        )
        assert "Hello, world!" in writer.output

    async def test_token_stream_flushed_per_token(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        tokens = ["a", "b", "c"]
        await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request(
                content=json.dumps(tokens),
                content_type=CLI_TOKEN_STREAM_CONTENT_TYPE,
            )
        )
        assert writer.flush_count >= len(tokens)

    async def test_token_stream_ends_with_newline(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request(
                content=json.dumps(["tok"]),
                content_type=CLI_TOKEN_STREAM_CONTENT_TYPE,
            )
        )
        assert writer.output.endswith("\n")

    async def test_tool_call_rendered_inline(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        call = {"name": "read_file", "input": {"path": "/etc/hosts"}, "id": "call_abc"}
        await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request(
                content=json.dumps(call),
                content_type=CLI_TOOL_CALL_CONTENT_TYPE,
            )
        )
        assert "[tool: read_file(" in writer.output
        assert "id=call_abc" in writer.output

    async def test_tool_call_without_id_omits_id_field(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        call = {"name": "list_dir", "input": {"path": "/"}}
        await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request(
                content=json.dumps(call),
                content_type=CLI_TOOL_CALL_CONTENT_TYPE,
            )
        )
        assert "id=" not in writer.output
        assert "[tool: list_dir(" in writer.output

    async def test_tool_call_ends_with_newline(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        call = {"name": "ping", "input": {}}
        await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request(
                content=json.dumps(call),
                content_type=CLI_TOOL_CALL_CONTENT_TYPE,
            )
        )
        assert writer.output.endswith("\n")

    async def test_assistant_prefix_written_before_content(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root, assistant_prefix="Assistant: ")
        await _make_driver(storage_root, stdout_writer=writer).send(
            _make_send_request(content="hi")
        )
        assert writer.output.startswith("Assistant: ")

    async def test_emits_otel_span(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        await _make_driver(storage_root, stdout_writer=writer).send(_make_send_request())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "cli.channel.send" in span_names

    async def test_span_has_channel_id_attribute(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root, channel_id="ch_cli_attr")
        req = SendRequest(
            channel_id="ch_cli_attr",
            channel_kind=_CLI_KIND,
            session_id="sess_s",
            recipient="r",
            content="c",
        )
        await _make_driver(storage_root, stdout_writer=writer).send(req)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("cli.channel.send")
        assert span is not None
        assert span.attributes["channel.id"] == "ch_cli_attr"

    async def test_span_has_invocation_event(self, storage_root: Path) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        await _make_driver(storage_root, stdout_writer=writer).send(_make_send_request())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("cli.channel.send")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names


# ---------------------------------------------------------------------------
# Driver: outbound send — failures
# ---------------------------------------------------------------------------


class TestDriverSendFailures:
    @pytest.fixture(autouse=True)
    def _otel_clear(self) -> None:
        _otel_exporter.clear()

    async def test_missing_config_raises_chan_config_not_found(
        self, storage_root: Path
    ) -> None:
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.send(
                SendRequest(
                    channel_id="ch_nonexistent",
                    channel_kind=_CLI_KIND,
                    session_id="s",
                    recipient="r",
                    content="c",
                )
            )
        assert exc_info.value.code == "CHAN_CONFIG_NOT_FOUND"

    async def test_invalid_token_stream_json_raises_chan_send_failed(
        self, storage_root: Path
    ) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await _make_driver(storage_root, stdout_writer=writer).send(
                _make_send_request(
                    content="not-json",
                    content_type=CLI_TOKEN_STREAM_CONTENT_TYPE,
                )
            )
        assert exc_info.value.code == "CHAN_SEND_FAILED"

    async def test_token_stream_non_array_raises_chan_send_failed(
        self, storage_root: Path
    ) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await _make_driver(storage_root, stdout_writer=writer).send(
                _make_send_request(
                    content='{"not": "array"}',
                    content_type=CLI_TOKEN_STREAM_CONTENT_TYPE,
                )
            )
        assert exc_info.value.code == "CHAN_SEND_FAILED"

    async def test_invalid_tool_call_json_raises_chan_send_failed(
        self, storage_root: Path
    ) -> None:
        writer = CapturingStdoutWriter()
        _make_channel_file(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await _make_driver(storage_root, stdout_writer=writer).send(
                _make_send_request(
                    content="bad json{",
                    content_type=CLI_TOOL_CALL_CONTENT_TYPE,
                )
            )
        assert exc_info.value.code == "CHAN_SEND_FAILED"

    async def test_stdout_write_exception_raises_chan_send_failed(
        self, storage_root: Path
    ) -> None:
        _make_channel_file(storage_root)
        driver = _make_driver(storage_root, stdout_writer=FailingStdoutWriter())
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.send(_make_send_request(content="anything"))
        assert exc_info.value.code == "CHAN_SEND_FAILED"

    async def test_failure_writes_error_line_to_stdout(self, storage_root: Path) -> None:
        chunks: list[str] = []

        class PartialWriter:
            """Fails on first write, succeeds on subsequent writes."""

            call_count = 0

            def write(self, text: str) -> None:
                self.call_count += 1
                if self.call_count == 1:
                    raise OSError("broken pipe")
                chunks.append(text)

            def flush(self) -> None:
                pass

        _make_channel_file(storage_root)
        with pytest.raises(ChannelFailure):
            await _make_driver(storage_root, stdout_writer=PartialWriter()).send(
                _make_send_request(content="fail me")
            )
        error_output = "".join(chunks)
        assert "meridian error" in error_output

    async def test_failure_writes_audit_log(self, storage_root: Path) -> None:
        log = CapturingAuditLog()
        _make_channel_file(storage_root)
        with pytest.raises(ChannelFailure):
            await _make_driver(
                storage_root, stdout_writer=FailingStdoutWriter(), audit_log=log
            ).send(_make_send_request())
        assert len(log.entries) == 1
        entry = log.entries[0]
        assert entry.level == "error"
        assert entry.event == "cli.channel.send.failed"
        assert entry.code == "CHAN_SEND_FAILED"

    async def test_failure_span_marked_error(self, storage_root: Path) -> None:
        _make_channel_file(storage_root)
        with pytest.raises(ChannelFailure):
            await _make_driver(storage_root, stdout_writer=FailingStdoutWriter()).send(
                _make_send_request()
            )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("cli.channel.send")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Driver: start() — stdin reader lifecycle
# ---------------------------------------------------------------------------


class TestDriverStart:
    async def test_start_calls_stdin_reader_run(self, storage_root: Path) -> None:
        run_called = asyncio.Event()

        class TrackingStdinReaderClient:
            async def run(self) -> None:
                run_called.set()

            async def stop(self) -> None:
                pass

        _make_channel_file(storage_root)
        driver = _make_driver(storage_root, stdin_reader_client=TrackingStdinReaderClient())
        await driver.start(
            StartRequest(channel_id="ch_cli_1", channel_kind=_CLI_KIND, session_id="s")
        )
        await asyncio.sleep(0)
        assert run_called.is_set()

    async def test_start_idempotent_does_not_duplicate_task(
        self, storage_root: Path
    ) -> None:
        run_count = 0

        class CountingStdinReaderClient:
            async def run(self) -> None:
                nonlocal run_count
                run_count += 1
                await asyncio.sleep(10)

            async def stop(self) -> None:
                pass

        _make_channel_file(storage_root)
        driver = _make_driver(
            storage_root, stdin_reader_client=CountingStdinReaderClient()
        )
        req = StartRequest(channel_id="ch_cli_1", channel_kind=_CLI_KIND, session_id="s1")
        await driver.start(req)
        await asyncio.sleep(0)
        await driver.start(
            StartRequest(channel_id="ch_cli_1", channel_kind=_CLI_KIND, session_id="s2")
        )
        await asyncio.sleep(0)
        assert run_count == 1
        for task in driver._read_tasks.values():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def test_start_missing_config_raises_chan_config_not_found(
        self, storage_root: Path
    ) -> None:
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.start(
                StartRequest(
                    channel_id="ch_nonexistent",
                    channel_kind=_CLI_KIND,
                    session_id="s",
                )
            )
        assert exc_info.value.code == "CHAN_CONFIG_NOT_FOUND"


# ---------------------------------------------------------------------------
# Driver: stop() — task lifecycle
# ---------------------------------------------------------------------------


class TestDriverStop:
    async def test_stop_cancels_reader_task(self, storage_root: Path) -> None:
        connected = asyncio.Event()
        cancelled = asyncio.Event()

        class LongRunningStdinReaderClient:
            async def run(self) -> None:
                connected.set()
                try:
                    await asyncio.sleep(999)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            async def stop(self) -> None:
                pass

        _make_channel_file(storage_root)
        driver = _make_driver(
            storage_root, stdin_reader_client=LongRunningStdinReaderClient()
        )
        await driver.start(
            StartRequest(channel_id="ch_cli_1", channel_kind=_CLI_KIND, session_id="s")
        )
        await connected.wait()
        await driver.stop(
            StopRequest(channel_id="ch_cli_1", channel_kind=_CLI_KIND, session_id="s")
        )
        assert cancelled.is_set()
        assert "ch_cli_1" not in driver._read_tasks

    async def test_stop_noop_when_no_task(self, storage_root: Path) -> None:
        driver = _make_driver(storage_root, stdin_reader_client=NoopStdinReaderClient())
        await driver.stop(
            StopRequest(channel_id="ch_cli_1", channel_kind=_CLI_KIND, session_id="s")
        )

    async def test_stop_calls_stdin_reader_stop(self, storage_root: Path) -> None:
        stop_called = asyncio.Event()

        class TrackingStdinReaderClient:
            async def run(self) -> None:
                pass

            async def stop(self) -> None:
                stop_called.set()

        _make_channel_file(storage_root)
        driver = _make_driver(
            storage_root, stdin_reader_client=TrackingStdinReaderClient()
        )
        await driver.stop(
            StopRequest(channel_id="ch_cli_1", channel_kind=_CLI_KIND, session_id="s")
        )
        assert stop_called.is_set()
