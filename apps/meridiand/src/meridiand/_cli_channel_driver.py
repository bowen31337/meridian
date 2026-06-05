"""
CLI Channel Driver: meridian.cli

Synchronous TTY interface.  Inbound: a background asyncio task calls
StdinReaderClient.run() on start(), running a stdin read loop until stop()
cancels it.  The StdinReaderClient protocol abstracts the blocking read loop;
inject NoopStdinReaderClient (or a mock) for unit tests.

Outbound (send()): writes assistant content blocks to stdout via the injected
StdoutWriter.  The StdoutWriter protocol abstracts sys.stdout; inject
CapturingStdoutWriter (or any mock) for unit tests.

Supported outbound content types:
  text/plain
    Write content verbatim, append a newline if absent, then flush.

  application/vnd.meridian.token-stream+json
    Payload is a JSON array of str tokens.  Each token is written to stdout
    and flushed immediately for a live streaming appearance.  A single
    trailing newline is appended after the last token.

  application/vnd.meridian.tool-call+json
    Payload is a JSON object with "name" (str), "input" (dict), and an
    optional "id" (str) field.  Rendered inline as:
      [tool: name({"k":"v"}) id=call_xyz]
    followed by a newline.

On failure the driver writes an error line to stdout so the terminal user
sees it immediately, then raises ChannelFailure and appends an audit-log
entry before re-raising.

Every send() emits an OTel span "cli.channel.send" with channel/session
attributes and a structured invocation event.  The span is marked ERROR and
the exception is recorded on failure.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any, Protocol, runtime_checkable
import uuid

from core_errors import (
    AuditLog,
    AuditLogEntry,
    NoopAuditLog,
    StructuredEvent,
    get_tracer,
    record_invocation_event,
)
from opentelemetry.trace import StatusCode
from sdk_channel import (
    ChannelCapabilities,
    ChannelDriver,
    ChannelFailure,
    SendRequest,
    SendResult,
    StartRequest,
    StopRequest,
)

CLI_TOKEN_STREAM_CONTENT_TYPE = "application/vnd.meridian.token-stream+json"
CLI_TOOL_CALL_CONTENT_TYPE = "application/vnd.meridian.tool-call+json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class StdoutWriter(Protocol):
    """Abstracts sys.stdout for testability."""

    def write(self, text: str) -> None: ...

    def flush(self) -> None: ...


@runtime_checkable
class StdinReaderClient(Protocol):
    """
    Abstracts the blocking stdin read loop.

    The driver calls run() in a background task started by start().  The
    implementation reads lines from stdin and dispatches them as inbound
    events.  Inject NoopStdinReaderClient (or a mock) for unit tests.
    """

    async def run(self) -> None:
        """Block reading stdin until cancelled or EOF."""
        ...

    async def stop(self) -> None:
        """Signal the read loop to stop gracefully."""
        ...


# ---------------------------------------------------------------------------
# Default implementations
# ---------------------------------------------------------------------------


class SysStdoutWriter:
    """Writes to sys.stdout directly."""

    def write(self, text: str) -> None:
        sys.stdout.write(text)

    def flush(self) -> None:
        sys.stdout.flush()


class NoopStdinReaderClient:
    """No-op stdin client; performs no I/O."""

    async def run(self) -> None:
        pass

    async def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class CliChannelDriver(ChannelDriver):
    """
    CLI channel driver (kind: meridian.cli).

    Channel config fields (nested under "config" in the channel record):
        assistant_prefix  (optional) String written to stdout before each
                          outbound message.  Defaults to empty string.
    """

    kind = "meridian.cli"

    def __init__(
        self,
        *,
        storage_root: Path,
        audit_log: AuditLog | None = None,
        stdout_writer: StdoutWriter | None = None,
        stdin_reader_client: StdinReaderClient | None = None,
    ) -> None:
        self._storage_root = storage_root
        self._audit_log = audit_log or NoopAuditLog()
        self._stdout_writer: StdoutWriter = stdout_writer or SysStdoutWriter()
        self._stdin_reader_client: StdinReaderClient = (
            stdin_reader_client or NoopStdinReaderClient()
        )
        self._read_tasks: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_driver_config(
        self, channel_id: str, channel_kind: str, session_id: str
    ) -> dict[str, Any]:
        channel_file = self._storage_root / "channels" / f"{channel_id}.json"
        if not channel_file.exists():
            raise ChannelFailure(
                code="CHAN_CONFIG_NOT_FOUND",
                message=f"Channel config not found for '{channel_id}'",
                channel_id=channel_id,
                channel_kind=channel_kind,
                session_id=session_id,
                timestamp=_now(),
            )
        return json.loads(channel_file.read_text()).get("config", {})

    def _write_content(self, request: SendRequest, driver_config: dict[str, Any]) -> None:
        prefix: str = driver_config.get("assistant_prefix", "")
        if prefix:
            self._stdout_writer.write(prefix)
            self._stdout_writer.flush()

        if request.content_type == CLI_TOKEN_STREAM_CONTENT_TYPE:
            self._write_token_stream(request.content)
        elif request.content_type == CLI_TOOL_CALL_CONTENT_TYPE:
            self._write_tool_call(request.content)
        else:
            text = request.content
            self._stdout_writer.write(text)
            if not text.endswith("\n"):
                self._stdout_writer.write("\n")
            self._stdout_writer.flush()

    def _write_token_stream(self, content: str) -> None:
        try:
            tokens: list[Any] = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid token stream JSON: {exc}") from exc
        if not isinstance(tokens, list):
            raise ValueError("Token stream content must be a JSON array")
        for token in tokens:
            if isinstance(token, str):
                self._stdout_writer.write(token)
                self._stdout_writer.flush()
        self._stdout_writer.write("\n")
        self._stdout_writer.flush()

    def _write_tool_call(self, content: str) -> None:
        try:
            call: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tool call JSON: {exc}") from exc
        if not isinstance(call, dict):
            raise ValueError("Tool call content must be a JSON object")
        name: str = call.get("name", "unknown")
        input_data: Any = call.get("input", {})
        call_id: str = call.get("id", "")
        input_json = json.dumps(input_data, separators=(",", ":"))
        if call_id:
            line = f"[tool: {name}({input_json}) id={call_id}]\n"
        else:
            line = f"[tool: {name}({input_json})]\n"
        self._stdout_writer.write(line)
        self._stdout_writer.flush()

    # ------------------------------------------------------------------
    # ChannelDriver interface
    # ------------------------------------------------------------------

    async def start(self, request: StartRequest) -> None:
        """
        Validate channel config and start the background stdin reader task.

        The task is idempotent: a second start() for the same channel_id
        while the task is still running is a no-op.
        """
        self._load_driver_config(request.channel_id, request.channel_kind, request.session_id)

        existing = self._read_tasks.get(request.channel_id)
        if existing is not None and not existing.done():
            return

        src = self._stdin_reader_client

        async def _run_read() -> None:
            await src.run()

        task: asyncio.Task[None] = asyncio.ensure_future(_run_read())
        self._read_tasks[request.channel_id] = task

    async def send(self, request: SendRequest) -> SendResult:
        """
        Write an assistant message to stdout.

        Renders plain text, token streams, and tool-call blocks according to
        content_type.  Emits OTel span "cli.channel.send"; on failure marks
        the span ERROR, writes an error line to stdout so the terminal user
        sees it, appends an audit-log entry, and raises ChannelFailure.
        """
        driver_config = self._load_driver_config(
            request.channel_id, request.channel_kind, request.session_id
        )

        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "cli.channel.send",
            attributes={
                "channel.id": request.channel_id,
                "channel.kind": request.channel_kind,
                "session.id": request.session_id,
                "cli.content_type": request.content_type,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="cli.channel.send.invocation",
                    code="cli_channel_send",
                    timestamp=now,
                ),
            )

            try:
                self._write_content(request, driver_config)

            except ChannelFailure:
                raise

            except Exception as exc:
                failure = ChannelFailure(
                    code="CHAN_SEND_FAILED",
                    message=f"CLI write to stdout failed: {exc}",
                    channel_id=request.channel_id,
                    channel_kind=request.channel_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    cause=exc,
                )
                span.set_status(StatusCode.ERROR, failure.message)
                span.record_exception(exc)
                try:
                    self._stdout_writer.write(
                        f"\n[meridian error: {failure.code} — {failure.message}]\n"
                    )
                    self._stdout_writer.flush()
                except Exception:
                    pass
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="cli.channel.send.failed",
                        code=failure.code,
                        timestamp=failure.timestamp,
                        detail={
                            "channel_id": request.channel_id,
                            "session_id": request.session_id,
                            "message": failure.message,
                        },
                    )
                )
                raise failure from exc

        return SendResult(
            message_id=f"cli_{uuid.uuid4().hex}",
            timestamp=now,
            delivered=True,
        )

    async def stop(self, request: StopRequest) -> None:
        """Cancel the background stdin reader task for this channel."""
        task = self._read_tasks.pop(request.channel_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self._stdin_reader_client.stop()

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            can_send_text=True,
            can_thread=False,
            max_message_length=None,
        )
