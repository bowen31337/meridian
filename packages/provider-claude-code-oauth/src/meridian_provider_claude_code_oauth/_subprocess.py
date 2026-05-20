"""Claude Code CLI subprocess lifecycle manager.

Architecture §15.1 — Subprocess Protocol
=========================================
The manager maintains a single long-running Claude Code CLI process.  Model
calls are serialized via an async lock (one call at a time).  A background
health-check task pings the process between calls and restarts it when the
process stops responding.

stdin  → {"type": "call",  "id": "<hex12>", "model": "...", "messages": [...], ...}\\n
stdout ← {"type": "message_start",   "call_id": "...", "model": "...", "input_tokens": N}\\n
         {"type": "text_delta",       "call_id": "...", "text": "..."}\\n
         {"type": "thinking_delta",   "call_id": "...", "thinking": "..."}\\n
         {"type": "tool_use_start",   "call_id": "...", "id": "...", "name": "..."}\\n
         {"type": "tool_input_delta", "call_id": "...", "id": "...", "partial_json": "..."}\\n
         {"type": "message_stop",     "call_id": "...", "input_tokens": N, "output_tokens": M,
                                      "stop_reason": "..."}\\n
         {"type": "done",             "call_id": "..."}\\n
      OR {"type": "error",            "call_id": "...", "code": "...", "message": "..."}\\n

Health-check:
    stdin  → {"type": "ping", "id": "<hex12>"}\\n
    stdout ← {"type": "pong", "id": "<hex12>"}\\n

Shutdown:
    stdin  → {"type": "shutdown"}\\n

Lifecycle events (Architecture §15.2):
  - spawn:            process started (initial or after restart)
  - health_check_ok:  pong received within health_timeout_s
  - health_check_fail:no pong → SIGTERM → grace → SIGKILL → respawn
  - call_timeout:     readline timed out → SIGTERM → grace → SIGKILL → respawn
  - call_cancelled:   CancelledError → SIGTERM → grace → SIGKILL → respawn
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from meridian_sdk_provider.errors import ProviderCallError, ProviderTimeoutError
from meridian_sdk_provider.types import (
    MessageStartEvent,
    MessageStopEvent,
    ModelCallOpts,
    ModelEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolInputDeltaEvent,
    ToolUseStartEvent,
)

_MAX_STDERR_BYTES = 64 * 1024
_SIGKILL_GRACE_S = 2.0


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class CliSubprocessError(ProviderCallError):
    """CLI subprocess crashed, closed stdout unexpectedly, or emitted invalid JSON."""

    def __init__(self, message: str, *, provider_name: str = "claude_code_oauth") -> None:
        super().__init__(message, provider_name=provider_name)


class CliCallTimeoutError(ProviderTimeoutError):
    """A model call to the CLI subprocess timed out (no event within call_timeout_s)."""

    def __init__(self, message: str, *, provider_name: str = "claude_code_oauth") -> None:
        super().__init__(message, provider_name=provider_name)


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------


def _call_id() -> str:
    return uuid.uuid4().hex[:12]


def _opts_to_dict(opts: ModelCallOpts) -> dict[str, Any]:
    """Serialize ModelCallOpts to a JSON-compatible dict for the subprocess protocol."""
    messages: list[dict[str, Any]] = []
    for msg in opts.messages:
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            blocks: list[dict[str, Any]] = []
            for block in content:
                btype = block.type
                if btype == "text":
                    b: dict[str, Any] = {"type": "text", "text": block.text}  # type: ignore[union-attr]
                    if getattr(block, "cache_control", None) is not None:
                        b["cache_control"] = {"type": "ephemeral"}
                    blocks.append(b)
                elif btype == "tool_use":
                    blocks.append({
                        "type": "tool_use",
                        "id": block.id,  # type: ignore[union-attr]
                        "name": block.name,  # type: ignore[union-attr]
                        "input": block.input,  # type: ignore[union-attr]
                    })
                elif btype == "tool_result":
                    raw = block.content  # type: ignore[union-attr]
                    tb: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,  # type: ignore[union-attr]
                    }
                    if isinstance(raw, str):
                        tb["content"] = raw
                    else:
                        tb["content"] = [{"type": "text", "text": str(c)} for c in raw]
                    blocks.append(tb)
                elif btype == "thinking":
                    blocks.append({
                        "type": "thinking",
                        "thinking": block.thinking,  # type: ignore[union-attr]
                        "signature": block.signature,  # type: ignore[union-attr]
                    })
            messages.append({"role": msg.role, "content": blocks})

    d: dict[str, Any] = {
        "model": opts.model,
        "messages": messages,
        "max_tokens": opts.max_tokens,
    }
    if opts.system:
        d["system"] = opts.system
    if opts.tools:
        d["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in opts.tools
        ]
    if opts.temperature is not None:
        d["temperature"] = opts.temperature
    if opts.enable_thinking and opts.thinking_budget_tokens:
        d["thinking"] = {"type": "enabled", "budget_tokens": opts.thinking_budget_tokens}
    if opts.session_id:
        d["session_id"] = opts.session_id
    return d


def _parse_event(msg: dict[str, Any], provider_name: str) -> ModelEvent | None:
    """Translate one line of subprocess stdout into a ModelEvent, or None to skip."""
    t = msg.get("type")
    if t == "message_start":
        return MessageStartEvent(
            type="message_start",
            model=msg.get("model", ""),
            provider=provider_name,
            input_tokens=msg.get("input_tokens"),
        )
    if t == "text_delta":
        return TextDeltaEvent(type="text_delta", text=msg.get("text", ""))
    if t == "thinking_delta":
        return ThinkingDeltaEvent(type="thinking_delta", thinking=msg.get("thinking", ""))
    if t == "tool_use_start":
        return ToolUseStartEvent(
            type="tool_use_start",
            id=msg.get("id", ""),
            name=msg.get("name", ""),
        )
    if t == "tool_input_delta":
        return ToolInputDeltaEvent(
            type="tool_input_delta",
            id=msg.get("id", ""),
            partial_json=msg.get("partial_json", ""),
        )
    if t == "message_stop":
        return MessageStopEvent(
            type="message_stop",
            input_tokens=msg.get("input_tokens"),
            output_tokens=msg.get("output_tokens"),
            stop_reason=msg.get("stop_reason"),
        )
    return None


# ---------------------------------------------------------------------------
# Subprocess lifecycle manager
# ---------------------------------------------------------------------------


class CliSubprocessManager:
    """Manages a long-running Claude Code CLI subprocess.

    Lifecycle contract
    ------------------
    * ``start()``:  spawn the process and begin the background health-check loop.
    * ``stop()``:   send shutdown, SIGTERM the process, and cancel the health loop.
    * ``call()``:   async generator that streams :class:`ModelEvent` objects.

    Health-check loop (background task)
    ------------------------------------
    Every *health_interval_s* seconds, if no call is active, the manager sends a
    ``{"type": "ping"}`` line and waits up to *health_timeout_s* for a pong.
    On failure (timeout or dead process) it kills with SIGTERM → *sigkill_grace_s*
    grace → SIGKILL and then respawns.

    Call error handling
    --------------------
    * Timeout (no event for *call_timeout_s*): kill + respawn, raise
      :class:`CliCallTimeoutError`.
    * ``asyncio.CancelledError``: kill + respawn, re-raise.
    * Subprocess crash / bad JSON: raise :class:`CliSubprocessError`.
    """

    def __init__(
        self,
        cli_path: str,
        cli_version: str,
        *,
        provider_name: str = "claude_code_oauth",
        health_interval_s: float = 30.0,
        health_timeout_s: float = 5.0,
        call_timeout_s: float = 120.0,
        sigkill_grace_s: float = _SIGKILL_GRACE_S,
    ) -> None:
        self._cli_path = cli_path
        self._cli_version = cli_version
        self._provider_name = provider_name
        self._health_interval_s = health_interval_s
        self._health_timeout_s = health_timeout_s
        self._call_timeout_s = call_timeout_s
        self._sigkill_grace_s = sigkill_grace_s

        self._proc: asyncio.subprocess.Process | None = None
        self._call_lock = asyncio.Lock()
        self._call_active = False
        self._health_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn the subprocess and start the background health-check loop."""
        await self._spawn()
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(
                self._health_loop(), name="cli-health-check"
            )

    async def stop(self) -> None:
        """Gracefully stop the subprocess and cancel the health-check loop."""
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None
        await self._kill()

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        """Stream model events from the subprocess (one call at a time).

        Kills and respawns the subprocess on timeout or cancellation so the
        next call starts with a clean process state.
        """
        async with self._call_lock:
            self._call_active = True
            aborted = False
            try:
                await self._ensure_alive()
                call_id = _call_id()
                payload = {"type": "call", "id": call_id, **_opts_to_dict(opts)}
                self._write_line(json.dumps(payload))
                await self._flush_stdin()
                async for event in self._read_events(call_id):
                    yield event
            except asyncio.CancelledError:
                aborted = True
                raise
            except CliCallTimeoutError:
                aborted = True
                raise
            finally:
                self._call_active = False
                if aborted:
                    await self._kill_and_spawn()

    # ------------------------------------------------------------------
    # Internal — I/O helpers
    # ------------------------------------------------------------------

    def _write_line(self, line: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((line + "\n").encode())

    async def _flush_stdin(self) -> None:
        if self._proc is not None and self._proc.stdin is not None:
            await self._proc.stdin.drain()

    async def _read_events(self, call_id: str) -> AsyncIterator[ModelEvent]:
        """Yield ModelEvents until the subprocess signals done (or error)."""
        assert self._proc is not None and self._proc.stdout is not None

        while True:
            try:
                raw = await asyncio.wait_for(
                    self._proc.stdout.readline(),
                    timeout=self._call_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                raise CliCallTimeoutError(
                    f"Call {call_id!r} timed out after {self._call_timeout_s}s "
                    f"waiting for next event from subprocess",
                    provider_name=self._provider_name,
                ) from exc

            if not raw:
                raise CliSubprocessError(
                    "CLI subprocess closed stdout unexpectedly during call",
                    provider_name=self._provider_name,
                )

            try:
                msg: dict[str, Any] = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError as exc:
                raise CliSubprocessError(
                    f"CLI subprocess emitted invalid JSON: {exc}",
                    provider_name=self._provider_name,
                ) from exc

            msg_type = msg.get("type")

            if msg_type == "done":
                return

            if msg_type == "error":
                code = msg.get("code", "cli_error")
                message = msg.get("message", "unknown error from subprocess")
                raise CliSubprocessError(
                    f"{code}: {message}",
                    provider_name=self._provider_name,
                )

            event = _parse_event(msg, self._provider_name)
            if event is not None:
                yield event

    # ------------------------------------------------------------------
    # Internal — process lifecycle
    # ------------------------------------------------------------------

    async def _ensure_alive(self) -> None:
        """Spawn a fresh process if none is running."""
        if self._proc is None or self._proc.returncode is not None:
            await self._spawn()

    async def _spawn(self) -> None:
        """Start a new subprocess; no-op if one is already alive."""
        if self._proc is not None and self._proc.returncode is None:
            return
        self._proc = await asyncio.create_subprocess_exec(
            self._cli_path,
            "--server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _kill(self) -> None:
        """Terminate the subprocess: send shutdown, SIGTERM, then SIGKILL after grace."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            self._proc = None
            return

        # Attempt graceful shutdown via protocol
        try:
            self._write_line(json.dumps({"type": "shutdown"}))
            await self._flush_stdin()
        except Exception:
            pass

        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._sigkill_grace_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        finally:
            self._proc = None

    async def _kill_and_spawn(self) -> None:
        """Kill the current subprocess and immediately spawn a replacement."""
        await self._kill()
        await self._spawn()

    # ------------------------------------------------------------------
    # Internal — health-check loop
    # ------------------------------------------------------------------

    async def _health_loop(self) -> None:
        """Background task: ping the subprocess every health_interval_s."""
        while True:
            await asyncio.sleep(self._health_interval_s)
            if self._call_active:
                # A call owns the process right now; skip this cycle.
                continue
            await self._do_health_check()

    async def _do_health_check(self) -> None:
        """Send a ping and wait for a pong; restart on any failure."""
        if self._proc is None or self._proc.returncode is not None:
            await self._spawn()
            return

        ping_id = _call_id()
        try:
            self._write_line(json.dumps({"type": "ping", "id": ping_id}))
            await self._flush_stdin()

            raw = await asyncio.wait_for(
                self._proc.stdout.readline(),  # type: ignore[union-attr]
                timeout=self._health_timeout_s,
            )
            if raw:
                msg = json.loads(raw.decode("utf-8", errors="replace"))
                if msg.get("type") == "pong" and msg.get("id") == ping_id:
                    return  # healthy
        except Exception:
            pass

        # Health check failed — restart
        await self._kill_and_spawn()
