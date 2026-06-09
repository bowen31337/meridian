"""Claude Code CLI subprocess manager (headless one-shot mode).

Drives the real Claude Code CLI in non-interactive mode:

    claude -p <prompt> --output-format json [--model <alias>]
           [--append-system-prompt <system>] --disallowed-tools <builtins...>

One process per model call (the CLI has no persistent server protocol). The
single JSON result object on stdout is translated into the ModelEvent stream
the Model Router consumes:

    {"result": "<text>", "stop_reason": "...",
     "usage": {"input_tokens": N, "output_tokens": M}, "is_error": false, ...}
        -> MessageStartEvent, TextDeltaEvent, MessageStopEvent

Safety: every built-in Claude Code tool (Bash, Edit, Write, ...) is passed to
``--disallowed-tools`` so an inbound chat message can never drive local code
execution through the CLI.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from meridian_sdk_provider.errors import ProviderCallError, ProviderTimeoutError
from meridian_sdk_provider.types import (
    MessageStartEvent,
    MessageStopEvent,
    ModelCallOpts,
    ModelEvent,
    TextDeltaEvent,
)

from ._disallowed_tools import ALL_CLAUDE_CODE_BUILTIN_TOOLS

_MAX_STDERR_CHARS = 2000


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class CliSubprocessError(ProviderCallError):
    """CLI subprocess crashed, exited non-zero, or emitted invalid JSON."""

    def __init__(self, message: str, *, provider_name: str = "claude_code_oauth") -> None:
        super().__init__(message, provider_name=provider_name)


class CliCallTimeoutError(ProviderTimeoutError):
    """A model call to the CLI subprocess exceeded call_timeout_s."""

    def __init__(self, message: str, *, provider_name: str = "claude_code_oauth") -> None:
        super().__init__(message, provider_name=provider_name)


class DisallowedToolError(CliSubprocessError):
    """Retained for interface compatibility; tools are blocked via --disallowed-tools."""

    def __init__(self, message: str, *, provider_name: str = "claude_code_oauth") -> None:
        super().__init__(message, provider_name=provider_name)


# ---------------------------------------------------------------------------
# Prompt + model mapping helpers
# ---------------------------------------------------------------------------


def _flatten_content(content: Any) -> str:
    """Reduce a message's content (str or block list) to plain text."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _build_prompt(opts: ModelCallOpts) -> str:
    """Serialize the conversation into a single prompt string for ``claude -p``."""
    messages = list(opts.messages)
    if len(messages) == 1:
        return _flatten_content(messages[0].content)
    return "\n\n".join(f"{m.role}: {_flatten_content(m.content)}" for m in messages)


def _map_model(model: str | None) -> str | None:
    """Map a Meridian model id to a CLI ``--model`` alias, or None for the CLI default."""
    m = (model or "").lower()
    if "opus" in m:
        return "opus"
    if "sonnet" in m:
        return "sonnet"
    if "haiku" in m:
        return "haiku"
    return None


# ---------------------------------------------------------------------------
# Subprocess manager
# ---------------------------------------------------------------------------


class CliSubprocessManager:
    """Runs one ``claude -p`` process per model call and yields ModelEvents.

    The constructor signature is kept stable for SystemOAuthProvider; the
    health-check parameters are accepted but unused (there is no persistent
    process to health-check in one-shot mode).
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
        sigkill_grace_s: float = 2.0,
    ) -> None:
        self._cli_path = cli_path
        self._cli_version = cli_version
        self._provider_name = provider_name
        self._call_timeout_s = call_timeout_s
        self._sigkill_grace_s = sigkill_grace_s

    async def start(self) -> None:
        """No-op: one-shot mode spawns a fresh process per call."""

    async def stop(self) -> None:
        """No-op: nothing persistent to tear down."""

    def _build_args(self, opts: ModelCallOpts) -> list[str]:
        args = [
            self._cli_path,
            "-p",
            _build_prompt(opts),
            "--output-format",
            "json",
            "--disallowed-tools",
            *sorted(ALL_CLAUDE_CODE_BUILTIN_TOOLS),
        ]
        model = _map_model(opts.model)
        if model is not None:
            args += ["--model", model]
        if opts.system:
            args += ["--append-system-prompt", opts.system]
        return args

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        """Spawn ``claude -p``, await its JSON result, and yield ModelEvents."""
        proc = await asyncio.create_subprocess_exec(
            *self._build_args(opts),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._call_timeout_s
            )
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise CliCallTimeoutError(
                f"claude CLI call timed out after {self._call_timeout_s}s",
                provider_name=self._provider_name,
            ) from exc
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise

        if proc.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace")[:_MAX_STDERR_CHARS].strip()
            raise CliSubprocessError(
                f"claude CLI exited with code {proc.returncode}: {detail}",
                provider_name=self._provider_name,
            )

        try:
            data: dict[str, Any] = json.loads(stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise CliSubprocessError(
                f"claude CLI emitted invalid JSON: {exc}",
                provider_name=self._provider_name,
            ) from exc

        if data.get("is_error"):
            message = data.get("result") or data.get("api_error_status") or "unknown CLI error"
            raise CliSubprocessError(
                f"claude CLI reported an error: {message}",
                provider_name=self._provider_name,
            )

        text = data.get("result", "") or ""
        usage: dict[str, Any] = data.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        model_used = next(iter(data.get("modelUsage", {})), opts.model)

        yield MessageStartEvent(
            type="message_start",
            model=model_used,
            provider=self._provider_name,
            input_tokens=input_tokens,
        )
        if text:
            yield TextDeltaEvent(type="text_delta", text=text)
        yield MessageStopEvent(
            type="message_stop",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=data.get("stop_reason"),
        )
