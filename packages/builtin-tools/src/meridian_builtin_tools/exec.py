"""exec — System built-in tool for running shell commands in the workspace environment.

Runs an arbitrary shell command with cwd set to the session workspace root and
returns stdout, stderr, and the exit code.

Command execution
-----------------
* **command** — Shell command string passed to ``/bin/sh -c`` (the POSIX shell).
* **timeout** — Optional per-call timeout in seconds (default 30, max 120).

Output limits
-------------
stdout and stderr are each capped at ``_MAX_OUTPUT_BYTES`` bytes; if either
stream exceeds the cap the returned string is truncated and ``truncated`` is set
to ``True``.

Capability
-----------
Requires ``exec.shell``.

Workspace confinement
----------------------
The subprocess is launched with ``cwd=ctx.workspace`` so relative paths in the
command resolve inside the workspace.  The sandbox profile enforced by the
platform provides the OS-level confinement boundary.

Error handling
--------------
If the command cannot be started the tool returns ``ToolResult(is_error=True)``;
the SDK execution pipeline writes the failure to the audit log
(Architecture §22.4).  Non-zero exit codes are **not** treated as errors — the
caller inspects ``exit_code`` to determine success.  Timeout is surfaced through
the ``timed_out`` field rather than as an error.
"""

from __future__ import annotations

import asyncio
from typing import Any

from meridian_sdk_tool import ToolContext, meridian_tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_SECONDS = 30
_MAX_TIMEOUT_SECONDS = 120
_MAX_OUTPUT_BYTES = 1 * 1024 * 1024  # 1 MiB per stream

# ---------------------------------------------------------------------------
# JSON Schema for tool I/O
# ---------------------------------------------------------------------------

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["command"],
    "properties": {
        "command": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Shell command to execute. Passed to /bin/sh -c; supports "
                "pipes, redirection, and all POSIX shell constructs."
            ),
        },
        "timeout": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": _MAX_TIMEOUT_SECONDS,
            "description": (
                f"Maximum seconds to wait for the command to complete "
                f"(default {_DEFAULT_TIMEOUT_SECONDS}, max {_MAX_TIMEOUT_SECONDS}). "
                "The process is killed if the timeout is exceeded."
            ),
        },
    },
    "additionalProperties": False,
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["stdout", "stderr", "exit_code", "command", "timed_out", "truncated"],
    "properties": {
        "stdout": {
            "type": "string",
            "description": "Captured standard output (may be truncated; see truncated field).",
        },
        "stderr": {
            "type": "string",
            "description": "Captured standard error (may be truncated; see truncated field).",
        },
        "exit_code": {
            "type": "integer",
            "description": "Process exit code (0 typically indicates success).",
        },
        "command": {
            "type": "string",
            "description": "The command as submitted.",
        },
        "timed_out": {
            "type": "boolean",
            "description": "True when the command was killed because it exceeded the timeout.",
        },
        "truncated": {
            "type": "boolean",
            "description": (
                "True when stdout or stderr exceeded the output cap and was trimmed."
            ),
        },
    },
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _run_command(
    command: str,
    workspace: str,
    timeout: float,
) -> tuple[str, str, int, bool, bool]:
    """Run *command* in a shell with cwd=*workspace*.

    Returns ``(stdout, stderr, exit_code, timed_out, truncated)``.
    Raises :class:`RuntimeError` if the subprocess cannot be started.
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to start subprocess: {exc}") from exc

    timed_out = False
    raw_stdout = b""
    raw_stderr = b""
    try:
        raw_stdout, raw_stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        timed_out = True
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()

    exit_code: int = proc.returncode if proc.returncode is not None else -1

    truncated = False
    if len(raw_stdout) > _MAX_OUTPUT_BYTES:
        raw_stdout = raw_stdout[:_MAX_OUTPUT_BYTES]
        truncated = True
    if len(raw_stderr) > _MAX_OUTPUT_BYTES:
        raw_stderr = raw_stderr[:_MAX_OUTPUT_BYTES]
        truncated = True

    stdout = raw_stdout.decode("utf-8", errors="replace")
    stderr = raw_stderr.decode("utf-8", errors="replace")

    return stdout, stderr, exit_code, timed_out, truncated


def _record_invocation(command: str, exit_code: int, timed_out: bool) -> None:
    """Attach an ``exec.invocation`` event to the active OTel span.

    Degrades gracefully when opentelemetry-api is not installed or no span is
    active in the current context.
    """
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]

        span = trace.get_current_span()
        span.add_event(
            "exec.invocation",
            {
                "exec.command_len": len(command),
                "exec.exit_code": exit_code,
                "exec.timed_out": timed_out,
            },
        )
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@meridian_tool(
    name="exec",
    description=(
        "Run a shell command in the session workspace and return stdout, stderr, and the exit code. "
        "The command is executed by /bin/sh with cwd set to the workspace root. "
        "Supports pipes, redirection, and all POSIX shell constructs. "
        "Non-zero exit codes are returned in exit_code, not treated as tool errors. "
        "The process is killed if the timeout is exceeded (default 30 s, max 120 s). "
        "Requires the exec.shell capability."
    ),
    input_schema=_INPUT_SCHEMA,
    output_schema=_OUTPUT_SCHEMA,
    capabilities=["exec.shell"],
)
async def exec_tool(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    command: str = args["command"]
    timeout: float = float(args.get("timeout", _DEFAULT_TIMEOUT_SECONDS))

    stdout, stderr, exit_code, timed_out, truncated = await _run_command(
        command=command,
        workspace=ctx.workspace,
        timeout=timeout,
    )

    _record_invocation(command, exit_code, timed_out)

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "command": command,
        "timed_out": timed_out,
        "truncated": truncated,
    }
