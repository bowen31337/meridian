"""Out-of-process subprocess tool helper.

Implements the stdin/stdout JSON protocol defined in Architecture §11.2::

    stdin  → {"args": ..., "context": {"workspace": ..., "session_id": ..., ...}}
    stdout ← {"result": ...} | {"error": {"code": "...", "message": "..."}}
    stderr → captured, truncated to 64 KB, attached to result on crash

The subprocess is terminated with SIGTERM on timeout; if it doesn't exit
within a 2-second grace period, SIGKILL is sent (Architecture §11.4).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ._execution import execute_tool
from ._types import Capability, SubprocessHandler, ToolContext, ToolDefinition, ToolResult

_MAX_STDERR_BYTES = 64 * 1024  # 64 KB cap per Architecture §11.2
_SIGKILL_GRACE_S = 2.0


class SubprocessCrashError(RuntimeError):
    """Raised when a subprocess exits non-zero or produces unparseable output.

    Carries *stderr_tail* as a structured field so the execution layer can
    surface it separately in OTel events and error details without needing to
    parse it back out of the exception message.
    """

    def __init__(self, message: str, *, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.stderr_tail = stderr_tail


async def _call_subprocess(path: str, args: Any, ctx: ToolContext, timeout_ms: int) -> Any:
    """Spawn *path*, write args+context to stdin, parse result from stdout."""
    payload = json.dumps(
        {
            "args": args,
            "context": {
                "workspace": ctx.workspace,
                "session_id": ctx.session_id,
                "thread_id": ctx.thread_id,
                "scratch_dir": ctx.scratch_dir,
            },
        }
    ).encode()

    timeout_s = timeout_ms / 1000.0

    proc = await asyncio.create_subprocess_exec(
        path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=payload),
            timeout=timeout_s,
        )
    except TimeoutError as exc:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=_SIGKILL_GRACE_S)
        except TimeoutError:
            proc.kill()
            await proc.wait()
        raise TimeoutError(f"Subprocess tool {path!r} timed out after {timeout_ms} ms") from exc

    if proc.returncode != 0:
        stderr_text = stderr[:_MAX_STDERR_BYTES].decode("utf-8", errors="replace")
        raise SubprocessCrashError(
            f"Subprocess exited with code {proc.returncode}. stderr: {stderr_text}",
            stderr_tail=stderr_text,
        )

    try:
        response: dict[str, Any] = json.loads(stdout)
    except json.JSONDecodeError as exc:
        stderr_text = stderr[:_MAX_STDERR_BYTES].decode("utf-8", errors="replace")
        raise SubprocessCrashError(
            f"Subprocess produced invalid JSON: {exc}. stderr: {stderr_text}",
            stderr_tail=stderr_text,
        ) from exc

    if "error" in response:
        err = response["error"]
        code = err.get("code", "subprocess_error")
        message = err.get("message", "unknown error from subprocess")
        raise RuntimeError(f"{code}: {message}")

    return response.get("result")


class SubprocessTool:
    """A Meridian tool that executes an out-of-process binary."""

    def __init__(
        self,
        definition: ToolDefinition,
        audit_log_path: str | None = None,
    ) -> None:
        self.definition = definition
        self._audit_log_path = audit_log_path

    async def execute(self, args: Any, ctx: ToolContext) -> ToolResult:
        handler_def = self.definition.handler
        assert isinstance(handler_def, SubprocessHandler)

        async def _handler(a: Any, c: ToolContext) -> Any:
            return await _call_subprocess(handler_def.path, a, c, self.definition.timeout_ms)

        return await execute_tool(
            self.definition,
            args,
            ctx,
            _handler,
            audit_log_path=self._audit_log_path,
        )

    def __repr__(self) -> str:
        return f"<SubprocessTool name={self.definition.name!r}>"


def subprocess_tool(
    name: str,
    description: str,
    path: str,
    input_schema: dict[str, Any],
    output_schema: dict[str, Any] | None = None,
    capabilities: list[Capability] | None = None,
    required_environment: str | None = None,
    timeout_ms: int = 30_000,
    memory_cap_mb: int | None = None,
    audit_log_path: str | None = None,
) -> SubprocessTool:
    """Build a :class:`SubprocessTool` from a path and a schema declaration.

    The binary at *path* must implement the stdin/stdout JSON protocol:

    * Read one JSON object from stdin: ``{"args": ..., "context": {...}}``.
    * Write one JSON object to stdout: ``{"result": ...}`` on success, or
      ``{"error": {"code": "...", "message": "..."}}`` on failure.
    * Non-zero exit code is treated as an error; stderr (up to 64 KB) is
      captured and included in the error message.

    Use :mod:`meridian_sdk_tool.subprocess_server` in the subprocess for
    the symmetric server-side helper.
    """
    definition = ToolDefinition(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        capabilities=capabilities or [],
        required_environment=required_environment,
        timeout_ms=timeout_ms,
        memory_cap_mb=memory_cap_mb,
        handler=SubprocessHandler(path=path),
    )
    return SubprocessTool(definition, audit_log_path)
