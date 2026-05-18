"""
Concrete ToolDispatcher implementations for all ToolHandler kinds.

Each dispatcher:
  - Opens an OTel child span named "{kind}.dispatch" with tool.name + session.id.
  - Attaches a "{kind}.dispatch" structured event on every invocation.
  - On failure: marks the span ERROR, writes an audit log entry, and returns
    SandboxResult(is_error=True) with the error message as content, so the
    orchestrator can surface it to the model as a tool_result — never raises,
    never silent.  Exception: InProcessDispatcher raises SandboxFailure when
    no handler is registered (developer misconfiguration, same class as
    DISPATCHER_KIND_NOT_REGISTERED).
  - On success: returns SandboxResult(content=result, duration_ms=...).

Handler protocols
-----------------
subprocess / container (docker-exec):
    stdin  → {"args": {...}, "context": {"workspace": ..., "session_id": ...,
                                          "scratch_dir": ...}}
    stdout ← {"result": ...}  |  {"error": {"code": "...", "message": "..."}}
    stderr → captured (≤ 64 KB), attached on crash

http:
    POST {url}  body={"args": {...}, "context": {...}}
    200         body={"result": ...}  |  {"error": {"code": ..., "message": ...}}

mcp/http (JSON-RPC 2.0 over HTTP):
    POST {server_url}  body={"jsonrpc":"2.0","id":"...","method":"tools/call",
                             "params":{"name":"...","arguments":{...}}}
    200               body={"jsonrpc":"2.0","id":"...","result":{"content":[...],"isError":false}}
                         | {"jsonrpc":"2.0","id":"...","error":{"code":...,"message":"..."}}

mcp/stdio (JSON-RPC 2.0 over newline-delimited stdin/stdout):
    Spawns command argv as subprocess.
    Performs MCP initialize / notifications/initialized handshake, then sends
    tools/call and reads the response.  Each message is one JSON line.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from opentelemetry.trace import Span, Status, StatusCode

from ._audit import AuditLog, NoopAuditLog
from ._contract import ToolDispatcher
from ._telemetry import get_tracer, record_dispatch_overhead
from ._types import (
    AuditLogEntry,
    ExecutionContext,
    SandboxFailure,
    SandboxResult,
    ToolDefinition,
)

try:
    import httpx as _httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

_MAX_STDERR_BYTES = 64 * 1024  # 64 KB — same cap as Architecture §11.2
_SIGKILL_GRACE_S = 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ms_since(start: float) -> float:
    return (time.monotonic() - start) * 1000.0


def _mark_error(span: Span, code: str, message: str, tool_name: str, session_id: str) -> None:
    span.set_status(Status(StatusCode.ERROR, message))
    span.add_event(
        "dispatch.error",
        {
            "tool.name": tool_name,
            "session.id": session_id,
            "error.code": code,
            "error.message": message,
        },
    )


def _write_audit(
    audit_log: AuditLog,
    *,
    event: str,
    code: str,
    message: str,
    tool_name: str,
    session_id: str,
    now: str,
    detail: dict[str, Any] | None = None,
    level: str = "error",
) -> None:
    audit_log.write(
        AuditLogEntry(
            level=level,  # type: ignore[arg-type]
            event=event,
            tool_name=tool_name,
            session_id=session_id,
            timestamp=now,
            detail={"code": code, "message": message, **(detail or {})},
        )
    )


def _error_result(
    span: Span,
    audit_log: AuditLog,
    *,
    audit_event: str,
    code: str,
    message: str,
    tool_name: str,
    session_id: str,
    now: str,
    detail: dict[str, Any] | None = None,
) -> SandboxResult:
    _mark_error(span, code, message, tool_name, session_id)
    _write_audit(
        audit_log,
        event=audit_event,
        code=code,
        message=message,
        tool_name=tool_name,
        session_id=session_id,
        now=now,
        detail=detail,
    )
    return SandboxResult(content=message, is_error=True, error_code=code, error_message=message)


# ---------------------------------------------------------------------------
# InProcessDispatcher
# ---------------------------------------------------------------------------


class InProcessDispatcher(ToolDispatcher):
    """
    Dispatcher for Python callables running in the host process.

    Register a coroutine function for each tool with register(tool_name, fn).
    The callable signature is:

        async def fn(input: dict[str, Any], context: ExecutionContext) -> Any
    """

    @property
    def kind(self) -> str:
        return "in_process"

    def __init__(self, *, audit_log: AuditLog | None = None) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any], ExecutionContext], Awaitable[Any]]] = {}
        self._audit_log: AuditLog = audit_log if audit_log is not None else NoopAuditLog()

    def register(
        self,
        tool_name: str,
        fn: Callable[[dict[str, Any], ExecutionContext], Awaitable[Any]],
    ) -> None:
        """Register an async callable for a tool name. Raises ValueError on duplicate."""
        if tool_name in self._handlers:
            raise ValueError(f'Handler for tool "{tool_name}" is already registered')
        self._handlers[tool_name] = fn

    async def dispatch(
        self,
        tool: ToolDefinition,
        input: dict[str, Any],
        context: ExecutionContext,
    ) -> SandboxResult:
        now = _now()
        tracer = get_tracer()
        dispatch_start = time.monotonic()
        with tracer.start_as_current_span(
            "in_process.dispatch",
            attributes={"tool.name": tool.name, "session.id": context.session_id},
        ) as span:
            span.add_event(
                "in_process.dispatch",
                {"tool.name": tool.name, "session.id": context.session_id, "timestamp": now},
            )

            fn = self._handlers.get(tool.name)
            if fn is None:
                failure = SandboxFailure(
                    code="IN_PROCESS_HANDLER_NOT_FOUND",
                    message=f'No in-process handler registered for tool "{tool.name}"',
                    tool_name=tool.name,
                    session_id=context.session_id,
                    timestamp=now,
                )
                _mark_error(span, failure.code, failure.message, tool.name, context.session_id)
                _write_audit(
                    self._audit_log,
                    event="in_process.handler.not_found",
                    code=failure.code,
                    message=failure.message,
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                )
                raise failure

            start = time.monotonic()
            try:
                result = await fn(input, context)
            except Exception as exc:
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="in_process.handler.failed",
                    code="in_process_handler_failed",
                    message=str(exc),
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                )

            overhead_ms = _ms_since(dispatch_start)
            breached = record_dispatch_overhead(span, "in_process", overhead_ms)
            if breached:
                _write_audit(
                    self._audit_log,
                    event="dispatch.overhead.target_breached",
                    code="dispatch_overhead_target_breached",
                    message=(
                        f"in_process dispatch overhead {overhead_ms:.1f}ms "
                        f"exceeded target 20.0ms"
                    ),
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                    detail={"kind": "in_process", "overhead_ms": overhead_ms, "target_ms": 20.0},
                    level="warn",
                )
            return SandboxResult(content=result, duration_ms=_ms_since(start))


# ---------------------------------------------------------------------------
# SubprocessDispatcher
# ---------------------------------------------------------------------------


class SubprocessDispatcher(ToolDispatcher):
    """
    Dispatcher for out-of-process binaries using the JSON stdin/stdout protocol.

    The binary at SubprocessHandler.path must read one JSON object from stdin
    ({"args": ..., "context": {...}}) and write one JSON object to stdout
    ({"result": ...} on success or {"error": {"code": ..., "message": ...}}
    on failure).  Non-zero exit or invalid JSON is treated as a crash.
    """

    @property
    def kind(self) -> str:
        return "subprocess"

    def __init__(self, *, audit_log: AuditLog | None = None) -> None:
        self._audit_log: AuditLog = audit_log if audit_log is not None else NoopAuditLog()

    async def dispatch(
        self,
        tool: ToolDefinition,
        input: dict[str, Any],
        context: ExecutionContext,
    ) -> SandboxResult:
        from ._types import SubprocessHandler

        now = _now()
        tracer = get_tracer()
        handler: SubprocessHandler = tool.handler
        dispatch_start = time.monotonic()

        with tracer.start_as_current_span(
            "subprocess.dispatch",
            attributes={
                "tool.name": tool.name,
                "session.id": context.session_id,
                "subprocess.path": handler.path,
            },
        ) as span:
            span.add_event(
                "subprocess.dispatch",
                {
                    "tool.name": tool.name,
                    "session.id": context.session_id,
                    "subprocess.path": handler.path,
                    "timestamp": now,
                },
            )

            payload = json.dumps(
                {
                    "args": input,
                    "context": {
                        "workspace": context.workspace,
                        "session_id": context.session_id,
                        "scratch_dir": context.scratch_dir,
                    },
                }
            ).encode()

            proc: asyncio.subprocess.Process | None = None
            start = time.monotonic()
            try:
                proc = await asyncio.create_subprocess_exec(
                    handler.path,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate(input=payload)
            except FileNotFoundError:
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="subprocess.binary.not_found",
                    code="subprocess_binary_not_found",
                    message=f'Subprocess binary not found: {handler.path!r}',
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                    detail={"path": handler.path},
                )
            except (asyncio.CancelledError, Exception) as exc:
                if proc is not None and proc.returncode is None:
                    proc.kill()
                raise exc
            finally:
                if proc is not None and proc.returncode is None:
                    proc.kill()

            if proc.returncode != 0:
                stderr_text = stderr[:_MAX_STDERR_BYTES].decode("utf-8", errors="replace")
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="subprocess.nonzero_exit",
                    code="subprocess_nonzero_exit",
                    message=(
                        f'Subprocess exited with code {proc.returncode}. stderr: {stderr_text}'
                    ),
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                    detail={"exit_code": proc.returncode, "stderr_tail": stderr_text},
                )

            try:
                response: dict[str, Any] = json.loads(stdout)
            except json.JSONDecodeError as exc:
                stderr_text = stderr[:_MAX_STDERR_BYTES].decode("utf-8", errors="replace")
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="subprocess.invalid_json",
                    code="subprocess_invalid_json",
                    message=f'Subprocess produced invalid JSON: {exc}. stderr: {stderr_text}',
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                    detail={"stderr_tail": stderr_text},
                )

            if "error" in response:
                err = response["error"]
                code = err.get("code", "subprocess_tool_error")
                message = err.get("message", "unknown error from subprocess")
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="subprocess.tool.error",
                    code=code,
                    message=message,
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                )

            overhead_ms = _ms_since(dispatch_start)
            breached = record_dispatch_overhead(span, "subprocess", overhead_ms)
            if breached:
                _write_audit(
                    self._audit_log,
                    event="dispatch.overhead.target_breached",
                    code="dispatch_overhead_target_breached",
                    message=(
                        f"subprocess dispatch overhead {overhead_ms:.1f}ms "
                        f"exceeded target 200.0ms"
                    ),
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                    detail={
                        "kind": "subprocess",
                        "overhead_ms": overhead_ms,
                        "target_ms": 200.0,
                        "path": handler.path,
                    },
                    level="warn",
                )
            return SandboxResult(content=response.get("result"), duration_ms=_ms_since(start))


# ---------------------------------------------------------------------------
# McpDispatcher
# ---------------------------------------------------------------------------


class McpDispatcher(ToolDispatcher):
    """
    Dispatcher for tools hosted on an MCP server (JSON-RPC 2.0).

    Supports two transports selected by McpHandler.transport:
      - "http": POST to McpHandler.server_url.  Requires ``httpx``;
        install with ``pip install 'meridian-sdk-sandbox[http]'``.
      - "stdio": Spawn McpHandler.command as a subprocess and communicate
        via newline-delimited JSON-RPC 2.0.  Performs the MCP
        initialize / notifications/initialized handshake before each call.

    MCP content blocks are collapsed to plain text in SandboxResult.content.
    """

    @property
    def kind(self) -> str:
        return "mcp"

    def __init__(
        self,
        *,
        timeout_s: float = 30.0,
        audit_log: AuditLog | None = None,
    ) -> None:
        self._timeout_s = timeout_s
        self._audit_log: AuditLog = audit_log if audit_log is not None else NoopAuditLog()

    def _normalize_mcp_result(
        self,
        span: Any,
        data: dict[str, Any],
        tool_name: str,
        session_id: str,
        now: str,
        start: float,
    ) -> SandboxResult:
        """Normalize a JSON-RPC tools/call response to SandboxResult."""
        if "error" in data:
            err = data["error"]
            message = err.get("message", "MCP JSON-RPC error")
            code = str(err.get("code", "mcp_rpc_error"))
            return _error_result(
                span,
                self._audit_log,
                audit_event="mcp.rpc.error",
                code=code,
                message=message,
                tool_name=tool_name,
                session_id=session_id,
                now=now,
            )

        result: dict[str, Any] = data.get("result", {})
        is_error = result.get("isError", False)
        content_blocks: list[dict[str, Any]] = result.get("content", [])
        text = "\n".join(
            b.get("text", "") for b in content_blocks if b.get("type") == "text"
        )
        content: Any = text if text else result

        if is_error:
            return _error_result(
                span,
                self._audit_log,
                audit_event="mcp.tool.error",
                code="mcp_tool_error",
                message=text or "MCP tool returned isError=true",
                tool_name=tool_name,
                session_id=session_id,
                now=now,
            )

        return SandboxResult(content=content, duration_ms=_ms_since(start))

    async def dispatch(
        self,
        tool: ToolDefinition,
        input: dict[str, Any],
        context: ExecutionContext,
    ) -> SandboxResult:
        from ._types import McpHandler

        now = _now()
        tracer = get_tracer()
        handler: McpHandler = tool.handler

        span_attrs: dict[str, Any] = {
            "tool.name": tool.name,
            "session.id": context.session_id,
            "mcp.server_url": handler.server_url,
            "mcp.tool_name": handler.tool_name,
            "mcp.transport": handler.transport,
        }
        if handler.transport == "stdio":
            span_attrs["mcp.command"] = " ".join(handler.command)

        with tracer.start_as_current_span("mcp.dispatch", attributes=span_attrs) as span:
            span.add_event(
                "mcp.dispatch",
                {
                    "tool.name": tool.name,
                    "session.id": context.session_id,
                    "mcp.server_url": handler.server_url,
                    "mcp.tool_name": handler.tool_name,
                    "mcp.transport": handler.transport,
                    "timestamp": now,
                },
            )

            # ------------------------------------------------------------------
            # stdio transport
            # ------------------------------------------------------------------
            if handler.transport == "stdio":
                from ._mcp_client import stdio_tools_call

                if not handler.command:
                    return _error_result(
                        span,
                        self._audit_log,
                        audit_event="mcp.stdio.no_command",
                        code="mcp_stdio_no_command",
                        message=(
                            "stdio McpHandler has an empty command; "
                            "set McpHandler.command to the argv for the MCP server"
                        ),
                        tool_name=tool.name,
                        session_id=context.session_id,
                        now=now,
                    )

                start = time.monotonic()
                try:
                    data = await stdio_tools_call(
                        handler.command,
                        handler.tool_name,
                        input,
                        self._timeout_s,
                    )
                except FileNotFoundError:
                    cmd_str = handler.command[0] if handler.command else ""
                    return _error_result(
                        span,
                        self._audit_log,
                        audit_event="mcp.stdio.command_not_found",
                        code="mcp_stdio_command_not_found",
                        message=f"MCP stdio command not found: {cmd_str!r}",
                        tool_name=tool.name,
                        session_id=context.session_id,
                        now=now,
                        detail={"command": list(handler.command)},
                    )
                except Exception as exc:
                    return _error_result(
                        span,
                        self._audit_log,
                        audit_event="mcp.stdio.request_failed",
                        code="mcp_stdio_request_failed",
                        message=str(exc),
                        tool_name=tool.name,
                        session_id=context.session_id,
                        now=now,
                        detail={"command": list(handler.command)},
                    )

                return self._normalize_mcp_result(
                    span, data, tool.name, context.session_id, now, start
                )

            # ------------------------------------------------------------------
            # HTTP transport (original path)
            # ------------------------------------------------------------------
            if not _HTTPX_AVAILABLE:
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="mcp.httpx.unavailable",
                    code="mcp_httpx_unavailable",
                    message=(
                        "httpx is required for MCP tools. "
                        "Install with: pip install 'meridian-sdk-sandbox[http]'"
                    ),
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                )

            rpc_id = f"{context.session_id}:{tool.name}:{now}"
            payload = {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "method": "tools/call",
                "params": {
                    "name": handler.tool_name,
                    "arguments": input,
                },
            }

            start = time.monotonic()
            try:
                async with _httpx.AsyncClient(timeout=self._timeout_s) as client:
                    response = await client.post(
                        handler.server_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    response.raise_for_status()
                    data: dict[str, Any] = response.json()
            except Exception as exc:
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="mcp.request.failed",
                    code="mcp_request_failed",
                    message=str(exc),
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                    detail={"server_url": handler.server_url},
                )

            return self._normalize_mcp_result(
                span, data, tool.name, context.session_id, now, start
            )


# ---------------------------------------------------------------------------
# HttpDispatcher
# ---------------------------------------------------------------------------


class HttpDispatcher(ToolDispatcher):
    """
    Dispatcher for tools exposed over HTTP (POST endpoint).

    POSTs {"args": ..., "context": {...}} to HttpHandler.url and expects
    {"result": ...} or {"error": {"code": ..., "message": ...}}.  Requires
    ``httpx``; install with ``pip install 'meridian-sdk-sandbox[http]'``.
    """

    @property
    def kind(self) -> str:
        return "http"

    def __init__(
        self,
        *,
        timeout_s: float = 30.0,
        audit_log: AuditLog | None = None,
    ) -> None:
        self._timeout_s = timeout_s
        self._audit_log: AuditLog = audit_log if audit_log is not None else NoopAuditLog()

    async def dispatch(
        self,
        tool: ToolDefinition,
        input: dict[str, Any],
        context: ExecutionContext,
    ) -> SandboxResult:
        from ._types import HttpHandler

        now = _now()
        tracer = get_tracer()
        handler: HttpHandler = tool.handler

        with tracer.start_as_current_span(
            "http.dispatch",
            attributes={
                "tool.name": tool.name,
                "session.id": context.session_id,
                "http.url": handler.url,
            },
        ) as span:
            span.add_event(
                "http.dispatch",
                {
                    "tool.name": tool.name,
                    "session.id": context.session_id,
                    "http.url": handler.url,
                    "timestamp": now,
                },
            )

            if not _HTTPX_AVAILABLE:
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="http.httpx.unavailable",
                    code="http_httpx_unavailable",
                    message=(
                        "httpx is required for HTTP tools. "
                        "Install with: pip install 'meridian-sdk-sandbox[http]'"
                    ),
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                )

            payload = {
                "args": input,
                "context": {
                    "workspace": context.workspace,
                    "session_id": context.session_id,
                    "scratch_dir": context.scratch_dir,
                },
            }

            start = time.monotonic()
            try:
                async with _httpx.AsyncClient(timeout=self._timeout_s) as client:
                    response = await client.post(
                        handler.url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    response.raise_for_status()
                    data: dict[str, Any] = response.json()
            except Exception as exc:
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="http.request.failed",
                    code="http_request_failed",
                    message=str(exc),
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                    detail={"url": handler.url},
                )

            if "error" in data:
                err = data["error"]
                code = err.get("code", "http_tool_error")
                message = err.get("message", "unknown error from HTTP tool")
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="http.tool.error",
                    code=code,
                    message=message,
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                )

            return SandboxResult(content=data.get("result"), duration_ms=_ms_since(start))


# ---------------------------------------------------------------------------
# ContainerDispatcher
# ---------------------------------------------------------------------------


class ContainerDispatcher(ToolDispatcher):
    """
    Dispatcher for tools running inside a container environment.

    Executes ``docker exec -i {environment_id} {entrypoint}`` and exchanges
    data via the JSON stdin/stdout protocol (same as SubprocessDispatcher).

    ContainerHandler.environment_id: running container name or ID.
    ContainerHandler.entrypoint: command path inside the container.
    """

    @property
    def kind(self) -> str:
        return "container"

    def __init__(
        self,
        *,
        docker_executable: str = "docker",
        audit_log: AuditLog | None = None,
    ) -> None:
        self._docker = docker_executable
        self._audit_log: AuditLog = audit_log if audit_log is not None else NoopAuditLog()

    async def dispatch(
        self,
        tool: ToolDefinition,
        input: dict[str, Any],
        context: ExecutionContext,
    ) -> SandboxResult:
        from ._types import ContainerHandler

        now = _now()
        tracer = get_tracer()
        handler: ContainerHandler = tool.handler
        dispatch_start = time.monotonic()

        with tracer.start_as_current_span(
            "container.dispatch",
            attributes={
                "tool.name": tool.name,
                "session.id": context.session_id,
                "container.environment_id": handler.environment_id,
                "container.entrypoint": handler.entrypoint,
            },
        ) as span:
            span.add_event(
                "container.dispatch",
                {
                    "tool.name": tool.name,
                    "session.id": context.session_id,
                    "container.environment_id": handler.environment_id,
                    "container.entrypoint": handler.entrypoint,
                    "timestamp": now,
                },
            )

            payload = json.dumps(
                {
                    "args": input,
                    "context": {
                        "workspace": context.workspace,
                        "session_id": context.session_id,
                        "scratch_dir": context.scratch_dir,
                    },
                }
            ).encode()

            proc: asyncio.subprocess.Process | None = None
            start = time.monotonic()
            try:
                proc = await asyncio.create_subprocess_exec(
                    self._docker,
                    "exec",
                    "-i",
                    handler.environment_id,
                    handler.entrypoint,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate(input=payload)
            except FileNotFoundError:
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="container.docker.not_found",
                    code="container_docker_not_found",
                    message=f'Docker executable not found: {self._docker!r}',
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                    detail={"docker_executable": self._docker},
                )
            except (asyncio.CancelledError, Exception) as exc:
                if proc is not None and proc.returncode is None:
                    proc.kill()
                raise exc
            finally:
                if proc is not None and proc.returncode is None:
                    proc.kill()

            if proc.returncode != 0:
                stderr_text = stderr[:_MAX_STDERR_BYTES].decode("utf-8", errors="replace")
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="container.nonzero_exit",
                    code="container_nonzero_exit",
                    message=(
                        f'Container exec exited with code {proc.returncode}. stderr: {stderr_text}'
                    ),
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                    detail={
                        "exit_code": proc.returncode,
                        "stderr_tail": stderr_text,
                        "environment_id": handler.environment_id,
                    },
                )

            try:
                response: dict[str, Any] = json.loads(stdout)
            except json.JSONDecodeError as exc:
                stderr_text = stderr[:_MAX_STDERR_BYTES].decode("utf-8", errors="replace")
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="container.invalid_json",
                    code="container_invalid_json",
                    message=f'Container produced invalid JSON: {exc}. stderr: {stderr_text}',
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                    detail={"stderr_tail": stderr_text},
                )

            if "error" in response:
                err = response["error"]
                code = err.get("code", "container_tool_error")
                message = err.get("message", "unknown error from container tool")
                return _error_result(
                    span,
                    self._audit_log,
                    audit_event="container.tool.error",
                    code=code,
                    message=message,
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                    detail={"environment_id": handler.environment_id},
                )

            overhead_ms = _ms_since(dispatch_start)
            breached = record_dispatch_overhead(span, "container", overhead_ms)
            if breached:
                _write_audit(
                    self._audit_log,
                    event="dispatch.overhead.target_breached",
                    code="dispatch_overhead_target_breached",
                    message=(
                        f"container dispatch overhead {overhead_ms:.1f}ms "
                        f"exceeded target 500.0ms"
                    ),
                    tool_name=tool.name,
                    session_id=context.session_id,
                    now=now,
                    detail={
                        "kind": "container",
                        "overhead_ms": overhead_ms,
                        "target_ms": 500.0,
                        "environment_id": handler.environment_id,
                    },
                    level="warn",
                )
            return SandboxResult(content=response.get("result"), duration_ms=_ms_since(start))
