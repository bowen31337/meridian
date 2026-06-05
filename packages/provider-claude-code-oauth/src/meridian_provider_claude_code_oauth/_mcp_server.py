"""Sandbox-proxy MCP bridge for the Claude Code CLI inner loop.

``create_sdk_mcp_server()`` returns an :class:`SdkMcpServer` that speaks MCP
stdio JSON-RPC 2.0 (newline-delimited) and exposes a single
``meridian_tool_proxy`` tool.  Every ``tools/call`` for that tool is forwarded
to ``Sandbox.execute()`` — full capability check, schema validation, hooks,
and audit are applied by the Sandbox before dispatch.

On each invocation the server emits an OTel span
``"sdk_mcp_server.tool_proxy"`` with a ``"mcp_server.invocation"`` structured
event attached.  On failure the span is marked ERROR, the audit log receives a
``"sdk_mcp_server.proxy.failed"`` entry, and an MCP ``isError=true`` response
is returned to the caller so the error message is surfaced to Claude.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from sdk_sandbox._audit import AuditLog, NoopAuditLog
from sdk_sandbox._runtime import RuntimeOptions, Sandbox
from sdk_sandbox._types import AuditLogEntry, ExecutionContext, SandboxFailure
from sdk_sandbox._version import SANDBOX_SDK_VERSION

_LOG = logging.getLogger(__name__)

_TRACER_NAME = "meridian.sdk-mcp-server"
_SERVER_VERSION = "0.1.0"
_MCP_PROTOCOL_VERSION = "2024-11-05"

# The single proxy tool exposed by this MCP server.
PROXY_TOOL_NAME = "meridian_tool_proxy"
_PROXY_TOOL_DESCRIPTION = (
    "Forward a tool call to the Meridian Sandbox. "
    "Passes tool_name and tool_input to Sandbox.execute() with full "
    "capability check, schema validation, hooks, and audit."
)
_PROXY_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_name": {
            "type": "string",
            "description": "Meridian-registered tool name to call",
        },
        "tool_input": {
            "type": "object",
            "description": "Input arguments for the tool",
        },
    },
    "required": ["tool_name", "tool_input"],
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, SANDBOX_SDK_VERSION)


def _ok(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_result(text: str, *, is_error: bool) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class SdkMcpServer:
    """MCP stdio server that proxies ``tools/call`` requests to ``Sandbox.execute()``.

    Speaks MCP JSON-RPC 2.0 on a pair of async streams (``asyncio.StreamReader``
    / ``asyncio.StreamWriter``).  Handles ``initialize``, ``tools/list``, and
    ``tools/call``.  Every ``tools/call`` for ``meridian_tool_proxy`` is
    forwarded to the injected :class:`~sdk_sandbox.Sandbox` — cap check,
    schema validation, hooks, and audit all happen inside ``Sandbox.execute()``.

    OTel: each ``tools/call`` opens a ``"sdk_mcp_server.tool_proxy"`` span with
    a ``"mcp_server.invocation"`` structured event.  Failure marks the span
    ERROR and writes a ``"sdk_mcp_server.proxy.failed"`` audit log entry.

    Parameters
    ----------
    sandbox:
        Meridian Sandbox instance with tools registered before ``serve()`` starts.
    context:
        Execution context forwarded to every ``Sandbox.execute()`` call.
    audit_log:
        Audit log sink for MCP-bridge-level failures.  Defaults to
        :class:`~sdk_sandbox.NoopAuditLog`.
    server_name:
        ``serverInfo.name`` returned during the MCP ``initialize`` handshake.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        context: ExecutionContext,
        *,
        audit_log: AuditLog | None = None,
        server_name: str = "meridian-sdk-mcp-server",
    ) -> None:
        self._sandbox = sandbox
        self._context = context
        self._audit_log: AuditLog = audit_log if audit_log is not None else NoopAuditLog()
        self._server_name = server_name

    # ------------------------------------------------------------------
    # Public serve entry point
    # ------------------------------------------------------------------

    async def serve(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Serve MCP JSON-RPC 2.0 on *reader*/*writer* until EOF.

        Reads newline-delimited JSON-RPC messages from *reader*, dispatches
        each, and writes response lines (if any) to *writer*.
        """
        while True:
            try:
                raw = await reader.readline()
            except Exception:
                break
            if not raw:
                break
            line = raw.strip()
            if not line:
                continue
            try:
                msg: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue

            response = await self._dispatch(msg)
            if response is not None:
                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()

    # ------------------------------------------------------------------
    # Request dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method = msg.get("method", "")
        request_id = msg.get("id")
        params: dict[str, Any] = msg.get("params") or {}

        if method == "initialize":
            return _ok(
                request_id,
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self._server_name, "version": _SERVER_VERSION},
                },
            )

        if method == "notifications/initialized":
            return None  # notification — no response expected

        if method == "tools/list":
            return _ok(
                request_id,
                {
                    "tools": [
                        {
                            "name": PROXY_TOOL_NAME,
                            "description": _PROXY_TOOL_DESCRIPTION,
                            "inputSchema": _PROXY_INPUT_SCHEMA,
                        }
                    ]
                },
            )

        if method == "tools/call":
            tool_name: str = params.get("name", "")
            arguments: dict[str, Any] = params.get("arguments") or {}
            if tool_name != PROXY_TOOL_NAME:
                # notifications (no id) never get a response
                if request_id is None:
                    return None
                return _rpc_error(request_id, -32601, f'Unknown tool: "{tool_name}"')
            result = await self._proxy_call(arguments)
            return _ok(request_id, result)

        # Unrecognised method.
        if request_id is None:
            return None
        return _rpc_error(request_id, -32601, f'Method not found: "{method}"')

    # ------------------------------------------------------------------
    # Proxy core: forward meridian_tool_proxy call to Sandbox.execute()
    # ------------------------------------------------------------------

    async def _proxy_call(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Forward one ``meridian_tool_proxy`` call to ``Sandbox.execute()``.

        Opens OTel span ``"sdk_mcp_server.tool_proxy"`` and attaches a
        ``"mcp_server.invocation"`` structured event.

        On success returns an MCP ``isError=false`` content block.
        On any failure:
          - marks span ERROR with a ``"mcp_server.proxy.error"`` event
          - writes ``"sdk_mcp_server.proxy.failed"`` to the audit log
          - returns an MCP ``isError=true`` content block (never raises)
        """
        now = _now()
        proxied_tool: str = arguments.get("tool_name", "")
        tool_input: dict[str, Any] = arguments.get("tool_input") or {}

        tracer = _get_tracer()
        with tracer.start_as_current_span(
            "sdk_mcp_server.tool_proxy",
            attributes={
                "tool.name": proxied_tool,
                "session.id": self._context.session_id,
                "mcp.proxy": True,
            },
        ) as span:
            span.add_event(
                "mcp_server.invocation",
                {
                    "tool.name": proxied_tool,
                    "session.id": self._context.session_id,
                    "timestamp": now,
                    "operation": "tool_proxy",
                },
            )

            if not proxied_tool:
                return self._fail(
                    span,
                    now,
                    proxied_tool,
                    'meridian_tool_proxy requires a non-empty "tool_name"',
                )

            options = RuntimeOptions(audit_log=self._audit_log)
            try:
                result = await self._sandbox.execute(
                    proxied_tool, tool_input, self._context, options
                )
            except SandboxFailure as exc:
                return self._fail(span, now, proxied_tool, exc.message)
            except Exception as exc:
                return self._fail(span, now, proxied_tool, str(exc))

            if result.is_error:
                msg = result.error_message or str(result.content)
                return self._fail(span, now, proxied_tool, msg)

            content = result.content if isinstance(result.content, str) else str(result.content)
            return _tool_result(content, is_error=False)

    def _fail(
        self,
        span: Any,
        now: str,
        tool_name: str,
        message: str,
    ) -> dict[str, Any]:
        """Record failure on span + audit log; return MCP ``isError=true`` block."""
        span.set_status(Status(StatusCode.ERROR, message))
        span.add_event(
            "mcp_server.proxy.error",
            {
                "tool.name": tool_name,
                "session.id": self._context.session_id,
                "error.message": message,
            },
        )
        self._audit_log.write(
            AuditLogEntry(
                level="error",
                event="sdk_mcp_server.proxy.failed",
                tool_name=tool_name,
                session_id=self._context.session_id,
                timestamp=now,
                detail={"message": message},
            )
        )
        return _tool_result(message, is_error=True)


def create_sdk_mcp_server(
    sandbox: Sandbox,
    context: ExecutionContext,
    *,
    audit_log: AuditLog | None = None,
    server_name: str = "meridian-sdk-mcp-server",
) -> SdkMcpServer:
    """Create a Sandbox-proxy MCP server for the Claude Code CLI inner loop.

    The returned :class:`SdkMcpServer` speaks MCP stdio JSON-RPC 2.0 and
    exposes a single ``meridian_tool_proxy`` tool.  When the Claude Code CLI
    subprocess calls that tool the server forwards the call to
    ``Sandbox.execute()`` with full capability check, schema validation,
    hooks, and audit.  On failure the span is marked ERROR, the audit log is
    written, and an ``isError=true`` response is returned to the CLI so the
    error message is surfaced to Claude.

    Parameters
    ----------
    sandbox:
        Meridian Sandbox instance.  Tools must be registered before
        :meth:`~SdkMcpServer.serve` starts accepting connections.
    context:
        Execution context forwarded to every ``Sandbox.execute()`` call.
    audit_log:
        Audit log sink.  Defaults to :class:`~sdk_sandbox.NoopAuditLog`.
    server_name:
        ``serverInfo.name`` in the MCP ``initialize`` response.

    Returns
    -------
    SdkMcpServer
        Call :meth:`~SdkMcpServer.serve` with an
        ``(asyncio.StreamReader, asyncio.StreamWriter)`` pair to begin serving.
    """
    return SdkMcpServer(
        sandbox,
        context,
        audit_log=audit_log,
        server_name=server_name,
    )
