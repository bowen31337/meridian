"""
MCP JSON-RPC client for tool discovery (tools/list) and stdio dispatch.

Supports HTTP and stdio transports.  HTTP uses httpx; stdio uses asyncio
subprocess with newline-delimited JSON-RPC 2.0.

MCP stdio protocol (per spec):
  1. Client → Server: initialize request (JSON-RPC, id="init")
  2. Server → Client: initialize response
  3. Client → Server: notifications/initialized (notification, no id, no response)
  4. Client → Server: tools/call or tools/list request
  5. Server → Client: response
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

try:
    import httpx as _httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _httpx = None  # type: ignore[assignment]
    _HTTPX_AVAILABLE = False

_MCP_PROTOCOL_VERSION = "2024-11-05"
_MAX_SKIP_LINES = 50  # max lines to scan before giving up on a response
_PROCESS_GRACE_S = 2.0  # seconds to wait for process exit before killing


@dataclass(frozen=True)
class McpToolSpec:
    """Metadata for a single tool returned by an MCP server's tools/list."""

    name: str
    description: str
    input_schema: dict[str, Any]


# ---------------------------------------------------------------------------
# Stdio low-level helpers
# ---------------------------------------------------------------------------


async def _stdio_read_response(
    reader: asyncio.StreamReader,
    request_id: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Read newline-delimited JSON-RPC lines until a response matching *request_id* arrives.

    Skips notification objects (no "id" field) and blank lines.  Raises
    ValueError on timeout, closed stream, or if no matching response is found
    within _MAX_SKIP_LINES lines.
    """
    for _ in range(_MAX_SKIP_LINES):
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
        except asyncio.TimeoutError:
            raise ValueError(f"Timed out waiting for MCP response id={request_id!r}")
        if not raw:
            raise ValueError("MCP server closed stdout before sending a response")
        line = raw.strip()
        if not line:
            continue
        try:
            msg: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("id") == request_id:
            return msg
    raise ValueError(
        f"No response with id={request_id!r} found after {_MAX_SKIP_LINES} lines"
    )


async def _stdio_handshake(proc: asyncio.subprocess.Process, timeout_s: float) -> None:
    """Perform the MCP initialize / notifications/initialized handshake.

    Raises ValueError if the server returns an error response.
    """
    assert proc.stdin is not None
    assert proc.stdout is not None

    init_req = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "meridian-sandbox", "version": "1.0"},
            },
        }
    ).encode()
    proc.stdin.write(init_req + b"\n")
    await proc.stdin.drain()

    resp = await _stdio_read_response(proc.stdout, "init", timeout_s)
    if "error" in resp:
        err = resp["error"]
        raise ValueError(f"MCP initialize failed: {err.get('message', resp)}")

    notif = json.dumps(
        {"jsonrpc": "2.0", "method": "notifications/initialized"}
    ).encode()
    proc.stdin.write(notif + b"\n")
    await proc.stdin.drain()


async def _stdio_rpc(
    proc: asyncio.subprocess.Process,
    request_id: str,
    method: str,
    params: dict[str, Any] | None,
    timeout_s: float,
) -> dict[str, Any]:
    """Send one JSON-RPC request and return the parsed response dict."""
    assert proc.stdin is not None

    req: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        req["params"] = params
    proc.stdin.write(json.dumps(req).encode() + b"\n")
    await proc.stdin.drain()
    return await _stdio_read_response(proc.stdout, request_id, timeout_s)  # type: ignore[arg-type]


async def _spawn_stdio_process(command: tuple[str, ...]) -> asyncio.subprocess.Process:
    """Spawn an MCP stdio server.  Raises ValueError for empty command."""
    if not command:
        raise ValueError("stdio McpHandler requires a non-empty command")
    return await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def _close_stdio_process(proc: asyncio.subprocess.Process) -> None:
    """Close stdin and wait for the process to exit; kill after grace period."""
    if proc.returncode is not None:
        return
    try:
        proc.stdin.close()  # type: ignore[union-attr]
        await proc.stdin.wait_closed()  # type: ignore[union-attr]
    except Exception:
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=_PROCESS_GRACE_S)
    except asyncio.TimeoutError:
        proc.kill()


# ---------------------------------------------------------------------------
# Public: stdio tools/call (used by McpDispatcher)
# ---------------------------------------------------------------------------


async def stdio_tools_call(
    command: tuple[str, ...],
    tool_name: str,
    arguments: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    """Spawn an MCP stdio server, perform handshake, call tools/call, and return the raw RPC response.

    Raises ValueError or OSError on transport failure.
    """
    proc = await _spawn_stdio_process(command)
    try:
        await _stdio_handshake(proc, timeout_s)
        return await _stdio_rpc(
            proc,
            request_id="call",
            method="tools/call",
            params={"name": tool_name, "arguments": arguments},
            timeout_s=timeout_s,
        )
    finally:
        await _close_stdio_process(proc)


# ---------------------------------------------------------------------------
# Public: tool discovery
# ---------------------------------------------------------------------------


async def discover_mcp_tools_stdio(
    command: tuple[str, ...],
    timeout_s: float = 30.0,
) -> list[McpToolSpec]:
    """Spawn an MCP stdio server, call tools/list, and return the tool specs."""
    proc = await _spawn_stdio_process(command)
    try:
        await _stdio_handshake(proc, timeout_s)
        response = await _stdio_rpc(proc, "list", "tools/list", None, timeout_s)
    finally:
        await _close_stdio_process(proc)

    if "error" in response:
        err = response["error"]
        raise ValueError(f"MCP tools/list failed: {err.get('message', response)}")

    tools_raw: list[dict[str, Any]] = response.get("result", {}).get("tools", [])
    return [
        McpToolSpec(
            name=t.get("name", ""),
            description=t.get("description", ""),
            input_schema=t.get("inputSchema", {}),
        )
        for t in tools_raw
    ]


async def discover_mcp_tools_http(
    server_url: str,
    timeout_s: float = 30.0,
) -> list[McpToolSpec]:
    """POST a tools/list JSON-RPC request to an HTTP MCP server and return the tool specs.

    Requires httpx; install with ``pip install 'meridian-sdk-sandbox[http]'``.
    """
    if not _HTTPX_AVAILABLE:
        raise ImportError(
            "httpx is required for MCP HTTP tool discovery. "
            "Install with: pip install 'meridian-sdk-sandbox[http]'"
        )

    payload = {"jsonrpc": "2.0", "id": "list", "method": "tools/list", "params": {}}

    async with _httpx.AsyncClient(timeout=timeout_s) as client:  # type: ignore[union-attr]
        response = await client.post(
            server_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

    if "error" in data:
        err = data["error"]
        raise ValueError(f"MCP tools/list failed: {err.get('message', data)}")

    tools_raw = data.get("result", {}).get("tools", [])
    return [
        McpToolSpec(
            name=t.get("name", ""),
            description=t.get("description", ""),
            input_schema=t.get("inputSchema", {}),
        )
        for t in tools_raw
    ]


async def discover_mcp_tools(
    handler: Any,
    timeout_s: float = 30.0,
) -> list[McpToolSpec]:
    """Discover tools from an MCP server, dispatching by handler.transport.

    Args:
        handler: A McpHandler instance (http or stdio transport).
        timeout_s: Per-operation timeout in seconds.
    """
    from ._types import McpHandler

    if not isinstance(handler, McpHandler):
        raise TypeError(f"Expected McpHandler, got {type(handler).__name__}")
    if handler.transport == "stdio":
        return await discover_mcp_tools_stdio(handler.command, timeout_s=timeout_s)
    return await discover_mcp_tools_http(handler.server_url, timeout_s=timeout_s)
