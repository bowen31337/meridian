from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

from ._contract import EnvironmentDriver
from ._types import (
    CapabilityEnvelope,
    ExecuteRequest,
    ExecuteResult,
    FilesystemPolicy,
    NetworkPolicy,
    ProvisionRequest,
    ReclaimRequest,
)

try:
    import httpx as _httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

_MCP_PROTOCOL_VERSION = "2024-11-05"
_PROCESS_GRACE_S = 2.0
_MAX_SKIP_LINES = 50


def _ms_since(start: float) -> float:
    return (time.monotonic() - start) * 1000.0


# ---------------------------------------------------------------------------
# Stdio transport helpers (MCP JSON-RPC 2.0 over subprocess stdin/stdout)
# ---------------------------------------------------------------------------


async def _stdio_read_response(
    reader: asyncio.StreamReader,
    request_id: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Read newline-delimited JSON-RPC lines until a response matching *request_id* arrives."""
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
    """Perform the MCP initialize / notifications/initialized handshake."""
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
                "clientInfo": {"name": "meridian-mcp-backend", "version": "1.0"},
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


async def _stdio_close(proc: asyncio.subprocess.Process) -> None:
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
# McpBackendDriver
# ---------------------------------------------------------------------------


class McpBackendDriver(EnvironmentDriver):
    """
    Environment backend that proxies tool calls to an MCP server.

    Supports two transports selected by *transport*:

      stdio — Spawns *command* as a subprocess on each execute call.
              Performs the MCP initialize / notifications/initialized handshake,
              sends tools/call, and returns the normalised ExecuteResult.
              The *env* dict from ExecuteRequest is forwarded to the process
              environment.

      sse   — Connects to an MCP server via the HTTP+SSE transport
              (RFC: GET /sse for the server-to-client event stream,
              POST to the returned endpoint URL for client-to-server messages).
              Requires httpx; install with:
                pip install 'meridian-sdk-environment[http]'

    ExecuteRequest convention
    -------------------------
      command[0]  MCP tool name to invoke
      stdin       JSON-encoded tool arguments dict (or empty / None for {})

    ExecuteResult mapping
    ---------------------
      stdout      joined text content from MCP content blocks
      stderr      error text on isError=true; empty on success
      exit_code   0 on success, 1 on MCP tool error (isError=true)
      duration_ms wall-clock time for the MCP call

    On any transport or protocol failure the driver raises; the runtime wraps
    the exception in EnvironmentFailure, surfaces the message to the caller,
    and writes the failure to the audit log.
    """

    KIND = "system.mcp"

    def __init__(
        self,
        *,
        transport: Literal["stdio", "sse"],
        command: tuple[str, ...] = (),
        server_url: str = "",
        timeout_s: float = 30.0,
        on_demand: bool = True,
        network_policy: NetworkPolicy | None = None,
        filesystem_policy: FilesystemPolicy | None = None,
        capability_envelope: CapabilityEnvelope | None = None,
    ) -> None:
        self._transport = transport
        self._command = command
        self._server_url = server_url.rstrip("/")
        self._timeout_s = timeout_s
        self._on_demand = on_demand
        self._network_policy = network_policy or NetworkPolicy()
        self._filesystem_policy = filesystem_policy or FilesystemPolicy()
        self._capability_envelope = capability_envelope or CapabilityEnvelope()

    @property
    def kind(self) -> str:
        return self.KIND

    @property
    def on_demand(self) -> bool:
        return self._on_demand

    def network_policy(self) -> NetworkPolicy:
        return self._network_policy

    def filesystem_policy(self) -> FilesystemPolicy:
        return self._filesystem_policy

    def capability_envelope(self) -> CapabilityEnvelope:
        return self._capability_envelope

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_call(request: ExecuteRequest) -> tuple[str, dict[str, Any]]:
        """Extract tool name and JSON arguments from ExecuteRequest.

        command[0] is the tool name; stdin is JSON-encoded arguments.
        """
        if not request.command:
            raise ValueError(
                "McpBackendDriver: command must be non-empty; "
                "command[0] is the MCP tool name"
            )
        tool_name = request.command[0]
        if request.stdin:
            try:
                arguments: dict[str, Any] = json.loads(request.stdin)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"McpBackendDriver: stdin must be JSON-encoded arguments: {exc}"
                ) from exc
        else:
            arguments = {}
        return tool_name, arguments

    @staticmethod
    def _mcp_to_execute_result(data: dict[str, Any], start: float) -> ExecuteResult:
        """Normalise a JSON-RPC tools/call response to ExecuteResult.

        JSON-RPC protocol errors ({"error": {...}}) raise RuntimeError so the
        runtime can surface them via EnvironmentFailure and the audit log.
        MCP tool errors ({"result": {"isError": true, ...}}) are returned as
        exit_code=1 so callers receive the error text without audit noise.
        """
        if "error" in data:
            err = data["error"]
            raise RuntimeError(err.get("message", "MCP JSON-RPC error"))

        result: dict[str, Any] = data.get("result", {})
        is_error: bool = bool(result.get("isError", False))
        raw_content: list[dict[str, Any]] | None = result.get("content")
        content_blocks: list[dict[str, Any]] = raw_content if raw_content is not None else []
        text = "\n".join(
            b.get("text", "") for b in content_blocks if b.get("type") == "text"
        )
        if not text and raw_content is None:
            # No content key at all — surface the whole result as text fallback
            text = str(result) if result else ""

        if is_error:
            return ExecuteResult(
                stdout="",
                stderr=text or "MCP tool returned isError=true",
                exit_code=1,
                duration_ms=_ms_since(start),
            )
        return ExecuteResult(
            stdout=text,
            stderr="",
            exit_code=0,
            duration_ms=_ms_since(start),
        )

    async def _call_stdio(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        env: dict[str, str] | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        if not self._command:
            raise ValueError(
                "McpBackendDriver with transport='stdio' requires a non-empty command"
            )
        import os

        proc_env = {**os.environ, **(env or {})} if env else None
        proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
        )
        try:
            await _stdio_handshake(proc, timeout_s)
            req = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": "call",
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                }
            ).encode()
            assert proc.stdin is not None
            proc.stdin.write(req + b"\n")
            await proc.stdin.drain()
            return await _stdio_read_response(
                proc.stdout,  # type: ignore[arg-type]
                "call",
                timeout_s,
            )
        finally:
            await _stdio_close(proc)

    async def _call_sse(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        """Full MCP HTTP+SSE handshake and tool call.

        Protocol flow (MCP spec 2024-11-05 §6.3.2):
          1. GET /sse  — open SSE stream; server emits "endpoint" event with
                         the URL for client-to-server POSTs.
          2. POST {endpoint}  initialize request
          3. SSE stream delivers the initialize response
          4. POST {endpoint}  notifications/initialized (no response)
          5. POST {endpoint}  tools/call request
          6. SSE stream delivers the tools/call response
        """
        if not _HTTPX_AVAILABLE:
            raise RuntimeError(
                "httpx is required for McpBackendDriver with transport='sse'. "
                "Install with: pip install 'meridian-sdk-environment[http]'"
            )
        if not self._server_url:
            raise ValueError(
                "McpBackendDriver with transport='sse' requires a non-empty server_url"
            )

        loop = asyncio.get_running_loop()
        endpoint_event: asyncio.Event = asyncio.Event()
        init_future: asyncio.Future[dict[str, Any]] = loop.create_future()
        call_future: asyncio.Future[dict[str, Any]] = loop.create_future()
        message_url_ref: list[str] = []  # single-element mutable box for closure

        async def _sse_reader(client: _httpx.AsyncClient) -> None:
            event_type: str | None = None
            data_lines: list[str] = []
            async with client.stream(
                "GET",
                f"{self._server_url}/sse",
                headers={"Accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                    elif not line:
                        # blank line = end of event; dispatch accumulated data
                        if data_lines:
                            data_str = "\n".join(data_lines)
                            _dispatch_sse_event(event_type, data_str)
                        event_type = None
                        data_lines = []
                    # id: and retry: lines are silently ignored

        def _dispatch_sse_event(etype: str | None, data_str: str) -> None:
            if etype == "endpoint":
                raw = data_str.strip()
                url = raw if raw.startswith("http") else f"{self._server_url}{raw}"
                message_url_ref.append(url)
                endpoint_event.set()
                return
            # "message" events or bare data lines carry JSON-RPC responses
            if etype not in ("message", None):
                return
            try:
                msg: dict[str, Any] = json.loads(data_str)
            except json.JSONDecodeError:
                return
            msg_id = msg.get("id")
            if msg_id == "init" and not init_future.done():
                init_future.set_result(msg)
            elif msg_id == "call" and not call_future.done():
                call_future.set_result(msg)

        async with _httpx.AsyncClient(timeout=timeout_s) as client:
            reader_task = asyncio.create_task(_sse_reader(client))
            try:
                # Step 1: wait for the server to emit its message endpoint URL
                await asyncio.wait_for(endpoint_event.wait(), timeout_s)
                message_url = message_url_ref[0]

                # Step 2: initialize
                await client.post(
                    message_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": "init",
                        "method": "initialize",
                        "params": {
                            "protocolVersion": _MCP_PROTOCOL_VERSION,
                            "capabilities": {},
                            "clientInfo": {
                                "name": "meridian-mcp-backend",
                                "version": "1.0",
                            },
                        },
                    },
                    headers={"Content-Type": "application/json"},
                )

                # Step 3: wait for initialize response on the SSE stream
                init_resp = await asyncio.wait_for(init_future, timeout_s)
                if "error" in init_resp:
                    err = init_resp["error"]
                    raise ValueError(
                        f"MCP initialize failed: {err.get('message', init_resp)}"
                    )

                # Step 4: notifications/initialized (no response expected)
                await client.post(
                    message_url,
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                    headers={"Content-Type": "application/json"},
                )

                # Step 5: tools/call
                await client.post(
                    message_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": "call",
                        "method": "tools/call",
                        "params": {"name": tool_name, "arguments": arguments},
                    },
                    headers={"Content-Type": "application/json"},
                )

                # Step 6: wait for tools/call response on the SSE stream
                return await asyncio.wait_for(call_future, timeout_s)

            finally:
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):
                    pass

    # ------------------------------------------------------------------
    # EnvironmentDriver
    # ------------------------------------------------------------------

    async def provision(self, request: ProvisionRequest) -> None:
        # on_demand=True: each execute is self-contained; validate config eagerly.
        if self._transport == "stdio" and not self._command:
            raise ValueError(
                "McpBackendDriver with transport='stdio' requires a non-empty command"
            )
        if self._transport == "sse" and not self._server_url:
            raise ValueError(
                "McpBackendDriver with transport='sse' requires a non-empty server_url"
            )

    async def execute(self, request: ExecuteRequest) -> ExecuteResult:
        tool_name, arguments = self._parse_tool_call(request)
        timeout_s = (
            float(request.timeout_seconds)
            if request.timeout_seconds is not None
            else self._timeout_s
        )

        start = time.monotonic()
        if self._transport == "stdio":
            data = await self._call_stdio(tool_name, arguments, request.env or None, timeout_s)
        else:
            data = await self._call_sse(tool_name, arguments, timeout_s)

        return self._mcp_to_execute_result(data, start)

    async def reclaim(self, request: ReclaimRequest) -> None:
        # on_demand=True: no persistent resources to release
        pass
