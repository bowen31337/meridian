"""
Tests for MCP server integration: stdio and HTTP transports.

Covers:
  - McpDispatcher with stdio transport: success, command not found, server error,
    process crash, empty command, span name / attributes / structured event,
    audit log on failure.
  - Tool discovery via tools/list: HTTP (mocked httpx) and stdio (real subprocess).
  - discover_mcp_tools dispatches by transport.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.trace import StatusCode

from sdk_sandbox import (
    ExecutionContext,
    McpHandler,
    McpToolSpec,
    SandboxResult,
    ToolDefinition,
    ToolDispatcher,
    discover_mcp_tools,
    discover_mcp_tools_http,
    discover_mcp_tools_stdio,
)
from sdk_sandbox._dispatchers import McpDispatcher
from sdk_sandbox._mcp_client import (
    _stdio_handshake,  # noqa: PLC2701 – tested directly
    _stdio_read_response,
)

from .conftest import CapturingAuditLog, MockSpan, MockTracer

# ---------------------------------------------------------------------------
# OTel mock
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_tracer(monkeypatch: pytest.MonkeyPatch) -> MockTracer:
    tracer = MockTracer()
    monkeypatch.setattr("sdk_sandbox._dispatchers.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_span(mock_tracer: MockTracer) -> MockSpan:
    return mock_tracer.span


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CTX = ExecutionContext(session_id="sess-mcp", workspace="/workspace", scratch_dir="/tmp")


def _tool(handler: Any, name: str = "test.tool") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Test MCP tool",
        input_schema={"type": "object"},
        handler=handler,
    )


def _stdio_handler(command: list[str], tool_name: str = "echo") -> McpHandler:
    return McpHandler(
        tool_name=tool_name,
        transport="stdio",
        command=tuple(command),
    )


# ---------------------------------------------------------------------------
# Minimal MCP stdio server scripts (used as real subprocesses in tests)
# ---------------------------------------------------------------------------

# Well-behaved server: handles initialize, notifications/initialized, tools/call
_MCP_ECHO_SERVER = """\
import sys, json

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    method = msg.get("method", "")
    id_ = msg.get("id")
    if method == "initialize":
        resp = {
            "jsonrpc": "2.0", "id": id_,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "serverInfo": {"name": "test", "version": "1.0"},
            },
        }
        print(json.dumps(resp), flush=True)
    elif method == "notifications/initialized":
        pass
    elif method == "tools/call":
        args = msg.get("params", {}).get("arguments", {})
        resp = {
            "jsonrpc": "2.0", "id": id_,
            "result": {
                "content": [{"type": "text", "text": json.dumps(args)}],
                "isError": False,
            },
        }
        print(json.dumps(resp), flush=True)
    elif method == "tools/list":
        resp = {
            "jsonrpc": "2.0", "id": id_,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo the input",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"msg": {"type": "string"}},
                        },
                    }
                ]
            },
        }
        print(json.dumps(resp), flush=True)
"""

# Server that returns isError=true from tools/call
_MCP_TOOL_ERROR_SERVER = """\
import sys, json

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    method = msg.get("method", "")
    id_ = msg.get("id")
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": id_, "result": {
            "protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {}
        }}), flush=True)
    elif method == "notifications/initialized":
        pass
    elif method == "tools/call":
        resp = {
            "jsonrpc": "2.0", "id": id_,
            "result": {
                "content": [{"type": "text", "text": "tool went wrong"}],
                "isError": True,
            },
        }
        print(json.dumps(resp), flush=True)
"""

# Server that sends a JSON-RPC error response for tools/call
_MCP_RPC_ERROR_SERVER = """\
import sys, json

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    method = msg.get("method", "")
    id_ = msg.get("id")
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": id_, "result": {
            "protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {}
        }}), flush=True)
    elif method == "notifications/initialized":
        pass
    elif method == "tools/call":
        resp = {
            "jsonrpc": "2.0", "id": id_,
            "error": {"code": -32000, "message": "server-side error"},
        }
        print(json.dumps(resp), flush=True)
"""

# Server that crashes immediately
_MCP_CRASH_SERVER = """\
import sys
sys.stderr.write("fatal error\\n")
sys.exit(1)
"""


def _write_server(tmp_path: Path, name: str, content: str) -> list[str]:
    """Write a Python MCP server script and return its argv."""
    p = tmp_path / name
    p.write_text(f"#!{sys.executable}\n{content}")
    p.chmod(0o755)
    return [str(p)]


# ---------------------------------------------------------------------------
# TestMcpStdioDispatch
# ---------------------------------------------------------------------------


class TestMcpStdioDispatch:
    def test_satisfies_abc(self) -> None:
        assert isinstance(McpDispatcher(), ToolDispatcher)

    def test_kind(self) -> None:
        assert McpDispatcher().kind == "mcp"

    # success

    async def test_success_returns_result(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "echo_server", _MCP_ECHO_SERVER)
        d = McpDispatcher()
        result = await d.dispatch(_tool(_stdio_handler(cmd)), {"key": "val"}, CTX)
        assert result.is_error is False
        assert result.content == '{"key": "val"}'

    async def test_success_no_audit_entry(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "echo_server", _MCP_ECHO_SERVER)
        audit = CapturingAuditLog()
        d = McpDispatcher(audit_log=audit)
        await d.dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        assert audit.entries == []

    async def test_success_span_name(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "echo_server", _MCP_ECHO_SERVER)
        await McpDispatcher().dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        assert mock_span.name == "mcp.dispatch"

    async def test_success_span_attributes(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "echo_server", _MCP_ECHO_SERVER)
        await McpDispatcher().dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        assert mock_span.attributes["tool.name"] == "test.tool"
        assert mock_span.attributes["session.id"] == "sess-mcp"
        assert mock_span.attributes["mcp.transport"] == "stdio"

    async def test_success_span_command_attribute(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "echo_server", _MCP_ECHO_SERVER)
        await McpDispatcher().dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        assert "mcp.command" in mock_span.attributes

    async def test_success_structured_event(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "echo_server", _MCP_ECHO_SERVER)
        await McpDispatcher().dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        event_names = [e[0] for e in mock_span.events]
        assert "mcp.dispatch" in event_names

    async def test_success_event_has_transport(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "echo_server", _MCP_ECHO_SERVER)
        await McpDispatcher().dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        event = next(e for e in mock_span.events if e[0] == "mcp.dispatch")
        assert event[1].get("mcp.transport") == "stdio"

    async def test_success_duration_ms_set(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "echo_server", _MCP_ECHO_SERVER)
        result = await McpDispatcher().dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        assert result.duration_ms >= 0.0

    # isError=true from tool

    async def test_tool_error_returns_is_error(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "err_server", _MCP_TOOL_ERROR_SERVER)
        result = await McpDispatcher().dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        assert result.is_error is True
        assert result.error_code == "mcp_tool_error"
        assert "tool went wrong" in (result.error_message or "")

    async def test_tool_error_audit_entry(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "err_server", _MCP_TOOL_ERROR_SERVER)
        audit = CapturingAuditLog()
        await McpDispatcher(audit_log=audit).dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "mcp.tool.error"

    async def test_tool_error_span_marked_error(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "err_server", _MCP_TOOL_ERROR_SERVER)
        await McpDispatcher().dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        assert mock_span.status.status_code == StatusCode.ERROR

    # JSON-RPC error from server

    async def test_rpc_error_returns_is_error(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "rpc_err_server", _MCP_RPC_ERROR_SERVER)
        result = await McpDispatcher().dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        assert result.is_error is True
        assert result.error_code == "-32000"
        assert "server-side error" in (result.error_message or "")

    async def test_rpc_error_audit_entry(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "rpc_err_server", _MCP_RPC_ERROR_SERVER)
        audit = CapturingAuditLog()
        await McpDispatcher(audit_log=audit).dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "mcp.rpc.error"

    # empty command

    async def test_empty_command_returns_is_error(self, mock_span: MockSpan) -> None:
        handler = McpHandler(tool_name="echo", transport="stdio", command=())
        result = await McpDispatcher().dispatch(_tool(handler), {}, CTX)
        assert result.is_error is True
        assert result.error_code == "mcp_stdio_no_command"

    async def test_empty_command_audit_entry(self, mock_span: MockSpan) -> None:
        audit = CapturingAuditLog()
        handler = McpHandler(tool_name="echo", transport="stdio", command=())
        await McpDispatcher(audit_log=audit).dispatch(_tool(handler), {}, CTX)
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "mcp.stdio.no_command"

    # command not found

    async def test_command_not_found_returns_is_error(self, mock_span: MockSpan) -> None:
        handler = McpHandler(
            tool_name="echo", transport="stdio", command=("/no/such/mcp_server",)
        )
        result = await McpDispatcher().dispatch(_tool(handler), {}, CTX)
        assert result.is_error is True
        assert result.error_code == "mcp_stdio_command_not_found"

    async def test_command_not_found_audit_entry(self, mock_span: MockSpan) -> None:
        audit = CapturingAuditLog()
        handler = McpHandler(
            tool_name="echo", transport="stdio", command=("/no/such/mcp_server",)
        )
        await McpDispatcher(audit_log=audit).dispatch(_tool(handler), {}, CTX)
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "mcp.stdio.command_not_found"

    async def test_command_not_found_span_marked_error(self, mock_span: MockSpan) -> None:
        handler = McpHandler(
            tool_name="echo", transport="stdio", command=("/no/such/mcp_server",)
        )
        await McpDispatcher().dispatch(_tool(handler), {}, CTX)
        assert mock_span.status.status_code == StatusCode.ERROR

    # process crash (exits non-zero before responding)

    async def test_process_crash_returns_is_error(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        cmd = _write_server(tmp_path, "crash_server", _MCP_CRASH_SERVER)
        result = await McpDispatcher().dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        assert result.is_error is True
        assert result.error_code == "mcp_stdio_request_failed"

    async def test_process_crash_audit_entry(self, tmp_path: Path, mock_span: MockSpan) -> None:
        cmd = _write_server(tmp_path, "crash_server", _MCP_CRASH_SERVER)
        audit = CapturingAuditLog()
        await McpDispatcher(audit_log=audit).dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "mcp.stdio.request_failed"

    async def test_process_crash_span_marked_error(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        cmd = _write_server(tmp_path, "crash_server", _MCP_CRASH_SERVER)
        await McpDispatcher().dispatch(_tool(_stdio_handler(cmd)), {}, CTX)
        assert mock_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# TestMcpHttpDispatch — existing HTTP path still works (regression guard)
# ---------------------------------------------------------------------------


class TestMcpHttpDispatch:
    def _http_tool(self) -> ToolDefinition:
        return _tool(McpHandler(server_url="http://mcp.local", tool_name="my_tool"))

    def _mock_response(self, data: dict[str, Any]) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=data)
        return resp

    async def test_http_success_returns_result(self, mock_span: MockSpan) -> None:
        data = {
            "jsonrpc": "2.0", "id": "x",
            "result": {"content": [{"type": "text", "text": "hello"}], "isError": False},
        }
        resp = self._mock_response(data)
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value = mock_client

            result = await McpDispatcher().dispatch(self._http_tool(), {}, CTX)

        assert result.is_error is False
        assert result.content == "hello"

    async def test_http_span_has_transport_attribute(self, mock_span: MockSpan) -> None:
        data = {"jsonrpc": "2.0", "id": "x", "result": {"content": [], "isError": False}}
        resp = self._mock_response(data)
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value = mock_client

            await McpDispatcher().dispatch(self._http_tool(), {}, CTX)

        assert mock_span.attributes.get("mcp.transport") == "http"


# ---------------------------------------------------------------------------
# TestMcpToolDiscoveryStdio
# ---------------------------------------------------------------------------


class TestMcpToolDiscoveryStdio:
    async def test_returns_tool_specs(self, tmp_path: Path) -> None:
        cmd = _write_server(tmp_path, "echo_server", _MCP_ECHO_SERVER)
        specs = await discover_mcp_tools_stdio(tuple(cmd))
        assert len(specs) == 1
        assert specs[0].name == "echo"
        assert specs[0].description == "Echo the input"
        assert specs[0].input_schema == {
            "type": "object",
            "properties": {"msg": {"type": "string"}},
        }

    async def test_returns_mcp_tool_spec_instances(self, tmp_path: Path) -> None:
        cmd = _write_server(tmp_path, "echo_server", _MCP_ECHO_SERVER)
        specs = await discover_mcp_tools_stdio(tuple(cmd))
        for spec in specs:
            assert isinstance(spec, McpToolSpec)

    async def test_command_not_found_raises(self) -> None:
        with pytest.raises((OSError, ValueError)):
            await discover_mcp_tools_stdio(("/no/such/mcp_server",))

    async def test_empty_command_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty command"):
            await discover_mcp_tools_stdio(())


# ---------------------------------------------------------------------------
# TestMcpToolDiscoveryHttp
# ---------------------------------------------------------------------------

_TOOLS_LIST_RESPONSE = {
    "jsonrpc": "2.0",
    "id": "list",
    "result": {
        "tools": [
            {
                "name": "search",
                "description": "Search the web",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "fetch",
                "description": "Fetch a URL",
                "inputSchema": {"type": "object"},
            },
        ]
    },
}


class TestMcpToolDiscoveryHttp:
    def _mock_httpx_client(self, data: dict[str, Any]) -> Any:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=data)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=resp)
        return mock_client

    async def test_returns_tool_specs(self) -> None:
        with patch("sdk_sandbox._mcp_client._httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = self._mock_httpx_client(_TOOLS_LIST_RESPONSE)
            specs = await discover_mcp_tools_http("http://mcp.local")

        assert len(specs) == 2
        assert specs[0].name == "search"
        assert specs[0].description == "Search the web"
        assert specs[1].name == "fetch"

    async def test_returns_mcp_tool_spec_instances(self) -> None:
        with patch("sdk_sandbox._mcp_client._httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = self._mock_httpx_client(_TOOLS_LIST_RESPONSE)
            specs = await discover_mcp_tools_http("http://mcp.local")

        for spec in specs:
            assert isinstance(spec, McpToolSpec)

    async def test_input_schema_populated(self) -> None:
        with patch("sdk_sandbox._mcp_client._httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = self._mock_httpx_client(_TOOLS_LIST_RESPONSE)
            specs = await discover_mcp_tools_http("http://mcp.local")

        assert specs[0].input_schema["properties"]["query"]["type"] == "string"

    async def test_rpc_error_raises_value_error(self) -> None:
        error_data = {
            "jsonrpc": "2.0", "id": "list",
            "error": {"code": -32601, "message": "Method not found"},
        }
        with patch("sdk_sandbox._mcp_client._httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = self._mock_httpx_client(error_data)
            with pytest.raises(ValueError, match="Method not found"):
                await discover_mcp_tools_http("http://mcp.local")

    async def test_network_error_propagates(self) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=OSError("connection refused"))
        with patch("sdk_sandbox._mcp_client._httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            with pytest.raises(OSError):
                await discover_mcp_tools_http("http://mcp.local")

    async def test_empty_tools_list(self) -> None:
        empty_data = {"jsonrpc": "2.0", "id": "list", "result": {"tools": []}}
        with patch("sdk_sandbox._mcp_client._httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = self._mock_httpx_client(empty_data)
            specs = await discover_mcp_tools_http("http://mcp.local")

        assert specs == []


# ---------------------------------------------------------------------------
# TestDiscoverMcpTools — combined dispatcher function
# ---------------------------------------------------------------------------


class TestDiscoverMcpTools:
    async def test_dispatches_to_http(self) -> None:
        handler = McpHandler(server_url="http://mcp.local", tool_name="t", transport="http")
        with patch("sdk_sandbox._mcp_client.discover_mcp_tools_http") as mock_http:
            mock_http.return_value = [McpToolSpec(name="t", description="", input_schema={})]
            result = await discover_mcp_tools(handler)

        mock_http.assert_called_once_with("http://mcp.local", timeout_s=30.0)
        assert result[0].name == "t"

    async def test_dispatches_to_stdio(self, tmp_path: Path) -> None:
        cmd = _write_server(tmp_path, "echo_server", _MCP_ECHO_SERVER)
        handler = McpHandler(tool_name="echo", transport="stdio", command=tuple(cmd))
        specs = await discover_mcp_tools(handler)
        assert len(specs) == 1
        assert specs[0].name == "echo"

    async def test_raises_for_non_mcp_handler(self) -> None:
        with pytest.raises(TypeError, match="McpHandler"):
            await discover_mcp_tools("not a handler")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestMcpToolSpec — dataclass
# ---------------------------------------------------------------------------


class TestMcpToolSpec:
    def test_frozen(self) -> None:
        spec = McpToolSpec(name="t", description="d", input_schema={})
        with pytest.raises((AttributeError, TypeError)):
            spec.name = "other"  # type: ignore[misc]

    def test_fields(self) -> None:
        spec = McpToolSpec(
            name="tool",
            description="A tool",
            input_schema={"type": "object"},
        )
        assert spec.name == "tool"
        assert spec.description == "A tool"
        assert spec.input_schema == {"type": "object"}
