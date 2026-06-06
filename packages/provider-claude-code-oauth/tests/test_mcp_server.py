"""
Tests for SdkMcpServer — Sandbox-proxy MCP bridge (Contract 2).

Coverage:
  MCP protocol handling:
  - initialize returns correct serverInfo and capabilities
  - tools/list returns meridian_tool_proxy descriptor
  - tools/call for unknown tool returns JSON-RPC error
  - notifications/initialized yields no response
  - unrecognised method yields JSON-RPC error
  - notification (no id) for unknown method yields no response

  meridian_tool_proxy — success path:
  - forwards tool_name + tool_input to Sandbox.execute()
  - returns MCP isError=false with result text
  - OTel span "sdk_mcp_server.tool_proxy" emitted
  - span carries tool.name + session.id + mcp.proxy attributes
  - span carries "mcp_server.invocation" structured event
  - no audit log entry written on success

  meridian_tool_proxy — sandbox is_error result:
  - returns MCP isError=true with error_message
  - writes "sdk_mcp_server.proxy.failed" audit entry
  - span marked ERROR

  meridian_tool_proxy — SandboxFailure raised:
  - returns MCP isError=true with failure message
  - writes audit entry
  - span marked ERROR

  meridian_tool_proxy — unexpected exception:
  - returns MCP isError=true
  - writes audit entry
  - span marked ERROR

  meridian_tool_proxy — missing tool_name:
  - returns MCP isError=true
  - writes audit entry
  - span marked ERROR

  create_sdk_mcp_server:
  - returns SdkMcpServer instance
  - default server_name used in initialize response
  - custom server_name reflected in initialize response
  - custom audit_log injected

  Module public API:
  - SdkMcpServer exported from package __init__
  - create_sdk_mcp_server exported from package __init__
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from opentelemetry.trace import StatusCode
from sdk_sandbox._audit import AuditLog
from sdk_sandbox._dispatchers import InProcessDispatcher
from sdk_sandbox._runtime import Sandbox
from sdk_sandbox._types import (
    AuditLogEntry,
    ExecutionContext,
    InProcessHandler,
    SandboxFailure,
    SandboxResult,
    ToolDefinition,
)

from meridian_provider_claude_code_oauth import SdkMcpServer, create_sdk_mcp_server
from meridian_provider_claude_code_oauth._mcp_server import PROXY_TOOL_NAME

# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------

CTX = ExecutionContext(session_id="sess-mcp-bridge", workspace="/ws")

# ---------------------------------------------------------------------------
# OTel mock
# ---------------------------------------------------------------------------


class MockSpan:
    def __init__(self) -> None:
        self.name: str = ""
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.status: Any = None

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, status: Any) -> None:
        self.status = status

    def record_exception(self, exc: BaseException, **_: Any) -> None:
        pass

    def __enter__(self) -> MockSpan:
        return self

    def __exit__(self, *_: Any) -> bool:
        return False


class MockTracer:
    def __init__(self) -> None:
        self.span = MockSpan()

    def start_as_current_span(
        self, name: str, *, attributes: dict[str, Any] | None = None, **_: Any
    ) -> MockSpan:
        self.span.name = name
        if attributes:
            self.span.attributes.update(attributes)
        return self.span


# ---------------------------------------------------------------------------
# Audit log capture
# ---------------------------------------------------------------------------


class CapturingAuditLog(AuditLog):
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


# ---------------------------------------------------------------------------
# Fake writer (duck-types asyncio.StreamWriter for tests)
# ---------------------------------------------------------------------------


class _FakeWriter:
    def __init__(self) -> None:
        self._buf = bytearray()

    def write(self, data: bytes) -> None:
        self._buf.extend(data)

    async def drain(self) -> None:
        pass

    def responses(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self._buf.split(b"\n") if line.strip()]


# ---------------------------------------------------------------------------
# Controllable fake Sandbox (not the real Sandbox — for isolation)
# ---------------------------------------------------------------------------


class _FakeSandbox:
    def __init__(
        self,
        result: SandboxResult | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._result = result if result is not None else SandboxResult(content="ok")
        self._raises = raises
        self.calls: list[tuple[str, dict[str, Any], ExecutionContext]] = []

    async def execute(
        self,
        name: str,
        input: dict[str, Any],
        context: ExecutionContext,
        options: Any = None,
    ) -> SandboxResult:
        self.calls.append((name, input, context))
        if self._raises is not None:
            raise self._raises
        return self._result


# ---------------------------------------------------------------------------
# Helper: feed messages through a server and collect responses
# ---------------------------------------------------------------------------


async def _exchange(
    server: SdkMcpServer,
    *messages: dict[str, Any],
) -> list[dict[str, Any]]:
    reader = asyncio.StreamReader()
    for msg in messages:
        reader.feed_data((json.dumps(msg) + "\n").encode())
    reader.feed_eof()

    writer = _FakeWriter()
    await server.serve(reader, writer)  # type: ignore[arg-type]
    return writer.responses()


def _make_server(
    *,
    sandbox: Any = None,
    audit_log: AuditLog | None = None,
    server_name: str = "test-server",
    tracer: MockTracer | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> SdkMcpServer:
    if sandbox is None:
        sandbox = _FakeSandbox()
    server = create_sdk_mcp_server(
        sandbox,  # type: ignore[arg-type]
        CTX,
        audit_log=audit_log,
        server_name=server_name,
    )
    if tracer is not None and monkeypatch is not None:
        monkeypatch.setattr(
            "meridian_provider_claude_code_oauth._mcp_server._get_tracer",
            lambda: tracer,
        )
    return server


# ---------------------------------------------------------------------------
# MCP protocol tests
# ---------------------------------------------------------------------------


class TestMcpProtocol:
    async def test_initialize_returns_ok(self) -> None:
        server = _make_server()
        responses = await _exchange(
            server, {"jsonrpc": "2.0", "id": "init", "method": "initialize", "params": {}}
        )
        assert len(responses) == 1
        resp = responses[0]
        assert resp["id"] == "init"
        assert "result" in resp
        assert "error" not in resp

    async def test_initialize_protocol_version(self) -> None:
        server = _make_server()
        responses = await _exchange(
            server, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        result = responses[0]["result"]
        assert result["protocolVersion"] == "2024-11-05"

    async def test_initialize_server_info_name(self) -> None:
        server = _make_server(server_name="my-bridge")
        responses = await _exchange(
            server, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        assert responses[0]["result"]["serverInfo"]["name"] == "my-bridge"

    async def test_initialize_capabilities_has_tools(self) -> None:
        server = _make_server()
        responses = await _exchange(
            server, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        assert "tools" in responses[0]["result"]["capabilities"]

    async def test_notifications_initialized_no_response(self) -> None:
        server = _make_server()
        responses = await _exchange(
            server, {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        assert responses == []

    async def test_tools_list_returns_proxy_tool(self) -> None:
        server = _make_server()
        responses = await _exchange(server, {"jsonrpc": "2.0", "id": "lst", "method": "tools/list"})
        tools = responses[0]["result"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == PROXY_TOOL_NAME

    async def test_tools_list_input_schema_has_required_fields(self) -> None:
        server = _make_server()
        responses = await _exchange(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        schema = responses[0]["result"]["tools"][0]["inputSchema"]
        assert "tool_name" in schema["properties"]
        assert "tool_input" in schema["properties"]
        assert "tool_name" in schema["required"]
        assert "tool_input" in schema["required"]

    async def test_tools_call_unknown_tool_returns_rpc_error(self) -> None:
        server = _make_server()
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": "x",
                "method": "tools/call",
                "params": {"name": "no_such_tool", "arguments": {}},
            },
        )
        assert "error" in responses[0]
        assert responses[0]["error"]["code"] == -32601

    async def test_unknown_method_returns_rpc_error(self) -> None:
        server = _make_server()
        responses = await _exchange(server, {"jsonrpc": "2.0", "id": "y", "method": "nonexistent"})
        assert "error" in responses[0]
        assert responses[0]["error"]["code"] == -32601

    async def test_notification_unknown_method_no_response(self) -> None:
        server = _make_server()
        responses = await _exchange(
            server,
            {"jsonrpc": "2.0", "method": "nonexistent"},  # no id → notification
        )
        assert responses == []

    async def test_multiple_messages_in_sequence(self) -> None:
        server = _make_server()
        responses = await _exchange(
            server,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        assert len(responses) == 2  # init + list; notification has no response
        assert responses[0]["id"] == 1
        assert responses[1]["id"] == 2


# ---------------------------------------------------------------------------
# meridian_tool_proxy — success
# ---------------------------------------------------------------------------


class TestProxySuccess:
    async def test_forwards_tool_name_to_sandbox(self) -> None:
        fake = _FakeSandbox()
        server = _make_server(sandbox=fake)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "my_tool", "tool_input": {"k": "v"}},
                },
            },
        )
        assert fake.calls[0][0] == "my_tool"

    async def test_forwards_tool_input_to_sandbox(self) -> None:
        fake = _FakeSandbox()
        server = _make_server(sandbox=fake)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {"x": 42}},
                },
            },
        )
        assert fake.calls[0][1] == {"x": 42}

    async def test_returns_mcp_is_error_false(self) -> None:
        server = _make_server(sandbox=_FakeSandbox(result=SandboxResult(content="done")))
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {}},
                },
            },
        )
        result = responses[0]["result"]
        assert result["isError"] is False

    async def test_returns_result_content_text(self) -> None:
        server = _make_server(sandbox=_FakeSandbox(result=SandboxResult(content="hello")))
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {}},
                },
            },
        )
        blocks = responses[0]["result"]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "hello"

    async def test_non_string_content_converted_to_str(self) -> None:
        server = _make_server(sandbox=_FakeSandbox(result=SandboxResult(content={"a": 1})))
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {}},
                },
            },
        )
        text = responses[0]["result"]["content"][0]["text"]
        assert "a" in text  # dict repr contains "a"

    async def test_no_audit_entry_on_success(self) -> None:
        audit = CapturingAuditLog()
        server = _make_server(sandbox=_FakeSandbox(), audit_log=audit)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {}},
                },
            },
        )
        assert audit.entries == []

    async def test_otel_span_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = MockTracer()
        server = _make_server(tracer=tracer, monkeypatch=monkeypatch)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "echo", "tool_input": {}},
                },
            },
        )
        assert tracer.span.name == "sdk_mcp_server.tool_proxy"

    async def test_span_tool_name_attribute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = MockTracer()
        server = _make_server(tracer=tracer, monkeypatch=monkeypatch)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "echo", "tool_input": {}},
                },
            },
        )
        assert tracer.span.attributes["tool.name"] == "echo"

    async def test_span_session_id_attribute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = MockTracer()
        server = _make_server(tracer=tracer, monkeypatch=monkeypatch)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {}},
                },
            },
        )
        assert tracer.span.attributes["session.id"] == CTX.session_id

    async def test_span_mcp_proxy_attribute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = MockTracer()
        server = _make_server(tracer=tracer, monkeypatch=monkeypatch)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {}},
                },
            },
        )
        assert tracer.span.attributes["mcp.proxy"] is True

    async def test_span_has_invocation_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = MockTracer()
        server = _make_server(tracer=tracer, monkeypatch=monkeypatch)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {}},
                },
            },
        )
        event_names = [e[0] for e in tracer.span.events]
        assert "mcp_server.invocation" in event_names

    async def test_invocation_event_has_tool_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = MockTracer()
        server = _make_server(tracer=tracer, monkeypatch=monkeypatch)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "my_tool", "tool_input": {}},
                },
            },
        )
        event = next(e for e in tracer.span.events if e[0] == "mcp_server.invocation")
        assert event[1]["tool.name"] == "my_tool"

    async def test_invocation_event_has_session_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = MockTracer()
        server = _make_server(tracer=tracer, monkeypatch=monkeypatch)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {}},
                },
            },
        )
        event = next(e for e in tracer.span.events if e[0] == "mcp_server.invocation")
        assert event[1]["session.id"] == CTX.session_id

    async def test_invocation_event_operation_is_tool_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tracer = MockTracer()
        server = _make_server(tracer=tracer, monkeypatch=monkeypatch)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {}},
                },
            },
        )
        event = next(e for e in tracer.span.events if e[0] == "mcp_server.invocation")
        assert event[1]["operation"] == "tool_proxy"


# ---------------------------------------------------------------------------
# meridian_tool_proxy — sandbox returns is_error
# ---------------------------------------------------------------------------


class TestProxySandboxIsError:
    def _error_sandbox(self) -> _FakeSandbox:
        return _FakeSandbox(
            result=SandboxResult(
                content="denied",
                is_error=True,
                error_code="capability_denied",
                error_message="missing cap: bash",
            )
        )

    async def test_returns_mcp_is_error_true(self) -> None:
        server = _make_server(sandbox=self._error_sandbox())
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "bash", "tool_input": {}},
                },
            },
        )
        assert responses[0]["result"]["isError"] is True

    async def test_error_message_in_content(self) -> None:
        server = _make_server(sandbox=self._error_sandbox())
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "bash", "tool_input": {}},
                },
            },
        )
        text = responses[0]["result"]["content"][0]["text"]
        assert "missing cap" in text

    async def test_writes_proxy_failed_audit_entry(self) -> None:
        audit = CapturingAuditLog()
        server = _make_server(sandbox=self._error_sandbox(), audit_log=audit)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "bash", "tool_input": {}},
                },
            },
        )
        proxy_entries = [e for e in audit.entries if e.event == "sdk_mcp_server.proxy.failed"]
        assert len(proxy_entries) >= 1

    async def test_audit_entry_level_is_error(self) -> None:
        audit = CapturingAuditLog()
        server = _make_server(sandbox=self._error_sandbox(), audit_log=audit)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "bash", "tool_input": {}},
                },
            },
        )
        proxy_entries = [e for e in audit.entries if e.event == "sdk_mcp_server.proxy.failed"]
        assert proxy_entries[0].level == "error"

    async def test_span_marked_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = MockTracer()
        server = _make_server(sandbox=self._error_sandbox(), tracer=tracer, monkeypatch=monkeypatch)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "bash", "tool_input": {}},
                },
            },
        )
        assert tracer.span.status is not None
        assert tracer.span.status.status_code == StatusCode.ERROR

    async def test_span_has_proxy_error_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = MockTracer()
        server = _make_server(sandbox=self._error_sandbox(), tracer=tracer, monkeypatch=monkeypatch)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "bash", "tool_input": {}},
                },
            },
        )
        event_names = [e[0] for e in tracer.span.events]
        assert "mcp_server.proxy.error" in event_names


# ---------------------------------------------------------------------------
# meridian_tool_proxy — SandboxFailure raised
# ---------------------------------------------------------------------------


class TestProxySandboxFailure:
    def _failure_sandbox(self) -> _FakeSandbox:
        return _FakeSandbox(
            raises=SandboxFailure(
                code="TOOL_NOT_REGISTERED",
                message='No tool registered with name "ghost"',
                tool_name="ghost",
                session_id="sess-mcp-bridge",
                timestamp="2026-01-01T00:00:00+00:00",
            )
        )

    async def test_returns_mcp_is_error_true(self) -> None:
        server = _make_server(sandbox=self._failure_sandbox())
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "ghost", "tool_input": {}},
                },
            },
        )
        assert responses[0]["result"]["isError"] is True

    async def test_failure_message_in_content(self) -> None:
        server = _make_server(sandbox=self._failure_sandbox())
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "ghost", "tool_input": {}},
                },
            },
        )
        text = responses[0]["result"]["content"][0]["text"]
        assert "ghost" in text

    async def test_writes_audit_entry(self) -> None:
        audit = CapturingAuditLog()
        server = _make_server(sandbox=self._failure_sandbox(), audit_log=audit)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "ghost", "tool_input": {}},
                },
            },
        )
        assert any(e.event == "sdk_mcp_server.proxy.failed" for e in audit.entries)

    async def test_audit_entry_tool_name(self) -> None:
        audit = CapturingAuditLog()
        server = _make_server(sandbox=self._failure_sandbox(), audit_log=audit)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "ghost", "tool_input": {}},
                },
            },
        )
        proxy_entry = next(e for e in audit.entries if e.event == "sdk_mcp_server.proxy.failed")
        assert proxy_entry.tool_name == "ghost"

    async def test_span_marked_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = MockTracer()
        server = _make_server(
            sandbox=self._failure_sandbox(), tracer=tracer, monkeypatch=monkeypatch
        )
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "ghost", "tool_input": {}},
                },
            },
        )
        assert tracer.span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# meridian_tool_proxy — unexpected exception from sandbox
# ---------------------------------------------------------------------------


class TestProxyUnexpectedException:
    async def test_returns_mcp_is_error_true(self) -> None:
        server = _make_server(sandbox=_FakeSandbox(raises=RuntimeError("internal meltdown")))
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {}},
                },
            },
        )
        assert responses[0]["result"]["isError"] is True

    async def test_exception_message_in_content(self) -> None:
        server = _make_server(sandbox=_FakeSandbox(raises=RuntimeError("internal meltdown")))
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {}},
                },
            },
        )
        assert "internal meltdown" in responses[0]["result"]["content"][0]["text"]

    async def test_writes_audit_entry(self) -> None:
        audit = CapturingAuditLog()
        server = _make_server(sandbox=_FakeSandbox(raises=RuntimeError("boom")), audit_log=audit)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {}},
                },
            },
        )
        assert any(e.event == "sdk_mcp_server.proxy.failed" for e in audit.entries)

    async def test_span_marked_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = MockTracer()
        server = _make_server(
            sandbox=_FakeSandbox(raises=RuntimeError("boom")),
            tracer=tracer,
            monkeypatch=monkeypatch,
        )
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "t", "tool_input": {}},
                },
            },
        )
        assert tracer.span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# meridian_tool_proxy — missing tool_name in arguments
# ---------------------------------------------------------------------------


class TestProxyMissingToolName:
    async def test_returns_mcp_is_error_true(self) -> None:
        server = _make_server()
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_input": {}},  # tool_name omitted
                },
            },
        )
        assert responses[0]["result"]["isError"] is True

    async def test_content_mentions_tool_name(self) -> None:
        server = _make_server()
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_input": {}},
                },
            },
        )
        text = responses[0]["result"]["content"][0]["text"]
        assert "tool_name" in text

    async def test_writes_audit_entry(self) -> None:
        audit = CapturingAuditLog()
        server = _make_server(audit_log=audit)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {},  # both fields omitted
                },
            },
        )
        assert any(e.event == "sdk_mcp_server.proxy.failed" for e in audit.entries)

    async def test_span_marked_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = MockTracer()
        server = _make_server(tracer=tracer, monkeypatch=monkeypatch)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {},
                },
            },
        )
        assert tracer.span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# create_sdk_mcp_server factory
# ---------------------------------------------------------------------------


class TestCreateSdkMcpServer:
    def test_returns_sdk_mcp_server_instance(self) -> None:
        server = create_sdk_mcp_server(_FakeSandbox(), CTX)  # type: ignore[arg-type]
        assert isinstance(server, SdkMcpServer)

    def test_default_server_name_in_initialize(self) -> None:
        server = create_sdk_mcp_server(_FakeSandbox(), CTX)  # type: ignore[arg-type]

        async def _run() -> list[dict[str, Any]]:
            return await _exchange(
                server,
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            )

        responses = asyncio.run(_run())
        assert responses[0]["result"]["serverInfo"]["name"] == "meridian-sdk-mcp-server"

    def test_custom_server_name_injected(self) -> None:
        server = create_sdk_mcp_server(  # type: ignore[arg-type]
            _FakeSandbox(), CTX, server_name="custom-name"
        )
        assert server._server_name == "custom-name"

    def test_custom_audit_log_injected(self) -> None:
        audit = CapturingAuditLog()
        server = create_sdk_mcp_server(_FakeSandbox(), CTX, audit_log=audit)  # type: ignore[arg-type]
        assert server._audit_log is audit

    def test_noop_audit_log_when_none(self) -> None:
        from sdk_sandbox._audit import NoopAuditLog

        server = create_sdk_mcp_server(_FakeSandbox(), CTX)  # type: ignore[arg-type]
        assert isinstance(server._audit_log, NoopAuditLog)

    def test_context_stored(self) -> None:
        server = create_sdk_mcp_server(_FakeSandbox(), CTX)  # type: ignore[arg-type]
        assert server._context is CTX


# ---------------------------------------------------------------------------
# Integration: real Sandbox with InProcessDispatcher
# ---------------------------------------------------------------------------


class TestRealSandboxIntegration:
    """Use the actual Sandbox + InProcessDispatcher (no mocks) for a golden-path test."""

    def _build_sandbox(self) -> Sandbox:
        sandbox = Sandbox()
        dispatcher = InProcessDispatcher()

        async def _greet(inp: dict[str, Any], ctx: ExecutionContext) -> str:
            return f"Hello, {inp.get('name', 'world')}!"

        dispatcher.register("greet", _greet)
        sandbox.register_dispatcher(dispatcher)
        sandbox.register_tool(
            ToolDefinition(
                name="greet",
                description="Greet someone",
                input_schema={"type": "object"},
                handler=InProcessHandler(module="test"),
            )
        )
        return sandbox

    async def test_full_round_trip_success(self) -> None:
        sandbox = self._build_sandbox()
        server = SdkMcpServer(sandbox, CTX)
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "greet", "tool_input": {"name": "Alice"}},
                },
            },
        )
        result = responses[0]["result"]
        assert result["isError"] is False
        assert "Alice" in result["content"][0]["text"]

    async def test_full_round_trip_unknown_tool_is_error(self) -> None:
        sandbox = self._build_sandbox()
        server = SdkMcpServer(sandbox, CTX)
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "no_such_tool", "tool_input": {}},
                },
            },
        )
        assert responses[0]["result"]["isError"] is True

    async def test_audit_log_written_on_unknown_tool(self) -> None:
        sandbox = self._build_sandbox()
        audit = CapturingAuditLog()
        server = SdkMcpServer(sandbox, CTX, audit_log=audit)
        await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": PROXY_TOOL_NAME,
                    "arguments": {"tool_name": "no_such_tool", "tool_input": {}},
                },
            },
        )
        assert any(e.event == "sdk_mcp_server.proxy.failed" for e in audit.entries)


# ---------------------------------------------------------------------------
# Module public API
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_sdk_mcp_server_exported(self) -> None:
        import meridian_provider_claude_code_oauth as pkg

        assert hasattr(pkg, "SdkMcpServer")
        assert pkg.SdkMcpServer is SdkMcpServer

    def test_create_sdk_mcp_server_exported(self) -> None:
        import meridian_provider_claude_code_oauth as pkg

        assert hasattr(pkg, "create_sdk_mcp_server")
        assert pkg.create_sdk_mcp_server is create_sdk_mcp_server


# ---------------------------------------------------------------------------
# serve() edge-cases — readline failure, blank line, invalid JSON
# ---------------------------------------------------------------------------


class _RawBytesReader:
    """Async reader that returns pre-supplied raw lines (or raises)."""

    def __init__(self, lines: list[bytes | type]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        item = self._lines.pop(0)
        if isinstance(item, type) and issubclass(item, Exception):
            raise item("boom")
        assert isinstance(item, bytes)
        return item


class TestServeEdgeCases:
    async def test_serve_breaks_on_readline_exception(self) -> None:
        server = _make_server()
        reader = _RawBytesReader([RuntimeError])
        writer = _FakeWriter()
        await server.serve(reader, writer)  # type: ignore[arg-type]
        assert writer.responses() == []

    async def test_serve_skips_blank_lines(self) -> None:
        server = _make_server()
        reader = _RawBytesReader(
            [
                b"\n",  # blank — skipped
                b"   \n",  # whitespace-only — skipped
                (json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n").encode(),
            ]
        )
        writer = _FakeWriter()
        await server.serve(reader, writer)  # type: ignore[arg-type]
        responses = writer.responses()
        assert len(responses) == 1
        assert "result" in responses[0]

    async def test_serve_skips_invalid_json(self) -> None:
        server = _make_server()
        reader = _RawBytesReader(
            [
                b"not json {{{\n",  # invalid — skipped, no response
                (json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n").encode(),
            ]
        )
        writer = _FakeWriter()
        await server.serve(reader, writer)  # type: ignore[arg-type]
        responses = writer.responses()
        assert len(responses) == 1
        assert responses[0]["id"] == 2

    async def test_tools_call_unknown_tool_notification_no_response(self) -> None:
        """tools/call with unknown name AND no id (notification) → no response."""
        server = _make_server()
        responses = await _exchange(
            server,
            {
                "jsonrpc": "2.0",
                # no "id" → this is a notification
                "method": "tools/call",
                "params": {"name": "no_such_tool", "arguments": {}},
            },
        )
        assert responses == []
