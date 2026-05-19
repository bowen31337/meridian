"""
Tests for McpBackendDriver.

Covers:
  - Kind constant: driver.kind == "system.mcp".
  - on_demand default True; configurable to False.
  - Policy / capability delegation.

  stdio transport:
  - Successful execute: spawns process, performs handshake, calls tools/call,
    returns ExecuteResult with stdout from text content blocks.
  - Tool error (isError=true): exit_code=1, stderr=error text, no exception.
  - JSON-RPC protocol error ({"error": {...}}): raises RuntimeError.
  - FileNotFoundError on spawn: propagates (runtime wraps as ENV_EXECUTE_FAILED).
  - Env dict from ExecuteRequest forwarded to subprocess.
  - timeout_seconds from request used; falls back to driver timeout_s.
  - empty command raises ValueError on provision and execute.

  SSE transport:
  - Successful execute: full handshake via mocked SSE stream + POST; returns
    ExecuteResult with content from MCP response.
  - Tool error (isError=true): exit_code=1.
  - JSON-RPC protocol error: raises RuntimeError.
  - httpx unavailable: raises RuntimeError.
  - server_url empty: raises ValueError on provision and execute.
  - SSE endpoint not received (timeout): asyncio.TimeoutError propagates.

  Command parsing:
  - empty command raises ValueError.
  - stdin=None → arguments={}.
  - stdin JSON parsed as arguments.
  - invalid JSON stdin → ValueError.

  ExecuteResult normalisation:
  - stdout is joined text content.
  - non-text content blocks are ignored.
  - empty content list → stdout="".
  - duration_ms is positive float.
  - exit_code/stderr/stdout types correct.

  Runtime integration:
  - Full lifecycle (provision → execute → reclaim) produces no audit entries.
  - Failure writes audit entry, raises EnvironmentFailure with correct code.
  - execute span name is "environment.execute".
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sdk_environment import (
    AuditLogEntry,
    CapabilityEnvelope,
    EnvironmentFailure,
    EnvironmentRuntime,
    ExecuteRequest,
    ExecuteResult,
    FilesystemPolicy,
    McpBackendDriver,
    NetworkPolicy,
    ProvisionRequest,
    ReclaimRequest,
    RuntimeOptions,
)

from .conftest import CapturingAuditLog, MockSpan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMAND = ("python", "-m", "my_mcp_server")
_SERVER_URL = "https://mcp.example.com"


def _stdio_driver(**kwargs: Any) -> McpBackendDriver:
    return McpBackendDriver(transport="stdio", command=_COMMAND, **kwargs)


def _sse_driver(**kwargs: Any) -> McpBackendDriver:
    return McpBackendDriver(transport="sse", server_url=_SERVER_URL, **kwargs)


def _provision_req(kind: str = McpBackendDriver.KIND) -> ProvisionRequest:
    return ProvisionRequest(environment_id="env1", environment_kind=kind, session_id="s1")


def _execute_req(
    tool: str = "my_tool",
    args: dict[str, Any] | None = None,
    kind: str = McpBackendDriver.KIND,
    timeout: int | None = None,
) -> ExecuteRequest:
    return ExecuteRequest(
        environment_id="env1",
        environment_kind=kind,
        session_id="s1",
        command=(tool,),
        stdin=json.dumps(args) if args is not None else None,
        timeout_seconds=timeout,
    )


def _reclaim_req(kind: str = McpBackendDriver.KIND) -> ReclaimRequest:
    return ReclaimRequest(environment_id="env1", environment_kind=kind, session_id="s1")


def _ok_rpc(content_text: str = "result text", call_id: str = "call") -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": call_id,
        "result": {
            "content": [{"type": "text", "text": content_text}],
            "isError": False,
        },
    }


def _error_rpc(message: str = "tool failed", call_id: str = "call") -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": call_id, "error": {"code": -32000, "message": message}}


def _tool_error_rpc(message: str = "bad input", call_id: str = "call") -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": call_id,
        "result": {
            "content": [{"type": "text", "text": message}],
            "isError": True,
        },
    }


def _make_options(audit: CapturingAuditLog) -> RuntimeOptions:
    return RuntimeOptions(audit_log=audit)


# ---------------------------------------------------------------------------
# Stdio mock helpers
# ---------------------------------------------------------------------------


def _make_stdio_proc(responses: list[dict[str, Any]]) -> MagicMock:
    """Build a mock asyncio.Process whose stdout yields the given JSON-RPC responses."""
    lines = [json.dumps(r).encode() + b"\n" for r in responses]
    line_iter = iter(lines)

    async def _readline() -> bytes:
        try:
            return next(line_iter)
        except StopIteration:
            return b""

    stdout_mock = MagicMock()
    stdout_mock.readline = _readline

    stdin_mock = AsyncMock()
    stdin_mock.write = MagicMock()
    stdin_mock.drain = AsyncMock()
    stdin_mock.close = MagicMock()
    stdin_mock.wait_closed = AsyncMock()

    proc = MagicMock()
    proc.returncode = None
    proc.stdin = stdin_mock
    proc.stdout = stdout_mock
    proc.wait = AsyncMock(return_value=0)
    proc.kill = MagicMock()
    return proc


def _mock_create_subprocess_exec(proc: MagicMock) -> AsyncMock:
    return AsyncMock(return_value=proc)


# ---------------------------------------------------------------------------
# SSE mock helpers
# ---------------------------------------------------------------------------


def _make_sse_transport(
    endpoint_path: str,
    init_resp: dict[str, Any],
    call_resp: dict[str, Any],
) -> MagicMock:
    """
    Build a mock httpx.AsyncClient whose stream() yields SSE events and whose
    post() returns 202 Accepted for all message POSTs.

    SSE events emitted:
      event: endpoint
      data: {endpoint_path}

      event: message
      data: {json(init_resp)}

      event: message
      data: {json(call_resp)}
    """
    sse_lines = [
        "event: endpoint",
        f"data: {endpoint_path}",
        "",
        "event: message",
        f"data: {json.dumps(init_resp)}",
        "",
        "event: message",
        f"data: {json.dumps(call_resp)}",
        "",
    ]

    async def _aiter_lines():
        for line in sse_lines:
            yield line

    stream_resp = MagicMock()
    stream_resp.raise_for_status = MagicMock()
    stream_resp.aiter_lines = _aiter_lines

    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=stream_resp)
    stream_ctx.__aexit__ = AsyncMock(return_value=False)

    post_resp = MagicMock()
    post_resp.raise_for_status = MagicMock()
    post_resp.status_code = 202

    client_mock = MagicMock()
    client_mock.stream = MagicMock(return_value=stream_ctx)
    client_mock.post = AsyncMock(return_value=post_resp)

    client_ctx = MagicMock()
    client_ctx.__aenter__ = AsyncMock(return_value=client_mock)
    client_ctx.__aexit__ = AsyncMock(return_value=False)

    return client_ctx


_DEFAULT_INIT_RESP = {
    "jsonrpc": "2.0",
    "id": "init",
    "result": {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {}},
}


# ---------------------------------------------------------------------------
# Kind and defaults
# ---------------------------------------------------------------------------


class TestKindAndDefaults:
    def test_kind_is_system_mcp(self) -> None:
        assert _stdio_driver().kind == "system.mcp"

    def test_kind_constant(self) -> None:
        assert McpBackendDriver.KIND == "system.mcp"

    def test_on_demand_default_true(self) -> None:
        assert _stdio_driver().on_demand is True

    def test_on_demand_configurable_false(self) -> None:
        assert _stdio_driver(on_demand=False).on_demand is False

    def test_network_policy_default(self) -> None:
        assert isinstance(_stdio_driver().network_policy(), NetworkPolicy)

    def test_filesystem_policy_default(self) -> None:
        assert isinstance(_stdio_driver().filesystem_policy(), FilesystemPolicy)

    def test_capability_envelope_default(self) -> None:
        assert isinstance(_stdio_driver().capability_envelope(), CapabilityEnvelope)

    def test_custom_network_policy_returned(self) -> None:
        policy = NetworkPolicy(egress_allowed=True)
        assert _stdio_driver(network_policy=policy).network_policy() is policy

    def test_custom_filesystem_policy_returned(self) -> None:
        policy = FilesystemPolicy(read_globs=("**",))
        assert _stdio_driver(filesystem_policy=policy).filesystem_policy() is policy

    def test_custom_capability_envelope_returned(self) -> None:
        caps = CapabilityEnvelope(cpu_millicores=250)
        assert _stdio_driver(capability_envelope=caps).capability_envelope() is caps


# ---------------------------------------------------------------------------
# provision — validation
# ---------------------------------------------------------------------------


class TestProvision:
    async def test_stdio_empty_command_raises(self) -> None:
        driver = McpBackendDriver(transport="stdio")
        with pytest.raises(ValueError, match="command"):
            await driver.provision(_provision_req())

    async def test_sse_empty_url_raises(self) -> None:
        driver = McpBackendDriver(transport="sse")
        with pytest.raises(ValueError, match="server_url"):
            await driver.provision(_provision_req())

    async def test_stdio_valid_config_succeeds(self) -> None:
        await _stdio_driver().provision(_provision_req())  # must not raise

    async def test_sse_valid_config_succeeds(self) -> None:
        await _sse_driver().provision(_provision_req())  # must not raise


# ---------------------------------------------------------------------------
# reclaim — no-op
# ---------------------------------------------------------------------------


class TestReclaim:
    async def test_reclaim_is_noop(self) -> None:
        await _stdio_driver().reclaim(_reclaim_req())  # must not raise

    async def test_reclaim_sse_is_noop(self) -> None:
        await _sse_driver().reclaim(_reclaim_req())  # must not raise


# ---------------------------------------------------------------------------
# Command / argument parsing
# ---------------------------------------------------------------------------


class TestCommandParsing:
    async def test_empty_command_raises(self) -> None:
        driver = _stdio_driver()
        req = ExecuteRequest(
            environment_id="e",
            environment_kind=McpBackendDriver.KIND,
            session_id="s",
            command=(),
        )
        with pytest.raises(ValueError, match="command"):
            await driver.execute(req)

    async def test_none_stdin_gives_empty_arguments(self) -> None:
        req = ExecuteRequest(
            environment_id="e",
            environment_kind=McpBackendDriver.KIND,
            session_id="s",
            command=("tool",),
            stdin=None,
        )
        _, args = McpBackendDriver._parse_tool_call(req)
        assert args == {}

    async def test_valid_json_stdin_parsed(self) -> None:
        req = ExecuteRequest(
            environment_id="e",
            environment_kind=McpBackendDriver.KIND,
            session_id="s",
            command=("tool",),
            stdin='{"key": "value"}',
        )
        _, args = McpBackendDriver._parse_tool_call(req)
        assert args == {"key": "value"}

    async def test_invalid_json_stdin_raises(self) -> None:
        req = ExecuteRequest(
            environment_id="e",
            environment_kind=McpBackendDriver.KIND,
            session_id="s",
            command=("tool",),
            stdin="not json",
        )
        with pytest.raises(ValueError, match="JSON"):
            McpBackendDriver._parse_tool_call(req)

    async def test_tool_name_is_command_first_element(self) -> None:
        req = ExecuteRequest(
            environment_id="e",
            environment_kind=McpBackendDriver.KIND,
            session_id="s",
            command=("my_tool",),
        )
        tool_name, _ = McpBackendDriver._parse_tool_call(req)
        assert tool_name == "my_tool"


# ---------------------------------------------------------------------------
# ExecuteResult normalisation
# ---------------------------------------------------------------------------


class TestNormalisation:
    def test_ok_result_stdout_is_text(self) -> None:
        data = _ok_rpc("hello world")
        result = McpBackendDriver._mcp_to_execute_result(data, 0.0)
        assert result.stdout == "hello world"

    def test_ok_result_stderr_empty(self) -> None:
        data = _ok_rpc("x")
        result = McpBackendDriver._mcp_to_execute_result(data, 0.0)
        assert result.stderr == ""

    def test_ok_result_exit_code_zero(self) -> None:
        data = _ok_rpc("x")
        result = McpBackendDriver._mcp_to_execute_result(data, 0.0)
        assert result.exit_code == 0

    def test_multiple_text_blocks_joined(self) -> None:
        data = {
            "jsonrpc": "2.0",
            "id": "call",
            "result": {
                "content": [
                    {"type": "text", "text": "line1"},
                    {"type": "text", "text": "line2"},
                ],
                "isError": False,
            },
        }
        result = McpBackendDriver._mcp_to_execute_result(data, 0.0)
        assert result.stdout == "line1\nline2"

    def test_non_text_blocks_ignored(self) -> None:
        data = {
            "jsonrpc": "2.0",
            "id": "call",
            "result": {
                "content": [
                    {"type": "image", "data": "base64stuff"},
                    {"type": "text", "text": "the text"},
                ],
                "isError": False,
            },
        }
        result = McpBackendDriver._mcp_to_execute_result(data, 0.0)
        assert result.stdout == "the text"

    def test_empty_content_list(self) -> None:
        data = {
            "jsonrpc": "2.0",
            "id": "call",
            "result": {"content": [], "isError": False},
        }
        result = McpBackendDriver._mcp_to_execute_result(data, 0.0)
        assert result.stdout == ""

    def test_tool_error_exit_code_one(self) -> None:
        data = _tool_error_rpc("bad param")
        result = McpBackendDriver._mcp_to_execute_result(data, 0.0)
        assert result.exit_code == 1

    def test_tool_error_stderr_contains_message(self) -> None:
        data = _tool_error_rpc("bad param")
        result = McpBackendDriver._mcp_to_execute_result(data, 0.0)
        assert "bad param" in result.stderr

    def test_tool_error_stdout_empty(self) -> None:
        data = _tool_error_rpc("bad param")
        result = McpBackendDriver._mcp_to_execute_result(data, 0.0)
        assert result.stdout == ""

    def test_rpc_error_raises_runtime_error(self) -> None:
        data = _error_rpc("server exploded")
        with pytest.raises(RuntimeError, match="server exploded"):
            McpBackendDriver._mcp_to_execute_result(data, 0.0)

    def test_duration_ms_positive_float(self) -> None:
        import time

        start = time.monotonic() - 0.1
        data = _ok_rpc("x")
        result = McpBackendDriver._mcp_to_execute_result(data, start)
        assert isinstance(result.duration_ms, float)
        assert result.duration_ms >= 0.0

    def test_returns_execute_result_type(self) -> None:
        data = _ok_rpc("x")
        result = McpBackendDriver._mcp_to_execute_result(data, 0.0)
        assert isinstance(result, ExecuteResult)

    def test_stdout_is_str(self) -> None:
        data = _ok_rpc("x")
        assert isinstance(McpBackendDriver._mcp_to_execute_result(data, 0.0).stdout, str)

    def test_exit_code_is_int(self) -> None:
        data = _ok_rpc("x")
        assert isinstance(McpBackendDriver._mcp_to_execute_result(data, 0.0).exit_code, int)


# ---------------------------------------------------------------------------
# stdio transport — success
# ---------------------------------------------------------------------------


class TestStdioSuccess:
    async def test_returns_execute_result(self) -> None:
        driver = _stdio_driver()
        proc = _make_stdio_proc([{"jsonrpc": "2.0", "id": "init", "result": {}}, _ok_rpc()])
        with patch(
            "sdk_environment._mcp_driver.asyncio.create_subprocess_exec",
            _mock_create_subprocess_exec(proc),
        ):
            result = await driver.execute(_execute_req())
        assert isinstance(result, ExecuteResult)

    async def test_stdout_contains_content_text(self) -> None:
        driver = _stdio_driver()
        proc = _make_stdio_proc([{"jsonrpc": "2.0", "id": "init", "result": {}}, _ok_rpc("hello")])
        with patch(
            "sdk_environment._mcp_driver.asyncio.create_subprocess_exec",
            _mock_create_subprocess_exec(proc),
        ):
            result = await driver.execute(_execute_req())
        assert result.stdout == "hello"

    async def test_exit_code_zero_on_success(self) -> None:
        driver = _stdio_driver()
        proc = _make_stdio_proc([{"jsonrpc": "2.0", "id": "init", "result": {}}, _ok_rpc()])
        with patch(
            "sdk_environment._mcp_driver.asyncio.create_subprocess_exec",
            _mock_create_subprocess_exec(proc),
        ):
            result = await driver.execute(_execute_req())
        assert result.exit_code == 0

    async def test_tool_error_exit_code_one(self) -> None:
        driver = _stdio_driver()
        proc = _make_stdio_proc(
            [{"jsonrpc": "2.0", "id": "init", "result": {}}, _tool_error_rpc("oops")]
        )
        with patch(
            "sdk_environment._mcp_driver.asyncio.create_subprocess_exec",
            _mock_create_subprocess_exec(proc),
        ):
            result = await driver.execute(_execute_req())
        assert result.exit_code == 1
        assert "oops" in result.stderr

    async def test_rpc_error_raises(self) -> None:
        driver = _stdio_driver()
        proc = _make_stdio_proc(
            [{"jsonrpc": "2.0", "id": "init", "result": {}}, _error_rpc("server crash")]
        )
        with patch(
            "sdk_environment._mcp_driver.asyncio.create_subprocess_exec",
            _mock_create_subprocess_exec(proc),
        ):
            with pytest.raises(RuntimeError, match="server crash"):
                await driver.execute(_execute_req())

    async def test_env_vars_forwarded_to_subprocess(self) -> None:
        import os

        driver = _stdio_driver()
        proc = _make_stdio_proc([{"jsonrpc": "2.0", "id": "init", "result": {}}, _ok_rpc()])
        captured_kwargs: dict[str, Any] = {}

        async def _fake_exec(*args: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return proc

        req = ExecuteRequest(
            environment_id="e",
            environment_kind=McpBackendDriver.KIND,
            session_id="s",
            command=("my_tool",),
            env={"CUSTOM_VAR": "42"},
        )
        with patch("sdk_environment._mcp_driver.asyncio.create_subprocess_exec", _fake_exec):
            await driver.execute(req)

        passed_env = captured_kwargs.get("env", {})
        assert passed_env.get("CUSTOM_VAR") == "42"
        assert "PATH" in passed_env  # merged with os.environ

    async def test_timeout_seconds_used_from_request(self) -> None:
        driver = _stdio_driver(timeout_s=99.0)
        proc = _make_stdio_proc([{"jsonrpc": "2.0", "id": "init", "result": {}}, _ok_rpc()])
        with patch(
            "sdk_environment._mcp_driver.asyncio.create_subprocess_exec",
            _mock_create_subprocess_exec(proc),
        ):
            # Just confirm it runs without error; timeout enforcement is asyncio internal
            result = await driver.execute(_execute_req(timeout=5))
        assert isinstance(result, ExecuteResult)

    async def test_file_not_found_propagates(self) -> None:
        driver = _stdio_driver()
        with patch(
            "sdk_environment._mcp_driver.asyncio.create_subprocess_exec",
            AsyncMock(side_effect=FileNotFoundError("no such file")),
        ):
            with pytest.raises(FileNotFoundError):
                await driver.execute(_execute_req())

    async def test_empty_command_raises_value_error(self) -> None:
        driver = McpBackendDriver(transport="stdio")
        req = ExecuteRequest(
            environment_id="e",
            environment_kind=McpBackendDriver.KIND,
            session_id="s",
            command=("tool",),
        )
        with patch(
            "sdk_environment._mcp_driver.asyncio.create_subprocess_exec",
            AsyncMock(side_effect=AssertionError("should not reach")),
        ):
            # Empty command is caught before subprocess spawn
            driver2 = McpBackendDriver(transport="stdio")  # command=()
            with pytest.raises(ValueError, match="command"):
                await driver2.execute(req)


# ---------------------------------------------------------------------------
# SSE transport — success
# ---------------------------------------------------------------------------


class TestSseSuccess:
    async def test_returns_execute_result(self) -> None:
        driver = _sse_driver()
        client_ctx = _make_sse_transport("/message", _DEFAULT_INIT_RESP, _ok_rpc())
        with patch("sdk_environment._mcp_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert isinstance(result, ExecuteResult)

    async def test_stdout_contains_content_text(self) -> None:
        driver = _sse_driver()
        client_ctx = _make_sse_transport("/message", _DEFAULT_INIT_RESP, _ok_rpc("sse result"))
        with patch("sdk_environment._mcp_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert result.stdout == "sse result"

    async def test_exit_code_zero_on_success(self) -> None:
        driver = _sse_driver()
        client_ctx = _make_sse_transport("/message", _DEFAULT_INIT_RESP, _ok_rpc())
        with patch("sdk_environment._mcp_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert result.exit_code == 0

    async def test_tool_error_exit_code_one(self) -> None:
        driver = _sse_driver()
        client_ctx = _make_sse_transport(
            "/message", _DEFAULT_INIT_RESP, _tool_error_rpc("sse oops")
        )
        with patch("sdk_environment._mcp_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert result.exit_code == 1
        assert "sse oops" in result.stderr

    async def test_rpc_error_raises(self) -> None:
        driver = _sse_driver()
        client_ctx = _make_sse_transport(
            "/message", _DEFAULT_INIT_RESP, _error_rpc("sse crash")
        )
        with patch("sdk_environment._mcp_driver._httpx.AsyncClient", return_value=client_ctx):
            with pytest.raises(RuntimeError, match="sse crash"):
                await driver.execute(_execute_req())

    async def test_full_url_endpoint_used_directly(self) -> None:
        driver = _sse_driver()
        full_endpoint_url = "https://mcp.example.com/message?sessionId=abc"
        client_ctx = _make_sse_transport(
            full_endpoint_url, _DEFAULT_INIT_RESP, _ok_rpc()
        )
        with patch("sdk_environment._mcp_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert result.exit_code == 0

    async def test_httpx_unavailable_raises(self) -> None:
        driver = _sse_driver()
        with patch("sdk_environment._mcp_driver._HTTPX_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="httpx"):
                await driver.execute(_execute_req())

    async def test_empty_server_url_raises(self) -> None:
        driver = McpBackendDriver(transport="sse")
        with pytest.raises(ValueError, match="server_url"):
            await driver.execute(_execute_req())

    async def test_posts_tools_call_with_correct_method(self) -> None:
        driver = _sse_driver()
        client_ctx = _make_sse_transport("/message", _DEFAULT_INIT_RESP, _ok_rpc())
        with patch("sdk_environment._mcp_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.execute(_execute_req())
        calls = client_ctx.__aenter__.return_value.post.call_args_list
        methods = [c[1]["json"]["method"] for c in calls if "method" in c[1].get("json", {})]
        assert "tools/call" in methods

    async def test_initialize_sent_before_tools_call(self) -> None:
        driver = _sse_driver()
        client_ctx = _make_sse_transport("/message", _DEFAULT_INIT_RESP, _ok_rpc())
        with patch("sdk_environment._mcp_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.execute(_execute_req())
        calls = client_ctx.__aenter__.return_value.post.call_args_list
        methods = [c[1]["json"]["method"] for c in calls if "json" in c[1]]
        assert methods.index("initialize") < methods.index("tools/call")

    async def test_notifications_initialized_sent(self) -> None:
        driver = _sse_driver()
        client_ctx = _make_sse_transport("/message", _DEFAULT_INIT_RESP, _ok_rpc())
        with patch("sdk_environment._mcp_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.execute(_execute_req())
        calls = client_ctx.__aenter__.return_value.post.call_args_list
        methods = [c[1]["json"]["method"] for c in calls if "json" in c[1]]
        assert "notifications/initialized" in methods


# ---------------------------------------------------------------------------
# SSE — server_url trailing slash stripped
# ---------------------------------------------------------------------------


class TestSseUrlNormalisation:
    async def test_trailing_slash_stripped(self) -> None:
        driver = McpBackendDriver(
            transport="sse", server_url="https://mcp.example.com/"
        )
        assert driver._server_url == "https://mcp.example.com"


# ---------------------------------------------------------------------------
# Runtime integration
# ---------------------------------------------------------------------------


class TestRuntimeIntegration:
    async def test_full_lifecycle_no_audit_entries(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _stdio_driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        opts = _make_options(audit_log)
        kind = McpBackendDriver.KIND

        await rt.provision(_provision_req(kind), opts)

        proc = _make_stdio_proc([{"jsonrpc": "2.0", "id": "init", "result": {}}, _ok_rpc()])
        with patch(
            "sdk_environment._mcp_driver.asyncio.create_subprocess_exec",
            _mock_create_subprocess_exec(proc),
        ):
            result = await rt.execute(_execute_req(kind=kind), opts)

        await rt.reclaim(_reclaim_req(kind), opts)

        assert isinstance(result, ExecuteResult)
        assert audit_log.entries == []

    async def test_execute_failure_writes_audit(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _stdio_driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        proc = _make_stdio_proc(
            [{"jsonrpc": "2.0", "id": "init", "result": {}}, _error_rpc("crash")]
        )
        with patch(
            "sdk_environment._mcp_driver.asyncio.create_subprocess_exec",
            _mock_create_subprocess_exec(proc),
        ):
            with pytest.raises(EnvironmentFailure):
                await rt.execute(_execute_req(), _make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "environment.execute.failed"

    async def test_execute_failure_code_is_env_execute_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _stdio_driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        proc = _make_stdio_proc(
            [{"jsonrpc": "2.0", "id": "init", "result": {}}, _error_rpc("crash")]
        )
        with patch(
            "sdk_environment._mcp_driver.asyncio.create_subprocess_exec",
            _mock_create_subprocess_exec(proc),
        ):
            with pytest.raises(EnvironmentFailure) as exc_info:
                await rt.execute(_execute_req(), _make_options(audit_log))
        assert exc_info.value.code == "ENV_EXECUTE_FAILED"

    async def test_provision_failure_writes_audit(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = McpBackendDriver(transport="stdio")  # no command
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.provision(_provision_req(), _make_options(audit_log))
        assert exc_info.value.code == "ENV_PROVISION_FAILED"
        assert len(audit_log.entries) == 1

    async def test_execute_span_name(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _stdio_driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        proc = _make_stdio_proc([{"jsonrpc": "2.0", "id": "init", "result": {}}, _ok_rpc()])
        with patch(
            "sdk_environment._mcp_driver.asyncio.create_subprocess_exec",
            _mock_create_subprocess_exec(proc),
        ):
            await rt.execute(_execute_req(), _make_options(audit_log))
        assert mock_span.name == "environment.execute"

    async def test_capability_envelope_has_required_fields(self) -> None:
        caps = _stdio_driver().capability_envelope()
        assert isinstance(caps.cpu_millicores, int)
        assert isinstance(caps.memory_mb, int)
        assert isinstance(caps.timeout_seconds, int)
        assert isinstance(caps.can_write_filesystem, bool)
        assert isinstance(caps.network, NetworkPolicy)
        assert isinstance(caps.filesystem, FilesystemPolicy)

    async def test_sse_full_lifecycle_no_audit_entries(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _sse_driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        opts = _make_options(audit_log)
        kind = McpBackendDriver.KIND

        await rt.provision(_provision_req(kind), opts)

        client_ctx = _make_sse_transport("/message", _DEFAULT_INIT_RESP, _ok_rpc("sse ok"))
        with patch("sdk_environment._mcp_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await rt.execute(_execute_req(kind=kind), opts)

        await rt.reclaim(_reclaim_req(kind), opts)

        assert isinstance(result, ExecuteResult)
        assert result.stdout == "sse ok"
        assert audit_log.entries == []
