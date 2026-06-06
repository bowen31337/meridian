"""
Tests for concrete ToolDispatcher implementations.

Covers per-dispatcher:
  - kind property returns the correct string
  - Satisfies ToolDispatcher ABC
  - Successful dispatch: returns SandboxResult with content, no audit entry
  - OTel span opened with correct name, attributes, structured event
  - Each failure mode: is_error=True, correct error_code, audit entry written,
    span marked ERROR
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from opentelemetry.trace import StatusCode
import pytest
from sdk_sandbox import (
    ContainerHandler,
    ExecutionContext,
    HttpHandler,
    InProcessHandler,
    McpHandler,
    SandboxFailure,
    SandboxResult,
    SubprocessHandler,
    ToolDefinition,
    ToolDispatcher,
)
from sdk_sandbox._dispatchers import (
    ContainerDispatcher,
    HttpDispatcher,
    InProcessDispatcher,
    McpDispatcher,
    SubprocessDispatcher,
)

from .conftest import CapturingAuditLog, MockSpan, MockTracer

# ---------------------------------------------------------------------------
# OTel mock fixtures — one tracer per dispatcher module
# ---------------------------------------------------------------------------


def _make_tracer_fixture(module_path: str) -> tuple:
    @pytest.fixture()
    def mock_tracer(monkeypatch: pytest.MonkeyPatch) -> MockTracer:
        tracer = MockTracer()
        monkeypatch.setattr(module_path, lambda: tracer)
        return tracer

    @pytest.fixture()
    def mock_span(mock_tracer: MockTracer) -> MockSpan:
        return mock_tracer.span

    return mock_tracer, mock_span


# All dispatcher classes import get_tracer from ._dispatchers
@pytest.fixture()
def mock_tracer(monkeypatch: pytest.MonkeyPatch) -> MockTracer:
    tracer = MockTracer()
    monkeypatch.setattr("sdk_sandbox._dispatchers.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_span(mock_tracer: MockTracer) -> MockSpan:
    return mock_tracer.span


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

CTX = ExecutionContext(session_id="sess-disp", workspace="/workspace", scratch_dir="/tmp/scratch")


def _tool(handler: Any, name: str = "test.tool") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Test tool",
        input_schema={"type": "object"},
        handler=handler,
    )


# ---------------------------------------------------------------------------
# InProcessDispatcher
# ---------------------------------------------------------------------------


class TestInProcessDispatcher:
    def test_satisfies_abc(self) -> None:
        d = InProcessDispatcher()
        assert isinstance(d, ToolDispatcher)

    def test_kind(self) -> None:
        assert InProcessDispatcher().kind == "in_process"

    def test_register_duplicate_raises(self) -> None:
        d = InProcessDispatcher()
        d.register("test.tool", AsyncMock(return_value="ok"))
        with pytest.raises(ValueError, match="already registered"):
            d.register("test.tool", AsyncMock(return_value="ok"))

    # success

    async def test_success_returns_result(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        d.register("test.tool", AsyncMock(return_value={"count": 3}))
        result = await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        assert isinstance(result, SandboxResult)
        assert result.content == {"count": 3}
        assert result.is_error is False

    async def test_success_no_audit_entry(self, mock_span: MockSpan) -> None:
        audit = CapturingAuditLog()
        d = InProcessDispatcher(audit_log=audit)
        d.register("test.tool", AsyncMock(return_value="ok"))
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        assert audit.entries == []

    async def test_success_span_name(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        d.register("test.tool", AsyncMock(return_value="ok"))
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        assert mock_span.name == "in_process.dispatch"

    async def test_success_span_attributes(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        d.register("test.tool", AsyncMock(return_value="ok"))
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        assert mock_span.attributes["tool.name"] == "test.tool"
        assert mock_span.attributes["session.id"] == "sess-disp"

    async def test_success_structured_event(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        d.register("test.tool", AsyncMock(return_value="ok"))
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        event_names = [e[0] for e in mock_span.events]
        assert "in_process.dispatch" in event_names

    async def test_success_forwards_input_and_context(self, mock_span: MockSpan) -> None:
        captured: list[tuple] = []

        async def fn(inp: dict, ctx: ExecutionContext) -> str:
            captured.append((inp, ctx))
            return "done"

        d = InProcessDispatcher()
        d.register("test.tool", fn)
        await d.dispatch(_tool(InProcessHandler()), {"x": 1}, CTX)
        assert len(captured) == 1
        assert captured[0][0] == {"x": 1}
        assert captured[0][1] is CTX

    # handler not found — raises SandboxFailure

    async def test_handler_not_found_raises(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        with pytest.raises(SandboxFailure) as exc_info:
            await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        assert exc_info.value.code == "IN_PROCESS_HANDLER_NOT_FOUND"

    async def test_handler_not_found_span_marked_error(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        with pytest.raises(SandboxFailure):
            await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_handler_not_found_audit_entry(self, mock_span: MockSpan) -> None:
        audit = CapturingAuditLog()
        d = InProcessDispatcher(audit_log=audit)
        with pytest.raises(SandboxFailure):
            await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        assert len(audit.entries) == 1
        assert audit.entries[0].level == "error"
        assert audit.entries[0].event == "in_process.handler.not_found"

    # callable raises — returns is_error

    async def test_callable_raises_returns_is_error(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        d.register("test.tool", AsyncMock(side_effect=RuntimeError("boom")))
        result = await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        assert result.is_error is True
        assert result.error_code == "in_process_handler_failed"
        assert "boom" in (result.error_message or "")

    async def test_callable_raises_span_marked_error(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        d.register("test.tool", AsyncMock(side_effect=RuntimeError("boom")))
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_callable_raises_audit_entry(self, mock_span: MockSpan) -> None:
        audit = CapturingAuditLog()
        d = InProcessDispatcher(audit_log=audit)
        d.register("test.tool", AsyncMock(side_effect=RuntimeError("oops")))
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "in_process.handler.failed"

    async def test_callable_raises_error_in_content(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        d.register("test.tool", AsyncMock(side_effect=ValueError("bad value")))
        result = await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        assert "bad value" in result.content


# ---------------------------------------------------------------------------
# SubprocessDispatcher — uses a real python child process
# ---------------------------------------------------------------------------

_ECHO_SCRIPT = """\
import sys, json
req = json.load(sys.stdin)
json.dump({"result": req["args"]}, sys.stdout)
"""

_ERROR_SCRIPT = """\
import sys, json
json.dump({"error": {"code": "tool_error", "message": "bad input"}}, sys.stdout)
"""

_CRASH_SCRIPT = """\
import sys
sys.stderr.write("something went wrong\\n")
sys.exit(1)
"""

_BADJSON_SCRIPT = """\
import sys
sys.stdout.write("not json")
"""


def _write_script(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content)
    p.chmod(0o755)
    return str(p)


class TestSubprocessDispatcher:
    def test_satisfies_abc(self) -> None:
        assert isinstance(SubprocessDispatcher(), ToolDispatcher)

    def test_kind(self) -> None:
        assert SubprocessDispatcher().kind == "subprocess"

    # success

    async def test_success_returns_result(self, tmp_path: Path, mock_span: MockSpan) -> None:
        script = _write_script(tmp_path, "echo.py", _ECHO_SCRIPT)
        d = SubprocessDispatcher()
        _tool(SubprocessHandler(path=f"{sys.executable} {script}".split()[0]))
        # run via python interpreter directly
        _tool(SubprocessHandler(path=sys.executable))
        # easier: write a proper executable
        exe = tmp_path / "echo_tool"
        exe.write_text(f"#!{sys.executable}\n{_ECHO_SCRIPT}")
        exe.chmod(0o755)
        result = await d.dispatch(_tool(SubprocessHandler(path=str(exe))), {"key": "val"}, CTX)
        assert result.is_error is False
        assert result.content == {"key": "val"}

    async def test_success_span_name(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = tmp_path / "echo_tool"
        exe.write_text(f"#!{sys.executable}\n{_ECHO_SCRIPT}")
        exe.chmod(0o755)
        d = SubprocessDispatcher()
        await d.dispatch(_tool(SubprocessHandler(path=str(exe))), {}, CTX)
        assert mock_span.name == "subprocess.dispatch"

    async def test_success_structured_event(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = tmp_path / "echo_tool"
        exe.write_text(f"#!{sys.executable}\n{_ECHO_SCRIPT}")
        exe.chmod(0o755)
        d = SubprocessDispatcher()
        await d.dispatch(_tool(SubprocessHandler(path=str(exe))), {}, CTX)
        event_names = [e[0] for e in mock_span.events]
        assert "subprocess.dispatch" in event_names

    async def test_success_no_audit_entry(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = tmp_path / "echo_tool"
        exe.write_text(f"#!{sys.executable}\n{_ECHO_SCRIPT}")
        exe.chmod(0o755)
        audit = CapturingAuditLog()
        d = SubprocessDispatcher(audit_log=audit)
        await d.dispatch(_tool(SubprocessHandler(path=str(exe))), {}, CTX)
        assert audit.entries == []

    # tool error response

    async def test_tool_error_returns_is_error(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = tmp_path / "error_tool"
        exe.write_text(f"#!{sys.executable}\n{_ERROR_SCRIPT}")
        exe.chmod(0o755)
        d = SubprocessDispatcher()
        result = await d.dispatch(_tool(SubprocessHandler(path=str(exe))), {}, CTX)
        assert result.is_error is True
        assert result.error_code == "tool_error"
        assert "bad input" in (result.error_message or "")

    async def test_tool_error_audit_entry(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = tmp_path / "error_tool"
        exe.write_text(f"#!{sys.executable}\n{_ERROR_SCRIPT}")
        exe.chmod(0o755)
        audit = CapturingAuditLog()
        d = SubprocessDispatcher(audit_log=audit)
        await d.dispatch(_tool(SubprocessHandler(path=str(exe))), {}, CTX)
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "subprocess.tool.error"

    # nonzero exit

    async def test_nonzero_exit_returns_is_error(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = tmp_path / "crash_tool"
        exe.write_text(f"#!{sys.executable}\n{_CRASH_SCRIPT}")
        exe.chmod(0o755)
        d = SubprocessDispatcher()
        result = await d.dispatch(_tool(SubprocessHandler(path=str(exe))), {}, CTX)
        assert result.is_error is True
        assert result.error_code == "subprocess_nonzero_exit"

    async def test_nonzero_exit_audit_entry(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = tmp_path / "crash_tool"
        exe.write_text(f"#!{sys.executable}\n{_CRASH_SCRIPT}")
        exe.chmod(0o755)
        audit = CapturingAuditLog()
        d = SubprocessDispatcher(audit_log=audit)
        await d.dispatch(_tool(SubprocessHandler(path=str(exe))), {}, CTX)
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "subprocess.nonzero_exit"

    async def test_nonzero_exit_stderr_in_detail(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = tmp_path / "crash_tool"
        exe.write_text(f"#!{sys.executable}\n{_CRASH_SCRIPT}")
        exe.chmod(0o755)
        audit = CapturingAuditLog()
        d = SubprocessDispatcher(audit_log=audit)
        await d.dispatch(_tool(SubprocessHandler(path=str(exe))), {}, CTX)
        assert audit.entries[0].detail is not None
        assert "stderr_tail" in audit.entries[0].detail

    # invalid JSON

    async def test_invalid_json_returns_is_error(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = tmp_path / "badjson_tool"
        exe.write_text(f"#!{sys.executable}\n{_BADJSON_SCRIPT}")
        exe.chmod(0o755)
        d = SubprocessDispatcher()
        result = await d.dispatch(_tool(SubprocessHandler(path=str(exe))), {}, CTX)
        assert result.is_error is True
        assert result.error_code == "subprocess_invalid_json"

    # binary not found

    async def test_binary_not_found_returns_is_error(self, mock_span: MockSpan) -> None:
        d = SubprocessDispatcher()
        result = await d.dispatch(
            _tool(SubprocessHandler(path="/nonexistent/bin/missing")), {}, CTX
        )
        assert result.is_error is True
        assert result.error_code == "subprocess_binary_not_found"

    async def test_binary_not_found_audit_entry(self, mock_span: MockSpan) -> None:
        audit = CapturingAuditLog()
        d = SubprocessDispatcher(audit_log=audit)
        await d.dispatch(_tool(SubprocessHandler(path="/no/such/binary")), {}, CTX)
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "subprocess.binary.not_found"

    async def test_binary_not_found_span_error(self, mock_span: MockSpan) -> None:
        d = SubprocessDispatcher()
        await d.dispatch(_tool(SubprocessHandler(path="/no/such/binary")), {}, CTX)
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_payload_includes_thread_id(self, tmp_path: Path, mock_span: MockSpan) -> None:
        script_text = (
            "import sys, json\n"
            "req = json.load(sys.stdin)\n"
            "import json as _j; open('/dev/null','w').write('')\n"
            "sys.stdout.write(_j.dumps({'result': req['context']}))\n"
        )
        exe = tmp_path / "ctx_tool"
        exe.write_text(f"#!{sys.executable}\n{script_text}")
        exe.chmod(0o755)
        ctx_with_thread = ExecutionContext(
            session_id="sess-disp",
            workspace="/workspace",
            thread_id="thread-123",
            scratch_dir="/tmp/scratch",
        )
        d = SubprocessDispatcher()
        result = await d.dispatch(_tool(SubprocessHandler(path=str(exe))), {}, ctx_with_thread)
        assert result.is_error is False
        assert result.content["thread_id"] == "thread-123"
        assert result.content["session_id"] == "sess-disp"
        assert result.content["workspace"] == "/workspace"
        assert result.content["scratch_dir"] == "/tmp/scratch"

    async def test_unexpected_spawn_error_propagates(self, mock_span: MockSpan) -> None:
        # A non-FileNotFoundError before proc is assigned: proc stays None, so
        # the cleanup guard is skipped and the exception re-raises.
        d = SubprocessDispatcher()
        with (
            patch("asyncio.create_subprocess_exec", side_effect=RuntimeError("spawn boom")),
            pytest.raises(RuntimeError, match="spawn boom"),
        ):
            await d.dispatch(_tool(SubprocessHandler(path=sys.executable)), {}, CTX)


# ---------------------------------------------------------------------------
# McpDispatcher — mocked httpx
# ---------------------------------------------------------------------------


class TestMcpDispatcher:
    def test_satisfies_abc(self) -> None:
        assert isinstance(McpDispatcher(), ToolDispatcher)

    def test_kind(self) -> None:
        assert McpDispatcher().kind == "mcp"

    def _tool(self) -> ToolDefinition:
        return _tool(McpHandler(server_url="http://mcp.local", tool_name="my_tool"))

    def _mock_response(self, data: dict[str, Any]) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=data)
        return resp

    async def test_success_returns_result(self, mock_span: MockSpan) -> None:
        data = {
            "jsonrpc": "2.0",
            "id": "x",
            "result": {"content": [{"type": "text", "text": "hello"}], "isError": False},
        }
        resp = self._mock_response(data)
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value = mock_client

            d = McpDispatcher()
            result = await d.dispatch(self._tool(), {}, CTX)

        assert result.is_error is False
        assert result.content == "hello"

    async def test_success_span_name(self, mock_span: MockSpan) -> None:
        data = {
            "jsonrpc": "2.0",
            "id": "x",
            "result": {"content": [], "isError": False},
        }
        resp = self._mock_response(data)
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value = mock_client

            await McpDispatcher().dispatch(self._tool(), {}, CTX)

        assert mock_span.name == "mcp.dispatch"

    async def test_success_structured_event(self, mock_span: MockSpan) -> None:
        data = {"jsonrpc": "2.0", "id": "x", "result": {"content": [], "isError": False}}
        resp = self._mock_response(data)
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value = mock_client

            await McpDispatcher().dispatch(self._tool(), {}, CTX)

        event_names = [e[0] for e in mock_span.events]
        assert "mcp.dispatch" in event_names

    async def test_rpc_error_returns_is_error(self, mock_span: MockSpan) -> None:
        data = {
            "jsonrpc": "2.0",
            "id": "x",
            "error": {"code": -32601, "message": "Method not found"},
        }
        resp = self._mock_response(data)
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value = mock_client

            d = McpDispatcher()
            result = await d.dispatch(self._tool(), {}, CTX)

        assert result.is_error is True
        assert "Method not found" in (result.error_message or "")

    async def test_tool_is_error_returns_is_error(self, mock_span: MockSpan) -> None:
        data = {
            "jsonrpc": "2.0",
            "id": "x",
            "result": {
                "content": [{"type": "text", "text": "tool failed"}],
                "isError": True,
            },
        }
        resp = self._mock_response(data)
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value = mock_client

            result = await McpDispatcher().dispatch(self._tool(), {}, CTX)

        assert result.is_error is True
        assert result.error_code == "mcp_tool_error"

    async def test_network_failure_returns_is_error(self, mock_span: MockSpan) -> None:
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=OSError("connection refused"))
            mock_httpx.AsyncClient.return_value = mock_client

            result = await McpDispatcher().dispatch(self._tool(), {}, CTX)

        assert result.is_error is True
        assert result.error_code == "mcp_request_failed"

    async def test_network_failure_audit_entry(self, mock_span: MockSpan) -> None:
        audit = CapturingAuditLog()
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=OSError("refused"))
            mock_httpx.AsyncClient.return_value = mock_client

            await McpDispatcher(audit_log=audit).dispatch(self._tool(), {}, CTX)

        assert len(audit.entries) == 1
        assert audit.entries[0].event == "mcp.request.failed"

    async def test_httpx_unavailable_returns_is_error(self, mock_span: MockSpan) -> None:
        with patch("sdk_sandbox._dispatchers._HTTPX_AVAILABLE", False):
            result = await McpDispatcher().dispatch(self._tool(), {}, CTX)
        assert result.is_error is True
        assert result.error_code == "mcp_httpx_unavailable"


# ---------------------------------------------------------------------------
# HttpDispatcher — mocked httpx
# ---------------------------------------------------------------------------


class TestHttpDispatcher:
    def test_satisfies_abc(self) -> None:
        assert isinstance(HttpDispatcher(), ToolDispatcher)

    def test_kind(self) -> None:
        assert HttpDispatcher().kind == "http"

    def _tool(self) -> ToolDefinition:
        return _tool(HttpHandler(url="http://tool.local/run"))

    def _mock_response(self, data: dict[str, Any]) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=data)
        return resp

    async def test_success_returns_result(self, mock_span: MockSpan) -> None:
        resp = self._mock_response({"result": "done"})
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value = mock_client

            result = await HttpDispatcher().dispatch(self._tool(), {"x": 1}, CTX)

        assert result.is_error is False
        assert result.content == "done"

    async def test_success_span_name(self, mock_span: MockSpan) -> None:
        resp = self._mock_response({"result": "ok"})
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value = mock_client

            await HttpDispatcher().dispatch(self._tool(), {}, CTX)

        assert mock_span.name == "http.dispatch"

    async def test_success_structured_event(self, mock_span: MockSpan) -> None:
        resp = self._mock_response({"result": "ok"})
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value = mock_client

            await HttpDispatcher().dispatch(self._tool(), {}, CTX)

        event_names = [e[0] for e in mock_span.events]
        assert "http.dispatch" in event_names

    async def test_error_body_returns_is_error(self, mock_span: MockSpan) -> None:
        resp = self._mock_response({"error": {"code": "bad_request", "message": "invalid args"}})
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value = mock_client

            result = await HttpDispatcher().dispatch(self._tool(), {}, CTX)

        assert result.is_error is True
        assert result.error_code == "bad_request"
        assert "invalid args" in (result.error_message or "")

    async def test_error_body_audit_entry(self, mock_span: MockSpan) -> None:
        resp = self._mock_response({"error": {"code": "oops", "message": "fail"}})
        audit = CapturingAuditLog()
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value = mock_client

            await HttpDispatcher(audit_log=audit).dispatch(self._tool(), {}, CTX)

        assert len(audit.entries) == 1
        assert audit.entries[0].event == "http.tool.error"

    async def test_network_failure_returns_is_error(self, mock_span: MockSpan) -> None:
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=OSError("connection refused"))
            mock_httpx.AsyncClient.return_value = mock_client

            result = await HttpDispatcher().dispatch(self._tool(), {}, CTX)

        assert result.is_error is True
        assert result.error_code == "http_request_failed"

    async def test_network_failure_span_marked_error(self, mock_span: MockSpan) -> None:
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=OSError("refused"))
            mock_httpx.AsyncClient.return_value = mock_client

            await HttpDispatcher().dispatch(self._tool(), {}, CTX)

        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_httpx_unavailable_returns_is_error(self, mock_span: MockSpan) -> None:
        with patch("sdk_sandbox._dispatchers._HTTPX_AVAILABLE", False):
            result = await HttpDispatcher().dispatch(self._tool(), {}, CTX)
        assert result.is_error is True
        assert result.error_code == "http_httpx_unavailable"

    async def test_success_no_audit_entry(self, mock_span: MockSpan) -> None:
        resp = self._mock_response({"result": "ok"})
        audit = CapturingAuditLog()
        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value = mock_client

            await HttpDispatcher(audit_log=audit).dispatch(self._tool(), {}, CTX)

        assert audit.entries == []

    async def test_posts_args_and_context(self, mock_span: MockSpan) -> None:
        resp = self._mock_response({"result": "ok"})
        captured_payload: list[dict] = []

        with patch("sdk_sandbox._dispatchers._httpx") as mock_httpx:

            async def fake_post(url: str, *, json: dict, headers: dict) -> Any:
                captured_payload.append(json)
                return resp

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = fake_post
            mock_httpx.AsyncClient.return_value = mock_client

            await HttpDispatcher().dispatch(self._tool(), {"key": "value"}, CTX)

        assert len(captured_payload) == 1
        body = captured_payload[0]
        assert body["args"] == {"key": "value"}
        assert body["context"]["session_id"] == "sess-disp"
        assert body["context"]["workspace"] == "/workspace"


# ---------------------------------------------------------------------------
# ContainerDispatcher — real docker exec via python script
# ---------------------------------------------------------------------------

_CONTAINER_ECHO_SCRIPT = """\
import sys, json
req = json.load(sys.stdin)
json.dump({"result": req["args"]}, sys.stdout)
"""

_CONTAINER_ERROR_SCRIPT = """\
import sys, json
json.dump({"error": {"code": "container_tool_error", "message": "container error"}}, sys.stdout)
"""

_CONTAINER_CRASH_SCRIPT = """\
import sys
sys.stderr.write("crash\\n")
sys.exit(2)
"""


class TestContainerDispatcher:
    def test_satisfies_abc(self) -> None:
        assert isinstance(ContainerDispatcher(), ToolDispatcher)

    def test_kind(self) -> None:
        assert ContainerDispatcher().kind == "container"

    def _tool_with_handler(self, env_id: str, entrypoint: str) -> ToolDefinition:
        return _tool(ContainerHandler(environment_id=env_id, entrypoint=entrypoint))

    async def test_success_returns_result(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = tmp_path / "tool"
        exe.write_text(f"#!{sys.executable}\n{_CONTAINER_ECHO_SCRIPT}")
        exe.chmod(0o755)

        # Patch docker to "exec" the entrypoint directly by substituting python
        # docker exec <env_id> <entrypoint> → python <exe>
        captured: list[list[str]] = []
        orig_create = asyncio.create_subprocess_exec

        async def fake_exec(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
            captured.append(list(args))
            # Replace "docker exec -i {env} {entrypoint}" with direct exe call
            return await orig_create(str(exe), **kwargs)

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            d = ContainerDispatcher(docker_executable="docker")
            result = await d.dispatch(
                self._tool_with_handler("my-container", str(exe)), {"val": 42}, CTX
            )

        assert result.is_error is False
        assert result.content == {"val": 42}

    async def test_success_span_name(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = tmp_path / "tool"
        exe.write_text(f"#!{sys.executable}\n{_CONTAINER_ECHO_SCRIPT}")
        exe.chmod(0o755)
        orig = asyncio.create_subprocess_exec

        async def fake_exec(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
            return await orig(str(exe), **kwargs)

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            d = ContainerDispatcher()
            await d.dispatch(self._tool_with_handler("c1", str(exe)), {}, CTX)

        assert mock_span.name == "container.dispatch"

    async def test_success_structured_event(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = tmp_path / "tool"
        exe.write_text(f"#!{sys.executable}\n{_CONTAINER_ECHO_SCRIPT}")
        exe.chmod(0o755)
        orig = asyncio.create_subprocess_exec

        async def fake_exec(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
            return await orig(str(exe), **kwargs)

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await ContainerDispatcher().dispatch(self._tool_with_handler("c1", str(exe)), {}, CTX)

        event_names = [e[0] for e in mock_span.events]
        assert "container.dispatch" in event_names

    async def test_docker_not_found_returns_is_error(self, mock_span: MockSpan) -> None:
        d = ContainerDispatcher(docker_executable="/no/such/docker")
        result = await d.dispatch(self._tool_with_handler("c1", "/entrypoint"), {}, CTX)
        assert result.is_error is True
        assert result.error_code == "container_docker_not_found"

    async def test_docker_not_found_audit_entry(self, mock_span: MockSpan) -> None:
        audit = CapturingAuditLog()
        d = ContainerDispatcher(docker_executable="/no/such/docker", audit_log=audit)
        await d.dispatch(self._tool_with_handler("c1", "/ep"), {}, CTX)
        assert len(audit.entries) == 1
        assert audit.entries[0].event == "container.docker.not_found"

    async def test_docker_not_found_span_marked_error(self, mock_span: MockSpan) -> None:
        d = ContainerDispatcher(docker_executable="/no/such/docker")
        await d.dispatch(self._tool_with_handler("c1", "/ep"), {}, CTX)
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_nonzero_exit_returns_is_error(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = tmp_path / "crash_tool"
        exe.write_text(f"#!{sys.executable}\n{_CONTAINER_CRASH_SCRIPT}")
        exe.chmod(0o755)
        orig = asyncio.create_subprocess_exec

        async def fake_exec(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
            return await orig(str(exe), **kwargs)

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            result = await ContainerDispatcher().dispatch(
                self._tool_with_handler("c1", str(exe)), {}, CTX
            )

        assert result.is_error is True
        assert result.error_code == "container_nonzero_exit"

    async def test_tool_error_response_returns_is_error(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        exe = tmp_path / "error_tool"
        exe.write_text(f"#!{sys.executable}\n{_CONTAINER_ERROR_SCRIPT}")
        exe.chmod(0o755)
        orig = asyncio.create_subprocess_exec

        async def fake_exec(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
            return await orig(str(exe), **kwargs)

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            result = await ContainerDispatcher().dispatch(
                self._tool_with_handler("c1", str(exe)), {}, CTX
            )

        assert result.is_error is True
        assert result.error_code == "container_tool_error"

    async def test_payload_includes_thread_id(self, tmp_path: Path, mock_span: MockSpan) -> None:
        script_text = (
            "import sys, json\n"
            "req = json.load(sys.stdin)\n"
            "sys.stdout.write(json.dumps({'result': req['context']}))\n"
        )
        exe = tmp_path / "ctx_tool"
        exe.write_text(f"#!{sys.executable}\n{script_text}")
        exe.chmod(0o755)
        orig = asyncio.create_subprocess_exec

        async def fake_exec(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
            return await orig(str(exe), **kwargs)

        ctx_with_thread = ExecutionContext(
            session_id="sess-disp",
            workspace="/workspace",
            thread_id="thread-456",
            scratch_dir="/tmp/scratch",
        )
        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            result = await ContainerDispatcher().dispatch(
                self._tool_with_handler("c1", str(exe)), {}, ctx_with_thread
            )

        assert result.is_error is False
        assert result.content["thread_id"] == "thread-456"
        assert result.content["session_id"] == "sess-disp"
        assert result.content["workspace"] == "/workspace"
        assert result.content["scratch_dir"] == "/tmp/scratch"

    async def test_unexpected_spawn_error_propagates(self, mock_span: MockSpan) -> None:
        # A non-FileNotFoundError before proc is assigned: proc stays None, so
        # the cleanup guard is skipped and the exception re-raises.
        with (
            patch("asyncio.create_subprocess_exec", side_effect=RuntimeError("docker boom")),
            pytest.raises(RuntimeError, match="docker boom"),
        ):
            await ContainerDispatcher().dispatch(
                self._tool_with_handler("c1", "/entrypoint"), {}, CTX
            )
