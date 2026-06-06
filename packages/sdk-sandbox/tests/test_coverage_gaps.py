"""Unit coverage for sdk-sandbox telemetry, schema/secret helpers, dispatcher
kill paths, MCP stdio helpers, and optional-dependency import fallbacks."""

from __future__ import annotations

import asyncio
import builtins
import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from opentelemetry import trace
import pytest
from sdk_sandbox import (
    ContainerHandler,
    ExecutionContext,
    InProcessHandler,
    SandboxFailure,
    SandboxResult,
    ToolDefinition,
    ToolDispatcher,
)
from sdk_sandbox._dispatchers import ContainerDispatcher, SubprocessDispatcher
from sdk_sandbox._runtime import Sandbox
from sdk_sandbox._schema import _fmt_path
from sdk_sandbox._secret_refs import _substitute_value
from sdk_sandbox._telemetry import get_tracer, record_dispatch_overhead

from .conftest import CapturingAuditLog, MockSpan, MockTracer

CTX = ExecutionContext(session_id="sess-gap", workspace="/workspace", scratch_dir="/tmp/scratch")


def _tool(handler: Any, name: str = "test.tool") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Test tool",
        input_schema={"type": "object"},
        handler=handler,
    )


@pytest.fixture()
def disp_span(monkeypatch: pytest.MonkeyPatch) -> MockSpan:
    tracer = MockTracer()
    monkeypatch.setattr("sdk_sandbox._dispatchers.get_tracer", lambda: tracer)
    return tracer.span


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def test_get_tracer_returns_tracer() -> None:
    assert isinstance(get_tracer(), trace.Tracer)


def test_record_dispatch_overhead_unknown_kind_returns_false() -> None:
    assert record_dispatch_overhead(MockSpan(), "unknown", 99999.0) is False


# ---------------------------------------------------------------------------
# Schema / secret-ref helper leaf branches
# ---------------------------------------------------------------------------


def test_fmt_path_int_index() -> None:
    assert _fmt_path([0, "key"]) == "$[0].key"


def test_substitute_value_passthrough_non_container() -> None:
    assert _substitute_value(123, {}) == 123


# ---------------------------------------------------------------------------
# Runtime: dispatcher raising SandboxFailure is re-raised unchanged
# ---------------------------------------------------------------------------


class _RaisingDispatcher(ToolDispatcher):
    @property
    def kind(self) -> str:
        return "in_process"

    async def dispatch(
        self, tool: ToolDefinition, input: dict[str, Any], context: ExecutionContext
    ) -> SandboxResult:
        raise SandboxFailure(
            code="CUSTOM",
            message="direct failure",
            tool_name=tool.name,
            session_id=context.session_id,
            timestamp="2026-01-01T00:00:00Z",
        )


async def test_execute_reraises_sandbox_failure(mock_span: MockSpan) -> None:
    sbx = Sandbox()
    sbx.register_dispatcher(_RaisingDispatcher())
    sbx.register_tool(_tool(InProcessHandler(module="x")))
    with pytest.raises(SandboxFailure) as exc_info:
        await sbx.execute("test.tool", {}, CTX)
    assert exc_info.value.code == "CUSTOM"


# ---------------------------------------------------------------------------
# Subprocess / container dispatcher except+finally kill paths
# ---------------------------------------------------------------------------


def _fake_proc_communicate_raises() -> MagicMock:
    proc = MagicMock()
    proc.returncode = None
    proc.communicate = AsyncMock(side_effect=RuntimeError("boom"))
    proc.kill = MagicMock()
    return proc


async def test_subprocess_communicate_error_kills_and_raises(disp_span: MockSpan) -> None:
    proc = _fake_proc_communicate_raises()
    with (
        patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
        pytest.raises(RuntimeError),
    ):
        await SubprocessDispatcher().dispatch(_tool(__import_subprocess_handler()), {}, CTX)
    assert proc.kill.called


async def test_container_communicate_error_kills_and_raises(disp_span: MockSpan) -> None:
    proc = _fake_proc_communicate_raises()
    with (
        patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
        pytest.raises(RuntimeError),
    ):
        await ContainerDispatcher().dispatch(
            _tool(ContainerHandler(environment_id="c1", entrypoint="/ep")), {}, CTX
        )
    assert proc.kill.called


def __import_subprocess_handler() -> Any:
    from sdk_sandbox import SubprocessHandler

    return SubprocessHandler(path="/bin/whatever")


# ---------------------------------------------------------------------------
# Container: invalid JSON output + overhead-target-breached audit
# ---------------------------------------------------------------------------

_CONTAINER_BADJSON_SCRIPT = """\
import sys
sys.stdout.write("not json")
"""

_CONTAINER_ECHO_SCRIPT = """\
import sys, json
req = json.load(sys.stdin)
json.dump({"result": req["args"]}, sys.stdout)
"""


async def test_container_invalid_json_returns_is_error(tmp_path: Path, disp_span: MockSpan) -> None:
    exe = tmp_path / "badjson_tool"
    exe.write_text(f"#!{sys.executable}\n{_CONTAINER_BADJSON_SCRIPT}")
    exe.chmod(0o755)
    orig = asyncio.create_subprocess_exec

    async def fake_exec(*_a: str, **kw: Any) -> asyncio.subprocess.Process:
        return await orig(str(exe), **kw)

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await ContainerDispatcher().dispatch(
            _tool(ContainerHandler(environment_id="c1", entrypoint=str(exe))), {}, CTX
        )
    assert result.is_error is True
    assert result.error_code == "container_invalid_json"


async def test_container_overhead_breach_audits(tmp_path: Path, disp_span: MockSpan) -> None:
    exe = tmp_path / "echo_tool"
    exe.write_text(f"#!{sys.executable}\n{_CONTAINER_ECHO_SCRIPT}")
    exe.chmod(0o755)
    orig = asyncio.create_subprocess_exec

    async def fake_exec(*_a: str, **kw: Any) -> asyncio.subprocess.Process:
        return await orig(str(exe), **kw)

    audit = CapturingAuditLog()
    with (
        patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
        patch("sdk_sandbox._dispatchers.record_dispatch_overhead", return_value=True),
    ):
        result = await ContainerDispatcher(audit_log=audit).dispatch(
            _tool(ContainerHandler(environment_id="c1", entrypoint=str(exe))), {"v": 1}, CTX
        )
    assert result.is_error is False
    assert any(e.event == "dispatch.overhead.target_breached" for e in audit.entries)


# ---------------------------------------------------------------------------
# MCP stdio helpers
# ---------------------------------------------------------------------------


def _mcp() -> Any:
    return importlib.import_module("sdk_sandbox._mcp_client")


class _FakeReader:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class _HangReader:
    async def readline(self) -> bytes:
        await asyncio.sleep(10)
        return b""


class _FakeStdin:
    def __init__(self, close_raises: bool = False) -> None:
        self._close_raises = close_raises

    def write(self, _data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        if self._close_raises:
            raise RuntimeError("close boom")

    async def wait_closed(self) -> None:
        pass


class _FakeProc:
    def __init__(
        self,
        *,
        stdout: Any = None,
        stdin: Any = None,
        returncode: int | None = None,
        wait_hangs: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stdin = stdin
        self.returncode = returncode
        self._wait_hangs = wait_hangs
        self.kill = MagicMock()

    async def wait(self) -> int:
        if self._wait_hangs:
            await asyncio.sleep(10)
        return 0


async def test_stdio_read_response_timeout() -> None:
    mcp = _mcp()
    with pytest.raises(ValueError, match="Timed out"):
        await mcp._stdio_read_response(_HangReader(), "x", 0.01)


async def test_stdio_read_response_skips_blank_and_bad_json() -> None:
    mcp = _mcp()
    reader = _FakeReader([b"\n", b"garbage not json\n", b'{"id":"call","ok":1}\n'])
    resp = await mcp._stdio_read_response(reader, "call", 1.0)
    assert resp["ok"] == 1


async def test_stdio_read_response_no_match() -> None:
    mcp = _mcp()
    reader = _FakeReader([b'{"id":"other"}\n'] * 60)
    with pytest.raises(ValueError, match="No response"):
        await mcp._stdio_read_response(reader, "call", 1.0)


async def test_stdio_handshake_error_response() -> None:
    mcp = _mcp()
    proc = _FakeProc(
        stdin=_FakeStdin(),
        stdout=_FakeReader([b'{"id":"init","error":{"message":"boom"}}\n']),
    )
    with pytest.raises(ValueError, match="initialize failed"):
        await mcp._stdio_handshake(proc, 1.0)


async def test_close_stdio_process_swallows_stdin_close_error() -> None:
    mcp = _mcp()
    proc = _FakeProc(stdin=_FakeStdin(close_raises=True), returncode=None)
    await mcp._close_stdio_process(proc)
    assert not proc.kill.called


async def test_close_stdio_process_timeout_kills(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp = _mcp()
    monkeypatch.setattr(mcp, "_PROCESS_GRACE_S", 0.01)
    proc = _FakeProc(stdin=_FakeStdin(), returncode=None, wait_hangs=True)
    await mcp._close_stdio_process(proc)
    assert proc.kill.called


async def test_discover_stdio_error_response(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp = _mcp()
    monkeypatch.setattr(mcp, "_spawn_stdio_process", AsyncMock(return_value=_FakeProc()))
    monkeypatch.setattr(mcp, "_stdio_handshake", AsyncMock(return_value=None))
    monkeypatch.setattr(mcp, "_stdio_rpc", AsyncMock(return_value={"error": {"message": "nope"}}))
    monkeypatch.setattr(mcp, "_close_stdio_process", AsyncMock(return_value=None))
    with pytest.raises(ValueError, match="tools/list failed"):
        await mcp.discover_mcp_tools_stdio(("server",), timeout_s=1.0)


async def test_discover_http_without_httpx_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp = _mcp()
    monkeypatch.setattr(mcp, "_HTTPX_AVAILABLE", False)
    with pytest.raises(ImportError, match="httpx is required"):
        await mcp.discover_mcp_tools_http("http://localhost", timeout_s=1.0)


# ---------------------------------------------------------------------------
# Optional-dependency ImportError module-load fallbacks
# ---------------------------------------------------------------------------


def _reload_flag(module_name: str, blocked: str, flag_attr: str) -> bool:
    """Re-execute *module_name* with *blocked* unimportable, into a throwaway
    module object so the live module (and its shared classes) stays intact."""
    real_import = builtins.__import__

    def fake_import(name: str, *a: Any, **k: Any) -> Any:
        if name == blocked or name.split(".")[0] == blocked:
            raise ImportError(f"blocked {blocked}")
        return real_import(name, *a, **k)

    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.object(builtins, "__import__", fake_import):
        spec.loader.exec_module(module)
    return bool(getattr(module, flag_attr))


def test_dispatchers_without_httpx_sets_flag_false() -> None:
    assert _reload_flag("sdk_sandbox._dispatchers", "httpx", "_HTTPX_AVAILABLE") is False


def test_mcp_client_without_httpx_sets_flag_false() -> None:
    assert _reload_flag("sdk_sandbox._mcp_client", "httpx", "_HTTPX_AVAILABLE") is False
