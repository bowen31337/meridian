"""Unit coverage for sdk-environment telemetry, audit, fs-enforcer, runtime,
ssh timeout, and optional-dependency import fallbacks."""

from __future__ import annotations

import builtins
import importlib
from unittest.mock import AsyncMock, patch

from opentelemetry import trace
import pytest
from sdk_environment import EnvironmentFailure, EnvironmentRuntime
from sdk_environment._audit import NoopAuditLog
from sdk_environment._fs_enforcer import FilesystemEnforcer
from sdk_environment._telemetry import get_tracer
from sdk_environment._types import (
    AgentFilesystemPolicy,
    AuditLogEntry,
    FilesystemPolicy,
)

from .conftest import CapturingAuditLog
from .test_conformance import (
    StubDriver,
    make_execute,
    make_options,
    make_provision,
    make_reclaim,
)
from .test_ssh_driver import (
    _driver as _ssh_driver,
    _execute_req,
    _make_conn,
    _make_rsync_proc,
    _make_ssh_result,
    _patch_asyncssh,
    _patch_rsync,
    _provision_req,
)


def test_get_tracer_returns_tracer() -> None:
    assert isinstance(get_tracer(), trace.Tracer)


def test_noop_audit_log_write_is_silent() -> None:
    entry = AuditLogEntry(
        level="error",
        event="test.event",
        environment_id="e1",
        environment_kind="test.stub",
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
    )
    assert NoopAuditLog().write(entry) is None


# ---------------------------------------------------------------------------
# FilesystemEnforcer glob fallbacks
# ---------------------------------------------------------------------------


def test_env_globs_unknown_operation_returns_empty(tmp_path) -> None:
    enf = FilesystemEnforcer(tmp_path, FilesystemPolicy(read_globs=("**",)))
    assert enf._env_globs("teleport") == ()


def test_agent_globs_branches(tmp_path) -> None:
    agent = AgentFilesystemPolicy(
        agent_id="a1",
        read_globs=("r",),
        write_globs=("w",),
        delete_globs=("d",),
    )
    enf = FilesystemEnforcer(tmp_path, FilesystemPolicy(read_globs=("**",)), agent)
    assert enf._agent_globs("write") == ("w",)
    assert enf._agent_globs("delete") == ("d",)
    assert enf._agent_globs("teleport") is None


# ---------------------------------------------------------------------------
# Runtime — unknown kind and EnvironmentFailure re-raise
# ---------------------------------------------------------------------------


def test_filesystem_policy_unknown_kind_raises() -> None:
    rt = EnvironmentRuntime()
    with pytest.raises(EnvironmentFailure) as exc_info:
        rt.filesystem_policy("acme.unknown")
    assert exc_info.value.code == "ENV_KIND_NOT_REGISTERED"


def _failure() -> EnvironmentFailure:
    return EnvironmentFailure(
        code="ENV_CUSTOM",
        message="driver said no",
        environment_id="env1",
        environment_kind="test.stub",
        session_id="sess1",
        timestamp="2026-01-01T00:00:00Z",
    )


class TestRuntimeReraisesEnvironmentFailure:
    async def test_provision_reraises(self, mock_span, audit_log: CapturingAuditLog) -> None:
        orig = _failure()
        rt = EnvironmentRuntime()
        rt.register(StubDriver(provision_raises=orig))
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.provision(make_provision(), make_options(audit_log))
        assert exc_info.value is orig

    async def test_execute_reraises(self, mock_span, audit_log: CapturingAuditLog) -> None:
        orig = _failure()
        rt = EnvironmentRuntime()
        rt.register(StubDriver(execute_raises=orig))
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.execute(make_execute(), make_options(audit_log))
        assert exc_info.value is orig

    async def test_reclaim_reraises(self, mock_span, audit_log: CapturingAuditLog) -> None:
        orig = _failure()
        rt = EnvironmentRuntime()
        rt.register(StubDriver(reclaim_raises=orig))
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.reclaim(make_reclaim(), make_options(audit_log))
        assert exc_info.value is orig


# ---------------------------------------------------------------------------
# SSH execute timeout re-raise
# ---------------------------------------------------------------------------


async def test_ssh_execute_timeout_reraises() -> None:
    conn = _make_conn()
    conn.run = AsyncMock(side_effect=[_make_ssh_result(), TimeoutError()])
    with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
        driver = _ssh_driver()
        await driver.provision(_provision_req())
        with pytest.raises(TimeoutError):
            await driver.execute(_execute_req())


# ---------------------------------------------------------------------------
# Optional-dependency ImportError fallbacks (reload module with dep blocked)
# ---------------------------------------------------------------------------


def _reload_without(module_name: str, blocked: str, flag_attr: str) -> object:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == blocked:
            raise ImportError(f"blocked {blocked}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    mod = importlib.import_module(module_name)
    try:
        with patch.object(builtins, "__import__", fake_import):
            reloaded = importlib.reload(mod)
            captured = getattr(reloaded, flag_attr)
        return captured
    finally:
        importlib.reload(mod)


def test_http_driver_without_httpx() -> None:
    assert _reload_without("sdk_environment._http_driver", "httpx", "_HTTPX_AVAILABLE") is False


def test_ssh_driver_without_asyncssh() -> None:
    assert (
        _reload_without("sdk_environment._ssh_driver", "asyncssh", "_ASYNCSSH_AVAILABLE") is False
    )


def test_mcp_driver_without_httpx() -> None:
    assert _reload_without("sdk_environment._mcp_driver", "httpx", "_HTTPX_AVAILABLE") is False


# ---------------------------------------------------------------------------
# Pool — cached provision_error and reaper loop
# ---------------------------------------------------------------------------


async def test_pool_acquire_raises_cached_provision_error() -> None:
    from sdk_environment import EnvironmentRuntime as _RT, RuntimeOptions, WorkerPool
    from sdk_environment._pool import _WorkerEntry

    from .test_lifecycle import PoolDriver, _exec_req

    rt = _RT()
    rt.register(PoolDriver())
    pool = WorkerPool(rt)
    cached = _failure()
    entry = _WorkerEntry(
        environment_id="env1",
        environment_kind="test.pool",
        session_id="sess1",
    )
    entry.provision_error = cached
    pool._workers["env1"] = entry
    with pytest.raises(EnvironmentFailure) as exc_info:
        await pool._get_or_provision(_exec_req("test.pool"), RuntimeOptions())
    assert exc_info.value is cached


async def test_pool_reaper_loop_runs_then_cancels() -> None:
    import asyncio as _asyncio
    import contextlib

    from sdk_environment import EnvironmentRuntime as _RT, PoolOptions, WorkerPool

    from .test_lifecycle import PoolDriver

    rt = _RT()
    rt.register(PoolDriver())
    pool = WorkerPool(rt, PoolOptions(reap_interval_seconds=0.0))
    task = _asyncio.create_task(pool._reaper_loop())
    await _asyncio.sleep(0.01)
    task.cancel()
    with contextlib.suppress(_asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# Runtime — filesystem_policy success path
# ---------------------------------------------------------------------------


def test_filesystem_policy_known_kind_returns_policy() -> None:
    rt = EnvironmentRuntime()
    rt.register(StubDriver())
    assert isinstance(rt.filesystem_policy("test.stub"), FilesystemPolicy)


# ---------------------------------------------------------------------------
# MCP stdio helpers
# ---------------------------------------------------------------------------


def _mcp() -> object:
    import importlib

    return importlib.import_module("sdk_environment._mcp_driver")


class _FakeReader:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class _HangReader:
    async def readline(self) -> bytes:
        import asyncio as _asyncio

        await _asyncio.sleep(10)
        return b""


async def test_stdio_read_response_times_out() -> None:
    mcp = _mcp()
    with pytest.raises(ValueError, match="Timed out"):
        await mcp._stdio_read_response(_HangReader(), "x", 0.01)


async def test_stdio_read_response_closed_stdout() -> None:
    mcp = _mcp()
    with pytest.raises(ValueError, match="closed stdout"):
        await mcp._stdio_read_response(_FakeReader([]), "x", 1.0)


async def test_stdio_read_response_skips_blank_and_bad_json() -> None:
    import json

    mcp = _mcp()
    reader = _FakeReader([b"\n", b"not-json\n", json.dumps({"id": "x", "ok": 1}).encode() + b"\n"])
    msg = await mcp._stdio_read_response(reader, "x", 1.0)
    assert msg["ok"] == 1


async def test_stdio_read_response_no_match_after_max_lines() -> None:
    import json

    mcp = _mcp()
    lines = [json.dumps({"id": "other"}).encode() + b"\n" for _ in range(60)]
    with pytest.raises(ValueError, match="No response"):
        await mcp._stdio_read_response(_FakeReader(lines), "x", 1.0)


async def test_stdio_handshake_raises_on_init_error() -> None:
    import json
    from unittest.mock import AsyncMock as _AM, MagicMock as _MM

    mcp = _mcp()
    proc = _MM()
    proc.stdin = _MM()
    proc.stdin.drain = _AM()
    proc.stdout = _FakeReader(
        [json.dumps({"id": "init", "error": {"message": "boom"}}).encode() + b"\n"]
    )
    with pytest.raises(ValueError, match="initialize failed"):
        await mcp._stdio_handshake(proc, 1.0)


async def test_stdio_close_already_exited_returns() -> None:
    from unittest.mock import MagicMock as _MM

    mcp = _mcp()
    proc = _MM()
    proc.returncode = 0
    await mcp._stdio_close(proc)  # returns immediately


async def test_stdio_close_handles_close_error_and_timeout() -> None:
    import asyncio as _asyncio
    from unittest.mock import MagicMock as _MM

    mcp = _mcp()

    class _Proc:
        def __init__(self) -> None:
            self.returncode = None
            self.killed = False
            self.stdin = _MM()
            self.stdin.close.side_effect = RuntimeError("close boom")

        async def wait(self) -> int:
            await _asyncio.sleep(10)
            return 0

        def kill(self) -> None:
            self.killed = True

    proc = _Proc()
    with patch.object(mcp, "_PROCESS_GRACE_S", 0.01):
        await mcp._stdio_close(proc)
    assert proc.killed is True


# ---------------------------------------------------------------------------
# MCP — _mcp_to_execute_result no-content fallback (str(result))
# ---------------------------------------------------------------------------


def test_mcp_to_execute_result_no_content_fallback() -> None:
    from sdk_environment._mcp_driver import McpBackendDriver

    data = {"jsonrpc": "2.0", "id": "call", "result": {"status": "done"}}
    result = McpBackendDriver._mcp_to_execute_result(data, 0.0)
    assert "status" in result.stdout


# ---------------------------------------------------------------------------
# MCP SSE — non-message events, bad JSON, and init error
# ---------------------------------------------------------------------------


def _sse_transport_with_extra_lines(call_resp: dict) -> object:
    import json
    from unittest.mock import AsyncMock as _AM, MagicMock as _MM

    sse_lines = [
        "event: endpoint",
        "data: /message",
        "",
        "event: ping",  # non-message event with data -> hits the etype guard
        "data: keepalive",
        "",
        "event: message",  # malformed JSON -> hits JSONDecodeError guard
        "data: {not json",
        "",
        "event: message",
        f"data: {json.dumps({'jsonrpc': '2.0', 'id': 'init', 'result': {}})}",
        "",
        "event: message",
        f"data: {json.dumps(call_resp)}",
        "",
    ]

    async def _aiter_lines():
        for line in sse_lines:
            yield line

    stream_resp = _MM()
    stream_resp.raise_for_status = _MM()
    stream_resp.aiter_lines = _aiter_lines
    stream_ctx = _MM()
    stream_ctx.__aenter__ = _AM(return_value=stream_resp)
    stream_ctx.__aexit__ = _AM(return_value=False)
    post_resp = _MM()
    post_resp.raise_for_status = _MM()
    post_resp.status_code = 202
    client_mock = _MM()
    client_mock.stream = _MM(return_value=stream_ctx)
    client_mock.post = _AM(return_value=post_resp)
    client_ctx = _MM()
    client_ctx.__aenter__ = _AM(return_value=client_mock)
    client_ctx.__aexit__ = _AM(return_value=False)
    return client_ctx


async def test_sse_ignores_non_message_and_bad_json() -> None:
    from sdk_environment._mcp_driver import McpBackendDriver

    from .test_mcp_driver import _SERVER_URL, _execute_req, _ok_rpc

    driver = McpBackendDriver(transport="sse", server_url=_SERVER_URL)
    client_ctx = _sse_transport_with_extra_lines(_ok_rpc("done"))
    with patch("sdk_environment._mcp_driver._httpx.AsyncClient", return_value=client_ctx):
        result = await driver.execute(_execute_req())
    assert result.exit_code == 0


async def test_sse_init_error_raises() -> None:
    import json
    from unittest.mock import AsyncMock as _AM, MagicMock as _MM

    from sdk_environment._mcp_driver import McpBackendDriver

    from .test_mcp_driver import _SERVER_URL, _execute_req, _ok_rpc

    sse_lines = [
        "event: endpoint",
        "data: /message",
        "",
        "event: message",
        f"data: {json.dumps({'jsonrpc': '2.0', 'id': 'init', 'error': {'message': 'nope'}})}",
        "",
        "event: message",
        f"data: {json.dumps(_ok_rpc('x'))}",
        "",
    ]

    async def _aiter_lines():
        for line in sse_lines:
            yield line

    stream_resp = _MM()
    stream_resp.raise_for_status = _MM()
    stream_resp.aiter_lines = _aiter_lines
    stream_ctx = _MM()
    stream_ctx.__aenter__ = _AM(return_value=stream_resp)
    stream_ctx.__aexit__ = _AM(return_value=False)
    post_resp = _MM()
    post_resp.raise_for_status = _MM()
    post_resp.status_code = 202
    client_mock = _MM()
    client_mock.stream = _MM(return_value=stream_ctx)
    client_mock.post = _AM(return_value=post_resp)
    client_ctx = _MM()
    client_ctx.__aenter__ = _AM(return_value=client_mock)
    client_ctx.__aexit__ = _AM(return_value=False)

    driver = McpBackendDriver(transport="sse", server_url=_SERVER_URL)
    with (
        patch("sdk_environment._mcp_driver._httpx.AsyncClient", return_value=client_ctx),
        pytest.raises(ValueError, match="initialize failed"),
    ):
        await driver.execute(_execute_req())
