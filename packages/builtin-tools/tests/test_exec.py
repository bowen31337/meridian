"""Tests for the exec built-in tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from meridian_builtin_tools.exec import (
    _INPUT_SCHEMA,
    _MAX_OUTPUT_BYTES,
    _MAX_TIMEOUT_SECONDS,
    _OUTPUT_SCHEMA,
    _run_command,
    exec_tool,
)
from meridian_sdk_tool import ToolContext

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

_CTX = ToolContext(workspace="/tmp", session_id="sess_exec_test")


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture()
def ws_ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=str(workspace), session_id="sess_exec_int")


# ---------------------------------------------------------------------------
# _run_command unit tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_command_captures_stdout(tmp_path: Path) -> None:
    stdout, stderr, exit_code, timed_out, truncated = await _run_command(
        "echo hello", str(tmp_path), timeout=10
    )
    assert stdout.strip() == "hello"
    assert stderr == ""
    assert exit_code == 0
    assert not timed_out
    assert not truncated


@pytest.mark.anyio
async def test_run_command_captures_stderr(tmp_path: Path) -> None:
    stdout, stderr, exit_code, timed_out, truncated = await _run_command(
        "echo err >&2", str(tmp_path), timeout=10
    )
    assert stdout == ""
    assert stderr.strip() == "err"
    assert exit_code == 0


@pytest.mark.anyio
async def test_run_command_nonzero_exit_code(tmp_path: Path) -> None:
    _, _, exit_code, timed_out, _ = await _run_command(
        "exit 42", str(tmp_path), timeout=10
    )
    assert exit_code == 42
    assert not timed_out


@pytest.mark.anyio
async def test_run_command_cwd_is_workspace(workspace: Path) -> None:
    stdout, _, exit_code, _, _ = await _run_command(
        "pwd", str(workspace), timeout=10
    )
    assert exit_code == 0
    assert str(workspace.resolve()) in stdout.strip()


@pytest.mark.anyio
async def test_run_command_pipe_works(tmp_path: Path) -> None:
    stdout, _, exit_code, _, _ = await _run_command(
        "echo hello | tr a-z A-Z", str(tmp_path), timeout=10
    )
    assert exit_code == 0
    assert stdout.strip() == "HELLO"


@pytest.mark.anyio
async def test_run_command_timeout_sets_timed_out(tmp_path: Path) -> None:
    _, _, exit_code, timed_out, _ = await _run_command(
        "sleep 60", str(tmp_path), timeout=0.1
    )
    assert timed_out is True
    assert exit_code != 0


@pytest.mark.anyio
async def test_run_command_timeout_stdout_empty_on_timeout(tmp_path: Path) -> None:
    stdout, stderr, _, timed_out, _ = await _run_command(
        "sleep 60", str(tmp_path), timeout=0.1
    )
    assert timed_out is True
    assert stdout == ""
    assert stderr == ""


@pytest.mark.anyio
async def test_run_command_truncates_large_stdout(tmp_path: Path) -> None:
    # Generate output larger than _MAX_OUTPUT_BYTES
    big_write = f"python3 -c \"print('x' * {_MAX_OUTPUT_BYTES + 100})\""
    stdout, _, _, _, truncated = await _run_command(
        big_write, str(tmp_path), timeout=30
    )
    assert truncated is True
    assert len(stdout.encode("utf-8")) <= _MAX_OUTPUT_BYTES


@pytest.mark.anyio
async def test_run_command_truncates_large_stderr(tmp_path: Path) -> None:
    big_write = f"python3 -c \"import sys; sys.stderr.write('x' * {_MAX_OUTPUT_BYTES + 100})\""
    _, stderr, _, _, truncated = await _run_command(
        big_write, str(tmp_path), timeout=30
    )
    assert truncated is True
    assert len(stderr.encode("utf-8")) <= _MAX_OUTPUT_BYTES


@pytest.mark.anyio
async def test_run_command_not_truncated_for_small_output(tmp_path: Path) -> None:
    stdout, _, _, _, truncated = await _run_command(
        "echo small", str(tmp_path), timeout=10
    )
    assert not truncated
    assert "small" in stdout


# ---------------------------------------------------------------------------
# exec_tool integration tests — happy path
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_no_error_on_successful_command(ws_ctx: ToolContext) -> None:
    result = await exec_tool.execute({"command": "echo hi"}, ws_ctx)
    assert not result.is_error


@pytest.mark.anyio
async def test_stdout_contains_command_output(ws_ctx: ToolContext) -> None:
    result = await exec_tool.execute({"command": "echo hello_world"}, ws_ctx)
    assert not result.is_error
    assert "hello_world" in result.result["stdout"]


@pytest.mark.anyio
async def test_exit_code_zero_on_success(ws_ctx: ToolContext) -> None:
    result = await exec_tool.execute({"command": "true"}, ws_ctx)
    assert not result.is_error
    assert result.result["exit_code"] == 0


@pytest.mark.anyio
async def test_exit_code_nonzero_not_treated_as_tool_error(ws_ctx: ToolContext) -> None:
    result = await exec_tool.execute({"command": "false"}, ws_ctx)
    assert not result.is_error
    assert result.result["exit_code"] != 0


@pytest.mark.anyio
async def test_stderr_captured_separately(ws_ctx: ToolContext) -> None:
    result = await exec_tool.execute({"command": "echo out; echo err >&2"}, ws_ctx)
    assert not result.is_error
    assert "out" in result.result["stdout"]
    assert "err" in result.result["stderr"]


@pytest.mark.anyio
async def test_command_is_echoed_in_result(ws_ctx: ToolContext) -> None:
    cmd = "echo test_echo"
    result = await exec_tool.execute({"command": cmd}, ws_ctx)
    assert not result.is_error
    assert result.result["command"] == cmd


@pytest.mark.anyio
async def test_timed_out_false_on_fast_command(ws_ctx: ToolContext) -> None:
    result = await exec_tool.execute({"command": "echo fast"}, ws_ctx)
    assert not result.is_error
    assert result.result["timed_out"] is False


@pytest.mark.anyio
async def test_truncated_false_on_small_output(ws_ctx: ToolContext) -> None:
    result = await exec_tool.execute({"command": "echo small"}, ws_ctx)
    assert not result.is_error
    assert result.result["truncated"] is False


@pytest.mark.anyio
async def test_result_has_all_required_fields(ws_ctx: ToolContext) -> None:
    result = await exec_tool.execute({"command": "echo check"}, ws_ctx)
    assert not result.is_error
    for field in ("stdout", "stderr", "exit_code", "command", "timed_out", "truncated"):
        assert field in result.result


# ---------------------------------------------------------------------------
# Workspace confinement — cwd is the workspace
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cwd_is_workspace(ws_ctx: ToolContext, workspace: Path) -> None:
    result = await exec_tool.execute({"command": "pwd"}, ws_ctx)
    assert not result.is_error
    assert str(workspace.resolve()) in result.result["stdout"]


@pytest.mark.anyio
async def test_relative_paths_resolve_in_workspace(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "hello.txt").write_text("from workspace")
    result = await exec_tool.execute({"command": "cat hello.txt"}, ws_ctx)
    assert not result.is_error
    assert "from workspace" in result.result["stdout"]


# ---------------------------------------------------------------------------
# Timeout behaviour
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_timed_out_true_on_slow_command(ws_ctx: ToolContext) -> None:
    result = await exec_tool.execute(
        {"command": "sleep 60", "timeout": 0.1}, ws_ctx
    )
    assert not result.is_error
    assert result.result["timed_out"] is True


@pytest.mark.anyio
async def test_timeout_exit_code_nonzero_on_kill(ws_ctx: ToolContext) -> None:
    result = await exec_tool.execute(
        {"command": "sleep 60", "timeout": 0.1}, ws_ctx
    )
    assert not result.is_error
    assert result.result["exit_code"] != 0


@pytest.mark.anyio
async def test_custom_timeout_respected(ws_ctx: ToolContext) -> None:
    # Command finishes before the generous timeout — should not time out.
    result = await exec_tool.execute(
        {"command": "echo done", "timeout": 10}, ws_ctx
    )
    assert not result.is_error
    assert result.result["timed_out"] is False


# ---------------------------------------------------------------------------
# Input schema validation (pre-dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_missing_command_returns_is_error() -> None:
    result = await exec_tool.execute({}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_empty_command_returns_is_error() -> None:
    result = await exec_tool.execute({"command": ""}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_timeout_exceeds_max_returns_is_error() -> None:
    result = await exec_tool.execute(
        {"command": "echo x", "timeout": _MAX_TIMEOUT_SECONDS + 1}, _CTX
    )
    assert result.is_error


@pytest.mark.anyio
async def test_timeout_zero_returns_is_error() -> None:
    result = await exec_tool.execute({"command": "echo x", "timeout": 0}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_extra_field_returns_is_error() -> None:
    result = await exec_tool.execute(
        {"command": "echo x", "unknown_field": True}, _CTX
    )
    assert result.is_error


@pytest.mark.anyio
async def test_validation_error_code_on_bad_input() -> None:
    result = await exec_tool.execute({}, _CTX)
    assert result.is_error
    assert result.error is not None
    assert "validation" in result.error.code


# ---------------------------------------------------------------------------
# Failure → audit log written
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bad_cwd_writes_audit_log(tmp_path: Path) -> None:
    from meridian_sdk_tool import meridian_tool as _mk_tool

    audit_path = str(tmp_path / "audit.ndjson")

    @_mk_tool(
        name="exec",
        input_schema=_INPUT_SCHEMA,
        output_schema=_OUTPUT_SCHEMA,
        audit_log_path=audit_path,
    )
    async def _tool_with_audit(
        args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        from meridian_builtin_tools.exec import _run_command as _rc

        stdout, stderr, exit_code, timed_out, truncated = await _rc(
            command=args["command"],
            workspace=ctx.workspace,  # non-existent → OSError → RuntimeError
            timeout=float(args.get("timeout", 30)),
        )
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "command": args["command"],
            "timed_out": timed_out,
            "truncated": truncated,
        }

    bad_ctx = ToolContext(
        workspace="/nonexistent/path/that/cannot/exist",
        session_id="sess_audit_test",
    )
    result = await _tool_with_audit.execute({"command": "echo hi"}, bad_ctx)
    assert result.is_error

    lines = Path(audit_path).read_text().strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert "exec" in entry.get("tool_name", "")
    assert "error" in entry
