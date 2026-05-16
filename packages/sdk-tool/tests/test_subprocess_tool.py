"""Tests for subprocess_tool and subprocess_server (Architecture §11.2)."""

from __future__ import annotations

import io
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from meridian_sdk_tool import ToolContext, subprocess_tool
from meridian_sdk_tool.subprocess_server import run_subprocess_tool

_CTX = ToolContext(workspace="/workspace", session_id="sess_sub")


# ---------------------------------------------------------------------------
# subprocess_server — unit tests (no subprocess spawned)
# ---------------------------------------------------------------------------


class _IO:
    """Simple in-memory stdin/stdout pair for testing run_subprocess_tool."""

    def __init__(self, payload: str) -> None:
        self.stdin = io.StringIO(payload)
        self.stdout = io.StringIO()

    def result(self) -> dict[str, Any]:
        self.stdout.seek(0)
        return json.loads(self.stdout.read())


def test_server_success() -> None:
    request = json.dumps({"args": {"x": 21}, "context": {}})
    io_ = _IO(request)
    run_subprocess_tool(lambda args, ctx: {"doubled": args["x"] * 2}, input_stream=io_.stdin, output_stream=io_.stdout)
    assert io_.result() == {"result": {"doubled": 42}}


def test_server_handler_exception_writes_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # run_subprocess_tool calls sys.exit(1) on handler exception; capture it
    calls: list[int] = []
    monkeypatch.setattr(sys, "exit", lambda code: calls.append(code))

    request = json.dumps({"args": {}, "context": {}})
    io_ = _IO(request)

    def bad_handler(args: dict, ctx: dict) -> Any:
        raise RuntimeError("boom")

    run_subprocess_tool(bad_handler, input_stream=io_.stdin, output_stream=io_.stdout)

    data = io_.result()
    assert "error" in data
    assert data["error"]["code"] == "execution_failed"
    assert "boom" in data["error"]["message"]
    assert calls == [1]


def test_server_invalid_json_input_writes_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(sys, "exit", lambda code: calls.append(code))

    io_ = _IO("NOT JSON")
    run_subprocess_tool(lambda a, c: {}, input_stream=io_.stdin, output_stream=io_.stdout)

    data = io_.result()
    assert data["error"]["code"] == "invalid_request"
    assert calls == [1]


# ---------------------------------------------------------------------------
# subprocess_tool — integration test with a real subprocess
# ---------------------------------------------------------------------------


def _write_echo_script(tmp_path: Path) -> Path:
    """Write a minimal subprocess tool server script to a temp file."""
    script = tmp_path / "echo_tool.py"
    script.write_text(
        textwrap.dedent(f"""\
            #!{sys.executable}
            import sys
            sys.path.insert(0, {str(Path(__file__).parent.parent / "src")!r})
            from meridian_sdk_tool.subprocess_server import run_subprocess_tool

            def handle(args, ctx):
                return {{"echo": args, "workspace": ctx.get("workspace")}}

            run_subprocess_tool(handle)
        """)
    )
    script.chmod(0o755)
    return script


@pytest.mark.anyio
async def test_subprocess_tool_round_trip(tmp_path: Path) -> None:
    script = _write_echo_script(tmp_path)

    tool = subprocess_tool(
        name="echo",
        description="Echo args back",
        path=sys.executable,  # we'll pass the script as an arg via path trick
        input_schema={"type": "object"},
        output_schema=None,
        timeout_ms=10_000,
    )
    # Override path to call python <script> since the script may not be +x everywhere
    tool.definition.handler.path = str(script)  # type: ignore[union-attr]

    result = await tool.execute({"hello": "world"}, _CTX)
    assert not result.is_error, result.error
    assert result.result["echo"] == {"hello": "world"}
    assert result.result["workspace"] == "/workspace"


@pytest.mark.anyio
async def test_subprocess_tool_error_response(tmp_path: Path) -> None:
    script = tmp_path / "failing_tool.py"
    script.write_text(
        textwrap.dedent(f"""\
            #!{sys.executable}
            import sys
            sys.path.insert(0, {str(Path(__file__).parent.parent / "src")!r})
            from meridian_sdk_tool.subprocess_server import run_subprocess_tool

            def handle(args, ctx):
                raise ValueError("intentional failure")

            run_subprocess_tool(handle)
        """)
    )
    script.chmod(0o755)

    tool = subprocess_tool(
        name="failing",
        description="Always fails",
        path=str(script),
        input_schema={"type": "object"},
        timeout_ms=5_000,
    )

    result = await tool.execute({}, _CTX)
    assert result.is_error
    assert result.error is not None
    assert "execution_failed" in result.error.code or "intentional failure" in result.error.message


@pytest.mark.anyio
async def test_subprocess_tool_input_validation_failure() -> None:
    tool = subprocess_tool(
        name="strict",
        description="Strict schema",
        path="/usr/bin/env",  # won't be called because validation fails first
        input_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
        timeout_ms=5_000,
    )

    result = await tool.execute({"n": "not-an-int"}, _CTX)
    assert result.is_error
    assert result.error is not None
    assert "validation" in result.error.code
