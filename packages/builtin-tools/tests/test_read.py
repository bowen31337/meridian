"""Tests for the read built-in tool."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from meridian_sdk_tool import ToolContext

from meridian_builtin_tools.read import (
    _INPUT_SCHEMA,
    _OUTPUT_SCHEMA,
    _record_invocation,
    _resolve_safe,
    read_tool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CTX = ToolContext(workspace="/workspace", session_id="sess_read_test")


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


# ---------------------------------------------------------------------------
# _resolve_safe unit tests
# ---------------------------------------------------------------------------


def test_resolve_safe_simple_relative_path(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    result = _resolve_safe(str(ws), "src/main.py")
    assert result == (ws / "src" / "main.py").resolve()


def test_resolve_safe_nested_path(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    result = _resolve_safe(str(ws), "a/b/c/d.txt")
    assert result == (ws / "a" / "b" / "c" / "d.txt").resolve()


def test_resolve_safe_rejects_dotdot_escape(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ValueError, match="resolves outside"):
        _resolve_safe(str(ws), "../outside.py")


def test_resolve_safe_rejects_deep_dotdot_escape(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ValueError):
        _resolve_safe(str(ws), "a/../../outside.py")


def test_resolve_safe_rejects_absolute_path_escape(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ValueError):
        _resolve_safe(str(ws), "/etc/passwd")


def test_resolve_safe_allows_absolute_path_in_allowed_root(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    dev = tmp_path / "dev"
    dev.mkdir()
    f = dev / "code.py"
    f.write_text("x")
    assert _resolve_safe(str(ws), str(f), allowed_roots=[str(dev)]) == f.resolve()


def test_resolve_safe_allows_absolute_path_in_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    f = ws / "a.py"
    f.write_text("x")
    assert _resolve_safe(str(ws), str(f)) == f.resolve()


def test_resolve_safe_rejects_absolute_path_outside_allowed_roots(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    dev = tmp_path / "dev"
    dev.mkdir()
    with pytest.raises(ValueError, match="outside the allowed roots"):
        _resolve_safe(str(ws), "/etc/passwd", allowed_roots=[str(dev)])


def test_resolve_safe_rejects_symlink_outside_jail(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    link = ws / "evil_link"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink"):
        _resolve_safe(str(ws), "evil_link")


def test_resolve_safe_rejects_intermediate_symlink_outside_jail(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    link = ws / "link_dir"
    link.symlink_to(outside_dir)
    with pytest.raises(ValueError, match="Symlink"):
        _resolve_safe(str(ws), "link_dir/file.txt")


def test_resolve_safe_allows_symlink_inside_jail(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    real = ws / "real.txt"
    real.write_text("hello")
    link = ws / "link.txt"
    link.symlink_to(real)
    result = _resolve_safe(str(ws), "link.txt")
    assert result == real.resolve()


def test_resolve_safe_returns_path_object(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    result = _resolve_safe(str(ws), "file.txt")
    assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# Happy path — integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return _make_workspace(tmp_path)


@pytest.fixture()
def ws_ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=str(workspace), session_id="sess_read_int")


@pytest.mark.anyio
async def test_read_returns_file_content(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "hello.txt").write_text("hello world", encoding="utf-8")
    result = await read_tool.execute({"path": "hello.txt"}, ws_ctx)
    assert not result.is_error
    assert result.result["content"] == "hello world"


@pytest.mark.anyio
async def test_read_rejects_file_over_size_cap(
    ws_ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "big.txt").write_text("toolong", encoding="utf-8")
    monkeypatch.setattr("meridian_builtin_tools.read._MAX_FILE_BYTES", 4)
    result = await read_tool.execute({"path": "big.txt"}, ws_ctx)
    assert result.is_error
    assert result.error is not None


def test_record_invocation_swallows_telemetry_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> None:
        raise RuntimeError("span unavailable")

    monkeypatch.setattr("opentelemetry.trace.get_current_span", _boom)
    _record_invocation("p.txt", "utf-8", 3)


@pytest.mark.anyio
async def test_read_returns_path(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a.txt").write_text("x", encoding="utf-8")
    result = await read_tool.execute({"path": "a.txt"}, ws_ctx)
    assert not result.is_error
    assert result.result["path"] == "a.txt"


@pytest.mark.anyio
async def test_read_returns_size(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "b.txt").write_bytes(b"hello")
    result = await read_tool.execute({"path": "b.txt"}, ws_ctx)
    assert not result.is_error
    assert result.result["size"] == 5


@pytest.mark.anyio
async def test_read_returns_encoding(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "c.txt").write_text("x", encoding="utf-8")
    result = await read_tool.execute({"path": "c.txt"}, ws_ctx)
    assert not result.is_error
    assert result.result["encoding"] == "utf-8"


@pytest.mark.anyio
async def test_read_default_encoding_is_utf8(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "d.txt").write_text("x", encoding="utf-8")
    result = await read_tool.execute({"path": "d.txt"}, ws_ctx)
    assert not result.is_error
    assert result.result["encoding"] == "utf-8"


@pytest.mark.anyio
async def test_read_empty_file(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "empty.txt").write_bytes(b"")
    result = await read_tool.execute({"path": "empty.txt"}, ws_ctx)
    assert not result.is_error
    assert result.result["content"] == ""
    assert result.result["size"] == 0


@pytest.mark.anyio
async def test_read_nested_path(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a").mkdir()
    (workspace / "a" / "b.txt").write_text("nested", encoding="utf-8")
    result = await read_tool.execute({"path": "a/b.txt"}, ws_ctx)
    assert not result.is_error
    assert result.result["content"] == "nested"


@pytest.mark.anyio
async def test_read_multibyte_content(ws_ctx: ToolContext, workspace: Path) -> None:
    # "é" is 2 bytes in UTF-8
    (workspace / "multi.txt").write_text("café", encoding="utf-8")
    result = await read_tool.execute({"path": "multi.txt"}, ws_ctx)
    assert not result.is_error
    assert result.result["content"] == "café"
    assert result.result["size"] == len("café".encode())


@pytest.mark.anyio
async def test_read_custom_text_encoding(ws_ctx: ToolContext, workspace: Path) -> None:
    raw = "café".encode("latin-1")
    (workspace / "latin.txt").write_bytes(raw)
    result = await read_tool.execute({"path": "latin.txt", "encoding": "latin-1"}, ws_ctx)
    assert not result.is_error
    assert result.result["content"] == "café"
    assert result.result["encoding"] == "latin-1"


# ---------------------------------------------------------------------------
# Base64 encoding
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_read_base64_binary_file(ws_ctx: ToolContext, workspace: Path) -> None:
    raw = bytes(range(256))
    (workspace / "binary.bin").write_bytes(raw)
    result = await read_tool.execute({"path": "binary.bin", "encoding": "base64"}, ws_ctx)
    assert not result.is_error
    assert result.result["encoding"] == "base64"
    assert base64.b64decode(result.result["content"]) == raw


@pytest.mark.anyio
async def test_read_base64_returns_ascii_string(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "f.bin").write_bytes(b"\x00\x01\x02")
    result = await read_tool.execute({"path": "f.bin", "encoding": "base64"}, ws_ctx)
    assert not result.is_error
    content = result.result["content"]
    assert isinstance(content, str)
    content.encode("ascii")  # Must be pure ASCII


@pytest.mark.anyio
async def test_read_base64_text_file_roundtrips(ws_ctx: ToolContext, workspace: Path) -> None:
    original = "hello world"
    (workspace / "text.txt").write_text(original, encoding="utf-8")
    result = await read_tool.execute({"path": "text.txt", "encoding": "base64"}, ws_ctx)
    assert not result.is_error
    decoded = base64.b64decode(result.result["content"]).decode("utf-8")
    assert decoded == original


@pytest.mark.anyio
async def test_read_base64_size_reflects_raw_bytes(ws_ctx: ToolContext, workspace: Path) -> None:
    raw = b"\xff\xfe\xfd"
    (workspace / "g.bin").write_bytes(raw)
    result = await read_tool.execute({"path": "g.bin", "encoding": "base64"}, ws_ctx)
    assert not result.is_error
    assert result.result["size"] == len(raw)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_read_missing_file_returns_is_error(ws_ctx: ToolContext) -> None:
    result = await read_tool.execute({"path": "nonexistent.txt"}, ws_ctx)
    assert result.is_error


@pytest.mark.anyio
async def test_read_missing_file_error_code_is_execution_failed(
    ws_ctx: ToolContext,
) -> None:
    result = await read_tool.execute({"path": "nonexistent.txt"}, ws_ctx)
    assert result.is_error
    assert result.error is not None
    assert "execution" in result.error.code


@pytest.mark.anyio
async def test_read_directory_returns_is_error(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "subdir").mkdir()
    result = await read_tool.execute({"path": "subdir"}, ws_ctx)
    assert result.is_error


# ---------------------------------------------------------------------------
# Workspace confinement
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_dotdot_path_returns_is_error(ws_ctx: ToolContext) -> None:
    result = await read_tool.execute({"path": "../escape.txt"}, ws_ctx)
    assert result.is_error


@pytest.mark.anyio
async def test_dotdot_path_error_code_is_execution_failed(ws_ctx: ToolContext) -> None:
    result = await read_tool.execute({"path": "../escape.txt"}, ws_ctx)
    assert result.is_error
    assert result.error is not None
    assert "execution" in result.error.code


@pytest.mark.anyio
async def test_symlink_escape_returns_is_error(
    ws_ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = workspace / "evil"
    link.symlink_to(outside)
    result = await read_tool.execute({"path": "evil"}, ws_ctx)
    assert result.is_error


@pytest.mark.anyio
async def test_symlink_escape_does_not_leak_content(
    ws_ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    secret = "very secret content"
    outside = tmp_path / "outside.txt"
    outside.write_text(secret, encoding="utf-8")
    link = workspace / "evil"
    link.symlink_to(outside)
    result = await read_tool.execute({"path": "evil"}, ws_ctx)
    assert result.is_error
    assert secret not in str(result.result or "")


# ---------------------------------------------------------------------------
# Input schema validation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_missing_path_returns_is_error() -> None:
    result = await read_tool.execute({}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_empty_path_returns_is_error() -> None:
    result = await read_tool.execute({"path": ""}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_extra_field_returns_is_error() -> None:
    result = await read_tool.execute({"path": "f.txt", "unknown": True}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_validation_error_code_contains_validation() -> None:
    result = await read_tool.execute({}, _CTX)
    assert result.is_error
    assert result.error is not None
    assert "validation" in result.error.code


# ---------------------------------------------------------------------------
# Failure → audit log written
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_confinement_failure_writes_audit_log(tmp_path: Path, ws_ctx: ToolContext) -> None:
    from meridian_sdk_tool import meridian_tool as _mk_tool

    audit_path = str(tmp_path / "audit.ndjson")

    @_mk_tool(
        name="read",
        input_schema=_INPUT_SCHEMA,
        output_schema=_OUTPUT_SCHEMA,
        audit_log_path=audit_path,
    )
    async def _tool_with_audit(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        from meridian_builtin_tools.read import _resolve_safe

        target = _resolve_safe(ctx.workspace, args["path"])
        raw = target.read_bytes()
        return {
            "path": args["path"],
            "content": raw.decode("utf-8"),
            "size": len(raw),
            "encoding": "utf-8",
        }

    result = await _tool_with_audit.execute({"path": "../outside.txt"}, ws_ctx)
    assert result.is_error

    lines = Path(audit_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert "read" in entry.get("tool_name", "")
    assert "error" in entry


@pytest.mark.anyio
async def test_validation_failure_writes_audit_log(tmp_path: Path) -> None:
    from meridian_sdk_tool import meridian_tool as _mk_tool

    audit_path = str(tmp_path / "audit_val.ndjson")

    @_mk_tool(
        name="read",
        input_schema=_INPUT_SCHEMA,
        output_schema=_OUTPUT_SCHEMA,
        audit_log_path=audit_path,
    )
    async def _tool_with_audit(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        return {"path": "x", "content": "", "size": 0, "encoding": "utf-8"}

    result = await _tool_with_audit.execute({}, _CTX)
    assert result.is_error

    lines = Path(audit_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert "read" in entry.get("tool_name", "")
    assert "error" in entry
