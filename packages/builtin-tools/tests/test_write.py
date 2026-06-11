"""Tests for the write built-in tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from meridian_sdk_tool import ToolContext

from meridian_builtin_tools.write import (
    _INPUT_SCHEMA,
    _OUTPUT_SCHEMA,
    _record_invocation,
    _resolve_safe,
    write_tool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CTX = ToolContext(workspace="/workspace", session_id="sess_write_test")


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
    target = dev / "out.txt"
    assert _resolve_safe(str(ws), str(target), allowed_roots=[str(dev)]) == target.resolve()


def test_resolve_safe_rejects_absolute_path_outside_allowed_roots(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    dev = tmp_path / "dev"
    dev.mkdir()
    with pytest.raises(ValueError, match="outside the allowed roots"):
        _resolve_safe(str(ws), "/etc/evil", allowed_roots=[str(dev)])


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
    return ToolContext(workspace=str(workspace), session_id="sess_write_int")


@pytest.mark.anyio
async def test_write_creates_file(ws_ctx: ToolContext, workspace: Path) -> None:
    result = await write_tool.execute({"path": "hello.txt", "content": "hello"}, ws_ctx)
    assert not result.is_error
    assert (workspace / "hello.txt").read_text(encoding="utf-8") == "hello"


@pytest.mark.anyio
async def test_write_returns_path(ws_ctx: ToolContext) -> None:
    result = await write_tool.execute({"path": "a.txt", "content": "x"}, ws_ctx)
    assert not result.is_error
    assert result.result["path"] == "a.txt"


@pytest.mark.anyio
async def test_write_returns_bytes_written(ws_ctx: ToolContext) -> None:
    result = await write_tool.execute({"path": "b.txt", "content": "hello"}, ws_ctx)
    assert not result.is_error
    assert result.result["bytes_written"] == 5


@pytest.mark.anyio
async def test_write_bytes_written_reflects_encoding(ws_ctx: ToolContext) -> None:
    # "é" is 2 bytes in UTF-8
    result = await write_tool.execute({"path": "c.txt", "content": "é"}, ws_ctx)
    assert not result.is_error
    assert result.result["bytes_written"] == 2


@pytest.mark.anyio
async def test_write_created_true_for_new_file(ws_ctx: ToolContext) -> None:
    result = await write_tool.execute({"path": "new.txt", "content": "x"}, ws_ctx)
    assert not result.is_error
    assert result.result["created"] is True


@pytest.mark.anyio
async def test_write_created_false_when_overwriting(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "existing.txt").write_text("old content", encoding="utf-8")
    result = await write_tool.execute({"path": "existing.txt", "content": "new content"}, ws_ctx)
    assert not result.is_error
    assert result.result["created"] is False


@pytest.mark.anyio
async def test_write_overwrite_replaces_content(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "f.txt").write_text("old", encoding="utf-8")
    await write_tool.execute({"path": "f.txt", "content": "new", "mode": "overwrite"}, ws_ctx)
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "new"


@pytest.mark.anyio
async def test_write_creates_parent_directories(ws_ctx: ToolContext, workspace: Path) -> None:
    result = await write_tool.execute({"path": "a/b/c/nested.txt", "content": "deep"}, ws_ctx)
    assert not result.is_error
    assert (workspace / "a" / "b" / "c" / "nested.txt").read_text(encoding="utf-8") == "deep"


@pytest.mark.anyio
async def test_write_empty_content(ws_ctx: ToolContext, workspace: Path) -> None:
    result = await write_tool.execute({"path": "empty.txt", "content": ""}, ws_ctx)
    assert not result.is_error
    assert result.result["bytes_written"] == 0
    assert (workspace / "empty.txt").read_bytes() == b""


@pytest.mark.anyio
async def test_write_custom_encoding(ws_ctx: ToolContext, workspace: Path) -> None:
    result = await write_tool.execute(
        {"path": "latin.txt", "content": "café", "encoding": "latin-1"}, ws_ctx
    )
    assert not result.is_error
    assert (workspace / "latin.txt").read_bytes() == "café".encode("latin-1")


# ---------------------------------------------------------------------------
# Mode: create
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_write_mode_create_fails_if_exists(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "exists.txt").write_text("already here", encoding="utf-8")
    result = await write_tool.execute(
        {"path": "exists.txt", "content": "new", "mode": "create"}, ws_ctx
    )
    assert result.is_error


@pytest.mark.anyio
async def test_write_mode_create_does_not_overwrite(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "preserve.txt").write_text("original", encoding="utf-8")
    await write_tool.execute(
        {"path": "preserve.txt", "content": "replaced", "mode": "create"}, ws_ctx
    )
    assert (workspace / "preserve.txt").read_text(encoding="utf-8") == "original"


@pytest.mark.anyio
async def test_write_mode_create_succeeds_for_new_file(ws_ctx: ToolContext) -> None:
    result = await write_tool.execute(
        {"path": "brand_new.txt", "content": "fresh", "mode": "create"}, ws_ctx
    )
    assert not result.is_error
    assert result.result["created"] is True


@pytest.mark.anyio
async def test_write_default_mode_is_overwrite(ws_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "default.txt").write_text("old", encoding="utf-8")
    result = await write_tool.execute({"path": "default.txt", "content": "new"}, ws_ctx)
    assert not result.is_error
    assert (workspace / "default.txt").read_text(encoding="utf-8") == "new"


@pytest.mark.anyio
async def test_write_rejects_content_over_size_cap(
    ws_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("meridian_builtin_tools.write._MAX_CONTENT_BYTES", 4)
    result = await write_tool.execute({"path": "big.txt", "content": "toolong"}, ws_ctx)
    assert result.is_error
    assert result.error is not None


def test_record_invocation_swallows_telemetry_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> None:
        raise RuntimeError("span unavailable")

    monkeypatch.setattr("opentelemetry.trace.get_current_span", _boom)
    # Must not raise despite the telemetry backend failing.
    _record_invocation("p.txt", "overwrite", 3, True)


# ---------------------------------------------------------------------------
# Workspace confinement
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_dotdot_path_returns_is_error(ws_ctx: ToolContext) -> None:
    result = await write_tool.execute({"path": "../escape.txt", "content": "x"}, ws_ctx)
    assert result.is_error


@pytest.mark.anyio
async def test_dotdot_path_error_code_is_execution_failed(ws_ctx: ToolContext) -> None:
    result = await write_tool.execute({"path": "../escape.txt", "content": "x"}, ws_ctx)
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
    result = await write_tool.execute({"path": "evil", "content": "x"}, ws_ctx)
    assert result.is_error


@pytest.mark.anyio
async def test_symlink_escape_does_not_write_file(
    ws_ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("original", encoding="utf-8")
    link = workspace / "evil"
    link.symlink_to(outside)
    await write_tool.execute({"path": "evil", "content": "injected"}, ws_ctx)
    assert outside.read_text(encoding="utf-8") == "original"


# ---------------------------------------------------------------------------
# Input schema validation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_missing_path_returns_is_error() -> None:
    result = await write_tool.execute({"content": "x"}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_missing_content_returns_is_error() -> None:
    result = await write_tool.execute({"path": "f.txt"}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_empty_path_returns_is_error() -> None:
    result = await write_tool.execute({"path": "", "content": "x"}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_invalid_mode_returns_is_error() -> None:
    result = await write_tool.execute({"path": "f.txt", "content": "x", "mode": "append"}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_extra_field_returns_is_error() -> None:
    result = await write_tool.execute({"path": "f.txt", "content": "x", "unknown": True}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_missing_both_required_fields_returns_is_error() -> None:
    result = await write_tool.execute({}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_validation_error_code_contains_validation() -> None:
    result = await write_tool.execute({}, _CTX)
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
        name="write",
        input_schema=_INPUT_SCHEMA,
        output_schema=_OUTPUT_SCHEMA,
        audit_log_path=audit_path,
    )
    async def _tool_with_audit(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        from meridian_builtin_tools.write import _resolve_safe

        target = _resolve_safe(ctx.workspace, args["path"])
        raw = args["content"].encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {
            "path": args["path"],
            "bytes_written": len(raw),
            "created": True,
        }

    result = await _tool_with_audit.execute({"path": "../outside.txt", "content": "x"}, ws_ctx)
    assert result.is_error

    lines = Path(audit_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert "write" in entry.get("tool_name", "")
    assert "error" in entry


@pytest.mark.anyio
async def test_validation_failure_writes_audit_log(tmp_path: Path) -> None:
    from meridian_sdk_tool import meridian_tool as _mk_tool

    audit_path = str(tmp_path / "audit_val.ndjson")

    @_mk_tool(
        name="write",
        input_schema=_INPUT_SCHEMA,
        output_schema=_OUTPUT_SCHEMA,
        audit_log_path=audit_path,
    )
    async def _tool_with_audit(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        return {"path": "x", "bytes_written": 0, "created": True}

    result = await _tool_with_audit.execute({}, _CTX)
    assert result.is_error

    lines = Path(audit_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert "write" in entry.get("tool_name", "")
    assert "error" in entry
