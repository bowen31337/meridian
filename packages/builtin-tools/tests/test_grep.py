"""Tests for the grep built-in tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from meridian_sdk_tool import ToolContext

from meridian_builtin_tools.grep import (
    _INPUT_SCHEMA,
    _OUTPUT_SCHEMA,
    _parse_rg_json,
    _record_invocation,
    _run_rg,
    _text_or_bytes,
    grep_tool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CTX = ToolContext(workspace="/workspace", session_id="sess_grep_test")


def _make_workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace with known content for ripgrep to search."""
    ws = tmp_path / "workspace"
    ws.mkdir()

    (ws / "src").mkdir()
    (ws / "src" / "auth.py").write_text(
        "def authenticate(user, password):\n"
        "    # validate credentials\n"
        "    return check_db(user, password)\n",
        encoding="utf-8",
    )
    (ws / "src" / "logout.py").write_text(
        "def logout(session):\n    session.clear()\n",
        encoding="utf-8",
    )

    (ws / "docs").mkdir()
    (ws / "docs" / "README.md").write_text(
        "# Auth Guide\n\nAuthentication is required for all API endpoints.\n",
        encoding="utf-8",
    )

    (ws / "config.json").write_text('{"debug": false, "auth_required": true}\n', encoding="utf-8")

    return ws


# ---------------------------------------------------------------------------
# _text_or_bytes unit tests
# ---------------------------------------------------------------------------


def test_text_or_bytes_returns_text_field() -> None:
    assert _text_or_bytes({"text": "hello\n"}) == "hello\n"


def test_text_or_bytes_decodes_bytes_field() -> None:
    import base64

    raw = base64.b64encode(b"hello").decode()
    assert _text_or_bytes({"bytes": raw}) == "hello"


def test_text_or_bytes_empty_dict_returns_empty_string() -> None:
    assert _text_or_bytes({}) == ""


def test_text_or_bytes_prefers_text_over_bytes() -> None:
    assert _text_or_bytes({"text": "ok", "bytes": "aGVsbG8="}) == "ok"


# ---------------------------------------------------------------------------
# _parse_rg_json unit tests
# ---------------------------------------------------------------------------


def _make_rg_json(*messages: dict[str, Any]) -> bytes:
    """Encode a sequence of ripgrep JSON messages as NDJSON bytes."""
    return b"\n".join(json.dumps(m).encode() for m in messages)


def _begin(path: str) -> dict[str, Any]:
    return {"type": "begin", "data": {"path": {"text": path}}}


def _match(path: str, lineno: int, text: str) -> dict[str, Any]:
    return {
        "type": "match",
        "data": {
            "path": {"text": path},
            "lines": {"text": text},
            "line_number": lineno,
            "absolute_offset": 0,
            "submatches": [],
        },
    }


def _context(path: str, lineno: int, text: str) -> dict[str, Any]:
    return {
        "type": "context",
        "data": {
            "path": {"text": path},
            "lines": {"text": text},
            "line_number": lineno,
            "absolute_offset": 0,
            "submatches": [],
        },
    }


def _end(path: str) -> dict[str, Any]:
    return {"type": "end", "data": {"path": {"text": path}, "binary_offset": None, "stats": {}}}


def test_parse_rg_json_single_match() -> None:
    stdout = _make_rg_json(
        _begin("/ws/a.py"),
        _match("/ws/a.py", 3, "def foo():\n"),
        _end("/ws/a.py"),
    )
    matches, truncated = _parse_rg_json(stdout, "/ws", max_results=50)
    assert len(matches) == 1
    assert matches[0]["file_path"] == "a.py"
    assert matches[0]["line_number"] == 3
    assert matches[0]["line"] == "def foo():"
    assert matches[0]["context_before"] == []
    assert matches[0]["context_after"] == []
    assert not truncated


def test_parse_rg_json_strips_workspace_prefix() -> None:
    stdout = _make_rg_json(
        _begin("/my/workspace/src/main.py"),
        _match("/my/workspace/src/main.py", 1, "hello\n"),
        _end("/my/workspace/src/main.py"),
    )
    matches, _ = _parse_rg_json(stdout, "/my/workspace", max_results=50)
    assert matches[0]["file_path"] == "src/main.py"


def test_parse_rg_json_context_lines_attached() -> None:
    stdout = _make_rg_json(
        _begin("/ws/f.py"),
        _context("/ws/f.py", 1, "line one\n"),
        _context("/ws/f.py", 2, "line two\n"),
        _match("/ws/f.py", 3, "MATCH\n"),
        _context("/ws/f.py", 4, "line four\n"),
        _context("/ws/f.py", 5, "line five\n"),
        _end("/ws/f.py"),
    )
    matches, _ = _parse_rg_json(stdout, "/ws", max_results=50)
    assert matches[0]["context_before"] == ["line one", "line two"]
    assert matches[0]["context_after"] == ["line four", "line five"]


def test_parse_rg_json_adjacent_matches_share_no_context() -> None:
    stdout = _make_rg_json(
        _begin("/ws/f.py"),
        _match("/ws/f.py", 1, "first\n"),
        _context("/ws/f.py", 2, "between\n"),
        _match("/ws/f.py", 3, "second\n"),
        _end("/ws/f.py"),
    )
    matches, _ = _parse_rg_json(stdout, "/ws", max_results=50)
    assert len(matches) == 2
    assert matches[0]["context_after"] == ["between"]
    assert matches[1]["context_before"] == ["between"]


def test_parse_rg_json_truncates_at_max_results() -> None:
    msgs: list[dict[str, Any]] = []
    for i in range(5):
        path = f"/ws/f{i}.py"
        msgs.extend([_begin(path), _match(path, 1, f"match {i}\n"), _end(path)])
    stdout = _make_rg_json(*msgs)
    matches, truncated = _parse_rg_json(stdout, "/ws", max_results=3)
    assert len(matches) == 3
    assert truncated


def test_parse_rg_json_not_truncated_when_under_limit() -> None:
    stdout = _make_rg_json(
        _begin("/ws/a.py"),
        _match("/ws/a.py", 1, "x\n"),
        _end("/ws/a.py"),
    )
    matches, truncated = _parse_rg_json(stdout, "/ws", max_results=10)
    assert len(matches) == 1
    assert not truncated


def test_parse_rg_json_ignores_malformed_lines() -> None:
    good = json.dumps(_match("/ws/a.py", 1, "x\n")).encode()
    stdout = b"\n".join(
        [
            json.dumps(_begin("/ws/a.py")).encode(),
            b"NOT JSON {{{",
            good,
            json.dumps(_end("/ws/a.py")).encode(),
        ]
    )
    matches, _ = _parse_rg_json(stdout, "/ws", max_results=50)
    assert len(matches) == 1


def test_parse_rg_json_empty_stdout_returns_empty() -> None:
    matches, truncated = _parse_rg_json(b"", "/ws", max_results=50)
    assert matches == []
    assert not truncated


# ---------------------------------------------------------------------------
# Happy path — integration with real ripgrep
# ---------------------------------------------------------------------------

_RG_AVAILABLE = __import__("shutil").which("rg") is not None


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return _make_workspace(tmp_path)


@pytest.fixture()
def ws_ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=str(workspace), session_id="sess_grep_int")


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_returns_no_error_on_match(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "authenticate"}, ws_ctx)
    assert not result.is_error


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_matches_is_a_list(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "authenticate"}, ws_ctx)
    assert isinstance(result.result["matches"], list)


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_total_matches_len_of_matches(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "authenticate"}, ws_ctx)
    assert result.result["total"] == len(result.result["matches"])


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_pattern_is_echoed(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "authenticate"}, ws_ctx)
    assert result.result["pattern"] == "authenticate"


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_glob_is_echoed_when_provided(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "authenticate", "glob": "**/*.py"}, ws_ctx)
    assert result.result["glob"] == "**/*.py"


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_glob_is_null_when_omitted(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "authenticate"}, ws_ctx)
    assert result.result["glob"] is None


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_result_items_have_required_fields(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "authenticate"}, ws_ctx)
    assert result.result["total"] > 0
    for item in result.result["matches"]:
        assert "file_path" in item
        assert "line_number" in item
        assert "line" in item
        assert "context_before" in item
        assert "context_after" in item


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_match_line_contains_pattern(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "authenticate"}, ws_ctx)
    assert result.result["total"] > 0
    for item in result.result["matches"]:
        assert "authenticate" in item["line"].lower()


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_file_path_is_relative(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "authenticate"}, ws_ctx)
    for item in result.result["matches"]:
        assert not item["file_path"].startswith("/")


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_line_number_is_positive_integer(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "authenticate"}, ws_ctx)
    for item in result.result["matches"]:
        assert isinstance(item["line_number"], int)
        assert item["line_number"] >= 1


# ---------------------------------------------------------------------------
# Context lines
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_context_lines_are_lists(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "authenticate", "context_lines": 1}, ws_ctx)
    for item in result.result["matches"]:
        assert isinstance(item["context_before"], list)
        assert isinstance(item["context_after"], list)


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_zero_context_lines_returns_empty_context(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "authenticate", "context_lines": 0}, ws_ctx)
    for item in result.result["matches"]:
        assert item["context_before"] == []
        assert item["context_after"] == []


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_context_lines_bounded_by_file_start(ws_ctx: ToolContext) -> None:
    # authenticate appears on line 1 of auth.py — context_before must be empty
    result = await grep_tool.execute({"pattern": "def authenticate", "context_lines": 3}, ws_ctx)
    auth_matches = [m for m in result.result["matches"] if "auth.py" in m["file_path"]]
    assert auth_matches
    assert auth_matches[0]["context_before"] == []


# ---------------------------------------------------------------------------
# Glob filter
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_glob_restricts_to_python_files(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "auth", "glob": "**/*.py"}, ws_ctx)
    assert not result.is_error
    for item in result.result["matches"]:
        assert item["file_path"].endswith(".py")


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_glob_restricts_to_markdown_files(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "auth", "glob": "**/*.md"}, ws_ctx)
    assert not result.is_error
    for item in result.result["matches"]:
        assert item["file_path"].endswith(".md")


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_nonmatching_glob_returns_empty_results(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "auth", "glob": "**/*.nonexistent"}, ws_ctx)
    assert not result.is_error
    assert result.result["total"] == 0


# ---------------------------------------------------------------------------
# Fixed strings
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_fixed_strings_matches_literal_text(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "auth_required", "fixed_strings": True}, ws_ctx)
    assert not result.is_error
    assert result.result["total"] > 0
    for item in result.result["matches"]:
        assert "auth_required" in item["line"]


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_fixed_strings_does_not_interpret_regex(ws_ctx: ToolContext) -> None:
    # "auth.required" as regex would match "auth_required" (. matches any char);
    # as a literal string it should not.
    result = await grep_tool.execute({"pattern": "auth.required", "fixed_strings": True}, ws_ctx)
    assert not result.is_error
    assert result.result["total"] == 0


# ---------------------------------------------------------------------------
# Case insensitive
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_case_insensitive_finds_mixed_case(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "AUTHENTICATE", "case_insensitive": True}, ws_ctx)
    assert not result.is_error
    assert result.result["total"] > 0


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_case_sensitive_misses_different_case(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "AUTHENTICATE", "case_insensitive": False}, ws_ctx)
    assert not result.is_error
    assert result.result["total"] == 0


# ---------------------------------------------------------------------------
# max_results / truncation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_max_results_caps_matches(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "a", "max_results": 1}, ws_ctx)
    assert not result.is_error
    assert result.result["total"] <= 1


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_truncated_true_when_limit_reached(ws_ctx: ToolContext) -> None:
    # "a" should appear many times; limit to 1 to force truncation.
    result = await grep_tool.execute({"pattern": "a", "max_results": 1}, ws_ctx)
    assert not result.is_error
    assert result.result["truncated"] is True


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_truncated_false_when_all_results_fit(ws_ctx: ToolContext) -> None:
    # A very specific pattern unlikely to exceed 200 results.
    result = await grep_tool.execute({"pattern": "def authenticate"}, ws_ctx)
    assert not result.is_error
    assert result.result["truncated"] is False


# ---------------------------------------------------------------------------
# No matches
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RG_AVAILABLE, reason="ripgrep not installed")
@pytest.mark.anyio
async def test_no_matches_returns_empty_list(ws_ctx: ToolContext) -> None:
    result = await grep_tool.execute({"pattern": "ZZZZNOMATCH_UNIQUE_STRING"}, ws_ctx)
    assert not result.is_error
    assert result.result["matches"] == []
    assert result.result["total"] == 0
    assert result.result["truncated"] is False


# ---------------------------------------------------------------------------
# Input schema validation (pre-dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_missing_pattern_returns_is_error() -> None:
    result = await grep_tool.execute({}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_empty_pattern_returns_is_error() -> None:
    result = await grep_tool.execute({"pattern": ""}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_context_lines_over_max_returns_is_error() -> None:
    result = await grep_tool.execute({"pattern": "x", "context_lines": 11}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_max_results_zero_returns_is_error() -> None:
    result = await grep_tool.execute({"pattern": "x", "max_results": 0}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_extra_field_returns_is_error() -> None:
    result = await grep_tool.execute({"pattern": "x", "unknown_field": True}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_error_code_is_validation_related_on_bad_input() -> None:
    result = await grep_tool.execute({}, _CTX)
    assert result.is_error
    assert result.error is not None
    assert "validation" in result.error.code


# ---------------------------------------------------------------------------
# rg not found → execution_failed
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_rg_not_found_returns_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import meridian_builtin_tools.grep as grep_mod

    monkeypatch.setattr(grep_mod.shutil, "which", lambda _: None)
    result = await grep_tool.execute({"pattern": "foo"}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_rg_not_found_error_code_is_execution_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import meridian_builtin_tools.grep as grep_mod

    monkeypatch.setattr(grep_mod.shutil, "which", lambda _: None)
    result = await grep_tool.execute({"pattern": "foo"}, _CTX)
    assert result.is_error
    assert result.error is not None
    assert "execution" in result.error.code


# ---------------------------------------------------------------------------
# Failure → audit log written
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_rg_not_found_writes_audit_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from meridian_sdk_tool import meridian_tool as _mk_tool

    import meridian_builtin_tools.grep as grep_mod

    monkeypatch.setattr(grep_mod.shutil, "which", lambda _: None)
    audit_path = str(tmp_path / "audit.ndjson")

    @_mk_tool(
        name="grep",
        input_schema=_INPUT_SCHEMA,
        output_schema=_OUTPUT_SCHEMA,
        audit_log_path=audit_path,
    )
    async def _tool_with_audit(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        from meridian_builtin_tools.grep import _run_rg

        matches, truncated = await _run_rg(
            pattern=args["pattern"],
            workspace=ctx.workspace,
            glob=args.get("glob"),
            context_lines=int(args.get("context_lines", 2)),
            fixed_strings=bool(args.get("fixed_strings", False)),
            case_insensitive=bool(args.get("case_insensitive", False)),
            max_results=int(args.get("max_results", 50)),
        )
        return {
            "matches": matches,
            "total": len(matches),
            "pattern": args["pattern"],
            "glob": args.get("glob"),
            "truncated": truncated,
        }

    result = await _tool_with_audit.execute({"pattern": "foo"}, _CTX)
    assert result.is_error

    lines = Path(audit_path).read_text().strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert "grep" in entry.get("tool_name", "")
    assert "error" in entry


# ---------------------------------------------------------------------------
# _parse_rg_json branch coverage
# ---------------------------------------------------------------------------


def _ndjson(*objs: dict[str, Any]) -> bytes:
    return "\n".join(json.dumps(o) for o in objs).encode("utf-8")


def test_parse_rg_json_skips_blank_lines() -> None:
    # A blank line between NDJSON records must be skipped (line 200).
    body = (
        json.dumps({"type": "begin", "data": {"path": {"text": "a.py"}}})
        + "\n\n   \n"
        + json.dumps({"type": "match", "data": {"line_number": 1, "lines": {"text": "hit\n"}}})
        + "\n"
        + json.dumps({"type": "end", "data": {}})
    ).encode("utf-8")
    matches, truncated = _parse_rg_json(body, "", 50)
    assert not truncated
    assert len(matches) == 1
    assert matches[0]["line"] == "hit"


def test_parse_rg_json_path_not_under_workspace_left_unstripped() -> None:
    # workspace set but path does not start with it → no stripping (214->218 False).
    body = _ndjson(
        {"type": "begin", "data": {"path": {"text": "/other/a.py"}}},
        {"type": "match", "data": {"line_number": 2, "lines": {"text": "x\n"}}},
        {"type": "end", "data": {}},
    )
    matches, _ = _parse_rg_json(body, "/workspace", 50)
    assert matches[0]["file_path"] == "/other/a.py"


def test_parse_rg_json_ignores_unknown_message_type_inside_file() -> None:
    # A message that is neither "match" nor "context" inside the file block is
    # ignored (223->228 False).
    body = _ndjson(
        {"type": "begin", "data": {"path": {"text": "a.py"}}},
        {"type": "summary", "data": {}},
        {"type": "match", "data": {"line_number": 3, "lines": {"text": "y\n"}}},
        {"type": "end", "data": {}},
    )
    matches, _ = _parse_rg_json(body, "", 50)
    assert len(matches) == 1
    assert matches[0]["line_number"] == 3


# ---------------------------------------------------------------------------
# _run_rg error paths
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", b"boom on stderr"

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return 0


@pytest.mark.anyio
async def test_run_rg_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(returncode=0)

    async def _fake_exec(*_a: Any, **_k: Any) -> _FakeProc:
        return proc

    async def _fake_wait_for(coro: Any, *_a: Any, **_k: Any) -> Any:
        coro.close()
        raise TimeoutError

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/rg")
    monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_exec)
    monkeypatch.setattr("asyncio.wait_for", _fake_wait_for)

    with pytest.raises(RuntimeError, match="timed out"):
        await _run_rg("p", "/ws", None, 0, False, False, 50)
    assert proc.killed is True


@pytest.mark.anyio
async def test_run_rg_error_exit_code_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(returncode=2)

    async def _fake_exec(*_a: Any, **_k: Any) -> _FakeProc:
        return proc

    async def _fake_wait_for(coro: Any, *, timeout: float) -> tuple[bytes, bytes]:
        return await coro

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/rg")
    monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_exec)
    monkeypatch.setattr("asyncio.wait_for", _fake_wait_for)

    with pytest.raises(RuntimeError, match="ripgrep error"):
        await _run_rg("p", "/ws", None, 0, False, False, 50)


# ---------------------------------------------------------------------------
# _record_invocation telemetry-error swallowing
# ---------------------------------------------------------------------------


def test_record_invocation_swallows_telemetry_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> None:
        raise RuntimeError("span unavailable")

    monkeypatch.setattr("opentelemetry.trace.get_current_span", _boom)
    _record_invocation("pattern", None, 0)
