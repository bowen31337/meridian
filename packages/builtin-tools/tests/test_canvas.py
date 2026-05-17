"""Tests for the canvas_op built-in tool."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest

from meridian_builtin_tools.canvas import SUPPORTED_KINDS, _reset_sequence_counters, canvas_op_tool
from meridian_sdk_tool import ToolContext

_CTX = ToolContext(workspace="/workspace", session_id="sess_canvas_test")
_CTX2 = ToolContext(workspace="/workspace", session_id="sess_canvas_other")


@pytest.fixture(autouse=True)
def reset_sequences() -> None:
    """Reset the per-session sequence counters before every test."""
    _reset_sequence_counters()


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_returns_canvas_op_content_block() -> None:
    result = await canvas_op_tool.execute(
        {"widget_id": "w1", "widget_kind": "meridian.text", "op": "set", "props": {"text": "hi"}},
        _CTX,
    )
    assert not result.is_error
    block = result.result
    assert block["type"] == "canvas_op"


@pytest.mark.anyio
async def test_canvas_op_fields_match_input() -> None:
    result = await canvas_op_tool.execute(
        {
            "widget_id": "my-widget",
            "widget_kind": "meridian.markdown",
            "op": "patch",
            "props": {"content": "# Hello"},
        },
        _CTX,
    )
    assert not result.is_error
    op = result.result["canvas_op"]
    assert op["widget_id"] == "my-widget"
    assert op["widget_kind"] == "meridian.markdown"
    assert op["op"] == "patch"
    assert op["props"] == {"content": "# Hello"}


@pytest.mark.anyio
async def test_session_id_comes_from_ctx() -> None:
    result = await canvas_op_tool.execute(
        {"widget_id": "w1", "widget_kind": "meridian.text", "op": "set", "props": {"text": "x"}},
        _CTX,
    )
    assert result.result["canvas_op"]["session_id"] == "sess_canvas_test"


@pytest.mark.anyio
async def test_timestamp_is_iso8601() -> None:
    result = await canvas_op_tool.execute(
        {"widget_id": "w1", "widget_kind": "meridian.text", "op": "set", "props": {"text": "x"}},
        _CTX,
    )
    ts = result.result["canvas_op"]["timestamp"]
    # e.g. "2026-05-17T12:00:00Z"
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", ts), f"Bad timestamp: {ts!r}"


@pytest.mark.anyio
async def test_sequence_starts_at_one() -> None:
    result = await canvas_op_tool.execute(
        {"widget_id": "w1", "widget_kind": "meridian.text", "op": "set"},
        _CTX,
    )
    assert result.result["canvas_op"]["sequence"] == 1


@pytest.mark.anyio
async def test_sequence_increments_within_session() -> None:
    for expected in (1, 2, 3):
        result = await canvas_op_tool.execute(
            {"widget_id": "w1", "widget_kind": "meridian.text", "op": "set"},
            _CTX,
        )
        assert result.result["canvas_op"]["sequence"] == expected


@pytest.mark.anyio
async def test_sequence_is_independent_across_sessions() -> None:
    r1 = await canvas_op_tool.execute(
        {"widget_id": "w1", "widget_kind": "meridian.text", "op": "set"},
        _CTX,
    )
    r2 = await canvas_op_tool.execute(
        {"widget_id": "w1", "widget_kind": "meridian.text", "op": "set"},
        _CTX2,
    )
    assert r1.result["canvas_op"]["sequence"] == 1
    assert r2.result["canvas_op"]["sequence"] == 1


@pytest.mark.anyio
async def test_props_defaults_to_empty_dict_when_omitted() -> None:
    result = await canvas_op_tool.execute(
        {"widget_id": "w1", "widget_kind": "meridian.progress", "op": "clear"},
        _CTX,
    )
    assert not result.is_error
    assert result.result["canvas_op"]["props"] == {}


@pytest.mark.anyio
@pytest.mark.parametrize("kind", sorted(SUPPORTED_KINDS))
async def test_all_supported_widget_kinds_accepted(kind: str) -> None:
    result = await canvas_op_tool.execute(
        {"widget_id": "w1", "widget_kind": kind, "op": "set"},
        _CTX,
    )
    assert not result.is_error, f"Expected success for kind {kind!r}, got {result.error}"


@pytest.mark.anyio
@pytest.mark.parametrize("op", ["set", "patch", "append", "clear"])
async def test_all_ops_accepted(op: str) -> None:
    result = await canvas_op_tool.execute(
        {"widget_id": "w1", "widget_kind": "meridian.text", "op": op},
        _CTX,
    )
    assert not result.is_error, f"Expected success for op {op!r}, got {result.error}"


# ---------------------------------------------------------------------------
# Failure path — invalid input (pre-dispatch schema validation)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_unknown_widget_kind_returns_is_error() -> None:
    result = await canvas_op_tool.execute(
        {"widget_id": "w1", "widget_kind": "acme.unknown", "op": "set"},
        _CTX,
    )
    assert result.is_error
    assert result.error is not None
    assert "validation" in result.error.code


@pytest.mark.anyio
async def test_invalid_op_returns_is_error() -> None:
    result = await canvas_op_tool.execute(
        {"widget_id": "w1", "widget_kind": "meridian.text", "op": "delete"},
        _CTX,
    )
    assert result.is_error
    assert result.error is not None
    assert "validation" in result.error.code


@pytest.mark.anyio
async def test_missing_widget_id_returns_is_error() -> None:
    result = await canvas_op_tool.execute(
        {"widget_kind": "meridian.text", "op": "set"},
        _CTX,
    )
    assert result.is_error


@pytest.mark.anyio
async def test_missing_widget_kind_returns_is_error() -> None:
    result = await canvas_op_tool.execute(
        {"widget_id": "w1", "op": "set"},
        _CTX,
    )
    assert result.is_error


@pytest.mark.anyio
async def test_extra_top_level_field_returns_is_error() -> None:
    result = await canvas_op_tool.execute(
        {
            "widget_id": "w1",
            "widget_kind": "meridian.text",
            "op": "set",
            "unexpected": True,
        },
        _CTX,
    )
    assert result.is_error


# ---------------------------------------------------------------------------
# Audit log — failure path writes an entry
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_invalid_kind_writes_audit_log() -> None:
    with tempfile.NamedTemporaryFile(suffix=".ndjson", delete=False) as f:
        audit_path = f.name

    # We need a tool instance that targets our temp audit log.
    # Recreate one with the same implementation but a custom audit path.
    from meridian_builtin_tools.canvas import _INPUT_SCHEMA, _OUTPUT_SCHEMA, _next_sequence
    from meridian_sdk_tool import meridian_tool

    @meridian_tool(
        name="canvas_op",
        input_schema=_INPUT_SCHEMA,
        output_schema=_OUTPUT_SCHEMA,
        audit_log_path=audit_path,
    )
    async def _tool_with_audit(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        import time as _time

        return {
            "type": "canvas_op",
            "canvas_op": {
                "op": args["op"],
                "widget_id": args["widget_id"],
                "widget_kind": args["widget_kind"],
                "props": args.get("props") or {},
                "sequence": _next_sequence(ctx.session_id),
                "session_id": ctx.session_id,
                "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
            },
        }

    result = await _tool_with_audit.execute(
        {"widget_id": "w1", "widget_kind": "bad.kind", "op": "set"},
        _CTX,
    )
    assert result.is_error

    lines = Path(audit_path).read_text().strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert "canvas_op" in entry.get("tool_name", "")
    assert "error" in entry
