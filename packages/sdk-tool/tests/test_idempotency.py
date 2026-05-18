"""Tests for the idempotent-retry contract (Architecture §11.5)."""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path

import pytest

from meridian_sdk_tool import ToolContext, meridian_tool
from meridian_sdk_tool._idempotency import cache_result, clear, get_cached_result
from meridian_sdk_tool._types import ToolResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Clear the idempotency store before every test."""
    clear()


def _ctx(idempotency_key: str | None = None) -> ToolContext:
    return ToolContext(
        workspace="/workspace",
        session_id="sess_test",
        idempotency_key=idempotency_key,
    )


call_count = 0


@meridian_tool(input_schema={"type": "object"})
async def counting_tool(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    global call_count
    call_count += 1
    return {"n": call_count}


@meridian_tool(input_schema={"type": "object"})
async def failing_tool(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raise RuntimeError("handler always fails")


# ---------------------------------------------------------------------------
# Unit tests for _idempotency module
# ---------------------------------------------------------------------------


def test_cache_miss_returns_none() -> None:
    assert get_cached_result("my_tool", "key-1") is None


def test_cache_stores_and_retrieves() -> None:
    result = ToolResult.ok({"x": 1})
    cache_result("my_tool", "key-1", result)
    assert get_cached_result("my_tool", "key-1") == result


def test_cache_first_write_wins() -> None:
    r1 = ToolResult.ok({"v": 1})
    r2 = ToolResult.ok({"v": 2})
    cache_result("t", "k", r1)
    cache_result("t", "k", r2)  # must be a no-op
    assert get_cached_result("t", "k") == r1


def test_different_keys_are_independent() -> None:
    r1 = ToolResult.ok({"v": 1})
    r2 = ToolResult.ok({"v": 2})
    cache_result("t", "key-a", r1)
    cache_result("t", "key-b", r2)
    assert get_cached_result("t", "key-a") == r1
    assert get_cached_result("t", "key-b") == r2


def test_different_tools_share_same_key_independently() -> None:
    r1 = ToolResult.ok({"v": "tool_a"})
    r2 = ToolResult.ok({"v": "tool_b"})
    cache_result("tool_a", "shared-key", r1)
    cache_result("tool_b", "shared-key", r2)
    assert get_cached_result("tool_a", "shared-key") == r1
    assert get_cached_result("tool_b", "shared-key") == r2


def test_clear_removes_all_entries() -> None:
    cache_result("t", "k", ToolResult.ok({}))
    clear()
    assert get_cached_result("t", "k") is None


# ---------------------------------------------------------------------------
# Integration tests via execute_tool pipeline
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_handler_runs_once_on_idempotent_retries() -> None:
    global call_count
    call_count = 0
    ctx = _ctx("idem-key-1")
    r1 = await counting_tool.execute({}, ctx)
    r2 = await counting_tool.execute({}, ctx)
    assert not r1.is_error
    assert r1.result == r2.result
    assert call_count == 1  # handler invoked exactly once


@pytest.mark.anyio
async def test_no_idempotency_key_executes_every_time() -> None:
    global call_count
    call_count = 0
    ctx = _ctx(idempotency_key=None)
    await counting_tool.execute({}, ctx)
    await counting_tool.execute({}, ctx)
    assert call_count == 2


@pytest.mark.anyio
async def test_failure_result_is_cached() -> None:
    ctx = _ctx("fail-key")
    r1 = await failing_tool.execute({}, ctx)
    r2 = await failing_tool.execute({}, ctx)
    assert r1.is_error
    assert r1 == r2


@pytest.mark.anyio
async def test_input_validation_failure_not_cached() -> None:
    """Input-validation errors are caller-side; must not be cached."""
    global call_count
    call_count = 0

    @meridian_tool(
        input_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        }
    )
    async def typed_tool(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        global call_count
        call_count += 1
        return {"n": args["n"]}

    ctx = _ctx("val-key")
    bad = await typed_tool.execute({"n": "not-an-int"}, ctx)  # validation fails
    assert bad.is_error
    assert bad.error is not None
    assert "validation" in bad.error.code

    # Now send a valid payload — handler must execute (key was NOT cached)
    good = await typed_tool.execute({"n": 7}, ctx)
    assert not good.is_error
    assert call_count == 1


@pytest.mark.anyio
async def test_idempotency_key_appears_in_audit_log_on_failure(tmp_path: Path) -> None:
    log = tmp_path / "audit.ndjson"

    @meridian_tool(
        input_schema={"type": "object"},
        audit_log_path=str(log),
    )
    async def always_raises(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        raise ValueError("oops")

    ctx = _ctx("audit-key-99")
    await always_raises.execute({}, ctx)

    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["idempotency_key"] == "audit-key-99"
    assert records[0]["type"] == "tool.execution_failed"
