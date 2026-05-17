"""Tests for the spawn built-in tool."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest

from meridian_builtin_tools.spawn import _INPUT_SCHEMA, _OUTPUT_SCHEMA, spawn_tool
from meridian_sdk_tool import ToolContext

_CTX = ToolContext(workspace="/workspace", session_id="sess_spawn_test")
_CTX2 = ToolContext(workspace="/workspace", session_id="sess_spawn_other")


def _args(
    *,
    agent_id: str = "code-reviewer",
    parent_capabilities: list[str] | None = None,
    child_capabilities: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "parent_capabilities": parent_capabilities
        if parent_capabilities is not None
        else ["exec.shell", "fs.read", "net.listen"],
        "child_capabilities": child_capabilities
        if child_capabilities is not None
        else ["exec.shell"],
    }


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_returns_no_error_on_valid_subset() -> None:
    result = await spawn_tool.execute(_args(), _CTX)
    assert not result.is_error


@pytest.mark.anyio
async def test_response_has_child_session_id() -> None:
    result = await spawn_tool.execute(_args(), _CTX)
    assert not result.is_error
    body = result.result
    assert "child_session_id" in body
    assert isinstance(body["child_session_id"], str)
    assert len(body["child_session_id"]) > 0


@pytest.mark.anyio
async def test_child_session_id_is_uuid() -> None:
    result = await spawn_tool.execute(_args(), _CTX)
    cid = result.result["child_session_id"]
    assert re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", cid
    ), f"Not a UUID: {cid!r}"


@pytest.mark.anyio
async def test_parent_session_id_comes_from_ctx() -> None:
    result = await spawn_tool.execute(_args(), _CTX)
    assert result.result["parent_session_id"] == "sess_spawn_test"


@pytest.mark.anyio
async def test_status_is_spawned() -> None:
    result = await spawn_tool.execute(_args(), _CTX)
    assert result.result["status"] == "spawned"


@pytest.mark.anyio
async def test_capabilities_reflect_child_set() -> None:
    result = await spawn_tool.execute(
        _args(
            parent_capabilities=["exec.shell", "fs.read"],
            child_capabilities=["exec.shell", "fs.read"],
        ),
        _CTX,
    )
    assert not result.is_error
    assert sorted(result.result["capabilities"]) == ["exec.shell", "fs.read"]


@pytest.mark.anyio
async def test_capabilities_are_sorted() -> None:
    result = await spawn_tool.execute(
        _args(
            parent_capabilities=["fs.read", "exec.shell"],
            child_capabilities=["fs.read", "exec.shell"],
        ),
        _CTX,
    )
    caps = result.result["capabilities"]
    assert caps == sorted(caps)


@pytest.mark.anyio
async def test_empty_child_capabilities_always_valid() -> None:
    result = await spawn_tool.execute(
        _args(child_capabilities=[]),
        _CTX,
    )
    assert not result.is_error
    assert result.result["capabilities"] == []


@pytest.mark.anyio
async def test_equal_capabilities_is_valid() -> None:
    caps = ["exec.shell", "fs.read"]
    result = await spawn_tool.execute(
        _args(parent_capabilities=caps, child_capabilities=caps),
        _CTX,
    )
    assert not result.is_error


@pytest.mark.anyio
async def test_parameterized_child_under_unrestricted_parent() -> None:
    result = await spawn_tool.execute(
        _args(
            parent_capabilities=["fs.read"],
            child_capabilities=["fs.read[/workspace]"],
        ),
        _CTX,
    )
    assert not result.is_error


@pytest.mark.anyio
async def test_child_session_ids_are_unique() -> None:
    r1 = await spawn_tool.execute(_args(), _CTX)
    r2 = await spawn_tool.execute(_args(), _CTX)
    assert r1.result["child_session_id"] != r2.result["child_session_id"]


@pytest.mark.anyio
async def test_child_session_ids_unique_across_sessions() -> None:
    r1 = await spawn_tool.execute(_args(), _CTX)
    r2 = await spawn_tool.execute(_args(), _CTX2)
    assert r1.result["child_session_id"] != r2.result["child_session_id"]


# ---------------------------------------------------------------------------
# Escalation denial
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_escalation_returns_is_error() -> None:
    result = await spawn_tool.execute(
        _args(
            parent_capabilities=["exec.shell"],
            child_capabilities=["exec.shell", "exec.sudo"],
        ),
        _CTX,
    )
    assert result.is_error


@pytest.mark.anyio
async def test_escalation_error_mentions_missing_cap() -> None:
    result = await spawn_tool.execute(
        _args(
            parent_capabilities=["exec.shell"],
            child_capabilities=["exec.sudo"],
        ),
        _CTX,
    )
    assert result.is_error
    assert result.error is not None
    assert "exec.sudo" in result.error.message


@pytest.mark.anyio
async def test_fully_disjoint_child_rejected() -> None:
    result = await spawn_tool.execute(
        _args(
            parent_capabilities=["exec.shell"],
            child_capabilities=["net.listen"],
        ),
        _CTX,
    )
    assert result.is_error


@pytest.mark.anyio
async def test_scoped_parent_cannot_cover_unscoped_child() -> None:
    result = await spawn_tool.execute(
        _args(
            parent_capabilities=["fs.read[/workspace]"],
            child_capabilities=["fs.read"],
        ),
        _CTX,
    )
    assert result.is_error


@pytest.mark.anyio
async def test_escalation_error_mentions_agent_id() -> None:
    result = await spawn_tool.execute(
        _args(
            agent_id="my-agent",
            parent_capabilities=["exec.shell"],
            child_capabilities=["exec.sudo"],
        ),
        _CTX,
    )
    assert result.is_error
    assert result.error is not None
    assert "my-agent" in result.error.message


# ---------------------------------------------------------------------------
# Invalid capability strings
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_invalid_parent_cap_string_returns_is_error() -> None:
    result = await spawn_tool.execute(
        _args(parent_capabilities=["INVALID!!"]),
        _CTX,
    )
    assert result.is_error


@pytest.mark.anyio
async def test_invalid_child_cap_string_returns_is_error() -> None:
    result = await spawn_tool.execute(
        _args(child_capabilities=["INVALID!!"]),
        _CTX,
    )
    assert result.is_error


# ---------------------------------------------------------------------------
# Input schema validation (pre-dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_missing_agent_id_returns_is_error() -> None:
    result = await spawn_tool.execute(
        {
            "parent_capabilities": ["exec.shell"],
            "child_capabilities": ["exec.shell"],
        },
        _CTX,
    )
    assert result.is_error


@pytest.mark.anyio
async def test_missing_child_capabilities_returns_is_error() -> None:
    result = await spawn_tool.execute(
        {
            "agent_id": "my-agent",
            "parent_capabilities": ["exec.shell"],
        },
        _CTX,
    )
    assert result.is_error


@pytest.mark.anyio
async def test_missing_parent_capabilities_returns_is_error() -> None:
    result = await spawn_tool.execute(
        {
            "agent_id": "my-agent",
            "child_capabilities": ["exec.shell"],
        },
        _CTX,
    )
    assert result.is_error


@pytest.mark.anyio
async def test_extra_field_returns_is_error() -> None:
    result = await spawn_tool.execute(
        {**_args(), "unexpected": True},
        _CTX,
    )
    assert result.is_error


# ---------------------------------------------------------------------------
# Audit log — failure path writes an entry
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_escalation_writes_audit_log() -> None:
    with tempfile.NamedTemporaryFile(suffix=".ndjson", delete=False) as f:
        audit_path = f.name

    from meridian_sdk_tool import meridian_tool

    @meridian_tool(
        name="spawn",
        input_schema=_INPUT_SCHEMA,
        output_schema=_OUTPUT_SCHEMA,
        audit_log_path=audit_path,
    )
    async def _tool_with_audit(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        from sdk_capabilities import CapabilityParseError, missing, parse_set

        agent_id: str = args["agent_id"]
        try:
            parent_caps = parse_set(args["parent_capabilities"])
            child_caps = parse_set(args["child_capabilities"])
        except CapabilityParseError as exc:
            raise ValueError(f"Invalid capability string: {exc}") from exc

        escalating = missing(child_caps, parent_caps)
        if escalating:
            escalating_strs = sorted(str(c) for c in escalating)
            raise ValueError(
                f"Capability escalation denied for agent {agent_id!r}: "
                f"caps not held by parent: {', '.join(escalating_strs)}"
            )

        import uuid as _uuid

        return {
            "child_session_id": str(_uuid.uuid4()),
            "parent_session_id": ctx.session_id,
            "capabilities": sorted(str(c) for c in child_caps),
            "status": "spawned",
        }

    result = await _tool_with_audit.execute(
        _args(
            parent_capabilities=["exec.shell"],
            child_capabilities=["exec.sudo"],
        ),
        _CTX,
    )
    assert result.is_error

    lines = Path(audit_path).read_text().strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert "spawn" in entry.get("tool_name", "")
    assert "error" in entry
