"""spawn — System built-in tool for launching child Sessions.

Spawns a child Session with a declared capability subset. The caller must
hold the ``agent.spawn[agent_id]`` capability. Returns ``child_session_id``
immediately; the child session inherits only the declared subset of the
parent's capabilities (no upward escalation).

Capability enforcement
----------------------
``child_capabilities`` must be a subset of ``parent_capabilities``.  Any
capability the child requests that the parent does not hold is rejected with
a descriptive error message.

Error handling
--------------
Capability parse failures and escalation denials surface as
``ToolResult(is_error=True)``; the SDK execution pipeline writes the
failure to the audit log (Architecture §22.4).
"""

from __future__ import annotations

import uuid
from typing import Any

from meridian_sdk_tool import ToolContext, meridian_tool
from sdk_capabilities import CapabilityParseError, missing, parse_set

# ---------------------------------------------------------------------------
# JSON Schema for tool I/O
# ---------------------------------------------------------------------------

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["agent_id", "child_capabilities", "parent_capabilities"],
    "properties": {
        "agent_id": {
            "type": "string",
            "description": "Identifier for the child agent to spawn.",
        },
        "child_capabilities": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Capabilities to grant to the child session. "
                "Must be a subset of parent_capabilities."
            ),
        },
        "parent_capabilities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The calling session's full capability set.",
        },
    },
    "additionalProperties": False,
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["child_session_id", "parent_session_id", "capabilities", "status"],
    "properties": {
        "child_session_id": {
            "type": "string",
            "description": "Unique identifier for the newly created child session.",
        },
        "parent_session_id": {
            "type": "string",
            "description": "Session ID of the calling (parent) session.",
        },
        "capabilities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Sorted list of capability strings granted to the child.",
        },
        "status": {
            "type": "string",
            "const": "spawned",
        },
    },
}

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@meridian_tool(
    name="spawn",
    description=(
        "Launch a child Session with a declared capability subset. "
        "child_capabilities must be a subset of parent_capabilities — "
        "no upward escalation is permitted. "
        "Returns child_session_id immediately. "
        "Requires the agent.spawn[agent_id] capability."
    ),
    input_schema=_INPUT_SCHEMA,
    output_schema=_OUTPUT_SCHEMA,
    capabilities=["agent.spawn"],
)
async def spawn_tool(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
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

    child_session_id = str(uuid.uuid4())

    return {
        "child_session_id": child_session_id,
        "parent_session_id": ctx.session_id,
        "capabilities": sorted(str(c) for c in child_caps),
        "status": "spawned",
    }
