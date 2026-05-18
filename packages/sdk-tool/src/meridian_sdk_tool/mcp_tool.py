"""MCP tool helper.

MCP servers are handled by the Meridian daemon's Sandbox dispatcher
(Architecture §11.3).  From the SDK perspective, a tool author registers
an MCP tool by creating a :class:`~meridian_sdk_tool._types.ToolDefinition`
with a :class:`~meridian_sdk_tool._types.McpHandler` — the dispatch path
goes through the MCP client that the Sandbox proxy owns, not through this
SDK at runtime.

This module provides the :func:`mcp_tool` builder so tool authors have a
consistent API across all handler kinds.  The returned ``ToolDefinition``
is registered with the Meridian agent just like any other tool; the Sandbox
will route calls to the MCP server automatically.
"""

from __future__ import annotations

from typing import Any, Literal

from ._types import Capability, McpHandler, ToolDefinition


def mcp_tool(
    name: str,
    description: str,
    server_url: str = "",
    mcp_tool_name: str | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    capabilities: list[Capability] | None = None,
    required_environment: str | None = None,
    timeout_ms: int = 30_000,
    memory_cap_mb: int | None = None,
    transport: Literal["http", "stdio"] = "http",
    command: list[str] | None = None,
) -> ToolDefinition:
    """Return a :class:`~meridian_sdk_tool._types.ToolDefinition` backed by an MCP server.

    Args:
        name: Meridian-level tool name (what the agent calls).
        description: Human-readable description for the model.
        server_url: Base URL of the MCP server (e.g. ``http://localhost:3000``).
            Required when *transport* is ``"http"``; ignored for ``"stdio"``.
        mcp_tool_name: The tool name as advertised by the MCP server.
            Defaults to *name* if omitted.
        input_schema: JSON Schema for args.  If omitted, the Sandbox fetches
            the schema from the MCP server's ``tools/list`` response.
        output_schema: JSON Schema for the result.  Optional; post-dispatch
            validation is skipped if absent.
        capabilities: Capability strings required for this tool.
        required_environment: Environment backend constraint (e.g. ``"docker"``).
        timeout_ms: Per-call timeout in milliseconds.
        memory_cap_mb: Memory cap passed to the environment backend.
        transport: ``"http"`` (default) or ``"stdio"``.  Selects the MCP
            transport used by the Sandbox dispatcher.
        command: Argv list for the MCP stdio server process (e.g.
            ``["python", "my_server.py"]``).  Required when *transport* is
            ``"stdio"``; ignored for ``"http"``.
    """
    return ToolDefinition(
        name=name,
        description=description,
        input_schema=input_schema or {},
        output_schema=output_schema,
        capabilities=capabilities or [],
        required_environment=required_environment,
        timeout_ms=timeout_ms,
        memory_cap_mb=memory_cap_mb,
        handler=McpHandler(
            server_url=server_url,
            tool_name=mcp_tool_name or name,
            transport=transport,
            command=command or [],
        ),
    )
