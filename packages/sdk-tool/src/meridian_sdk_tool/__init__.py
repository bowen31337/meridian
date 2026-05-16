"""Meridian Tool Plugin SDK.

Provides everything a tool author needs to define, validate, and register
tools with the Meridian agent platform.

In-process Python tools::

    from meridian_sdk_tool import meridian_tool, ToolContext
    from pydantic import BaseModel

    class CountArgs(BaseModel):
        text: str

    @meridian_tool(
        description="Count words in a string",
        capabilities=["fs.read[/workspace/**]"],
    )
    async def word_count(args: CountArgs, ctx: ToolContext) -> dict:
        return {"count": len(args.text.split())}

    # Execute
    result = await word_count.execute({"text": "hello world"}, ctx)
    assert result.result == {"count": 2}

Out-of-process subprocess tools::

    from meridian_sdk_tool import subprocess_tool

    grep_tool = subprocess_tool(
        name="grep",
        description="Search files for a pattern",
        path="/usr/bin/grep_wrapper.py",
        input_schema={
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern", "path"],
        },
        capabilities=["fs.read[/workspace/**]"],
    )

Out-of-process HTTP tools::

    from meridian_sdk_tool import http_tool

    search_tool = http_tool(
        name="web_search",
        description="Search the web",
        url="http://localhost:8080/search",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        capabilities=["net.fetch[*]"],
    )

MCP tools (dispatched by the Sandbox, not this SDK)::

    from meridian_sdk_tool import mcp_tool

    fs_read_def = mcp_tool(
        name="read_file",
        description="Read a file from the workspace",
        server_url="http://localhost:3000",
        capabilities=["fs.read[/workspace/**]"],
    )

Subprocess tool servers (the other end of the wire)::

    # my_tool_server.py
    from meridian_sdk_tool.subprocess_server import run_subprocess_tool

    def handle(args: dict, ctx: dict) -> dict:
        return {"echo": args}

    if __name__ == "__main__":
        run_subprocess_tool(handle)
"""

from __future__ import annotations

from ._decorator import MeridianTool, meridian_tool
from ._types import (
    Capability,
    ContainerHandler,
    HttpHandler,
    InProcessHandler,
    McpHandler,
    SubprocessHandler,
    ToolContext,
    ToolDefinition,
    ToolError,
    ToolHandler,
    ToolResult,
)
from .http_tool import HttpTool, http_tool
from .mcp_tool import mcp_tool
from .subprocess_tool import SubprocessTool, subprocess_tool

__all__ = [
    # Decorator
    "meridian_tool",
    "MeridianTool",
    # Types
    "Capability",
    "ToolContext",
    "ToolDefinition",
    "ToolError",
    "ToolHandler",
    "ToolResult",
    # Handler kinds
    "InProcessHandler",
    "SubprocessHandler",
    "McpHandler",
    "HttpHandler",
    "ContainerHandler",
    # Tool builders
    "subprocess_tool",
    "SubprocessTool",
    "http_tool",
    "HttpTool",
    "mcp_tool",
]
