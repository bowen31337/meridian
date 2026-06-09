"""MCP stdio server exposing an Agent's granted tools to the Claude Code CLI.

Spawned by the OAuth provider via ``claude --mcp-config`` (§13.4 Contract 2):
reads the Agent record by id, builds a cap-gated, workspace-confined
AgentToolExecutor, and serves MCP JSON-RPC 2.0 on stdin/stdout. Each granted
built-in tool is exposed as an MCP tool, so the inner CLI loop calls Meridian's
tools (routed through capability + workspace enforcement) instead of the CLI's
own built-ins.

Usage:
    python -m meridiand._agent_tool_server --agent-id <id> --storage-root <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

from ._agent_tools import AgentToolExecutor

_PROTOCOL_VERSION = "2024-11-05"
_SERVER_NAME = "meridian"
_SERVER_VERSION = "1.0.0"


def _ok(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _err(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class McpToolServer:
    """Minimal MCP JSON-RPC 2.0 stdio server over an AgentToolExecutor."""

    def __init__(self, executor: AgentToolExecutor, tool_defs: list[dict[str, Any]]) -> None:
        self._executor = executor
        self._tool_defs = tool_defs

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            raw = await reader.readline()
            if not raw:
                break
            line = raw.strip()
            if not line:
                continue
            try:
                msg: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = await self._dispatch(msg)
            if response is not None:
                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()

    async def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method = msg.get("method", "")
        request_id = msg.get("id")
        params: dict[str, Any] = msg.get("params") or {}

        if method == "initialize":
            return _ok(
                request_id,
                {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": _SERVER_NAME, "version": _SERVER_VERSION},
                },
            )
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return _ok(request_id, {"tools": self._tool_defs})
        if method == "tools/call":
            name: str = params.get("name", "")
            arguments: dict[str, Any] = params.get("arguments") or {}
            result = await self._executor.execute(name, arguments)
            content = result["content"]
            text = content if isinstance(content, str) else json.dumps(content)
            return _ok(
                request_id,
                {"content": [{"type": "text", "text": text}], "isError": bool(result["is_error"])},
            )
        if request_id is None:
            return None
        return _err(request_id, -32601, f"Method not found: {method}")


def build_executor_and_defs(
    agent_id: str, storage_root: str
) -> tuple[AgentToolExecutor, list[dict[str, Any]]]:
    """Load the agent record, resolve its workspace, and build the executor + MCP tool defs."""
    from meridian_builtin_tools import ALL_TOOLS

    defs_by_name = {t.definition.name: t.definition for t in ALL_TOOLS}
    root = Path(storage_root)
    agent = json.loads((root / "agents" / f"{agent_id}.json").read_text())
    version = agent.get("version") or {}
    tool_names = [t.get("name") for t in version.get("tools", []) if t.get("name")]
    capabilities = list(version.get("capabilities", []))

    env_id = agent.get("default_environment_id") or version.get("default_environment_id")
    workspace = str(root)
    if env_id:
        env_file = root / "environments" / f"{env_id}.json"
        if env_file.exists():
            workspace = json.loads(env_file.read_text()).get("workspace_path") or workspace

    executor = AgentToolExecutor(
        workspace=workspace,
        tool_names=[n for n in tool_names if n],
        granted_capabilities=capabilities,
    )
    tool_defs = [
        {
            "name": name,
            "description": defs_by_name[name].description,
            "inputSchema": defs_by_name[name].input_schema,
        }
        for name in executor.tool_names()
        if name in defs_by_name
    ]
    return executor, tool_defs


async def _stdio_streams() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    w_transport, w_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(w_transport, w_protocol, reader, loop)
    return reader, writer


async def _amain(agent_id: str, storage_root: str) -> None:
    executor, tool_defs = build_executor_and_defs(agent_id, storage_root)
    reader, writer = await _stdio_streams()
    await McpToolServer(executor, tool_defs).serve(reader, writer)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="meridiand._agent_tool_server")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--storage-root", required=True)
    args = parser.parse_args(argv)
    asyncio.run(_amain(args.agent_id, args.storage_root))


if __name__ == "__main__":
    main()
