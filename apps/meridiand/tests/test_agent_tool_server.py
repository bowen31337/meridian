"""
Tests for the MCP stdio server (_agent_tool_server) backing the OAuth tool bridge.

Covers: JSON-RPC dispatch (initialize / initialized / tools/list / tools/call /
unknown method / notifications), the serve() read loop, build_executor_and_defs
record loading (with and without an Environment), and the main()/_amain entry.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from meridiand import _agent_tool_server as ats
from meridiand._agent_tool_server import (
    McpToolServer,
    _err,
    _ok,
    build_executor_and_defs,
    main,
)


def _seed_agent(
    root: Path,
    *,
    agent_id: str = "agent_t",
    tools: list[str] | None = None,
    workspace: Path | None = None,
    with_env: bool = True,
) -> str:
    ws = workspace or root
    (root / "agents").mkdir(parents=True, exist_ok=True)
    version: dict[str, Any] = {
        "tools": [{"name": t} for t in (tools or ["read", "write"])],
        "capabilities": [f"fs.read[{ws}/**]", f"fs.write[{ws}/**]"],
    }
    record: dict[str, Any] = {"id": agent_id, "version": version}
    if with_env:
        env_id = "env_t"
        (root / "environments").mkdir(parents=True, exist_ok=True)
        (root / "environments" / f"{env_id}.json").write_text(
            json.dumps({"id": env_id, "workspace_path": str(ws)})
        )
        record["default_environment_id"] = env_id
    (root / "agents" / f"{agent_id}.json").write_text(json.dumps(record))
    return agent_id


def _server(root: Path) -> McpToolServer:
    executor, defs = build_executor_and_defs("agent_t", str(root))
    return McpToolServer(executor, defs)


class _FakeWriter:
    def __init__(self) -> None:
        self.buf = b""

    def write(self, data: bytes) -> None:
        self.buf += data

    async def drain(self) -> None:
        pass


async def _serve_messages(server: McpToolServer, *raw: bytes) -> list[dict[str, Any]]:
    reader = asyncio.StreamReader()
    for line in raw:
        reader.feed_data(line + b"\n")
    reader.feed_eof()
    writer = _FakeWriter()
    await server.serve(reader, writer)  # type: ignore[arg-type]
    return [json.loads(x) for x in writer.buf.decode().splitlines() if x]


class TestHelpers:
    def test_ok_and_err(self) -> None:
        assert _ok(1, {"x": 1}) == {"jsonrpc": "2.0", "id": 1, "result": {"x": 1}}
        e = _err(2, -32601, "nope")
        assert e["error"] == {"code": -32601, "message": "nope"}


class TestBuildExecutor:
    def test_loads_tools_and_workspace(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        _seed_agent(tmp_path, tools=["read", "write"], workspace=ws)
        executor, defs = build_executor_and_defs("agent_t", str(tmp_path))
        assert executor.tool_names() == ["read", "write"]
        names = {d["name"] for d in defs}
        assert names == {"read", "write"}
        assert all("inputSchema" in d for d in defs)

    def test_without_environment_falls_back_to_root(self, tmp_path: Path) -> None:
        _seed_agent(tmp_path, tools=["read"], with_env=False)
        executor, _ = build_executor_and_defs("agent_t", str(tmp_path))
        assert executor.tool_names() == ["read"]


class TestDispatch:
    async def test_initialize(self, tmp_path: Path) -> None:
        _seed_agent(tmp_path)
        out = await _serve_messages(
            _server(tmp_path), b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'
        )
        assert out[0]["result"]["serverInfo"]["name"] == "meridian"

    async def test_initialized_notification_no_response(self, tmp_path: Path) -> None:
        _seed_agent(tmp_path)
        out = await _serve_messages(
            _server(tmp_path), b'{"jsonrpc":"2.0","method":"notifications/initialized"}'
        )
        assert out == []

    async def test_tools_list(self, tmp_path: Path) -> None:
        _seed_agent(tmp_path, tools=["read", "write"])
        out = await _serve_messages(
            _server(tmp_path), b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
        )
        assert {t["name"] for t in out[0]["result"]["tools"]} == {"read", "write"}

    async def test_tools_call_success(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        _seed_agent(tmp_path, tools=["read", "write"], workspace=ws)
        call = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "write", "arguments": {"path": "f.txt", "content": "hi"}},
        }
        out = await _serve_messages(_server(tmp_path), json.dumps(call).encode())
        assert out[0]["result"]["isError"] is False
        assert (ws / "f.txt").read_text() == "hi"

    async def test_unknown_method_with_id_errors(self, tmp_path: Path) -> None:
        _seed_agent(tmp_path)
        out = await _serve_messages(_server(tmp_path), b'{"jsonrpc":"2.0","id":9,"method":"bogus"}')
        assert out[0]["error"]["code"] == -32601

    async def test_unknown_notification_no_response(self, tmp_path: Path) -> None:
        _seed_agent(tmp_path)
        out = await _serve_messages(_server(tmp_path), b'{"jsonrpc":"2.0","method":"bogus"}')
        assert out == []

    async def test_blank_and_invalid_lines_skipped(self, tmp_path: Path) -> None:
        _seed_agent(tmp_path)
        out = await _serve_messages(
            _server(tmp_path),
            b"",
            b"not json",
            b'{"jsonrpc":"2.0","id":1,"method":"initialize"}',
        )
        assert len(out) == 1 and out[0]["id"] == 1


class TestMainEntry:
    async def test_amain_serves_until_eof(self, tmp_path: Path, monkeypatch: Any) -> None:
        _seed_agent(tmp_path)
        reader = asyncio.StreamReader()
        reader.feed_eof()
        writer = _FakeWriter()

        async def _fake_streams() -> tuple[Any, Any]:
            return reader, writer

        monkeypatch.setattr(ats, "_stdio_streams", _fake_streams)
        await ats._amain("agent_t", str(tmp_path))

    def test_main_invokes_asyncio_run(self, monkeypatch: Any) -> None:
        called: dict[str, Any] = {}

        def _fake_run(coro: Any) -> None:
            called["ran"] = True
            coro.close()

        monkeypatch.setattr(ats.asyncio, "run", _fake_run)
        main(["--agent-id", "a", "--storage-root", "/r"])
        assert called["ran"] is True

    async def test_stdio_streams_wires_pipes(self, monkeypatch: Any) -> None:
        loop = asyncio.get_running_loop()

        async def _crp(factory: Any, pipe: Any) -> tuple[Any, Any]:
            factory()  # exercise the StreamReaderProtocol factory
            return object(), object()

        async def _cwp(factory: Any, pipe: Any) -> tuple[Any, Any]:
            return object(), object()

        monkeypatch.setattr(loop, "connect_read_pipe", _crp)
        monkeypatch.setattr(loop, "connect_write_pipe", _cwp)
        reader, writer = await ats._stdio_streams()
        assert reader is not None and writer is not None
