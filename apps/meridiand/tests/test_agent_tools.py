"""
Tests for AgentToolExecutor (_agent_tools) — cap-gated, workspace-confined
dispatch over an agent's built-in tools.

Covers: tool filtering, action-family capability parsing, allowed read/write,
ungranted tool, capability denial (audited), tool-level error results, and
workspace path confinement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meridiand._agent_tools import AgentToolExecutor, _families


class _RecordingAudit:
    def __init__(self) -> None:
        self.entries: list[Any] = []

    def write(self, entry: Any) -> None:
        self.entries.append(entry)


def _caps(ws: Path) -> list[str]:
    return ["exec.shell", f"fs.read[{ws}/**]", f"fs.write[{ws}/**]", "net.fetch[*]"]


def _executor(
    ws: Path,
    *,
    tools: tuple[str, ...] = ("exec", "read", "write", "grep"),
    caps: list[str] | None = None,
    audit: Any = None,
) -> AgentToolExecutor:
    return AgentToolExecutor(
        workspace=str(ws),
        tool_names=list(tools),
        granted_capabilities=caps if caps is not None else _caps(ws),
        audit_log=audit,
    )


class TestFamilies:
    def test_parses_namespace_name(self) -> None:
        assert _families(["fs.read[/x]", "exec.shell"]) == {("fs", "read"), ("exec", "shell")}

    def test_skips_unparseable(self) -> None:
        assert _families(["not a capability!!"]) == set()


class TestAgentToolExecutor:
    def test_tool_names_filters_unknown(self, tmp_path: Path) -> None:
        ex = _executor(tmp_path, tools=("read", "does_not_exist"))
        assert ex.tool_names() == ["read"]

    async def test_write_then_read_in_workspace(self, tmp_path: Path) -> None:
        ex = _executor(tmp_path)
        wrote = await ex.execute("write", {"path": "a.txt", "content": "hello"})
        assert wrote["is_error"] is False
        read = await ex.execute("read", {"path": "a.txt"})
        assert read["is_error"] is False
        assert "hello" in str(read["content"])

    async def test_ungranted_tool_rejected(self, tmp_path: Path) -> None:
        ex = _executor(tmp_path, tools=("read",))
        res = await ex.execute("write", {"path": "a.txt", "content": "x"})
        assert res["is_error"] is True
        assert "not granted" in res["content"]

    async def test_capability_denied_is_audited(self, tmp_path: Path) -> None:
        audit = _RecordingAudit()
        # exec is granted as a tool, but exec.shell is NOT in capabilities.
        ex = _executor(
            tmp_path, tools=("exec", "read"), caps=[f"fs.read[{tmp_path}/**]"], audit=audit
        )
        res = await ex.execute("exec", {"command": "echo hi"})
        assert res["is_error"] is True
        assert "capability denied" in res["content"]
        assert any(e.event == "agent.tool.capability_denied" for e in audit.entries)

    async def test_tool_error_result_surfaced(self, tmp_path: Path) -> None:
        ex = _executor(tmp_path)
        res = await ex.execute("read", {"path": "missing.txt"})
        assert res["is_error"] is True

    async def test_workspace_confinement(self, tmp_path: Path) -> None:
        (tmp_path.parent / "secret.txt").write_text("nope")
        ex = _executor(tmp_path)
        res = await ex.execute("read", {"path": "../secret.txt"})
        assert res["is_error"] is True
