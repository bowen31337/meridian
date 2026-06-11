"""Agent tool executor — §13.4 Contract 2 backing for the OAuth MCP bridge.

Dispatches an Agent's granted built-in tools (exec / read / write / grep / …),
cap-gated and workspace-confined:

  * Capability gating is at the action-family level (``namespace.name`` —
    e.g. ``fs.read``, ``exec.shell``, ``net.fetch``). MeridianTool.execute does
    NOT enforce capabilities, so this layer does.
  * Path confinement is enforced by the tools themselves via their workspace
    jail (``ToolContext.workspace``); read/write reject paths escaping it.

This object is what the MCP-bridge stdio server proxies ``tools/call`` into, so
every tool the inner CLI loop runs is checked against the Agent's grants and
run inside the Agent's Environment workspace.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core_errors import AuditLog, AuditLogEntry, NoopAuditLog
from meridian_builtin_tools import ALL_TOOLS
from meridian_sdk_tool import ToolContext
from sdk_capabilities import parse as parse_capability

_BUILTIN = {t.definition.name: t for t in ALL_TOOLS}

_GLOB_CHARS = "*?["


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _root_from_param(param: str) -> str | None:
    """Derive an absolute directory root from a capability path pattern.

    ``/Users/bob/dev/**`` -> ``/Users/bob/dev``. Returns None for relative or
    empty params (the workspace already covers the relative case).
    """
    if not param or not param.startswith("/"):
        return None
    cut = len(param)
    for i, ch in enumerate(param):
        if ch in _GLOB_CHARS:
            cut = i
            break
    prefix = param[:cut]
    if not prefix.endswith("/"):
        # pattern cut mid-segment (e.g. /a/b*) -> the segment's parent dir
        prefix = str(Path(prefix).parent)
    root = prefix.rstrip("/")
    return root or "/"


def _families(caps: list[str]) -> set[tuple[str, str]]:
    """Reduce capability strings to their (namespace, name) action families."""
    families: set[tuple[str, str]] = set()
    for cap in caps:
        try:
            parsed = parse_capability(cap)
        except Exception:  # noqa: BLE001 - skip unparseable grants
            continue
        families.add((parsed.namespace, parsed.name))
    return families


class AgentToolExecutor:
    """Cap-gated, workspace-confined dispatch over an Agent's built-in tools."""

    def __init__(
        self,
        *,
        workspace: str,
        tool_names: list[str],
        granted_capabilities: list[str],
        session_id: str = "agent-tools",
        audit_log: AuditLog | None = None,
    ) -> None:
        self._workspace = workspace
        self._granted_caps = list(granted_capabilities)
        self._granted = _families(granted_capabilities)
        self._session_id = session_id
        self._audit = audit_log or NoopAuditLog()
        self._tools = {name: _BUILTIN[name] for name in tool_names if name in _BUILTIN}

    def tool_names(self) -> list[str]:
        return list(self._tools)

    def _required_families(self, tool: Any) -> set[tuple[str, str]]:
        return _families(list(tool.definition.capabilities))

    def _allowed_roots(self, required: set[tuple[str, str]]) -> list[str]:
        """Absolute roots from granted caps whose family the tool requires.

        Lets a tool reach paths the agent was explicitly granted (e.g.
        ``fs.read[/Users/bob/dev/**]``) beyond its workspace, while the path
        confinement in the tools still rejects anything outside these roots.
        """
        roots: list[str] = []
        for cap in self._granted_caps:
            try:
                parsed = parse_capability(cap)
            except Exception:  # noqa: BLE001 - skip unparseable grants
                continue
            if (parsed.namespace, parsed.name) in required and parsed.param:
                root = _root_from_param(parsed.param)
                if root and root not in roots:
                    roots.append(root)
        return roots

    async def execute(self, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Run a granted tool; returns {"is_error": bool, "content": <payload>}."""
        tool = self._tools.get(name)
        if tool is None:
            return {"is_error": True, "content": f"tool {name!r} is not granted to this agent"}

        required = self._required_families(tool)
        if not required <= self._granted:
            missing = sorted(f"{ns}.{nm}" for ns, nm in (required - self._granted))
            self._audit.write(
                AuditLogEntry(
                    level="error",
                    event="agent.tool.capability_denied",
                    code="capability_denied",
                    timestamp=_now(),
                    detail={"tool": name, "missing": missing},
                )
            )
            return {
                "is_error": True,
                "content": f"capability denied for {name!r}: missing {missing}",
            }

        ctx = ToolContext(
            workspace=self._workspace,
            allowed_roots=self._allowed_roots(required),
            session_id=self._session_id,
        )
        result = await tool.execute(tool_input, ctx)
        is_error = bool(getattr(result, "is_error", False))
        if is_error:
            payload: Any = getattr(result, "error_message", None) or getattr(result, "content", "")
        else:
            payload = getattr(result, "result", getattr(result, "content", ""))
        return {"is_error": is_error, "content": payload}
