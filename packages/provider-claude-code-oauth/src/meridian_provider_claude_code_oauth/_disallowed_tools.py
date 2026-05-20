"""Contract 1 — Claude Code built-in tool disallow list.

Architecture §13.4 Contract 1: Claude Code's own Read / Write / Bash / Edit
tools are placed in ``disallowed_tools`` on every subprocess call so the inner
loop cannot invoke them, routing all tool access through the MCP bridge
(Contract 2) and preserving Meridian's capability boundary.
"""

from __future__ import annotations

# All tool names that Claude Code exposes as built-ins.  Passed as
# ``disallowed_tools`` in every subprocess call payload so the CLI refuses to
# invoke them on behalf of the model.
ALL_CLAUDE_CODE_BUILTIN_TOOLS: frozenset[str] = frozenset({
    "Bash",
    "Edit",
    "Read",
    "Write",
})
