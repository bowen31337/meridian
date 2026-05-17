"""Meridian system built-in tools.

Pre-registered tools shipped with the platform.  Import individual tools
or use ``ALL_TOOLS`` to register them all with a Sandbox.

Example::

    from meridian_builtin_tools import canvas_op_tool, ALL_TOOLS

    # Register all built-in tools with a sandbox
    for tool in ALL_TOOLS:
        sandbox.register(tool.definition, tool)
"""

from __future__ import annotations

from .canvas import canvas_op_tool

ALL_TOOLS = [canvas_op_tool]

__all__ = ["canvas_op_tool", "ALL_TOOLS"]
