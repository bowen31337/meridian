"""canvas_op — System built-in tool for emitting Live Canvas operations.

Constructs a ``content_block`` of type ``"canvas_op"`` and returns it as the
tool result.  The daemon forwards the block onto the SSE stream; clients with
a ``WidgetRegistry`` render the appropriate widget component.

Supported widget kinds
----------------------
* ``meridian.text``       — plain-text paragraph
* ``meridian.markdown``   — markdown-formatted content
* ``meridian.form``       — simple labelled-input form
* ``meridian.code``       — syntax-highlighted code block
* ``meridian.image``      — embedded image
* ``meridian.table``      — row/column data table
* ``meridian.progress``   — progress bar

Error handling
--------------
Input validation failures (unknown ``widget_kind``, unknown ``op``) are
caught by the SDK's pre-dispatch schema check and returned as
``ToolResult(is_error=True, error.code="input_validation_failed")``.

Any unexpected runtime error surfaces as ``execution_failed`` via the SDK
execution pipeline.  Both paths write an entry to the audit log so operators
can reconstruct failures without live OTel infrastructure
(Architecture §22.4).
"""

from __future__ import annotations

import threading
import time
from typing import Any

from meridian_sdk_tool import ToolContext, meridian_tool

# ---------------------------------------------------------------------------
# Supported widget kinds (mirrored in input_schema enum)
# ---------------------------------------------------------------------------

SUPPORTED_KINDS: frozenset[str] = frozenset(
    {
        "meridian.text",
        "meridian.markdown",
        "meridian.form",
        "meridian.code",
        "meridian.image",
        "meridian.table",
        "meridian.progress",
    }
)

# ---------------------------------------------------------------------------
# Per-session monotonic sequence counter
# ---------------------------------------------------------------------------

_seq_lock = threading.Lock()
_seq_counters: dict[str, int] = {}


def _next_sequence(session_id: str) -> int:
    """Return the next sequence number for *session_id*, starting at 1."""
    with _seq_lock:
        n = _seq_counters.get(session_id, 0) + 1
        _seq_counters[session_id] = n
        return n


def _reset_sequence_counters() -> None:
    """Clear all per-session counters.  Intended for use in tests only."""
    with _seq_lock:
        _seq_counters.clear()


# ---------------------------------------------------------------------------
# JSON Schema for tool I/O
# ---------------------------------------------------------------------------

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["widget_id", "widget_kind", "op"],
    "properties": {
        "widget_id": {
            "type": "string",
            "description": "Stable identifier for the widget instance within the session.",
        },
        "widget_kind": {
            "type": "string",
            "enum": sorted(SUPPORTED_KINDS),
            "description": "Registered widget kind, e.g. 'meridian.text'.",
        },
        "op": {
            "type": "string",
            "enum": ["append", "clear", "patch", "set"],
            "description": "Canvas operation: set | patch | append | clear.",
        },
        "props": {
            "type": "object",
            "description": "Widget-specific props validated client-side by the WidgetRegistry.",
        },
    },
    "additionalProperties": False,
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["type", "canvas_op"],
    "properties": {
        "type": {"type": "string", "const": "canvas_op"},
        "canvas_op": {
            "type": "object",
            "required": [
                "op",
                "widget_id",
                "widget_kind",
                "props",
                "sequence",
                "session_id",
                "timestamp",
            ],
            "properties": {
                "op": {"type": "string"},
                "widget_id": {"type": "string"},
                "widget_kind": {"type": "string"},
                "props": {"type": "object"},
                "sequence": {"type": "integer"},
                "session_id": {"type": "string"},
                "timestamp": {"type": "string"},
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@meridian_tool(
    name="canvas_op",
    description=(
        "Emit a content_block.canvas_op that renders a widget in the Live Canvas. "
        "Supported widget kinds: meridian.text, meridian.markdown, meridian.form, "
        "meridian.code, meridian.image, meridian.table, meridian.progress. "
        "Operations: set (replace), patch (merge props), append (add item), clear (remove)."
    ),
    input_schema=_INPUT_SCHEMA,
    output_schema=_OUTPUT_SCHEMA,
)
async def canvas_op_tool(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    sequence = _next_sequence(ctx.session_id)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return {
        "type": "canvas_op",
        "canvas_op": {
            "op": args["op"],
            "widget_id": args["widget_id"],
            "widget_kind": args["widget_kind"],
            "props": args.get("props") or {},
            "sequence": sequence,
            "session_id": ctx.session_id,
            "timestamp": timestamp,
        },
    }
