"""read — System built-in tool for reading files from the session workspace.

Reads a file at a path within the session workspace and returns its content
as text or base64.  Text decoding uses ``utf-8`` by default; pass any valid
Python codec name (e.g. ``latin-1``) to override.  Specify
``encoding='base64'`` to receive raw bytes as a base64-encoded ASCII string,
which is required for binary files.

Capability
-----------
Requires ``fs.read[glob]``.

Workspace confinement
----------------------
Paths are resolved relative to ``ctx.workspace``.  The tool rejects any path
that resolves outside the workspace root, including paths containing ``..``
segments and symlinks whose resolved target escapes the jail.

Error handling
--------------
All file-system and confinement failures surface as
``ToolResult(is_error=True)``; the SDK execution pipeline writes the failure
to the audit log (Architecture §22.4).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from meridian_sdk_tool import ToolContext, meridian_tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB hard cap

# ---------------------------------------------------------------------------
# JSON Schema for tool I/O
# ---------------------------------------------------------------------------

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["path"],
    "properties": {
        "path": {
            "type": "string",
            "minLength": 1,
            "description": (
                "File path relative to the workspace root "
                "(e.g. 'src/main.py').  Must not escape the workspace."
            ),
        },
        "encoding": {
            "type": "string",
            "description": (
                "Text encoding used to decode file bytes (default 'utf-8'). "
                "Use 'base64' to return raw bytes as a base64-encoded ASCII "
                "string, which is required for binary files."
            ),
        },
    },
    "additionalProperties": False,
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["path", "content", "size", "encoding"],
    "properties": {
        "path": {
            "type": "string",
            "description": "Path of the read file, relative to the workspace root.",
        },
        "content": {
            "type": "string",
            "description": (
                "File content as a text string, or base64-encoded bytes when "
                "encoding='base64' was requested."
            ),
        },
        "size": {
            "type": "integer",
            "description": "File size in bytes.",
        },
        "encoding": {
            "type": "string",
            "description": "Encoding used to decode the file content.",
        },
    },
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_safe(workspace: str, rel_path: str) -> Path:
    """Return the resolved absolute path for *rel_path* inside *workspace*.

    Raises :class:`ValueError` if the path resolves outside the workspace
    root, or if any existing symlink component points outside the jail.
    """
    ws = Path(workspace).resolve()

    # Explicit symlink check first: walk each component of the *unresolved*
    # path so symlink escapes surface with a specific error message.
    current = ws
    for part in Path(rel_path).parts:
        current = current / part
        if current.is_symlink():
            resolved_link = current.resolve()
            try:
                resolved_link.relative_to(ws)
            except ValueError:
                try:
                    sym_rel = str(current.relative_to(ws))
                except ValueError:
                    sym_rel = str(current)
                raise ValueError(f"Symlink {sym_rel!r} points outside workspace root") from None

    # Final confinement check: resolved path (covers dotdot and other escapes).
    target = (ws / rel_path).resolve()
    try:
        target.relative_to(ws)
    except ValueError:
        raise ValueError(f"Path {rel_path!r} resolves outside workspace root") from None

    return target


def _record_invocation(path: str, encoding: str, size: int) -> None:
    """Attach a ``read.invocation`` event to the active OTel span.

    Degrades gracefully when opentelemetry-api is not installed or no span is
    active in the current context.
    """
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]

        span = trace.get_current_span()
        span.add_event(
            "read.invocation",
            {
                "read.path": path,
                "read.encoding": encoding,
                "read.size": size,
            },
        )
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@meridian_tool(
    name="read",
    description=(
        "Read a file from the session workspace and return its content as text "
        "or base64. Use encoding='base64' for binary files. "
        "Rejects paths that escape the workspace root, including symlinks "
        "pointing outside the jail. "
        "Requires the fs.read[glob] capability."
    ),
    input_schema=_INPUT_SCHEMA,
    output_schema=_OUTPUT_SCHEMA,
    capabilities=["fs.read[glob]"],
)
async def read_tool(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    rel_path: str = args["path"]
    encoding: str = args.get("encoding", "utf-8")

    target = _resolve_safe(ctx.workspace, rel_path)

    if not target.exists():
        raise FileNotFoundError(f"File not found: {rel_path!r}")

    if not target.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {rel_path!r}")

    raw: bytes = target.read_bytes()
    size = len(raw)

    if size > _MAX_FILE_BYTES:
        raise ValueError(
            f"File size {size} exceeds maximum readable size of "
            f"{_MAX_FILE_BYTES // (1024 * 1024)} MiB"
        )

    if encoding == "base64":
        content = base64.b64encode(raw).decode("ascii")
    else:
        content = raw.decode(encoding)

    _record_invocation(rel_path, encoding, size)

    return {
        "path": rel_path,
        "content": content,
        "size": size,
        "encoding": encoding,
    }
