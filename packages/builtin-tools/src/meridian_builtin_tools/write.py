"""write — System built-in tool for writing files to the session workspace.

Writes text content to a path within the session workspace.  Supports two
modes: ``create`` (fails if the target already exists) and ``overwrite``
(creates or replaces the target, which is the default).  Parent directories
are created automatically.

Capability
-----------
Requires ``fs.write[glob]``.

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

from pathlib import Path
from typing import Any

from meridian_sdk_tool import ToolContext, meridian_tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_CONTENT_BYTES = 10 * 1024 * 1024  # 10 MiB hard cap

# ---------------------------------------------------------------------------
# JSON Schema for tool I/O
# ---------------------------------------------------------------------------

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["path", "content"],
    "properties": {
        "path": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Target file path relative to the workspace root "
                "(e.g. 'src/main.py').  Must not escape the workspace."
            ),
        },
        "content": {
            "type": "string",
            "description": "Text content to write.  Encoded as UTF-8 by default.",
        },
        "mode": {
            "type": "string",
            "enum": ["create", "overwrite"],
            "description": (
                "'create' fails with an error if the file already exists. "
                "'overwrite' creates or replaces the file (default)."
            ),
        },
        "encoding": {
            "type": "string",
            "description": "Text encoding to use when writing (default 'utf-8').",
        },
    },
    "additionalProperties": False,
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["path", "bytes_written", "created"],
    "properties": {
        "path": {
            "type": "string",
            "description": "Path of the written file, relative to the workspace root.",
        },
        "bytes_written": {
            "type": "integer",
            "description": "Number of bytes written to the file.",
        },
        "created": {
            "type": "boolean",
            "description": (
                "True when the file was newly created; false when it was overwritten."
            ),
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
                raise ValueError(
                    f"Symlink {sym_rel!r} points outside workspace root"
                )

    # Final confinement check: resolved path (covers dotdot and other escapes).
    target = (ws / rel_path).resolve()
    try:
        target.relative_to(ws)
    except ValueError:
        raise ValueError(f"Path {rel_path!r} resolves outside workspace root")

    return target


def _record_invocation(
    path: str, mode: str, bytes_written: int, created: bool
) -> None:
    """Attach a ``write.invocation`` event to the active OTel span.

    Degrades gracefully when opentelemetry-api is not installed or no span is
    active in the current context.
    """
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]

        span = trace.get_current_span()
        span.add_event(
            "write.invocation",
            {
                "write.path": path,
                "write.mode": mode,
                "write.bytes_written": bytes_written,
                "write.created": created,
            },
        )
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@meridian_tool(
    name="write",
    description=(
        "Write text content to a file in the session workspace. "
        "Use mode='create' to fail if the file already exists, or "
        "mode='overwrite' (default) to create or replace the file. "
        "Parent directories are created automatically. "
        "Rejects paths that escape the workspace root, including symlinks "
        "pointing outside the jail. "
        "Requires the fs.write[glob] capability."
    ),
    input_schema=_INPUT_SCHEMA,
    output_schema=_OUTPUT_SCHEMA,
    capabilities=["fs.write[glob]"],
)
async def write_tool(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    rel_path: str = args["path"]
    content: str = args["content"]
    mode: str = args.get("mode", "overwrite")
    encoding: str = args.get("encoding", "utf-8")

    target = _resolve_safe(ctx.workspace, rel_path)

    already_exists = target.exists()
    if mode == "create" and already_exists:
        raise FileExistsError(f"File already exists: {rel_path!r}")

    # Encode before any I/O so we enforce the size cap atomically.
    raw: bytes = content.encode(encoding)
    if len(raw) > _MAX_CONTENT_BYTES:
        raise ValueError(
            f"Content exceeds maximum size of "
            f"{_MAX_CONTENT_BYTES // (1024 * 1024)} MiB"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)

    _record_invocation(rel_path, mode, len(raw), not already_exists)

    return {
        "path": rel_path,
        "bytes_written": len(raw),
        "created": not already_exists,
    }
