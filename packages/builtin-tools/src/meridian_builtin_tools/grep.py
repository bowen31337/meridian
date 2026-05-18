"""grep — System built-in tool for searching workspace files with ripgrep.

Runs ``rg --json`` against the session workspace root, returning matches
with file path, line number, matched line text, and surrounding context lines.

Search options
--------------
* **pattern** — Rust regex (default) or literal string (``fixed_strings=true``).
* **glob** — Restricts files searched via a glob pattern (e.g. ``**/*.py``).
* **context_lines** — Lines of surrounding context returned with each match.
* **case_insensitive** — Case-fold the search.
* **max_results** — Hard cap on total matches returned; ``truncated`` flag is
  set when the result set was trimmed.

Capability
-----------
Requires ``fs.read[glob]``.

Workspace confinement
----------------------
The search root is always ``ctx.workspace``; callers cannot escape the session
workspace by supplying absolute paths.

Error handling
--------------
If ripgrep is not installed, times out, or exits with an error code the tool
returns ``ToolResult(is_error=True)``; the SDK execution pipeline writes the
failure to the audit log (Architecture §22.4).
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
from typing import Any

from meridian_sdk_tool import ToolContext, meridian_tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CONTEXT_LINES = 2
_MAX_CONTEXT_LINES = 10
_DEFAULT_MAX_RESULTS = 50
_MAX_RESULTS_LIMIT = 200
_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# JSON Schema for tool I/O
# ---------------------------------------------------------------------------

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["pattern"],
    "properties": {
        "pattern": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Search pattern. Supports full Rust regex syntax by default; "
                "use fixed_strings=true for a literal string search."
            ),
        },
        "glob": {
            "type": "string",
            "description": (
                "Glob pattern to restrict which files are searched "
                "(e.g. '**/*.py', '*.md'). Supports *, **, and ?."
            ),
        },
        "context_lines": {
            "type": "integer",
            "minimum": 0,
            "maximum": _MAX_CONTEXT_LINES,
            "description": (
                f"Number of surrounding context lines to include with each match "
                f"(default {_DEFAULT_CONTEXT_LINES}, max {_MAX_CONTEXT_LINES})."
            ),
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": _MAX_RESULTS_LIMIT,
            "description": (
                f"Maximum number of matches to return "
                f"(default {_DEFAULT_MAX_RESULTS}, max {_MAX_RESULTS_LIMIT}). "
                "Set 'truncated' to true in output when the limit is reached."
            ),
        },
        "fixed_strings": {
            "type": "boolean",
            "description": (
                "Treat pattern as a literal string instead of a regex (default false)."
            ),
        },
        "case_insensitive": {
            "type": "boolean",
            "description": "Case-insensitive search (default false).",
        },
    },
    "additionalProperties": False,
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["matches", "total", "pattern", "glob", "truncated"],
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "file_path",
                    "line_number",
                    "line",
                    "context_before",
                    "context_after",
                ],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    },
                    "line_number": {
                        "type": "integer",
                        "description": "1-based line number of the match.",
                    },
                    "line": {
                        "type": "string",
                        "description": "Content of the matching line (trailing newline stripped).",
                    },
                    "context_before": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lines immediately before the match.",
                    },
                    "context_after": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lines immediately after the match.",
                    },
                },
            },
        },
        "total": {
            "type": "integer",
            "description": "Number of matches returned.",
        },
        "pattern": {
            "type": "string",
            "description": "The search pattern as submitted.",
        },
        "glob": {
            "type": ["string", "null"],
            "description": "The file glob filter, or null if none was applied.",
        },
        "truncated": {
            "type": "boolean",
            "description": "True when the result set was trimmed to max_results.",
        },
    },
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _text_or_bytes(obj: dict[str, Any]) -> str:
    """Extract a string from ripgrep's ``{text: …}`` or ``{bytes: …}`` object."""
    if not obj:
        return ""
    if "text" in obj:
        return obj["text"]
    raw = base64.b64decode(obj.get("bytes", b""))
    return raw.decode("utf-8", errors="replace")


def _parse_rg_json(
    stdout: bytes,
    workspace: str,
    max_results: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Parse ``rg --json`` output into structured match dicts.

    Returns ``(matches, truncated)`` where *truncated* is ``True`` when the
    result list was cut short at *max_results*.
    """
    matches: list[dict[str, Any]] = []
    truncated = False

    # Decode all NDJSON lines up front.
    messages: list[dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            messages.append(json.loads(raw_line))
        except json.JSONDecodeError:
            continue

    idx = 0
    while idx < len(messages):
        msg = messages[idx]
        if msg.get("type") != "begin":
            idx += 1
            continue

        file_path = _text_or_bytes(msg.get("data", {}).get("path", {}))
        if workspace and file_path.startswith(workspace):
            file_path = file_path[len(workspace):].lstrip("/\\")

        # Collect (type, line_number, text) items until the paired "end".
        file_items: list[tuple[str, int, str]] = []
        idx += 1
        while idx < len(messages) and messages[idx].get("type") != "end":
            m = messages[idx]
            mtype = m.get("type", "")
            if mtype in ("match", "context"):
                data = m.get("data", {})
                lineno: int = data.get("line_number", 0)
                text: str = _text_or_bytes(data.get("lines", {}))
                file_items.append((mtype, lineno, text))
            idx += 1
        idx += 1  # step past "end"

        # Build one result dict per "match" item.
        for i, (itype, lineno, text) in enumerate(file_items):
            if itype != "match":
                continue
            if len(matches) >= max_results:
                truncated = True
                break

            # context_before: context items between the previous match and here.
            prev_m = -1
            for j in range(i - 1, -1, -1):
                if file_items[j][0] == "match":
                    prev_m = j
                    break
            ctx_before = [
                file_items[j][2].rstrip("\n")
                for j in range(prev_m + 1, i)
                if file_items[j][0] == "context"
            ]

            # context_after: context items between here and the next match.
            next_m = len(file_items)
            for j in range(i + 1, len(file_items)):
                if file_items[j][0] == "match":
                    next_m = j
                    break
            ctx_after = [
                file_items[j][2].rstrip("\n")
                for j in range(i + 1, next_m)
                if file_items[j][0] == "context"
            ]

            matches.append(
                {
                    "file_path": file_path,
                    "line_number": lineno,
                    "line": text.rstrip("\n"),
                    "context_before": ctx_before,
                    "context_after": ctx_after,
                }
            )

        if truncated:
            break

    return matches, truncated


async def _run_rg(
    pattern: str,
    workspace: str,
    glob: str | None,
    context_lines: int,
    fixed_strings: bool,
    case_insensitive: bool,
    max_results: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Invoke ripgrep and return ``(matches, truncated)``.

    Raises :class:`RuntimeError` on timeout, missing binary, or rg exit code 2.
    """
    rg = shutil.which("rg")
    if rg is None:
        raise RuntimeError("ripgrep (rg) is not installed or not on PATH")

    cmd: list[str] = [rg, "--json"]
    if context_lines > 0:
        cmd.extend(["-C", str(context_lines)])
    if glob:
        cmd.extend(["-g", glob])
    if fixed_strings:
        cmd.append("-F")
    if case_insensitive:
        cmd.append("-i")
    cmd.extend(["--", pattern, workspace])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise RuntimeError(
            f"ripgrep timed out after {_TIMEOUT_SECONDS}s"
        ) from exc

    # Exit code 0 = matches found, 1 = no matches (not an error), 2 = error.
    if proc.returncode not in (0, 1):
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ripgrep error (exit {proc.returncode}): {stderr_text}")

    return _parse_rg_json(stdout, workspace, max_results)


def _record_invocation(pattern: str, glob: str | None, match_count: int) -> None:
    """Attach a ``grep.invocation`` event to the active OTel span.

    Degrades gracefully when opentelemetry-api is not installed or no span is
    active in the current context.
    """
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]

        span = trace.get_current_span()
        attrs: dict[str, str | int] = {
            "grep.pattern_len": len(pattern),
            "grep.match_count": match_count,
        }
        if glob is not None:
            attrs["grep.glob"] = glob
        span.add_event("grep.invocation", attrs)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@meridian_tool(
    name="grep",
    description=(
        "Search workspace files using ripgrep. "
        "Returns matches with file path, line number, matched line, and surrounding context. "
        "Supports full Rust regex syntax by default; set fixed_strings=true for literal search. "
        "Use 'glob' to restrict the search to files matching a pattern (e.g. '**/*.py'). "
        "Results are capped at max_results; check 'truncated' to know if the set was trimmed. "
        "Requires the fs.read[glob] capability."
    ),
    input_schema=_INPUT_SCHEMA,
    output_schema=_OUTPUT_SCHEMA,
    capabilities=["fs.read[glob]"],
)
async def grep_tool(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    pattern: str = args["pattern"]
    glob: str | None = args.get("glob")
    context_lines: int = int(args.get("context_lines", _DEFAULT_CONTEXT_LINES))
    max_results: int = min(
        int(args.get("max_results", _DEFAULT_MAX_RESULTS)), _MAX_RESULTS_LIMIT
    )
    fixed_strings: bool = bool(args.get("fixed_strings", False))
    case_insensitive: bool = bool(args.get("case_insensitive", False))

    matches, truncated = await _run_rg(
        pattern=pattern,
        workspace=ctx.workspace,
        glob=glob,
        context_lines=context_lines,
        fixed_strings=fixed_strings,
        case_insensitive=case_insensitive,
        max_results=max_results,
    )

    _record_invocation(pattern, glob, len(matches))

    return {
        "matches": matches,
        "total": len(matches),
        "pattern": pattern,
        "glob": glob,
        "truncated": truncated,
    }
