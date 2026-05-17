"""kb_search — System built-in tool for querying the Knowledge Base hybrid index.

Queries the KB hybrid index (BM25 via SQLite FTS5 + glob scope filter) for
chunks matching *query*.  An optional *scope* glob restricts results to file
paths matching the pattern (e.g. ``src/**/*.py``).

Hybrid search strategy
-----------------------
1. **BM25** (SQLite FTS5 ``rank``) — primary relevance signal; SQLite's
   built-in BM25 ranking is used directly via the virtual ``rank`` column.
2. **Glob** — Python post-query filter applied to ``file_path`` using the
   *scope* pattern; supports ``*``, ``**``, and ``?`` wildcards.
3. **Vector** — reserved; requires a ``vec0`` virtual table in the database.
   The current release ranks by BM25 only and ignores any vector columns.

Capability
-----------
Requires ``kb.read[scope]``.  When *scope* is omitted the caller must hold an
unrestricted ``kb.read`` capability.

Database path
--------------
Resolved in order:

1. ``MERIDIAN_KB_PATH`` environment variable (absolute path to the .sqlite file).
2. ``{workspace}/.meridian/kb.sqlite`` (default relative to the session workspace).

If the database does not yet exist the tool returns an empty result list rather
than an error — the index may simply not have been built yet.

Error handling
--------------
Unexpected SQL errors (corrupt DB, missing FTS5 module, invalid query syntax)
surface as ``ToolResult(is_error=True)``; the SDK execution pipeline writes the
failure to the audit log (Architecture §22.4).
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from meridian_sdk_tool import ToolContext, meridian_tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DB_ENV = "MERIDIAN_KB_PATH"
_DB_SUBPATH = os.path.join(".meridian", "kb.sqlite")
_FTS_TABLE = "kb_chunks"
_DEFAULT_LIMIT = 10
_MAX_LIMIT = 50
_SCAN_MULTIPLIER = 10  # over-fetch before glob filtering to reach target limit

# ---------------------------------------------------------------------------
# JSON Schema for tool I/O
# ---------------------------------------------------------------------------

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["query"],
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Full-text search query.  Supports FTS5 syntax: phrase quotes, "
                "AND/OR/NOT operators, and prefix search (e.g. 'authen*')."
            ),
        },
        "scope": {
            "type": "string",
            "description": (
                "Glob pattern to restrict results to matching file paths "
                "(e.g. 'src/**/*.py' or '*.md').  Supports *, **, and ?."
            ),
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": _MAX_LIMIT,
            "description": (
                f"Maximum number of results to return "
                f"(default {_DEFAULT_LIMIT}, max {_MAX_LIMIT})."
            ),
        },
    },
    "additionalProperties": False,
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["results", "total", "query", "scope"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "file_path",
                    "kind",
                    "content",
                    "start_line",
                    "end_line",
                    "score",
                ],
                "properties": {
                    "file_path": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["symbol", "heading", "text"],
                    },
                    "content": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "score": {"type": "number"},
                    "symbol_name": {"type": ["string", "null"]},
                    "symbol_kind": {"type": ["string", "null"]},
                    "heading_text": {"type": ["string", "null"]},
                    "language": {"type": ["string", "null"]},
                },
            },
        },
        "total": {
            "type": "integer",
            "description": "Number of results returned.",
        },
        "query": {
            "type": "string",
            "description": "The search query as submitted.",
        },
        "scope": {
            "type": ["string", "null"],
            "description": "The scope filter pattern, or null if none was applied.",
        },
    },
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SQL_SEARCH = (
    "SELECT file_path, kind, content, start_line, end_line, "
    "symbol_name, symbol_kind, heading_text, language, rank "
    "FROM kb_chunks WHERE kb_chunks MATCH ? ORDER BY rank LIMIT ?"
)

_SQL_TABLE_EXISTS = (
    "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
)


def _resolve_db_path(workspace: str) -> str:
    """Return the KB SQLite database path for *workspace*."""
    env = os.environ.get(_DB_ENV)
    if env:
        return env
    return str(Path(workspace) / _DB_SUBPATH)


def _glob_matches(path: str, scope: str) -> bool:
    """Return True if *path* matches the glob *scope* pattern.

    Unlike :func:`fnmatch.fnmatch`, ``*`` never matches ``/`` (only ``**``
    does), matching standard shell glob semantics for file paths.
    Supports ``*``, ``**`` (recursive), and ``?`` wildcards.
    """
    # Build regex: split on ** first, then within each segment convert
    # * → [^/]* and ? → [^/] so single-star doesn't cross path separators.
    parts = scope.split("**")
    regex_frags: list[str] = []
    for part in parts:
        escaped = re.escape(part)
        escaped = escaped.replace(r"\*", "[^/]*")
        escaped = escaped.replace(r"\?", "[^/]")
        regex_frags.append(escaped)
    # ** segments join with .* (matches anything including /)
    pattern = re.compile("^" + ".*".join(regex_frags) + "$")
    return bool(pattern.match(path))


def _scope_matches(file_path: str, scope: str, workspace: str) -> bool:
    """Match *file_path* against *scope*, stripping *workspace* prefix first."""
    path = file_path
    if workspace and path.startswith(workspace):
        path = path[len(workspace):].lstrip("/\\")
    return _glob_matches(path, scope)


def _sync_search(
    db_path: str,
    query: str,
    scope: str | None,
    limit: int,
    workspace: str,
) -> list[dict[str, Any]]:
    """Run BM25 + glob search against the SQLite KB index synchronously.

    Returns an empty list when the database or FTS table does not yet exist.
    Raises :class:`sqlite3.Error` on unexpected database errors (caught by the
    SDK execution pipeline and surfaced as is_error=True).
    """
    if not Path(db_path).exists():
        return []

    con = sqlite3.connect(db_path, check_same_thread=False)
    try:
        if con.execute(_SQL_TABLE_EXISTS, (_FTS_TABLE,)).fetchone() is None:
            return []

        # Over-fetch so that glob filtering still delivers up to *limit* rows.
        fetch_limit = limit * _SCAN_MULTIPLIER if scope else limit
        cursor = con.execute(_SQL_SEARCH, (query, fetch_limit))

        results: list[dict[str, Any]] = []
        for row in cursor:
            file_path: str = row[0]
            if scope and not _scope_matches(file_path, scope, workspace):
                continue
            # FTS5 rank is negative BM25 (more negative = more relevant).
            # Negate so callers receive a positive relevance score.
            results.append(
                {
                    "file_path": file_path,
                    "kind": row[1],
                    "content": row[2],
                    "start_line": row[3],
                    "end_line": row[4],
                    "score": round(-row[9], 4),
                    "symbol_name": row[5],
                    "symbol_kind": row[6],
                    "heading_text": row[7],
                    "language": row[8],
                }
            )
            if len(results) >= limit:
                break

        return results
    finally:
        con.close()


def _record_invocation(query: str, scope: str | None, result_count: int) -> None:
    """Attach a ``kb_search.invocation`` event to the active OTel span.

    Degrades gracefully when opentelemetry-api is not installed or no span is
    active in the current context.
    """
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]

        span = trace.get_current_span()
        attrs: dict[str, str | int] = {
            "kb_search.query_len": len(query),
            "kb_search.result_count": result_count,
        }
        if scope is not None:
            attrs["kb_search.scope"] = scope
        span.add_event("kb_search.invocation", attrs)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@meridian_tool(
    name="kb_search",
    description=(
        "Query the Knowledge Base hybrid index (BM25 + glob scope filter) for "
        "chunks matching the query string. "
        "Returns ranked Chunk results with file path, content, line range, and "
        "relevance score. "
        "Use 'scope' to restrict results to a file-path glob (e.g. 'src/**/*.py'). "
        "Requires the kb.read[scope] capability."
    ),
    input_schema=_INPUT_SCHEMA,
    output_schema=_OUTPUT_SCHEMA,
    capabilities=["kb.read"],
)
async def kb_search_tool(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    import asyncio

    query: str = args["query"]
    scope: str | None = args.get("scope")
    limit: int = min(int(args.get("limit", _DEFAULT_LIMIT)), _MAX_LIMIT)

    db_path = _resolve_db_path(ctx.workspace)
    results = await asyncio.to_thread(
        _sync_search, db_path, query, scope, limit, ctx.workspace
    )

    _record_invocation(query, scope, len(results))

    return {
        "results": results,
        "total": len(results),
        "query": query,
        "scope": scope,
    }
