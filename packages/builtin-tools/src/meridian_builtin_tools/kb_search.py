"""kb_search — System built-in tool for querying the Knowledge Base hybrid index.

Queries the KB hybrid index (BM25 via SQLite FTS5 + vector via sqlite-vec +
glob scope filter) for chunks matching *query*.  An optional *scope* glob
restricts results to file paths matching the pattern (e.g. ``src/**/*.py``).

Hybrid search strategy
-----------------------
1. **BM25** (SQLite FTS5 ``rank``) — primary relevance signal; SQLite's
   built-in BM25 ranking is used directly via the virtual ``rank`` column.
2. **Vector** — hashing-trick dense embedding (128-dim normalized float32)
   stored in the ``kb_chunks_vec`` companion ``vec0`` virtual table.  Requires
   the ``sqlite-vec`` extension.  When the vector table is absent or
   ``sqlite-vec`` is not installed the tool degrades gracefully to BM25-only.
3. **Glob** — Python post-query filter applied to ``file_path`` using the
   *scope* pattern; supports ``*``, ``**``, and ``?`` wildcards.

When both BM25 and vector results are available they are fused via Reciprocal
Rank Fusion (RRF, k=60) and the ``score`` field reflects the combined RRF
score.  In BM25-only mode ``score`` is the negated FTS5 rank (positive float).

Capability
-----------
Requires ``kb.read[scope]``.

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

from typing import Any

from meridian_kb_indexer._reader import (
    _glob_matches,
    _hash_embed,
    _resolve_db_path,
    _rrf_fuse,
    _scope_matches,
    _sync_search,
)
from meridian_sdk_tool import ToolContext, meridian_tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LIMIT = 10
_MAX_LIMIT = 50

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

# Re-export reader helpers so existing callers (e.g. tests) can import them
# from this module without needing to know the underlying package.
__all__ = [
    "_glob_matches",
    "_hash_embed",
    "_resolve_db_path",
    "_rrf_fuse",
    "_scope_matches",
    "_sync_search",
    "_INPUT_SCHEMA",
    "_OUTPUT_SCHEMA",
    "kb_search_tool",
]


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
        "Query the Knowledge Base hybrid index (BM25 + vector + glob scope filter) "
        "for chunks matching the query string. "
        "BM25 and vector results are fused via Reciprocal Rank Fusion (RRF); "
        "degrades to BM25-only when the vector table is absent. "
        "Returns ranked Chunk results with file path, content, line range, and "
        "relevance score. "
        "Use 'scope' to restrict results to a file-path glob (e.g. 'src/**/*.py'). "
        "Requires the kb.read[scope] capability."
    ),
    input_schema=_INPUT_SCHEMA,
    output_schema=_OUTPUT_SCHEMA,
    capabilities=["kb.read[scope]"],
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
