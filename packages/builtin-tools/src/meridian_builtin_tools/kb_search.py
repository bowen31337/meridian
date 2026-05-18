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

import hashlib
import os
import re
import sqlite3
import struct
from pathlib import Path
from typing import Any

from meridian_sdk_tool import ToolContext, meridian_tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DB_ENV = "MERIDIAN_KB_PATH"
_DB_SUBPATH = os.path.join(".meridian", "kb.sqlite")
_FTS_TABLE = "kb_chunks"
_VEC_TABLE = "kb_chunks_vec"
_DEFAULT_LIMIT = 10
_MAX_LIMIT = 50
_SCAN_MULTIPLIER = 10  # over-fetch before glob filtering to reach target limit
_EMBED_DIM = 128
_RRF_K = 60  # RRF constant; higher = less sensitive to top-rank position

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

# Vector KNN query: inner subquery pulls rowids from the vec0 virtual table,
# outer join fetches metadata from the FTS5 table.
_SQL_VEC_SEARCH = (
    "SELECT f.file_path, f.kind, f.content, f.start_line, f.end_line, "
    "f.symbol_name, f.symbol_kind, f.heading_text, f.language "
    "FROM ("
    "SELECT rowid, distance FROM kb_chunks_vec "
    "WHERE embedding MATCH ? AND k = ? ORDER BY distance"
    ") v "
    "JOIN kb_chunks f ON f.rowid = v.rowid "
    "LIMIT ?"
)


def _resolve_db_path(workspace: str) -> str:
    """Return the KB SQLite database path for *workspace*."""
    env = os.environ.get(_DB_ENV)
    if env:
        return env
    return str(Path(workspace) / _DB_SUBPATH)


def _hash_embed(text: str) -> bytes:
    """Hashing-trick 128-dim normalized float32 embedding as raw bytes.

    Tokenises *text*, maps each token to a bucket via MD5, accumulates counts,
    then L2-normalises.  Matches the backend's ``_hash_embed`` implementation
    so built-in-tool queries are comparable to backend-stored embeddings.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    vec = [0.0] * _EMBED_DIM
    for token in tokens:
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % _EMBED_DIM
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return struct.pack(f"{_EMBED_DIM}f", *(v / norm for v in vec))


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


def _run_bm25(
    con: sqlite3.Connection,
    query: str,
    fetch_limit: int,
) -> list[dict[str, Any]]:
    """Run FTS5 BM25 search; returns result dicts with positive ``score``."""
    cursor = con.execute(_SQL_SEARCH, (query, fetch_limit))
    results: list[dict[str, Any]] = []
    for row in cursor:
        # FTS5 rank is negative BM25 (more negative = more relevant).
        # Negate so callers receive a positive relevance score.
        results.append(
            {
                "file_path": row[0],
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
    return results


def _run_vector(
    con: sqlite3.Connection,
    query: str,
    fetch_limit: int,
) -> list[dict[str, Any]]:
    """Run sqlite-vec KNN search; returns [] on any error or missing dependency.

    Graceful-degrades when:
    - ``sqlite-vec`` is not installed (ImportError)
    - the ``kb_chunks_vec`` virtual table does not exist
    - extension loading is not supported by the current SQLite build
    - any SQL error occurs during the KNN query
    """
    try:
        import sqlite_vec as _sv  # type: ignore[import-not-found]
    except ImportError:
        return []

    if con.execute(_SQL_TABLE_EXISTS, (_VEC_TABLE,)).fetchone() is None:
        return []

    try:
        con.enable_load_extension(True)
        _sv.load(con)
        con.enable_load_extension(False)
    except Exception:  # noqa: BLE001
        return []

    q_embed = _hash_embed(query)
    try:
        cursor = con.execute(_SQL_VEC_SEARCH, (q_embed, fetch_limit, fetch_limit))
        results: list[dict[str, Any]] = []
        for row in cursor:
            results.append(
                {
                    "file_path": row[0],
                    "kind": row[1],
                    "content": row[2],
                    "start_line": row[3],
                    "end_line": row[4],
                    "score": 0.0,  # placeholder; _rrf_fuse assigns the real score
                    "symbol_name": row[5],
                    "symbol_kind": row[6],
                    "heading_text": row[7],
                    "language": row[8],
                }
            )
        return results
    except Exception:  # noqa: BLE001
        return []


def _rrf_fuse(
    ranked_lists: list[list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion of multiple ranked lists.

    Assigns each chunk an RRF score = Σ 1/(k + rank) over all lists it
    appears in, where rank is 1-based position.  Chunks missing from a list
    contribute 0 for that list.  The ``score`` field of each returned chunk
    is set to the fused RRF score (positive float, higher = more relevant).
    """

    def _key(c: dict[str, Any]) -> tuple[str, int, int]:
        return (c["file_path"], c["start_line"], c["end_line"])

    scores: dict[tuple[str, int, int], float] = {}
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, 1):
            key = _key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            by_key[key] = chunk

    sorted_keys = sorted(scores, key=lambda kk: scores[kk], reverse=True)
    results: list[dict[str, Any]] = []
    for kk in sorted_keys[:limit]:
        chunk = dict(by_key[kk])
        chunk["score"] = round(scores[kk], 6)
        results.append(chunk)
    return results


def _sync_search(
    db_path: str,
    query: str,
    scope: str | None,
    limit: int,
    workspace: str,
) -> list[dict[str, Any]]:
    """Run hybrid (BM25 + vector + glob) search against the SQLite KB index.

    Falls back to BM25-only when the ``kb_chunks_vec`` vector table is absent
    or ``sqlite-vec`` is not installed.  Returns an empty list when the
    database or FTS table does not yet exist.  Raises :class:`sqlite3.Error`
    on unexpected database errors (caught by the SDK execution pipeline and
    surfaced as ``is_error=True``).
    """
    if not Path(db_path).exists():
        return []

    con = sqlite3.connect(db_path, check_same_thread=False)
    try:
        if con.execute(_SQL_TABLE_EXISTS, (_FTS_TABLE,)).fetchone() is None:
            return []

        # Over-fetch so that glob filtering still delivers up to *limit* rows.
        fetch_limit = limit * _SCAN_MULTIPLIER if scope else limit

        # 1. BM25 (always available via FTS5)
        bm25_rows = _run_bm25(con, query, fetch_limit)

        # 2. Vector KNN (optional; degrades to [] when unavailable)
        vec_rows = _run_vector(con, query, fetch_limit)

        # 3. Glob scope filter applied to both candidate lists
        if scope:
            bm25_rows = [
                r for r in bm25_rows if _scope_matches(r["file_path"], scope, workspace)
            ]
            vec_rows = [
                r for r in vec_rows if _scope_matches(r["file_path"], scope, workspace)
            ]

        # 4. Hybrid RRF when vector results are present; BM25-only otherwise
        if vec_rows:
            return _rrf_fuse([bm25_rows, vec_rows], limit)

        return bm25_rows[:limit]
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
