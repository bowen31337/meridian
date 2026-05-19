"""Tests for the kb_search built-in tool."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pytest

from meridian_builtin_tools.kb_search import (
    _INPUT_SCHEMA,
    _OUTPUT_SCHEMA,
    kb_search_tool,
)
from meridian_kb_indexer._reader import (
    _glob_matches,
    _hash_embed,
    _resolve_db_path,
    _rrf_fuse,
    _scope_matches,
)

try:
    import sqlite3 as _sqlite3_check
    import sqlite_vec as _sqlite_vec  # type: ignore[import-not-found]

    _con_check = _sqlite3_check.connect(":memory:")
    _has_load_ext = hasattr(_con_check, "enable_load_extension")
    _con_check.close()
    del _con_check, _sqlite3_check
    _SQLITE_VEC_AVAILABLE = _has_load_ext
except ImportError:
    _sqlite_vec = None  # type: ignore[assignment]
    _SQLITE_VEC_AVAILABLE = False
from meridian_sdk_tool import ToolContext

_CTX = ToolContext(workspace="/workspace", session_id="sess_kb_test")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_SAMPLE_CHUNKS: list[dict[str, Any]] = [
    {
        "file_path": "src/auth/login.py",
        "kind": "symbol",
        "content": "def authenticate(user, password): ...",
        "start_line": 10,
        "end_line": 20,
        "symbol_name": "authenticate",
        "symbol_kind": "function",
        "heading_text": None,
        "language": "python",
    },
    {
        "file_path": "src/auth/logout.py",
        "kind": "symbol",
        "content": "def logout(session): ...",
        "start_line": 5,
        "end_line": 12,
        "symbol_name": "logout",
        "symbol_kind": "function",
        "heading_text": None,
        "language": "python",
    },
    {
        "file_path": "docs/README.md",
        "kind": "heading",
        "content": "Authentication Guide",
        "start_line": 1,
        "end_line": 1,
        "symbol_name": None,
        "symbol_kind": None,
        "heading_text": "Authentication Guide",
        "language": None,
    },
    {
        "file_path": "docs/README.md",
        "kind": "text",
        "content": "This guide describes how to authenticate users in the system.",
        "start_line": 3,
        "end_line": 5,
        "symbol_name": None,
        "symbol_kind": None,
        "heading_text": None,
        "language": None,
    },
]


def _make_test_db(path: str | Path, chunks: list[dict[str, Any]] | None = None) -> None:
    """Create a SQLite FTS5 KB database with optional test chunks."""
    con = sqlite3.connect(str(path))
    con.execute(
        """CREATE VIRTUAL TABLE kb_chunks USING fts5(
            file_path UNINDEXED,
            kind UNINDEXED,
            content,
            start_line UNINDEXED,
            end_line UNINDEXED,
            symbol_name UNINDEXED,
            symbol_kind UNINDEXED,
            heading_text UNINDEXED,
            language UNINDEXED,
            tokenize='unicode61'
        )"""
    )
    for chunk in chunks or []:
        con.execute(
            "INSERT INTO kb_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chunk["file_path"],
                chunk.get("kind", "text"),
                chunk["content"],
                chunk.get("start_line", 1),
                chunk.get("end_line", 1),
                chunk.get("symbol_name"),
                chunk.get("symbol_kind"),
                chunk.get("heading_text"),
                chunk.get("language"),
            ),
        )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# _glob_matches unit tests
# ---------------------------------------------------------------------------


def test_glob_matches_exact() -> None:
    assert _glob_matches("src/auth/login.py", "src/auth/login.py")


def test_glob_matches_star_wildcard() -> None:
    assert _glob_matches("src/auth/login.py", "src/auth/*.py")


def test_glob_matches_double_star() -> None:
    assert _glob_matches("src/auth/login.py", "src/**/*.py")


def test_glob_matches_double_star_deep() -> None:
    assert _glob_matches("a/b/c/d/e.py", "a/**/*.py")


def test_glob_matches_question_mark() -> None:
    assert _glob_matches("foo.py", "fo?.py")


def test_glob_no_match() -> None:
    assert not _glob_matches("docs/README.md", "src/**/*.py")


def test_glob_star_does_not_cross_separator() -> None:
    assert not _glob_matches("src/auth/login.py", "*.py")


# ---------------------------------------------------------------------------
# _scope_matches unit tests
# ---------------------------------------------------------------------------


def test_scope_matches_strips_workspace_prefix() -> None:
    assert _scope_matches("/workspace/src/auth/login.py", "src/**/*.py", "/workspace")


def test_scope_matches_no_workspace_prefix() -> None:
    assert _scope_matches("src/auth/login.py", "src/**/*.py", "/workspace")


def test_scope_matches_no_match_after_strip() -> None:
    assert not _scope_matches("/workspace/docs/README.md", "src/**/*.py", "/workspace")


# ---------------------------------------------------------------------------
# _resolve_db_path unit tests
# ---------------------------------------------------------------------------


def test_resolve_db_path_uses_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERIDIAN_KB_PATH", "/custom/kb.sqlite")
    assert _resolve_db_path("/workspace") == "/custom/kb.sqlite"


def test_resolve_db_path_uses_workspace_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MERIDIAN_KB_PATH", raising=False)
    result = _resolve_db_path("/workspace")
    assert result == "/workspace/.meridian/kb.sqlite"


# ---------------------------------------------------------------------------
# Success path — basic query
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_returns_no_error_on_valid_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert not result.is_error


@pytest.mark.anyio
async def test_results_key_is_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert isinstance(result.result["results"], list)


@pytest.mark.anyio
async def test_query_is_echoed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert result.result["query"] == "authenticate"


@pytest.mark.anyio
async def test_scope_is_echoed_when_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute(
        {"query": "authenticate", "scope": "src/**/*.py"}, _CTX
    )
    assert result.result["scope"] == "src/**/*.py"


@pytest.mark.anyio
async def test_scope_is_null_when_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert result.result["scope"] is None


@pytest.mark.anyio
async def test_total_matches_results_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert result.result["total"] == len(result.result["results"])


@pytest.mark.anyio
async def test_matching_chunks_are_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert not result.is_error
    assert result.result["total"] > 0


@pytest.mark.anyio
async def test_result_items_have_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert not result.is_error
    for item in result.result["results"]:
        assert "file_path" in item
        assert "kind" in item
        assert "content" in item
        assert "start_line" in item
        assert "end_line" in item
        assert "score" in item


@pytest.mark.anyio
async def test_score_is_positive_float(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert not result.is_error
    assert result.result["total"] > 0
    for item in result.result["results"]:
        assert isinstance(item["score"], float)
        assert item["score"] >= 0


# ---------------------------------------------------------------------------
# Scope filter
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_scope_filter_excludes_nonmatching_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    # Scope to docs/ only — should NOT return src/auth results
    result = await kb_search_tool.execute(
        {"query": "authenticate", "scope": "docs/**"}, _CTX
    )
    assert not result.is_error
    for item in result.result["results"]:
        assert item["file_path"].startswith("docs/")


@pytest.mark.anyio
async def test_scope_filter_includes_matching_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute(
        {"query": "authenticate", "scope": "src/**/*.py"}, _CTX
    )
    assert not result.is_error
    assert result.result["total"] > 0
    for item in result.result["results"]:
        assert item["file_path"].endswith(".py")


@pytest.mark.anyio
async def test_nonmatching_scope_returns_empty_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute(
        {"query": "authenticate", "scope": "nonexistent/**"}, _CTX
    )
    assert not result.is_error
    assert result.result["total"] == 0


# ---------------------------------------------------------------------------
# Limit
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_limit_caps_number_of_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    # Insert more chunks than the requested limit
    many_chunks = [
        {
            "file_path": f"src/file{i}.py",
            "kind": "text",
            "content": f"authenticate user number {i}",
            "start_line": i,
            "end_line": i,
        }
        for i in range(20)
    ]
    _make_test_db(db, many_chunks)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate", "limit": 3}, _CTX)
    assert not result.is_error
    assert result.result["total"] <= 3


@pytest.mark.anyio
async def test_default_limit_is_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    many_chunks = [
        {
            "file_path": f"src/file{i}.py",
            "kind": "text",
            "content": f"authenticate session {i}",
            "start_line": i,
            "end_line": i,
        }
        for i in range(30)
    ]
    _make_test_db(db, many_chunks)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert not result.is_error
    assert result.result["total"] <= 10  # _DEFAULT_LIMIT


# ---------------------------------------------------------------------------
# Missing or empty DB
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_missing_db_returns_empty_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(tmp_path / "nonexistent.sqlite"))

    result = await kb_search_tool.execute({"query": "anything"}, _CTX)
    assert not result.is_error
    assert result.result["results"] == []
    assert result.result["total"] == 0


@pytest.mark.anyio
async def test_db_without_fts_table_returns_empty_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    # Create a valid SQLite DB but without the kb_chunks FTS table
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE unrelated (id INTEGER)")
    con.commit()
    con.close()
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "anything"}, _CTX)
    assert not result.is_error
    assert result.result["total"] == 0


@pytest.mark.anyio
async def test_empty_index_returns_empty_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, [])  # table exists but no rows
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert not result.is_error
    assert result.result["total"] == 0


# ---------------------------------------------------------------------------
# Input schema validation (pre-dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_missing_query_returns_is_error() -> None:
    result = await kb_search_tool.execute({}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_empty_query_string_returns_is_error() -> None:
    result = await kb_search_tool.execute({"query": ""}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_limit_zero_returns_is_error() -> None:
    result = await kb_search_tool.execute({"query": "foo", "limit": 0}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_limit_over_max_returns_is_error() -> None:
    result = await kb_search_tool.execute({"query": "foo", "limit": 51}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_extra_field_returns_is_error() -> None:
    result = await kb_search_tool.execute({"query": "foo", "unexpected": True}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_error_code_is_validation_related_on_bad_input() -> None:
    result = await kb_search_tool.execute({}, _CTX)
    assert result.is_error
    assert result.error is not None
    assert "validation" in result.error.code


# ---------------------------------------------------------------------------
# Corrupt DB → execution_failed → audit log
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_corrupt_db_returns_is_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    db.write_text("this is not a sqlite database", encoding="utf-8")
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "anything"}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_corrupt_db_writes_audit_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    db.write_text("not a database", encoding="utf-8")
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    audit_path = str(tmp_path / "audit.ndjson")

    from meridian_sdk_tool import meridian_tool

    @meridian_tool(
        name="kb_search",
        input_schema=_INPUT_SCHEMA,
        output_schema=_OUTPUT_SCHEMA,
        audit_log_path=audit_path,
    )
    async def _tool_with_audit(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        import asyncio

        from meridian_kb_indexer._reader import _resolve_db_path, _sync_search

        query: str = args["query"]
        scope: str | None = args.get("scope")
        limit: int = args.get("limit", 10)
        db_path = _resolve_db_path(ctx.workspace)
        results = await asyncio.to_thread(
            _sync_search, db_path, query, scope, limit, ctx.workspace
        )
        return {
            "results": results,
            "total": len(results),
            "query": query,
            "scope": scope,
        }

    result = await _tool_with_audit.execute({"query": "anything"}, _CTX)
    assert result.is_error

    lines = Path(audit_path).read_text().strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert "kb_search" in entry.get("tool_name", "")
    assert "error" in entry


# ---------------------------------------------------------------------------
# MERIDIAN_KB_PATH env var
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_db_path_from_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "custom_kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert not result.is_error
    assert result.result["total"] > 0


# ---------------------------------------------------------------------------
# _hash_embed unit tests
# ---------------------------------------------------------------------------


def test_hash_embed_returns_bytes() -> None:
    data = _hash_embed("hello world")
    assert isinstance(data, bytes)


def test_hash_embed_correct_length() -> None:
    import struct

    data = _hash_embed("hello world")
    # 128 float32 values × 4 bytes each
    assert len(data) == struct.calcsize("128f")


def test_hash_embed_empty_string_returns_zero_vector() -> None:
    import struct

    data = _hash_embed("")
    vec = struct.unpack("128f", data)
    assert all(v == 0.0 for v in vec)


def test_hash_embed_same_text_is_deterministic() -> None:
    assert _hash_embed("authenticate") == _hash_embed("authenticate")


def test_hash_embed_different_texts_differ() -> None:
    assert _hash_embed("authenticate") != _hash_embed("logout")


# ---------------------------------------------------------------------------
# _rrf_fuse unit tests
# ---------------------------------------------------------------------------


def _make_chunk(file_path: str, start: int = 1, end: int = 5) -> dict[str, Any]:
    return {
        "file_path": file_path,
        "kind": "text",
        "content": "content",
        "start_line": start,
        "end_line": end,
        "score": 0.0,
        "symbol_name": None,
        "symbol_kind": None,
        "heading_text": None,
        "language": None,
    }


def test_rrf_fuse_single_list_preserves_order() -> None:
    chunks = [_make_chunk(f"file{i}.py", i) for i in range(5)]
    result = _rrf_fuse([chunks], limit=5)
    assert [r["file_path"] for r in result] == [c["file_path"] for c in chunks]


def test_rrf_fuse_assigns_positive_scores() -> None:
    a = [_make_chunk("a.py", 1), _make_chunk("b.py", 2)]
    b = [_make_chunk("b.py", 2), _make_chunk("a.py", 1)]
    result = _rrf_fuse([a, b], limit=5)
    assert all(r["score"] > 0 for r in result)


def test_rrf_fuse_chunk_in_both_lists_scores_higher() -> None:
    shared = _make_chunk("shared.py", 1)
    unique_a = _make_chunk("only_a.py", 2)
    unique_b = _make_chunk("only_b.py", 3)
    a = [shared, unique_a]
    b = [shared, unique_b]
    result = _rrf_fuse([a, b], limit=5)
    shared_score = next(r["score"] for r in result if r["file_path"] == "shared.py")
    for r in result:
        if r["file_path"] != "shared.py":
            assert shared_score > r["score"]


def test_rrf_fuse_respects_limit() -> None:
    chunks = [_make_chunk(f"f{i}.py", i) for i in range(10)]
    result = _rrf_fuse([chunks], limit=3)
    assert len(result) <= 3


def test_rrf_fuse_empty_lists_returns_empty() -> None:
    assert _rrf_fuse([[], []], limit=5) == []


# ---------------------------------------------------------------------------
# Vector table absent → graceful BM25-only fallback
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_no_vec_table_falls_back_to_bm25(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FTS5-only DB (no kb_chunks_vec) must still return BM25 results."""
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert not result.is_error
    assert result.result["total"] > 0


@pytest.mark.anyio
async def test_no_vec_table_scores_are_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert not result.is_error
    for item in result.result["results"]:
        assert item["score"] >= 0


# ---------------------------------------------------------------------------
# Hybrid search (sqlite-vec required)
# ---------------------------------------------------------------------------


def _make_test_db_with_vec(
    path: Path, chunks: list[dict[str, Any]] | None = None
) -> None:
    """Create a KB database with both kb_chunks (FTS5) and kb_chunks_vec (vec0)."""
    con = sqlite3.connect(str(path))
    con.enable_load_extension(True)
    _sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.execute(
        """CREATE VIRTUAL TABLE kb_chunks USING fts5(
            file_path UNINDEXED,
            kind UNINDEXED,
            content,
            start_line UNINDEXED,
            end_line UNINDEXED,
            symbol_name UNINDEXED,
            symbol_kind UNINDEXED,
            heading_text UNINDEXED,
            language UNINDEXED,
            tokenize='unicode61'
        )"""
    )
    con.execute("CREATE VIRTUAL TABLE kb_chunks_vec USING vec0(embedding FLOAT[128])")
    for chunk in chunks or []:
        con.execute(
            "INSERT INTO kb_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chunk["file_path"],
                chunk.get("kind", "text"),
                chunk["content"],
                chunk.get("start_line", 1),
                chunk.get("end_line", 1),
                chunk.get("symbol_name"),
                chunk.get("symbol_kind"),
                chunk.get("heading_text"),
                chunk.get("language"),
            ),
        )
        rowid = con.execute("SELECT last_insert_rowid()").fetchone()[0]
        embed = _hash_embed(chunk["content"])
        con.execute(
            "INSERT INTO kb_chunks_vec(rowid, embedding) VALUES (?, ?)",
            (rowid, embed),
        )
    con.commit()
    con.close()


@pytest.mark.skipif(not _SQLITE_VEC_AVAILABLE, reason="sqlite-vec not installed")
@pytest.mark.anyio
async def test_hybrid_returns_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db_with_vec(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert not result.is_error
    assert result.result["total"] > 0


@pytest.mark.skipif(not _SQLITE_VEC_AVAILABLE, reason="sqlite-vec not installed")
@pytest.mark.anyio
async def test_hybrid_scores_are_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db_with_vec(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute({"query": "authenticate"}, _CTX)
    assert not result.is_error
    for item in result.result["results"]:
        assert item["score"] > 0


@pytest.mark.skipif(not _SQLITE_VEC_AVAILABLE, reason="sqlite-vec not installed")
@pytest.mark.anyio
async def test_hybrid_scope_filter_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "kb.sqlite"
    _make_test_db_with_vec(db, _SAMPLE_CHUNKS)
    monkeypatch.setenv("MERIDIAN_KB_PATH", str(db))

    result = await kb_search_tool.execute(
        {"query": "authenticate", "scope": "src/**/*.py"}, _CTX
    )
    assert not result.is_error
    for item in result.result["results"]:
        assert item["file_path"].endswith(".py")
