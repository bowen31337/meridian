"""Full coverage for _reader.py — the SQLite-backed hybrid KB search used by
the kb_search built-in tool. Builds a real FTS5 + vec0 database and exercises
db-path resolution, the hashing embedder, glob/scope matching, RRF fusion,
BM25 and vector search (including every graceful-degradation branch), and the
top-level hybrid _sync_search."""

from __future__ import annotations

import sqlite3
import struct
import sys
from pathlib import Path
from typing import Any

import pytest
import sqlite_vec
from meridian_kb_indexer import _reader
from meridian_kb_indexer._reader import (
    _EMBED_DIM,
    _glob_matches,
    _hash_embed,
    _resolve_db_path,
    _rrf_fuse,
    _run_bm25,
    _run_vector,
    _scope_matches,
    _sync_search,
)

_CHUNK_COLS = (
    "file_path, kind, content, start_line, end_line, "
    "symbol_name, symbol_kind, heading_text, language"
)


def _make_kb_db(path: str, *, with_vec: bool = True) -> None:
    con = sqlite3.connect(path)
    try:
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        con.execute(f"CREATE VIRTUAL TABLE kb_chunks USING fts5({_CHUNK_COLS})")
        con.execute(
            f"INSERT INTO kb_chunks(rowid, {_CHUNK_COLS})"
            " VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("src/a.py", "code", "def hello world", "1", "5", "hello", "function", "", "python"),
        )
        con.execute(
            f"INSERT INTO kb_chunks(rowid, {_CHUNK_COLS})"
            " VALUES (2, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("docs/b.md", "doc", "hello docs", "1", "3", "", "", "Intro", "markdown"),
        )
        if with_vec:
            con.execute("CREATE VIRTUAL TABLE kb_chunks_vec USING vec0(embedding FLOAT[128])")
            emb = _hash_embed("hello")
            con.execute("INSERT INTO kb_chunks_vec(rowid, embedding) VALUES (1, ?)", (emb,))
            con.execute("INSERT INTO kb_chunks_vec(rowid, embedding) VALUES (2, ?)", (emb,))
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# _resolve_db_path
# ---------------------------------------------------------------------------


def test_resolve_db_path_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERIDIAN_KB_PATH", "/custom/kb.sqlite")
    assert _resolve_db_path("/workspace") == "/custom/kb.sqlite"


def test_resolve_db_path_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MERIDIAN_KB_PATH", raising=False)
    resolved = _resolve_db_path("/workspace")
    assert resolved.endswith("/.meridian/kb.sqlite")


# ---------------------------------------------------------------------------
# _hash_embed
# ---------------------------------------------------------------------------


def test_hash_embed_shape_and_norm() -> None:
    raw = _hash_embed("hello world hello")
    floats = struct.unpack(f"{_EMBED_DIM}f", raw)
    assert len(floats) == _EMBED_DIM
    norm = sum(f * f for f in floats) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_hash_embed_empty_text_unit_norm() -> None:
    # No tokens -> all-zero vector -> norm divisor falls back to 1.0 (no div0).
    raw = _hash_embed("")
    floats = struct.unpack(f"{_EMBED_DIM}f", raw)
    assert all(f == 0.0 for f in floats)


# ---------------------------------------------------------------------------
# _glob_matches / _scope_matches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "scope", "expected"),
    [
        ("src/a.py", "src/*.py", True),
        ("src/sub/a.py", "src/*.py", False),  # * does not cross /
        ("src/sub/a.py", "src/**", True),  # ** crosses /
        ("a.py", "?.py", True),
        ("ab.py", "?.py", False),
        ("src/a.py", "src/a.py", True),
    ],
)
def test_glob_matches(path: str, scope: str, expected: bool) -> None:
    assert _glob_matches(path, scope) is expected


def test_scope_matches_strips_workspace_prefix() -> None:
    assert _scope_matches("/workspace/src/a.py", "src/*.py", "/workspace") is True
    assert _scope_matches("src/a.py", "src/*.py", "") is True


# ---------------------------------------------------------------------------
# _rrf_fuse
# ---------------------------------------------------------------------------


def _chunk(fp: str, sl: int, el: int) -> dict[str, Any]:
    return {"file_path": fp, "start_line": sl, "end_line": el, "score": 0.0}


def test_rrf_fuse_merges_and_scores() -> None:
    list_a = [_chunk("a", 1, 2), _chunk("b", 1, 2)]
    list_b = [_chunk("b", 1, 2), _chunk("c", 1, 2)]
    fused = _rrf_fuse([list_a, list_b], limit=10)
    keys = [c["file_path"] for c in fused]
    # "b" appears in both lists -> highest fused score -> ranked first.
    assert keys[0] == "b"
    assert {"a", "b", "c"} == set(keys)
    assert all(c["score"] > 0 for c in fused)


def test_rrf_fuse_respects_limit() -> None:
    list_a = [_chunk("a", 1, 2), _chunk("b", 1, 2), _chunk("c", 1, 2)]
    assert len(_rrf_fuse([list_a], limit=2)) == 2


# ---------------------------------------------------------------------------
# _run_bm25
# ---------------------------------------------------------------------------


def test_run_bm25_returns_positive_scores(tmp_path: Path) -> None:
    db = str(tmp_path / "kb.sqlite")
    _make_kb_db(db)
    con = sqlite3.connect(db)
    try:
        rows = _run_bm25(con, "hello", 10)
        assert rows
        assert all(r["score"] >= 0 for r in rows)
        assert rows[0]["file_path"] in {"src/a.py", "docs/b.md"}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# _run_vector — success + every graceful-degradation branch
# ---------------------------------------------------------------------------


def test_run_vector_success(tmp_path: Path) -> None:
    db = str(tmp_path / "kb.sqlite")
    _make_kb_db(db)
    con = sqlite3.connect(db)
    try:
        rows = _run_vector(con, "hello", 10)
        assert rows
        assert all(r["score"] == 0.0 for r in rows)  # placeholder before RRF
    finally:
        con.close()


def test_run_vector_import_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "kb.sqlite")
    _make_kb_db(db)
    monkeypatch.setitem(sys.modules, "sqlite_vec", None)  # import -> ImportError
    con = sqlite3.connect(db)
    try:
        assert _run_vector(con, "hello", 10) == []
    finally:
        con.close()


def test_run_vector_missing_table(tmp_path: Path) -> None:
    db = str(tmp_path / "kb.sqlite")
    _make_kb_db(db, with_vec=False)
    con = sqlite3.connect(db)
    try:
        assert _run_vector(con, "hello", 10) == []
    finally:
        con.close()


def test_run_vector_extension_load_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "kb.sqlite")
    _make_kb_db(db)
    con = sqlite3.connect(db)

    def _boom(_con: Any) -> None:
        raise RuntimeError("load failed")

    monkeypatch.setattr(sqlite_vec, "load", _boom)
    try:
        assert _run_vector(con, "hello", 10) == []
    finally:
        con.close()


class _BoomVecConn(sqlite3.Connection):
    def execute(self, sql: str, *args: Any) -> Any:  # type: ignore[override]
        if "kb_chunks_vec" in sql and "MATCH" in sql:
            raise sqlite3.OperationalError("vec query boom")
        return super().execute(sql, *args)


def test_run_vector_query_error(tmp_path: Path) -> None:
    db = str(tmp_path / "kb.sqlite")
    _make_kb_db(db)
    con = sqlite3.connect(db, factory=_BoomVecConn)
    try:
        assert _run_vector(con, "hello", 10) == []
    finally:
        con.close()


# ---------------------------------------------------------------------------
# _sync_search — top-level hybrid flow
# ---------------------------------------------------------------------------


def test_sync_search_db_missing(tmp_path: Path) -> None:
    assert _sync_search(str(tmp_path / "nope.sqlite"), "hello", None, 10, "") == []


def test_sync_search_fts_table_missing(tmp_path: Path) -> None:
    db = str(tmp_path / "empty.sqlite")
    sqlite3.connect(db).close()  # exists but has no kb_chunks table
    assert _sync_search(db, "hello", None, 10, "") == []


def test_sync_search_hybrid_rrf(tmp_path: Path) -> None:
    db = str(tmp_path / "kb.sqlite")
    _make_kb_db(db)
    results = _sync_search(db, "hello", None, 10, "")
    assert results
    # vector rows present -> RRF fusion assigns positive scores.
    assert all(r["score"] > 0 for r in results)


def test_sync_search_bm25_only_when_no_vec(tmp_path: Path) -> None:
    db = str(tmp_path / "kb.sqlite")
    _make_kb_db(db, with_vec=False)
    results = _sync_search(db, "hello", None, 10, "")
    assert results
    assert all(r["file_path"] in {"src/a.py", "docs/b.md"} for r in results)


def test_sync_search_with_scope_filter(tmp_path: Path) -> None:
    db = str(tmp_path / "kb.sqlite")
    _make_kb_db(db)
    results = _sync_search(db, "hello", "src/*.py", 10, "")
    assert results
    assert all(r["file_path"] == "src/a.py" for r in results)
