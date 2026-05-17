from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import sqlite_vec
import sqlean
from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from meridian_kb_indexer import Chunk, WorkspaceIndexer, should_index_path
from pydantic import BaseModel

_WORKSPACE_ENV = "WORKSPACE"

KbScope = Literal["global", "project", "agent", "session"]
_DEFAULT_SCOPE: KbScope = "global"

# Embedding dimension for hashing-trick dense vectors stored in sqlite-vec.
_EMBED_DIM = 128


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class KbIndexError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="kb_index_failed", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 422


class KbStatusError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="kb_status_failed", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 422


class KbQueryError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="kb_query_failed", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class KbIndexRequest(BaseModel):
    path: str | None = None
    scope: KbScope | None = None


class KbQueryRequest(BaseModel):
    query: str
    scope: KbScope | None = None
    method: Literal["glob", "bm25", "vector", "hybrid"] = "hybrid"
    limit: int = 10


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def _status_path(storage_root: Path) -> Path:
    return storage_root / "kb" / "status.json"


def _load_status(storage_root: Path) -> dict[str, Any]:
    p = _status_path(storage_root)
    if p.exists():
        return json.loads(p.read_text())
    return {"status": "idle", "last_updated": None, "row_counts": {}}


def _write_status_atomic(storage_root: Path, data: dict[str, Any]) -> None:
    p = _status_path(storage_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, default=str).encode()
    with tempfile.NamedTemporaryFile(dir=p.parent, suffix=".tmp", delete=False) as tf:
        tf.write(encoded)
        tf.flush()
        os.fsync(tf.fileno())
        tmp = tf.name
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _hash_embed(text: str) -> bytes:
    """Hashing-trick dense embedding stored as sqlite-vec float32 bytes."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    vec = [0.0] * _EMBED_DIM
    for token in tokens:
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % _EMBED_DIM
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return sqlite_vec.serialize_float32([v / norm for v in vec])


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def _rrf_fuse(
    ranked_lists: list[list[dict[str, Any]]],
    limit: int,
    k: int = 60,
) -> list[dict[str, Any]]:
    def _key(c: dict[str, Any]) -> tuple[str, int, int]:
        return (c["file_path"], c["start_line"], c["end_line"])

    scores: dict[tuple[str, int, int], float] = {}
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, 1):
            key = _key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            by_key[key] = chunk

    sorted_keys = sorted(scores, key=lambda kk: scores[kk], reverse=True)
    return [by_key[kk] for kk in sorted_keys[:limit]]


# ---------------------------------------------------------------------------
# SQLite chunk store — FTS5 (BM25) + sqlite-vec (vector KNN)
# ---------------------------------------------------------------------------

# Columns stored in the FTS5 virtual table.  Only `content` is indexed for
# BM25; all others are UNINDEXED metadata.  The embedding lives in a
# companion vec0 virtual table linked by rowid.
_COLS = (
    "file_path",
    "scope",
    "kind",
    "content",
    "start_line",
    "end_line",
    "symbol_name",
    "symbol_kind",
    "heading_level",
    "heading_text",
    "language",
    "content_hash",
)

_CREATE_FTS5 = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    {", ".join(c if c == "content" else c + " UNINDEXED" for c in _COLS)},
    tokenize='unicode61'
)
"""

_CREATE_VEC = f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding FLOAT[{_EMBED_DIM}])"

_INSERT_SQL = (
    f"INSERT INTO chunks_fts ({', '.join(_COLS)}) "
    f"VALUES ({', '.join('?' * len(_COLS))})"
)

_SELECT_COLS = ", ".join(_COLS)
_SELECT_COLS_F = ", ".join(f"f.{c}" for c in _COLS)


def _open_conn(db_path: Path) -> sqlean.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn: sqlean.Connection = sqlean.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlean.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(_CREATE_FTS5)
    conn.execute(_CREATE_VEC)
    conn.commit()
    return conn


class KbStore:
    """Lazy-connecting store: FTS5 for BM25 tokens, vec0 for embedding vectors."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlean.Connection | None = None

    def _get_conn(self) -> sqlean.Connection:
        if self._conn is None:
            self._conn = _open_conn(self._db_path)
        return self._conn

    def upsert_chunks(self, file_path: str, scope: str, chunks: list[Chunk]) -> None:
        conn = self._get_conn()
        with conn:
            # Remove old FTS5 rows and their companion vec0 rows.
            old = conn.execute(
                "SELECT rowid FROM chunks_fts WHERE file_path = ?", [file_path]
            ).fetchall()
            if old:
                placeholders = ",".join("?" * len(old))
                old_ids = [r[0] for r in old]
                conn.execute(
                    f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})", old_ids
                )
            conn.execute("DELETE FROM chunks_fts WHERE file_path = ?", [file_path])

            # Insert new chunks; link each FTS5 rowid into chunks_vec.
            for c in chunks:
                content_hash = hashlib.sha256(c.content.encode()).hexdigest()
                conn.execute(
                    _INSERT_SQL,
                    (
                        file_path,
                        scope,
                        c.kind,
                        c.content,
                        c.start_line,
                        c.end_line,
                        c.symbol_name,
                        c.symbol_kind,
                        c.heading_level,
                        c.heading_text,
                        c.language,
                        content_hash,
                    ),
                )
                rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.execute(
                    "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                    [rowid, _hash_embed(c.content)],
                )

    def glob_search(
        self, pattern: str, scope: str | None, limit: int
    ) -> list[dict[str, Any]]:
        conn = self._get_conn()
        if scope:
            rows = conn.execute(
                f"SELECT {_SELECT_COLS} FROM chunks_fts WHERE scope = ?", [scope]
            ).fetchall()
        else:
            rows = conn.execute(f"SELECT {_SELECT_COLS} FROM chunks_fts").fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            if fnmatch.fnmatch(d["file_path"], pattern):
                results.append(d)
                if len(results) >= limit:
                    break
        return results

    def bm25_search(
        self, query: str, scope: str | None, limit: int
    ) -> list[dict[str, Any]]:
        words = re.findall(r"\w+", query)
        if not words:
            return []
        fts_query = " ".join(words)
        conn = self._get_conn()
        if scope:
            rows = conn.execute(
                f"SELECT {_SELECT_COLS} FROM chunks_fts "
                "WHERE chunks_fts MATCH ? AND scope = ? "
                "ORDER BY bm25(chunks_fts) LIMIT ?",
                [fts_query, scope, limit],
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_SELECT_COLS} FROM chunks_fts "
                "WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts) LIMIT ?",
                [fts_query, limit],
            ).fetchall()
        return [dict(row) for row in rows]

    def vector_search(
        self, query: str, scope: str | None, limit: int
    ) -> list[dict[str, Any]]:
        """KNN search via sqlite-vec, filtered by scope."""
        conn = self._get_conn()
        q_embed = _hash_embed(query)
        candidates_per_scope = limit * 10

        # Retrieve top-K candidates from the vec0 table, then join to get scope + metadata.
        if scope:
            rows = conn.execute(
                f"""
                SELECT {_SELECT_COLS_F}
                FROM (
                    SELECT rowid, distance
                    FROM chunks_vec
                    WHERE embedding MATCH ? AND k = ?
                    ORDER BY distance
                ) v
                JOIN chunks_fts f ON f.rowid = v.rowid
                WHERE f.scope = ?
                ORDER BY v.distance
                LIMIT ?
                """,
                [q_embed, candidates_per_scope, scope, limit],
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT {_SELECT_COLS_F}
                FROM (
                    SELECT rowid, distance
                    FROM chunks_vec
                    WHERE embedding MATCH ? AND k = ?
                    ORDER BY distance
                ) v
                JOIN chunks_fts f ON f.rowid = v.rowid
                ORDER BY v.distance
                LIMIT ?
                """,
                [q_embed, candidates_per_scope, limit],
            ).fetchall()
        return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_kb_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    store = KbStore(storage_root / "kb" / "chunks.db")

    @router.post("/v1/x/kb/index")
    async def kb_index(body: KbIndexRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        scope_key: KbScope = body.scope or _DEFAULT_SCOPE
        target_path = body.path

        with tracer.start_as_current_span(
            "kb.index",
            attributes={
                "kb.path": target_path or "",
                "kb.scope": scope_key,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(name="kb.index.invocation", code="kb_index", timestamp=now),
            )

            try:
                chunk_count = 0
                indexer = WorkspaceIndexer()

                if target_path:
                    chunks = await indexer.index_file(target_path)
                    chunk_count = len(chunks)
                    store.upsert_chunks(target_path, scope_key, chunks)
                else:
                    workspace = os.environ.get(_WORKSPACE_ENV, os.getcwd())
                    for p in Path(workspace).rglob("*"):
                        if p.is_file() and should_index_path(str(p)):
                            try:
                                file_chunks = await indexer.index_file(str(p))
                                chunk_count += len(file_chunks)
                                store.upsert_chunks(str(p), scope_key, file_chunks)
                            except Exception:
                                pass

                status = _load_status(storage_root)
                status["status"] = "idle"
                status["last_updated"] = _now()
                row_counts: dict[str, int] = status.get("row_counts") or {}
                row_counts[scope_key] = chunk_count
                status["row_counts"] = row_counts
                _write_status_atomic(storage_root, status)

            except KbIndexError:
                raise
            except Exception as exc:
                err = KbIndexError(
                    message=f"KB index failed for scope {scope_key!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="kb.index.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "scope": scope_key,
                            "path": target_path,
                            "message": err.message,
                        },
                    )
                )
                raise err

        return JSONResponse(
            content={
                "scope": scope_key,
                "row_count": chunk_count,
                "status": "indexed",
            }
        )

    @router.get("/v1/x/kb")
    async def kb_status() -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span("kb.status") as span:
            record_invocation_event(
                span,
                StructuredEvent(name="kb.status.invocation", code="kb_status", timestamp=now),
            )

            try:
                data = _load_status(storage_root)
            except Exception as exc:
                err = KbStatusError(
                    message=f"KB status read failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="kb.status.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"message": err.message},
                    )
                )
                raise err

        return JSONResponse(content=data)

    @router.post("/v1/x/kb/query")
    async def kb_query(body: KbQueryRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "kb.query",
            attributes={
                "kb.query": body.query,
                "kb.scope": body.scope or "",
                "kb.method": body.method,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(name="kb.query.invocation", code="kb_query", timestamp=now),
            )

            try:
                if body.method == "glob":
                    results = store.glob_search(body.query, body.scope, body.limit)
                elif body.method == "bm25":
                    results = store.bm25_search(body.query, body.scope, body.limit)
                elif body.method == "vector":
                    results = store.vector_search(body.query, body.scope, body.limit)
                else:  # hybrid
                    glob_r = store.glob_search(body.query, body.scope, body.limit)
                    bm25_r = store.bm25_search(body.query, body.scope, body.limit)
                    vec_r = store.vector_search(body.query, body.scope, body.limit)
                    results = _rrf_fuse([glob_r, bm25_r, vec_r], body.limit)

                span.set_attribute("kb.result_count", len(results))

            except KbQueryError:
                raise
            except Exception as exc:
                err = KbQueryError(
                    message=f"KB query failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="kb.query.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "query": body.query,
                            "scope": body.scope,
                            "message": err.message,
                        },
                    )
                )
                raise err

        return JSONResponse(
            content={
                "results": results,
                "query": body.query,
                "scope": body.scope,
                "method": body.method,
                "count": len(results),
            }
        )

    return router
