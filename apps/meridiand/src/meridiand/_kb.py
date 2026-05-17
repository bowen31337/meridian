from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from meridian_kb_indexer import WorkspaceIndexer, should_index_path
from pydantic import BaseModel

_WORKSPACE_ENV = "WORKSPACE"
_DEFAULT_SCOPE = "workspace"


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


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class KbIndexRequest(BaseModel):
    path: str | None = None
    scope: str | None = None


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
# Router factory
# ---------------------------------------------------------------------------


def make_kb_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/kb/index")
    async def kb_index(body: KbIndexRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        target_path = body.path
        if target_path:
            scope_key = target_path
        elif body.scope:
            scope_key = body.scope
        else:
            scope_key = _DEFAULT_SCOPE

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
                else:
                    workspace = body.scope or os.environ.get(_WORKSPACE_ENV, os.getcwd())
                    for p in Path(workspace).rglob("*"):
                        if p.is_file() and should_index_path(str(p)):
                            try:
                                file_chunks = await indexer.index_file(str(p))
                                chunk_count += len(file_chunks)
                            except Exception:
                                pass  # per-file failures already audited in WorkspaceIndexer

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

    return router
