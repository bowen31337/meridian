from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

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
from pydantic import BaseModel

from ._kb import KbStore


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class MemoryStoreCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="memory_store_create_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class MemoryStoreInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="memory_store_invalid_request", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 422


class MemoryStoreNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="memory_store_not_found", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 404


class MemoryStoreQueryError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="memory_store_query_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

MemoryStoreBackend = Literal["sqlite-vec", "pgvector", "http"]
MemoryStoreScope = Literal["global", "user", "agent", "project"]


class MemoryStoreCreateRequest(BaseModel):
    name: str
    backend: MemoryStoreBackend
    scope: MemoryStoreScope
    metadata: dict[str, Any] | None = None


class MemoryStoreQueryRequest(BaseModel):
    query: str
    scope: str | None = None
    limit: int = 10
    bm25_weight: float = 1.0
    vector_weight: float = 1.0
    rrf_k: int = 60


def _validate_request(body: MemoryStoreCreateRequest) -> MemoryStoreInvalidRequestError | None:
    if not body.name.strip():
        return MemoryStoreInvalidRequestError(
            message="'name' must not be empty",
            timestamp=_now(),
        )
    return None


# ---------------------------------------------------------------------------
# Weighted RRF fusion
# ---------------------------------------------------------------------------


def _weighted_rrf_fuse(
    ranked_lists: list[tuple[list[dict[str, Any]], float]],
    limit: int,
    k: int = 60,
) -> list[dict[str, Any]]:
    def _key(c: dict[str, Any]) -> tuple[str, int, int]:
        return (c["file_path"], c["start_line"], c["end_line"])

    scores: dict[tuple[str, int, int], float] = {}
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}

    for ranked, weight in ranked_lists:
        for rank, chunk in enumerate(ranked, 1):
            key = _key(chunk)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            by_key[key] = chunk

    sorted_keys = sorted(scores, key=lambda kk: scores[kk], reverse=True)
    return [by_key[kk] for kk in sorted_keys[:limit]]


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_memory_stores_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    stores_dir = storage_root / "memory_stores"

    @router.post("/v1/memory_stores", status_code=201)
    async def create_memory_store(body: MemoryStoreCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        store_id = f"memstore_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "memory_store.create",
            attributes={
                "memory_store.id": store_id,
                "memory_store.name": body.name,
                "memory_store.backend": body.backend,
                "memory_store.scope": body.scope,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="memory_store.create.invocation",
                    code="memory_store_create",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_request(body)
                if validation_err is not None:
                    raise validation_err

                stores_dir.mkdir(parents=True, exist_ok=True)

                store_record: dict[str, Any] = {
                    "id": store_id,
                    "name": body.name,
                    "backend": body.backend,
                    "scope": body.scope,
                    "metadata": body.metadata,
                    "created_at": now,
                }
                (stores_dir / f"{store_id}.json").write_text(json.dumps(store_record))

            except MemoryStoreInvalidRequestError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="memory_store.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "memory_store_id": store_id,
                            "name": body.name,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = MemoryStoreCreateError(
                    message=f"Failed to create memory store: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="memory_store.create.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "memory_store_id": store_id,
                            "name": body.name,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=store_record, status_code=201)

    @router.post("/v1/memory_stores/{store_id}/query_runs")
    async def query_memory_store(store_id: str, body: MemoryStoreQueryRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "memory_store.query",
            attributes={
                "memory_store.id": store_id,
                "memory_store.query": body.query,
                "memory_store.scope": body.scope or "",
                "memory_store.bm25_weight": body.bm25_weight,
                "memory_store.vector_weight": body.vector_weight,
                "memory_store.rrf_k": body.rrf_k,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="memory_store.query.invocation",
                    code="memory_store_query",
                    timestamp=now,
                ),
            )

            try:
                store_path = stores_dir / f"{store_id}.json"
                if not store_path.exists():
                    raise MemoryStoreNotFoundError(
                        message=f"Memory store '{store_id}' not found",
                        timestamp=now,
                    )

                kb_store = KbStore(stores_dir / store_id / "chunks.db")
                bm25_results = kb_store.bm25_search(body.query, body.scope, body.limit)
                vector_results = kb_store.vector_search(body.query, body.scope, body.limit)
                results = _weighted_rrf_fuse(
                    [(bm25_results, body.bm25_weight), (vector_results, body.vector_weight)],
                    body.limit,
                    k=body.rrf_k,
                )
                span.set_attribute("memory_store.result_count", len(results))

            except MemoryStoreNotFoundError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="memory_store.query.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "memory_store_id": store_id,
                            "query": body.query,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = MemoryStoreQueryError(
                    message=f"Memory store query failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="memory_store.query.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "memory_store_id": store_id,
                            "query": body.query,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(
            content={
                "results": results,
                "query": body.query,
                "scope": body.scope,
                "count": len(results),
                "store_id": store_id,
                "bm25_weight": body.bm25_weight,
                "vector_weight": body.vector_weight,
                "rrf_k": body.rrf_k,
            }
        )

    return router
