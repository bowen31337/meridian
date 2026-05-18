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


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

MemoryStoreBackend = Literal["sqlite-vec", "pgvector", "http"]
MemoryStoreScope = Literal["global", "user", "agent", "project"]


class MemoryStoreCreateRequest(BaseModel):
    name: str
    backend: MemoryStoreBackend
    scope: MemoryStoreScope
    metadata: dict[str, Any] | None = None


def _validate_request(body: MemoryStoreCreateRequest) -> MemoryStoreInvalidRequestError | None:
    if not body.name.strip():
        return MemoryStoreInvalidRequestError(
            message="'name' must not be empty",
            timestamp=_now(),
        )
    return None


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

    return router
