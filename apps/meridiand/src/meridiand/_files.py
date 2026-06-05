from __future__ import annotations

import base64
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
import uuid

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
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import UploadFile
from storage_blob import BlobFailure, LocalBlobStore


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class FilesUploadError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="files_upload_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


class FilesNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="files_not_found", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 404


class FilesInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="files_invalid_request", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Blob key helpers
# ---------------------------------------------------------------------------


def _blob_key(file_id: str) -> str:
    return f"files/{file_id}"


def _meta_key(file_id: str) -> str:
    return f"files/{file_id}.meta.json"


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_files_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    store = LocalBlobStore(storage_root)

    @router.post("/v1/files", status_code=201)
    async def upload_file(request: Request) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        meta: dict[str, Any] = {}

        with tracer.start_as_current_span("files.upload") as span:
            record_invocation_event(
                span,
                StructuredEvent(name="files.upload.invocation", code="files_upload", timestamp=now),
            )

            try:
                ct = request.headers.get("content-type", "")

                if "multipart/form-data" in ct:
                    form = await request.form()
                    upload = form.get("file")
                    if not isinstance(upload, UploadFile):
                        raise FilesInvalidRequestError(
                            message="multipart field 'file' is required and must be a file",
                            timestamp=_now(),
                        )
                    data = await upload.read()
                    name = str(form.get("name") or upload.filename or "upload")
                    content_type_value = upload.content_type or "application/octet-stream"

                elif "application/json" in ct:
                    body = await request.json()
                    name = str(body.get("name") or "upload")
                    raw_content = body.get("content")
                    if not raw_content:
                        raise FilesInvalidRequestError(
                            message="JSON body must include 'content' (base64-encoded bytes)",
                            timestamp=_now(),
                        )
                    try:
                        data = base64.b64decode(raw_content)
                    except Exception as exc:
                        raise FilesInvalidRequestError(
                            message=f"'content' is not valid base64: {exc}",
                            timestamp=_now(),
                        ) from exc
                    content_type_value = str(body.get("content_type") or "application/octet-stream")

                else:
                    raise FilesInvalidRequestError(
                        message="Content-Type must be multipart/form-data or application/json",
                        timestamp=_now(),
                    )

                file_id = f"file_{uuid.uuid4().hex}"
                created_at = _now()
                meta = {
                    "id": file_id,
                    "name": name,
                    "size": len(data),
                    "content_type": content_type_value,
                    "created_at": created_at,
                }

                await store.put(_blob_key(file_id), data)
                await store.put(_meta_key(file_id), json.dumps(meta).encode())

                span.set_attribute("file.id", file_id)
                span.set_attribute("file.size", len(data))

            except FilesInvalidRequestError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="files.upload.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"message": err.message},
                    )
                )
                raise
            except Exception as exc:
                err2 = FilesUploadError(
                    message=f"File upload failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="files.upload.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"message": err2.message},
                    )
                )
                raise err2 from exc

        return JSONResponse(content=meta, status_code=201)

    @router.get("/v1/files/{file_id}")
    async def get_file_metadata(file_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        meta: dict[str, Any] = {}

        with tracer.start_as_current_span(
            "files.get_metadata", attributes={"file.id": file_id}
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="files.get_metadata.invocation",
                    code="files_get_metadata",
                    timestamp=now,
                ),
            )

            try:
                raw = await store.get(_meta_key(file_id))
                meta = json.loads(raw)
            except BlobFailure as exc:
                if exc.code == "BLOB_KEY_NOT_FOUND":
                    err = FilesNotFoundError(
                        message=f"File not found: {file_id}",
                        timestamp=_now(),
                    )
                    record_error(span, err)
                    audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="files.get_metadata.failed",
                            code=err.code,
                            timestamp=err.timestamp,
                            detail={"file_id": file_id},
                        )
                    )
                    raise err from exc
                err2 = FilesUploadError(
                    message=f"Metadata read failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="files.get_metadata.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"file_id": file_id, "message": err2.message},
                    )
                )
                raise err2 from exc
            except Exception as exc:
                err3 = FilesUploadError(
                    message=f"Metadata read failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err3)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="files.get_metadata.failed",
                        code=err3.code,
                        timestamp=err3.timestamp,
                        detail={"file_id": file_id, "message": err3.message},
                    )
                )
                raise err3 from exc

        return JSONResponse(content=meta)

    @router.get("/v1/files/{file_id}/content")
    async def get_file_content(file_id: str) -> Response:
        now = _now()
        tracer = get_tracer()
        data: bytes = b""
        media_type = "application/octet-stream"

        with tracer.start_as_current_span(
            "files.get_content", attributes={"file.id": file_id}
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="files.get_content.invocation",
                    code="files_get_content",
                    timestamp=now,
                ),
            )

            try:
                try:
                    raw_meta = await store.get(_meta_key(file_id))
                    meta = json.loads(raw_meta)
                    media_type = meta.get("content_type", "application/octet-stream")
                except BlobFailure as exc:
                    if exc.code == "BLOB_KEY_NOT_FOUND":
                        raise FilesNotFoundError(
                            message=f"File not found: {file_id}",
                            timestamp=_now(),
                        ) from exc
                    raise

                data = await store.get(_blob_key(file_id))
                span.set_attribute("file.size", len(data))

            except FilesNotFoundError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="files.get_content.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"file_id": file_id},
                    )
                )
                raise
            except Exception as exc:
                err2 = FilesUploadError(
                    message=f"Content read failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="files.get_content.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"file_id": file_id, "message": err2.message},
                    )
                )
                raise err2 from exc

        return Response(content=data, media_type=media_type)

    return router
