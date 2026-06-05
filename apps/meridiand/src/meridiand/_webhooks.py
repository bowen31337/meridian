from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
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
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BackoffType(StrEnum):
    exponential = "exponential"
    linear = "linear"


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class WebhookCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="webhook_create_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


class WebhookInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="webhook_invalid_request", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 422


class WebhookNotFoundError(MeridianError):
    def __init__(self, *, webhook_id: str, timestamp: str) -> None:
        super().__init__(
            code="webhook_not_found",
            message=f"Webhook '{webhook_id}' not found",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class WebhookDeleteError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="webhook_delete_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class EventFilter(BaseModel):
    types: list[str]
    session_id: str | None = None


class WebhookCreateRequest(BaseModel):
    name: str
    url: str
    secret_ref: str | None = None
    event_filter: EventFilter
    max_retries: int
    backoff: BackoffType
    metadata: dict[str, Any] | None = None


def _validate_request(body: WebhookCreateRequest) -> WebhookInvalidRequestError | None:
    if not body.url.strip():
        return WebhookInvalidRequestError(
            message="'url' must not be empty",
            timestamp=_now(),
        )
    if not body.event_filter.types:
        return WebhookInvalidRequestError(
            message="'event_filter.types' must contain at least one event type",
            timestamp=_now(),
        )
    if body.max_retries < 0:
        return WebhookInvalidRequestError(
            message="'max_retries' must be >= 0",
            timestamp=_now(),
        )
    return None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_webhooks_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    webhooks_dir = storage_root / "webhooks"

    @router.post("/v1/webhooks", status_code=201)
    async def create_webhook(body: WebhookCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        webhook_id = f"webhook_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "webhook.create",
            attributes={
                "webhook.id": webhook_id,
                "webhook.url": body.url,
                "webhook.name": body.name,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="webhook.create.invocation",
                    code="webhook_create",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_request(body)
                if validation_err is not None:
                    raise validation_err

                webhooks_dir.mkdir(parents=True, exist_ok=True)
                resource: dict[str, Any] = {
                    "id": webhook_id,
                    "name": body.name,
                    "url": body.url,
                    "secret_ref": body.secret_ref,
                    "event_filter": {
                        "types": body.event_filter.types,
                        "session_id": body.event_filter.session_id,
                    },
                    "max_retries": body.max_retries,
                    "backoff": body.backoff.value,
                    "status": "active",
                    "created_at": now,
                    "metadata": body.metadata,
                }
                (webhooks_dir / f"{webhook_id}.json").write_text(json.dumps(resource))

            except WebhookInvalidRequestError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="webhook.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "webhook_id": webhook_id,
                            "url": body.url,
                            "name": body.name,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = WebhookCreateError(
                    message=f"Failed to create webhook: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="webhook.create.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "webhook_id": webhook_id,
                            "url": body.url,
                            "name": body.name,
                            "message": err2.message,
                        },
                    )
                )
                raise err2 from exc

        return JSONResponse(content=resource, status_code=201)

    @router.delete("/v1/webhooks/{webhook_id}", status_code=204)
    async def delete_webhook(webhook_id: str) -> Response:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "webhook.delete",
            attributes={"webhook.id": webhook_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="webhook.delete.invocation",
                    code="webhook_delete",
                    timestamp=now,
                ),
            )

            try:
                webhook_file = webhooks_dir / f"{webhook_id}.json"
                if not webhook_file.exists():
                    raise WebhookNotFoundError(webhook_id=webhook_id, timestamp=now)
                webhook_file.unlink()

            except WebhookNotFoundError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="webhook.delete.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"webhook_id": webhook_id, "message": err.message},
                    )
                )
                raise

            except Exception as exc:
                err2 = WebhookDeleteError(
                    message=f"Failed to delete webhook: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="webhook.delete.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"webhook_id": webhook_id, "message": err2.message},
                    )
                )
                raise err2 from exc

        return Response(status_code=204)

    return router
