from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import Enum
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
from pydantic import BaseModel


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Trigger type enum
# ---------------------------------------------------------------------------


class TriggerType(str, Enum):
    timestamp = "timestamp"
    interval = "interval"
    channel_event = "channel_event"
    file_change = "file_change"
    webhook = "webhook"
    memory_anniversary = "memory_anniversary"


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class CronCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="cron_create_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


class CronInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="cron_invalid_request", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class CronCreateRequest(BaseModel):
    trigger_type: TriggerType
    session_id: str
    name: str | None = None
    # timestamp trigger: ISO-8601 datetime for one-shot invocation
    timestamp: str | None = None
    # interval trigger: duration string such as "5m" or "1h"
    interval: str | None = None
    # channel_event trigger
    channel_id: str | None = None
    # file_change trigger
    path: str | None = None
    # webhook trigger
    webhook_id: str | None = None
    # memory_anniversary trigger
    memory_key: str | None = None
    metadata: dict[str, Any] | None = None


# Required fields per trigger type
_TRIGGER_REQUIRED: dict[TriggerType, tuple[str, str]] = {
    TriggerType.timestamp: ("timestamp", "ISO-8601 datetime"),
    TriggerType.interval: ("interval", "duration string (e.g. '5m', '1h')"),
    TriggerType.channel_event: ("channel_id", "channel identifier"),
    TriggerType.file_change: ("path", "file path to watch"),
    TriggerType.webhook: ("webhook_id", "webhook identifier"),
    TriggerType.memory_anniversary: ("memory_key", "memory key"),
}


def _validate_trigger(body: CronCreateRequest) -> CronInvalidRequestError | None:
    field, description = _TRIGGER_REQUIRED[body.trigger_type]
    if getattr(body, field) is None:
        return CronInvalidRequestError(
            message=(
                f"trigger_type '{body.trigger_type.value}' requires '{field}' "
                f"({description})"
            ),
            timestamp=_now(),
        )
    return None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_cron_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    cron_dir = storage_root / "cron"

    @router.post("/v1/x/cron", status_code=201)
    async def create_cron(body: CronCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        cron_id = f"cron_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "cron.create",
            attributes={
                "cron.id": cron_id,
                "cron.trigger_type": body.trigger_type.value,
                "cron.session_id": body.session_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="cron.create.invocation",
                    code="cron_create",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_trigger(body)
                if validation_err is not None:
                    raise validation_err

                cron_dir.mkdir(parents=True, exist_ok=True)
                resource: dict[str, Any] = {
                    "id": cron_id,
                    "trigger_type": body.trigger_type.value,
                    "session_id": body.session_id,
                    "name": body.name,
                    "status": "active",
                    "created_at": now,
                    "timestamp": body.timestamp,
                    "interval": body.interval,
                    "channel_id": body.channel_id,
                    "path": body.path,
                    "webhook_id": body.webhook_id,
                    "memory_key": body.memory_key,
                    "metadata": body.metadata,
                }
                (cron_dir / f"{cron_id}.json").write_text(json.dumps(resource))

                span.set_attribute("cron.name", body.name or "")

            except CronInvalidRequestError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="cron.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "cron_id": cron_id,
                            "trigger_type": body.trigger_type.value,
                            "session_id": body.session_id,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = CronCreateError(
                    message=f"Failed to create cron resource: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="cron.create.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "cron_id": cron_id,
                            "trigger_type": body.trigger_type.value,
                            "session_id": body.session_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=resource, status_code=201)

    return router
