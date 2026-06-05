from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import Any, Literal
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
# Duration parsing
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(
    r"^(?:(?P<days>\d+)d)?(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?$"
)


def _parse_duration(s: str) -> timedelta:
    """Parse a duration string like '5m', '1h', '2d', '1h30m' to timedelta."""
    stripped = s.strip()
    if not stripped:
        raise ValueError("Duration string is empty")
    m = _DURATION_RE.match(stripped)
    if not m or not any(m.group(k) for k in ("days", "hours", "minutes", "seconds")):
        raise ValueError(f"Invalid duration: {s!r} (expected e.g. '5m', '1h', '2d')")
    parts = {k: int(v) for k, v in m.groupdict(default="0").items()}
    td = timedelta(**parts)
    if td.total_seconds() <= 0:
        raise ValueError(f"Duration must be positive: {s!r}")
    return td


# ---------------------------------------------------------------------------
# Trigger type enum
# ---------------------------------------------------------------------------


class TriggerType(StrEnum):
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


class CronNotFoundError(MeridianError):
    def __init__(self, *, cron_id: str, timestamp: str) -> None:
        super().__init__(
            code="cron_not_found",
            message=f"Cron resource '{cron_id}' not found",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class CronDeleteError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="cron_delete_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


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
    # days_before: how many days before the anniversary date to fire
    days_before: int | None = None
    metadata: dict[str, Any] | None = None
    # Scheduler durability: what to do with fires missed during daemon downtime.
    # "catch_up" fires once per missed interval; "skip" skips the missed period.
    missed_fires_policy: Literal["catch_up", "skip"] = "skip"
    # Agent capabilities inherited by cron-triggered sessions; never escalated.
    capabilities: list[str] = []


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
                f"trigger_type '{body.trigger_type.value}' requires '{field}' ({description})"
            ),
            timestamp=_now(),
        )
    if body.trigger_type == TriggerType.memory_anniversary and body.days_before is None:
        return CronInvalidRequestError(
            message=(
                "trigger_type 'memory_anniversary' requires 'days_before' "
                "(number of days before the anniversary to fire)"
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

                # Compute next_fire_at for time-based triggers.
                next_fire_at: str | None = None
                if body.trigger_type == TriggerType.timestamp:
                    next_fire_at = body.timestamp
                elif body.trigger_type == TriggerType.interval:
                    try:
                        delta = _parse_duration(body.interval)  # type: ignore[arg-type]
                        next_fire_at = (datetime.now(UTC) + delta).isoformat()
                    except ValueError as exc:
                        raise CronInvalidRequestError(
                            message=f"Invalid interval duration {body.interval!r}: {exc}",
                            timestamp=_now(),
                        ) from exc

                cron_dir.mkdir(parents=True, exist_ok=True)
                resource: dict[str, Any] = {
                    "id": cron_id,
                    "trigger_type": body.trigger_type.value,
                    "session_id": body.session_id,
                    "name": body.name,
                    "status": "active",
                    "created_at": now,
                    "next_fire_at": next_fire_at,
                    "missed_fires_policy": body.missed_fires_policy,
                    "capabilities": body.capabilities,
                    "timestamp": body.timestamp,
                    "interval": body.interval,
                    "channel_id": body.channel_id,
                    "path": body.path,
                    "webhook_id": body.webhook_id,
                    "memory_key": body.memory_key,
                    "days_before": body.days_before,
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
                raise err2 from exc

        return JSONResponse(content=resource, status_code=201)

    @router.delete("/v1/x/cron/{cron_id}", status_code=204)
    async def delete_cron(cron_id: str) -> Response:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "cron.delete",
            attributes={"cron.id": cron_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="cron.delete.invocation",
                    code="cron_delete",
                    timestamp=now,
                ),
            )

            try:
                cron_file = cron_dir / f"{cron_id}.json"
                if not cron_file.exists():
                    raise CronNotFoundError(cron_id=cron_id, timestamp=now)
                cron_file.unlink()

            except CronNotFoundError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="cron.delete.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"cron_id": cron_id, "message": err.message},
                    )
                )
                raise

            except Exception as exc:
                err2 = CronDeleteError(
                    message=f"Failed to delete cron resource: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="cron.delete.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"cron_id": cron_id, "message": err2.message},
                    )
                )
                raise err2 from exc

        return Response(status_code=204)

    return router
