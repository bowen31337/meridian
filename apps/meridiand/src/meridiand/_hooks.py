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
# Enums
# ---------------------------------------------------------------------------


class HandlerType(str, Enum):
    in_process = "in_process"
    subprocess = "subprocess"
    mcp = "mcp"
    http = "http"
    container = "container"


class FailureMode(str, Enum):
    ignore = "ignore"
    warn = "warn"
    abort = "abort"


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class HookCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="hook_create_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


class HookInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="hook_invalid_request", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class MatchFilter(BaseModel):
    session_id: str | None = None
    agent_id: str | None = None


class HookCreateRequest(BaseModel):
    event: str
    name: str
    handler: HandlerType
    match: MatchFilter | None = None
    timeout_ms: int
    failure_mode: FailureMode
    secret_reads: list[str] | None = None
    metadata: dict[str, Any] | None = None


def _validate_request(body: HookCreateRequest) -> HookInvalidRequestError | None:
    if not body.event.strip():
        return HookInvalidRequestError(
            message="'event' must not be empty",
            timestamp=_now(),
        )
    if not body.name.strip():
        return HookInvalidRequestError(
            message="'name' must not be empty",
            timestamp=_now(),
        )
    if body.timeout_ms <= 0:
        return HookInvalidRequestError(
            message="'timeout_ms' must be > 0",
            timestamp=_now(),
        )
    return None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_hooks_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    hooks_dir = storage_root / "hooks"

    @router.post("/v1/x/hooks", status_code=201)
    async def create_hook(body: HookCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        hook_id = f"hook_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "hook.create",
            attributes={
                "hook.id": hook_id,
                "hook.event": body.event,
                "hook.name": body.name,
                "hook.handler": body.handler.value,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="hook.create.invocation",
                    code="hook_create",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_request(body)
                if validation_err is not None:
                    raise validation_err

                hooks_dir.mkdir(parents=True, exist_ok=True)
                resource: dict[str, Any] = {
                    "id": hook_id,
                    "event": body.event,
                    "name": body.name,
                    "handler": body.handler.value,
                    "match": body.match.model_dump() if body.match is not None else None,
                    "timeout_ms": body.timeout_ms,
                    "failure_mode": body.failure_mode.value,
                    "secret_reads": body.secret_reads,
                    "status": "active",
                    "created_at": now,
                    "metadata": body.metadata,
                }
                (hooks_dir / f"{hook_id}.json").write_text(json.dumps(resource))

            except HookInvalidRequestError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="hook.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "hook_id": hook_id,
                            "event": body.event,
                            "name": body.name,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = HookCreateError(
                    message=f"Failed to create hook: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="hook.create.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "hook_id": hook_id,
                            "event": body.event,
                            "name": body.name,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=resource, status_code=201)

    return router
