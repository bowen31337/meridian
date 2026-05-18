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


class ChannelCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="channel_create_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class ChannelInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="channel_invalid_request", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class ChannelCreateRequest(BaseModel):
    kind: str
    config: dict[str, Any]
    default_agent_id: str | None = None
    default_user_profile_id: str | None = None
    inbound_policy: Literal["open", "paired_only", "quarantine"] = "open"
    egress_policy: Literal["enabled", "disabled"] = "enabled"


def _validate_request(body: ChannelCreateRequest) -> ChannelInvalidRequestError | None:
    if not body.kind.strip():
        return ChannelInvalidRequestError(
            message="'kind' must not be empty",
            timestamp=_now(),
        )
    if "token_vault_ref" not in body.config:
        return ChannelInvalidRequestError(
            message="'config.token_vault_ref' is required",
            timestamp=_now(),
        )
    tvr = body.config["token_vault_ref"]
    if not isinstance(tvr, str) or not tvr.strip():
        return ChannelInvalidRequestError(
            message="'config.token_vault_ref' must be a non-empty string",
            timestamp=_now(),
        )
    return None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_channels_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    channels_dir = storage_root / "channels"

    @router.post("/v1/channels", status_code=201)
    async def create_channel(body: ChannelCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        channel_id = f"ch_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "channel.create",
            attributes={
                "channel.id": channel_id,
                "channel.kind": body.kind,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="channel.create.invocation",
                    code="channel_create",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_request(body)
                if validation_err is not None:
                    raise validation_err

                channels_dir.mkdir(parents=True, exist_ok=True)

                channel_record: dict[str, Any] = {
                    "id": channel_id,
                    "kind": body.kind,
                    "config": body.config,
                    "default_agent_id": body.default_agent_id,
                    "default_user_profile_id": body.default_user_profile_id,
                    "inbound_policy": body.inbound_policy,
                    "egress_policy": body.egress_policy,
                    "created_at": now,
                    "updated_at": now,
                }
                (channels_dir / f"{channel_id}.json").write_text(json.dumps(channel_record))

            except ChannelInvalidRequestError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="channel.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "channel_id": channel_id,
                            "kind": body.kind,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = ChannelCreateError(
                    message=f"Failed to create channel: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="channel.create.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "channel_id": channel_id,
                            "kind": body.kind,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=channel_record, status_code=201)

    return router
