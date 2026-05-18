from __future__ import annotations

import json
import uuid
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
from pydantic import BaseModel


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class UserProfileCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="user_profile_create_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class UserProfileInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="user_profile_invalid_request", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class UserProfileCreateRequest(BaseModel):
    username: str
    display_name: str | None = None
    email: str | None = None
    metadata: dict[str, Any] | None = None


def _validate_request(body: UserProfileCreateRequest) -> UserProfileInvalidRequestError | None:
    if not body.username.strip():
        return UserProfileInvalidRequestError(
            message="'username' must not be empty",
            timestamp=_now(),
        )
    return None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_user_profiles_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    profiles_dir = storage_root / "user_profiles"

    @router.post("/v1/user_profiles", status_code=201)
    async def create_user_profile(body: UserProfileCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        user_id = f"user_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "user_profile.create",
            attributes={
                "user_profile.id": user_id,
                "user_profile.username": body.username,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="user_profile.create.invocation",
                    code="user_profile_create",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_request(body)
                if validation_err is not None:
                    raise validation_err

                profiles_dir.mkdir(parents=True, exist_ok=True)

                existing = list(profiles_dir.glob("*.json"))
                is_primary = len(existing) == 0

                profile_record: dict[str, Any] = {
                    "id": user_id,
                    "username": body.username,
                    "display_name": body.display_name,
                    "email": body.email,
                    "metadata": body.metadata,
                    "is_primary": is_primary,
                    "created_at": now,
                    "updated_at": now,
                }
                (profiles_dir / f"{user_id}.json").write_text(json.dumps(profile_record))

            except UserProfileInvalidRequestError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="user_profile.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "user_profile_id": user_id,
                            "username": body.username,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = UserProfileCreateError(
                    message=f"Failed to create user profile: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="user_profile.create.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "user_profile_id": user_id,
                            "username": body.username,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=profile_record, status_code=201)

    return router
